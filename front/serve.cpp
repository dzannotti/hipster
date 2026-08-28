// hipster-serve: OpenAI-compatible endpoints over either engine (one slot). Prefill in chunks (GEMM path), decode greedy
// with drafts (Flash-Next: the in-checkpoint MTP block; 27B: the DFlash2 block draft) — exact — or sampled (temperature > 0,
// plain decode). The engine is picked from the GGUF architecture.
//   hipster-serve --model <gguf> [--draft <dflash2.gguf>] [--port 8090] [--host 127.0.0.1] [--ctx 16384] [--chunk 2048] [--mtp 2] [--ndraft 7]
#include "../engine/src/qwen4exp.h"
#include "../engine/src/qwen35.h"
#include "tokenizer.h"
#include "chat.h"
#include "http.h"
#include "json.h"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <fstream>
#include <sstream>

namespace {
constexpr int N_VOCAB = 248320;
// token-level view of an engine for the serving loop: prompt in chunks, then either plain steps or spec rounds
struct Backend {
    virtual ~Backend() {}
    virtual void reset() = 0;
    virtual void prefill(const int* toks, int n, int pos0, bool spec) = 0;     // one chunk (+ draft catch-up when spec)
    virtual const float* logits() = 0;                                          // rows of the last pass
    virtual void topk(const float* l, int T, int k, int* ids, float* vals) = 0;
    virtual int max_draft() = 0;                                                // 0: no drafts
    virtual int draft(int id_last, int pos, int* out) = 0;                      // drafts after id_last at pos, returns their count
    virtual void verify(const int* batch, int T, int pos) = 0;                  // target pass over [id_last, drafts...]
    virtual void accept(int mm, const int* drafts, int pos) = 0;                // commit mm drafts + bonus, draft catch-up
    virtual void step(int id, int pos) = 0;                                     // plain: one token
    virtual size_t weight_bytes() = 0;
};
struct FnBackend : Backend {   // Flash-Next + MTP
    hip::Qwen4Exp m; int mtp; float* h_pending = nullptr; std::vector<int> ids = std::vector<int>(16); std::vector<float> vals = std::vector<float>(16);
    FnBackend(const std::string& path, int ctx, int chunk, int mtp_) : m(path, ctx, 1, chunk), mtp(mtp_) {
        if (mtp > 0 && !m.has_mtp()) { fprintf(stderr, "no MTP block in this GGUF: decoding without drafts\n"); mtp = 0; }
        hipMalloc(&h_pending, (size_t)hip::FnDims::wide * 4);
    }
    void reset() override { m.reset(); }
    void prefill(const int* t, int n, int p, bool spec) override { m.prefill(t, n, p); if (spec && mtp > 0) m.mtp_catchup(t, n, p); hipMemcpy(h_pending, m.h_nextn(), (size_t)hip::FnDims::wide * 4, hipMemcpyDeviceToDevice); }
    const float* logits() override { return m.logits(); }
    void topk(const float* l, int T, int k, int* i, float* v) override { m.topk(l, T, k, i, v); }
    int max_draft() override { return mtp; }
    int draft(int id_last, int pos, int* out) override {
        int tok = id_last; const float* h = h_pending;
        for (int i = 0; i < mtp; ++i) { m.mtp_forward(&tok, h, 1, pos + i); m.topk(m.mtp_logits(), 1, 1, ids.data(), vals.data()); tok = ids[0]; out[i] = tok; h = m.mtp_h(); }
        return mtp;
    }
    void verify(const int* b, int T, int pos) override { m.forward(b, T, pos); }
    void accept(int mm, const int* drafts, int pos) override {
        m.accept(mm + 1); if (mm > 0) m.mtp_forward(drafts, m.h_nextn(), mm, pos + 1);
        hipMemcpy(h_pending, m.h_nextn() + (size_t)mm * hip::FnDims::wide, (size_t)hip::FnDims::wide * 4, hipMemcpyDeviceToDevice);
    }
    void step(int id, int pos) override { m.forward(&id, 1, pos); m.accept(1); }
    size_t weight_bytes() override { return m.weight_bytes(); }
};
struct B27Backend : Backend {   // 27B + DFlash2
    hip::Qwen35 m; int nd;
    B27Backend(const std::string& path, const std::string& draft_path, int ctx, int nd_) : m(path, ctx, 1), nd(nd_) {
        if (!draft_path.empty()) m.load_dflash(draft_path); else { fprintf(stderr, "no --draft: 27B decodes without drafts\n"); nd = 0; }
    }
    void reset() override { m.reset(); }
    void prefill(const int* t, int n, int p, bool spec) override { m.forward(t, n, p); m.accept(n); if (spec && nd > 0) m.dflash_encode(n, p); }
    const float* logits() override { return m.logits(); }
    void topk(const float* l, int T, int k, int* i, float* v) override { m.topk(l, T, k, i, v); }
    int max_draft() override { return nd; }
    int draft(int id_last, int pos, int* out) override { return m.dflash_draft(id_last, pos, nd, out); }
    void verify(const int* b, int T, int pos) override { m.forward(b, T, pos); }
    void accept(int mm, const int*, int pos) override { m.accept(mm + 1); if (nd > 0) m.dflash_encode(mm + 1, pos); }
    void step(int id, int pos) override { m.forward(&id, 1, pos); m.accept(1); if (nd > 0) m.dflash_encode(1, pos); }
    size_t weight_bytes() override { return m.weight_bytes(); }
};
struct Params { int max_tokens = 4096; float temperature = 0.f, top_p = 1.f; int top_k = 0; bool stream = false; };
struct Server {
    Backend* m; tok::Tokenizer* T; int ctx, chunk, mtp; std::string model_name;
    int eos, im_end, think_open, think_close;
    std::mt19937 rng{42};
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    // greedy top-1 / sampled next token from the logits on device
    int pick(const float* dlogits, const Params& p, std::vector<float>& host) {
        if (p.temperature <= 0.f) { int id; float v; m->topk(dlogits, 1, 1, &id, &v); return id; }
        host.resize(N_VOCAB); hipMemcpy(host.data(), dlogits, (size_t)N_VOCAB * 4, hipMemcpyDeviceToHost);
        std::vector<int> idx(N_VOCAB); for (int i = 0; i < N_VOCAB; ++i) idx[i] = i;
        const int k = p.top_k > 0 ? std::min(p.top_k, N_VOCAB) : std::min(N_VOCAB, 4096);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return host[a] > host[b]; });
        std::vector<double> pr(k); double mx = host[idx[0]], s = 0; for (int i = 0; i < k; ++i) { pr[i] = exp((host[idx[i]] - mx) / p.temperature); s += pr[i]; }
        double cum = 0; int n = k; for (int i = 0; i < k; ++i) { cum += pr[i] / s; if (cum >= p.top_p) { n = i + 1; break; } }
        std::discrete_distribution<int> dist(pr.begin(), pr.begin() + n); return idx[dist(rng)];
    }
    // run the prompt, then generate; emit(text, is_reasoning) per decoded piece; returns {generated ids, finish reason}
    struct Result { std::vector<int> ids; std::string finish; int prompt_tokens = 0; double prefill_s = 0, gen_s = 0; int rounds = 0, accepted = 0, drafted = 0; };
    Result generate(std::vector<int> prompt, const Params& p, bool thinking_open, const std::function<void(const std::string&, bool)>& emit) {
        Result r; r.prompt_tokens = (int)prompt.size();
        if ((int)prompt.size() + 8 > ctx) throw std::runtime_error("prompt longer than the context (" + std::to_string(prompt.size()) + " > " + std::to_string(ctx - 8) + ")");
        m->reset();
        const double t0 = now();
        for (int c0 = 0; c0 < (int)prompt.size(); c0 += chunk) { const int n = std::min(chunk, (int)prompt.size() - c0); m->prefill(&prompt[c0], n, c0, mtp > 0 && p.temperature <= 0.f); }
        r.prefill_s = now() - t0;
        int pos = (int)prompt.size(); bool reasoning = thinking_open;
        std::string pending;   // incomplete UTF-8 tail
        auto out = [&](int id) {
            r.ids.push_back(id);
            if (id == think_close && reasoning) { reasoning = false; return; }
            if (id == think_open) { reasoning = true; return; }
            pending += T->piece(id);
            // flush complete UTF-8 sequences: keep an incomplete trailing multi-byte character for the next piece
            size_t ok = pending.size();
            { size_t p = pending.size(); int back = 0; while (p > 0 && back < 4 && ((unsigned char)pending[p - 1] & 0xC0) == 0x80) { --p; ++back; }   // p: last lead byte (or ascii)
              if (p > 0) { unsigned char c = pending[p - 1]; int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1; if (pending.size() - (p - 1) < (size_t)need) ok = p - 1; } }
            if (ok > 0) { emit(pending.substr(0, ok), reasoning); pending.erase(0, ok); }
        };
        std::vector<float> host; int id_last; { id_last = pick(m->logits(), p, host); }
        const double t1 = now();
        std::vector<int> ids(16 * 16); std::vector<float> vals(16 * 16); int drafts[16];
        auto stop = [&](int id) { return id == eos || id == im_end; };
        bool done = false;
        while (!done && (int)r.ids.size() < p.max_tokens) {
            if (stop(id_last)) { r.finish = "stop"; done = true; break; }
            out(id_last);
            if (mtp > 0 && p.temperature <= 0.f) {   // spec round: draft n, verify n+1, accept the matching prefix + bonus
                const int nd = m->draft(id_last, pos, drafts);
                std::vector<int> batch; batch.push_back(id_last); for (int i = 0; i < nd; ++i) batch.push_back(drafts[i]);
                const int Tn = (int)batch.size(); m->verify(batch.data(), Tn, pos); m->topk(m->logits(), Tn, 1, ids.data(), vals.data());
                int mm = 0; while (mm < nd && ids[mm] == drafts[mm] && !stop(drafts[mm])) ++mm;
                m->accept(mm, drafts, pos);
                for (int i = 0; i < mm; ++i) { if ((int)r.ids.size() >= p.max_tokens) { done = true; break; } out(drafts[i]); }
                id_last = ids[mm]; pos += mm + 1; ++r.rounds; r.drafted += nd; r.accepted += mm;
            } else {
                m->step(id_last, pos); ++pos; id_last = pick(m->logits(), p, host);
            }
        }
        if (!pending.empty()) emit(pending, reasoning);
        if (r.finish.empty()) r.finish = "length";
        r.gen_s = now() - t1;
        return r;
    }
};
std::string sse(const js::Value& v) { return "data: " + js::dump(v) + "\n\n"; }
const char* mime(const std::string& p) {
    auto ends = [&](const char* e) { size_t n = strlen(e); return p.size() >= n && p.compare(p.size() - n, n, e) == 0; };
    if (ends(".html")) return "text/html; charset=utf-8"; if (ends(".js")) return "application/javascript"; if (ends(".css")) return "text/css"; if (ends(".json") || ends(".webmanifest")) return "application/json";
    if (ends(".svg")) return "image/svg+xml"; if (ends(".png")) return "image/png"; if (ends(".ico")) return "image/x-icon"; if (ends(".woff2")) return "font/woff2"; if (ends(".woff")) return "font/woff"; return "application/octet-stream";
}
bool serve_static(const std::string& dir, std::string path, http::Response& res) {   // llama.cpp's web UI (SvelteKit SPA): files, else index.html
    if (dir.empty() || path.find("..") != std::string::npos) return false;
    size_t q = path.find('?'); if (q != std::string::npos) path = path.substr(0, q);
    if (path == "/" || path.empty()) path = "/index.html";
    std::ifstream f(dir + path, std::ios::binary);
    if (!f && path.find('.') == std::string::npos) { path = "/index.html"; f.open(dir + path, std::ios::binary); }
    if (!f) return false;
    std::stringstream b; b << f.rdbuf(); res.send(200, mime(path), b.str()); return true;
}
js::Value timings_json(int prompt_n, double prefill_s, int predicted_n, double gen_s) {
    js::Value t = js::Value::object(); t["cache_n"] = 0; t["prompt_n"] = prompt_n; t["prompt_ms"] = prefill_s * 1000; t["prompt_per_second"] = prefill_s > 0 ? prompt_n / prefill_s : 0.0;
    t["predicted_n"] = predicted_n; t["predicted_ms"] = gen_s * 1000; t["predicted_per_second"] = gen_s > 0 ? predicted_n / gen_s : 0.0; return t;
}
}  // namespace

int main(int argc, char** argv) {
    std::string model, draft, host = "127.0.0.1", ui_dir = "/models/.work/fn-tree/build/tools/ui/dist"; int port = 8090, ctx = 16384, chunk = 2048, mtp = 2, ndraft = 7;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto next = [&] { return std::string(argv[++i]); };
        if (a == "--model") model = next(); else if (a == "--draft") draft = next(); else if (a == "--port") port = atoi(next().c_str()); else if (a == "--host") host = next(); else if (a == "--ctx") ctx = atoi(next().c_str());
        else if (a == "--chunk") chunk = atoi(next().c_str()); else if (a == "--mtp") mtp = atoi(next().c_str()); else if (a == "--ndraft") ndraft = atoi(next().c_str()); else if (a == "--ui-dir") ui_dir = next(); }
    if (model.empty()) { fprintf(stderr, "usage: hipster-serve --model <gguf> [--draft <dflash2.gguf>] [--port 8090] [--host 127.0.0.1] [--ctx 16384] [--chunk 2048] [--mtp 2] [--ndraft 7]\n"); return 2; }
    std::string arch; { hip::GGUF g(model); arch = g.arch(); }
    const bool is27b = arch == "qwen35";
    if (is27b) chunk = std::min(chunk, hip::Qwen35Dims::max_prefill);
    Backend* be = is27b ? (Backend*)new B27Backend(model, draft, ctx, std::min(ndraft, 7)) : (Backend*)new FnBackend(model, ctx, chunk, mtp);
    mtp = be->max_draft();
    tok::Tokenizer* Tp; { hip::GGUF g(model); Tp = new tok::Tokenizer(g); } tok::Tokenizer& T = *Tp;
    Server S{be, &T, ctx, chunk, mtp, is27b ? "qwen3.8-27b" : "qwen3.8-flash-next", T.eos(), T.special("<|im_end|>"), T.special("<think>"), T.special("</think>")};
    {   // warm-up: the GEMM shapes of a short and a full chunk (hipBLASLt autotune), then reset
        std::vector<int> w(chunk, 1); be->reset(); be->prefill(w.data(), 128, 0, false); be->reset(); be->prefill(w.data(), chunk, 0, false); be->reset();
        be->step(1, 0); be->reset();
    }
    fprintf(stderr, "ready: %s, ctx %d, chunk %d, drafts %d, weights %.1f GiB\n", S.model_name.c_str(), ctx, chunk, mtp, be->weight_bytes() / 1073741824.0);
    http::serve(host, port, [&](const http::Request& req0, http::Response& res) {
        http::Request req = req0; { size_t q = req.path.find('?'); if (q != std::string::npos) req.path = req.path.substr(0, q); }   // route without the query string
        if (req.method == "OPTIONS") { res.send(200, "text/plain", ""); return; }
        if (req.path == "/health") { res.send(200, "application/json", "{\"status\":\"ok\"}"); return; }
        if (req.path == "/props") {   // what llama.cpp's web UI reads
            js::Value v = js::Value::object(); js::Value dgs = js::Value::object(); dgs["params"] = js::Value::object(); dgs["n_ctx"] = ctx; v["default_generation_settings"] = dgs;
            v["total_slots"] = 1; v["model_alias"] = S.model_name; v["model_ftype"] = "UD-Q4_K_XL"; v["model_path"] = model;
            js::Value mod = js::Value::object(); mod["vision"] = false; mod["video"] = false; mod["audio"] = false; v["modalities"] = mod;
            v["endpoint_slots"] = false; v["endpoint_props"] = true; v["endpoint_metrics"] = false; v["ui"] = true; v["ui_settings"] = js::Value::object();
            v["chat_template"] = ""; v["chat_template_caps"] = js::Value::object(); v["bos_token"] = ""; v["eos_token"] = "<|im_end|>"; v["build_info"] = "hipster"; v["is_sleeping"] = false; v["cors_proxy_enabled"] = false;
            res.send(200, "application/json", js::dump(v)); return; }
        static const char* api_paths[] = {"/slots", "/metrics", "/tokenize", "/detokenize", "/apply-template", "/completion", "/infill", "/embedding", "/embeddings", "/rerank", "/models", "/api/", "/props", "/health"};
        bool api = req.path.rfind("/v1", 0) == 0; for (const char* a : api_paths) if (req.path.rfind(a, 0) == 0) api = true;
        if (req.method == "GET" && !api && serve_static(ui_dir, req.path, res)) return;
        if (req.path == "/v1/models" || req.path == "/models") { js::Value v = js::Value::object(); v["object"] = "list"; js::Value md = js::Value::object(); md["id"] = S.model_name; md["object"] = "model"; md["owned_by"] = "hipster"; v["data"] = js::Value::array(); v["data"].push(md); res.send(200, "application/json", js::dump(v)); return; }
        const bool chat_ep = req.path == "/v1/chat/completions", comp_ep = req.path == "/v1/completions";
        if (!chat_ep && !comp_ep) { res.send(404, "application/json", "{\"error\":{\"message\":\"not found\"}}"); return; }
        js::Value body = js::parse(req.body);
        Params p; p.max_tokens = (int)body.num("max_tokens", body.num("max_completion_tokens", 4096)); p.temperature = (float)body.num("temperature", 0.0); p.top_p = (float)body.num("top_p", 1.0); p.top_k = (int)body.num("top_k", 0); p.stream = body.boolean("stream", false);
        chat::Opts opts; opts.enable_thinking = body.boolean("enable_thinking", true); if (auto* kw = body.get("chat_template_kwargs")) { opts.enable_thinking = kw->boolean("enable_thinking", opts.enable_thinking); opts.reasoning_effort = kw->str("reasoning_effort", opts.reasoning_effort); }
        opts.reasoning_effort = body.str("reasoning_effort", opts.reasoning_effort);
        std::string prompt_text; bool thinking_open = false;
        if (chat_ep) { const js::Value* msgs = body.get("messages"); if (!msgs || msgs->type != js::Value::ARR) throw std::runtime_error("messages missing"); prompt_text = chat::render(*msgs, body.get("tools"), opts); thinking_open = opts.enable_thinking; }
        else { const js::Value* pr = body.get("prompt"); if (!pr || pr->type != js::Value::STR) throw std::runtime_error("prompt must be a string"); prompt_text = pr->s; }
        const std::vector<int> prompt = T.encode(prompt_text, true);
        const std::string id = "chatcmpl-" + std::to_string((long)Server::now()); const long created = (long)Server::now();
        std::string content, reasoning;
        if (p.stream) {
            res.begin_stream();
            int ntok = 0; double t_first = 0, t_start = Server::now(); int prompt_n = (int)prompt.size();
            auto delta = [&](const std::string& text, bool is_reasoning) { js::Value v = js::Value::object(); v["id"] = id; v["object"] = chat_ep ? "chat.completion.chunk" : "text_completion"; v["created"] = (double)created; v["model"] = S.model_name;
                js::Value ch = js::Value::object(); ch["index"] = 0; ch["finish_reason"] = js::Value();
                if (chat_ep) { js::Value d = js::Value::object(); if (content.empty() && reasoning.empty()) d["role"] = "assistant"; d[is_reasoning ? "reasoning_content" : "content"] = text; ch["delta"] = d; } else ch["text"] = text;
                v["choices"] = js::Value::array(); v["choices"].push(ch);
                if (!t_first) t_first = Server::now(); ++ntok; v["timings"] = timings_json(prompt_n, t_first - t_start, ntok, Server::now() - t_first);
                res.chunk(sse(v)); };
            auto r = S.generate(prompt, p, thinking_open, [&](const std::string& t, bool rs) { (rs ? reasoning : content) += t; delta(t, rs); });
            js::Value fin = js::Value::object(); fin["id"] = id; fin["object"] = chat_ep ? "chat.completion.chunk" : "text_completion"; fin["created"] = (double)created; fin["model"] = S.model_name;
            js::Value ch = js::Value::object(); ch["index"] = 0; ch["finish_reason"] = r.finish; js::Value d = js::Value::object();
            if (chat_ep) { auto parsed = chat::parse_output((thinking_open ? reasoning + "</think>" : "") + content, thinking_open); if (!parsed.tool_calls.a.empty()) { d["tool_calls"] = parsed.tool_calls; ch["finish_reason"] = "tool_calls"; } ch["delta"] = d; } else ch["text"] = "";
            fin["choices"] = js::Value::array(); fin["choices"].push(ch);
            js::Value u = js::Value::object(); u["prompt_tokens"] = r.prompt_tokens; u["completion_tokens"] = (int)r.ids.size(); u["total_tokens"] = r.prompt_tokens + (int)r.ids.size(); fin["usage"] = u;
            fin["timings"] = timings_json(r.prompt_tokens, r.prefill_s, (int)r.ids.size(), r.gen_s);
            res.chunk(sse(fin)); res.chunk("data: [DONE]\n\n"); res.end();
            fprintf(stderr, "[%s] prompt %d tok in %.2f s (%.0f t/s) | gen %zu tok in %.2f s (%.1f t/s) | mtp rounds %d acc %d/%d | %s\n", chat_ep ? "chat" : "comp", r.prompt_tokens, r.prefill_s, r.prompt_tokens / r.prefill_s, r.ids.size(), r.gen_s, r.ids.size() / r.gen_s, r.rounds, r.accepted, r.drafted, r.finish.c_str());
        } else {
            auto r = S.generate(prompt, p, thinking_open, [&](const std::string& t, bool rs) { (rs ? reasoning : content) += t; });
            js::Value v = js::Value::object(); v["id"] = id; v["object"] = chat_ep ? "chat.completion" : "text_completion"; v["created"] = (double)created; v["model"] = S.model_name;
            js::Value ch = js::Value::object(); ch["index"] = 0; ch["finish_reason"] = r.finish;
            if (chat_ep) { auto parsed = chat::parse_output((thinking_open ? reasoning + "</think>" : "") + content, thinking_open); js::Value msg = js::Value::object(); msg["role"] = "assistant"; msg["content"] = parsed.content; if (!parsed.reasoning.empty()) msg["reasoning_content"] = parsed.reasoning; if (!parsed.tool_calls.a.empty()) { msg["tool_calls"] = parsed.tool_calls; ch["finish_reason"] = "tool_calls"; } ch["message"] = msg; }
            else ch["text"] = content;
            v["choices"] = js::Value::array(); v["choices"].push(ch);
            js::Value u = js::Value::object(); u["prompt_tokens"] = r.prompt_tokens; u["completion_tokens"] = (int)r.ids.size(); u["total_tokens"] = r.prompt_tokens + (int)r.ids.size(); v["usage"] = u;
            v["timings"] = timings_json(r.prompt_tokens, r.prefill_s, (int)r.ids.size(), r.gen_s);
            res.send(200, "application/json", js::dump(v));
            fprintf(stderr, "[%s] prompt %d tok in %.2f s (%.0f t/s) | gen %zu tok in %.2f s (%.1f t/s) | mtp rounds %d acc %d/%d | %s\n", chat_ep ? "chat" : "comp", r.prompt_tokens, r.prefill_s, r.prompt_tokens / r.prefill_s, r.ids.size(), r.gen_s, r.ids.size() / r.gen_s, r.rounds, r.accepted, r.drafted, r.finish.c_str());
        }
    });
    return 0;
}
