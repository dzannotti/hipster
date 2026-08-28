// hipster-serve: OpenAI-compatible endpoints over the Flash-Next engine (one slot). Prefill in chunks (GEMM path),
// decode greedy with the in-checkpoint MTP drafts (exact) or sampled (temperature > 0, plain decode).
//   hipster-serve --model <shard1.gguf> [--port 8090] [--host 127.0.0.1] [--ctx 16384] [--chunk 2048] [--mtp 2]
#include "../engine/src/qwen4exp.h"
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

namespace {
using D = hip::FnDims;
struct Params { int max_tokens = 4096; float temperature = 0.f, top_p = 1.f; int top_k = 0; bool stream = false; };
struct Server {
    hip::Qwen4Exp* m; tok::Tokenizer* T; int ctx, chunk, mtp; std::string model_name;
    int eos, im_end, think_open, think_close;
    std::mt19937 rng{42};
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    // greedy top-1 / sampled next token from the logits on device
    int pick(const float* dlogits, const Params& p, std::vector<float>& host) {
        if (p.temperature <= 0.f) { int id; float v; m->topk(dlogits, 1, 1, &id, &v); return id; }
        host.resize(D::n_vocab); hipMemcpy(host.data(), dlogits, (size_t)D::n_vocab * 4, hipMemcpyDeviceToHost);
        std::vector<int> idx(D::n_vocab); for (int i = 0; i < D::n_vocab; ++i) idx[i] = i;
        const int k = p.top_k > 0 ? std::min(p.top_k, D::n_vocab) : std::min(D::n_vocab, 4096);
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
        for (int c0 = 0; c0 < (int)prompt.size(); c0 += chunk) { const int n = std::min(chunk, (int)prompt.size() - c0); m->prefill(&prompt[c0], n, c0); if (mtp > 0 && p.temperature <= 0.f) m->mtp_catchup(&prompt[c0], n, c0); }
        r.prefill_s = now() - t0;
        int pos = (int)prompt.size(); bool reasoning = thinking_open;
        std::string pending;   // incomplete UTF-8 tail
        auto out = [&](int id) {
            r.ids.push_back(id);
            if (id == think_close && reasoning) { reasoning = false; return; }
            if (id == think_open) { reasoning = true; return; }
            pending += T->piece(id);
            // flush complete UTF-8 sequences
            size_t ok = pending.size(); while (ok > 0) { unsigned char c = pending[ok - 1]; if ((c & 0xC0) != 0x80) { int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1; if (pending.size() - (ok - 1) < (size_t)need) ok -= 1; break; } if (pending.size() - ok >= 3) break; --ok; }
            if (ok > 0) { emit(pending.substr(0, ok), reasoning); pending.erase(0, ok); }
        };
        std::vector<float> host; int id_last; { id_last = pick(m->logits(), p, host); }
        const double t1 = now();
        float* h_pending = nullptr; hipMalloc(&h_pending, (size_t)D::wide * 4); hipMemcpy(h_pending, m->h_nextn(), (size_t)D::wide * 4, hipMemcpyDeviceToDevice);
        std::vector<int> ids(16 * 8); std::vector<float> vals(16 * 8);
        auto stop = [&](int id) { return id == eos || id == im_end; };
        bool done = false;
        while (!done && (int)r.ids.size() < p.max_tokens) {
            if (stop(id_last)) { r.finish = "stop"; done = true; break; }
            out(id_last);
            if (mtp > 0 && p.temperature <= 0.f) {   // MTP round: draft n, verify n+1, accept the matching prefix + bonus
                std::vector<int> drafts; int tok = id_last; const float* h = h_pending; int pp = pos;
                for (int i = 0; i < mtp; ++i) { m->mtp_forward(&tok, h, 1, pp); m->topk(m->mtp_logits(), 1, 1, ids.data(), vals.data()); tok = ids[0]; drafts.push_back(tok); h = m->mtp_h(); ++pp; }
                std::vector<int> batch; batch.push_back(id_last); for (int d : drafts) batch.push_back(d);
                const int Tn = (int)batch.size(); m->forward(batch.data(), Tn, pos); m->topk(m->logits(), Tn, 1, ids.data(), vals.data());
                int mm = 0; while (mm < (int)drafts.size() && ids[mm] == drafts[mm] && !stop(drafts[mm])) ++mm;
                m->accept(mm + 1);
                for (int i = 0; i < mm; ++i) { if ((int)r.ids.size() >= p.max_tokens) { done = true; break; } out(drafts[i]); }
                if (mm > 0) m->mtp_forward(drafts.data(), m->h_nextn(), mm, pos + 1);
                hipMemcpy(h_pending, m->h_nextn() + (size_t)mm * D::wide, (size_t)D::wide * 4, hipMemcpyDeviceToDevice);
                id_last = ids[mm]; pos += mm + 1; ++r.rounds; r.drafted += mtp; r.accepted += mm;
            } else {
                m->forward(&id_last, 1, pos); m->accept(1); ++pos; id_last = pick(m->logits(), p, host);
            }
        }
        hipFree(h_pending);
        if (!pending.empty()) emit(pending, reasoning);
        if (r.finish.empty()) r.finish = "length";
        r.gen_s = now() - t1;
        return r;
    }
};
std::string sse(const js::Value& v) { return "data: " + js::dump(v) + "\n\n"; }
}  // namespace

int main(int argc, char** argv) {
    std::string model, host = "127.0.0.1"; int port = 8090, ctx = 16384, chunk = 2048, mtp = 2;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto next = [&] { return std::string(argv[++i]); };
        if (a == "--model") model = next(); else if (a == "--port") port = atoi(next().c_str()); else if (a == "--host") host = next(); else if (a == "--ctx") ctx = atoi(next().c_str()); else if (a == "--chunk") chunk = atoi(next().c_str()); else if (a == "--mtp") mtp = atoi(next().c_str()); }
    if (model.empty()) { fprintf(stderr, "usage: hipster-serve --model <shard1.gguf> [--port 8090] [--host 127.0.0.1] [--ctx 16384] [--chunk 2048] [--mtp 2]\n"); return 2; }
    hip::Qwen4Exp m(model, ctx, 1, chunk);
    tok::Tokenizer T(m.gguf());
    if (mtp > 0 && !m.has_mtp()) { fprintf(stderr, "no MTP block in this GGUF: decoding without drafts\n"); mtp = 0; }
    Server S{&m, &T, ctx, chunk, mtp, "qwen3.8-flash-next", T.eos(), T.special("<|im_end|>"), T.special("<think>"), T.special("</think>")};
    {   // warm-up: the GEMM shapes of a short and a full chunk (hipBLASLt autotune), then reset
        std::vector<int> w(chunk, 1); m.reset(); m.prefill(w.data(), 128, 0); m.reset(); m.prefill(w.data(), chunk, 0); m.reset();
        std::vector<int> one(1, 1); m.forward(one.data(), 1, 0); m.accept(1); m.reset();
    }
    fprintf(stderr, "ready: ctx %d, chunk %d, mtp %d, weights %.1f GiB\n", ctx, chunk, mtp, m.weight_bytes() / 1073741824.0);
    http::serve(host, port, [&](const http::Request& req, http::Response& res) {
        if (req.method == "OPTIONS") { res.send(200, "text/plain", ""); return; }
        if (req.path == "/health") { res.send(200, "application/json", "{\"status\":\"ok\"}"); return; }
        if (req.path == "/v1/models") { js::Value v = js::Value::object(); v["object"] = "list"; js::Value md = js::Value::object(); md["id"] = S.model_name; md["object"] = "model"; md["owned_by"] = "hipster"; v["data"] = js::Value::array(); v["data"].push(md); res.send(200, "application/json", js::dump(v)); return; }
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
            auto delta = [&](const std::string& text, bool is_reasoning) { js::Value v = js::Value::object(); v["id"] = id; v["object"] = chat_ep ? "chat.completion.chunk" : "text_completion"; v["created"] = (double)created; v["model"] = S.model_name;
                js::Value ch = js::Value::object(); ch["index"] = 0; ch["finish_reason"] = js::Value();
                if (chat_ep) { js::Value d = js::Value::object(); if (content.empty() && reasoning.empty()) d["role"] = "assistant"; d[is_reasoning ? "reasoning_content" : "content"] = text; ch["delta"] = d; } else ch["text"] = text;
                v["choices"] = js::Value::array(); v["choices"].push(ch); res.chunk(sse(v)); };
            auto r = S.generate(prompt, p, thinking_open, [&](const std::string& t, bool rs) { (rs ? reasoning : content) += t; delta(t, rs); });
            js::Value fin = js::Value::object(); fin["id"] = id; fin["object"] = chat_ep ? "chat.completion.chunk" : "text_completion"; fin["created"] = (double)created; fin["model"] = S.model_name;
            js::Value ch = js::Value::object(); ch["index"] = 0; ch["finish_reason"] = r.finish; js::Value d = js::Value::object();
            if (chat_ep) { auto parsed = chat::parse_output((thinking_open ? reasoning + "</think>" : "") + content, thinking_open); if (!parsed.tool_calls.a.empty()) { d["tool_calls"] = parsed.tool_calls; ch["finish_reason"] = "tool_calls"; } ch["delta"] = d; } else ch["text"] = "";
            fin["choices"] = js::Value::array(); fin["choices"].push(ch);
            js::Value u = js::Value::object(); u["prompt_tokens"] = r.prompt_tokens; u["completion_tokens"] = (int)r.ids.size(); u["total_tokens"] = r.prompt_tokens + (int)r.ids.size(); fin["usage"] = u;
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
            res.send(200, "application/json", js::dump(v));
            fprintf(stderr, "[%s] prompt %d tok in %.2f s (%.0f t/s) | gen %zu tok in %.2f s (%.1f t/s) | mtp rounds %d acc %d/%d | %s\n", chat_ep ? "chat" : "comp", r.prompt_tokens, r.prefill_s, r.prompt_tokens / r.prefill_s, r.ids.size(), r.gen_s, r.ids.size() / r.gen_s, r.rounds, r.accepted, r.drafted, r.finish.c_str());
        }
    });
    return 0;
}
