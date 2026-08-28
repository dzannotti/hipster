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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

namespace {
constexpr int N_VOCAB = 248320;
// token-level view of an engine for the serving loop: slots, prompt in chunks, then rounds of (draft, verify, accept)
struct Backend {
    virtual ~Backend() {}
    virtual int n_slots() = 0;
    virtual void reset_slot(int s) = 0;
    virtual const float* prefill(int s, const int* toks, int n, int pos0, bool spec) = 0;   // one chunk of slot s (+ draft catch-up when spec); returns the last row's logits
    virtual const float* logits() = 0;                                              // rows of the last pass, in request order
    virtual void topk(const float* l, int T, int k, int* ids, float* vals) = 0;
    virtual int max_draft() = 0;                                                    // 0: no drafts
    virtual int draft(const int* slots, const int* id_last, const int* pos, int S, int* out) = 0;   // out[s*max_draft + i]; returns the count per slot
    virtual void verify(const hip::SlotReq* reqs, int S) = 0;                       // one pass over the active slots
    virtual void accept(int s, int mm, const int* drafts, int pos) = 0;             // commit mm drafts + bonus on slot s, draft catch-up
    virtual size_t weight_bytes() = 0;
};
struct FnBackend : Backend {   // Flash-Next + MTP, one slot
    hip::Qwen4Exp m; int mtp; float* h_pending = nullptr; std::vector<int> ids = std::vector<int>(16); std::vector<float> vals = std::vector<float>(16);
    FnBackend(const std::string& path, int ctx, int chunk, int mtp_) : m(path, ctx, 1, chunk), mtp(mtp_) {
        if (mtp > 0 && !m.has_mtp()) { fprintf(stderr, "no MTP block in this GGUF: decoding without drafts\n"); mtp = 0; }
        hipMalloc(&h_pending, (size_t)hip::FnDims::wide * 4);
    }
    int n_slots() override { return 1; }
    void reset_slot(int) override { m.reset(); }
    const float* prefill(int, const int* t, int n, int p, bool spec) override { m.prefill(t, n, p); if (spec && mtp > 0) m.mtp_catchup(t, n, p); hipMemcpy(h_pending, m.h_nextn(), (size_t)hip::FnDims::wide * 4, hipMemcpyDeviceToDevice); return m.logits(); }
    const float* logits() override { return m.logits(); }
    void topk(const float* l, int T, int k, int* i, float* v) override { m.topk(l, T, k, i, v); }
    int max_draft() override { return mtp; }
    int draft(const int*, const int* id_last, const int* pos, int, int* out) override {
        int tok = id_last[0]; const float* h = h_pending;
        for (int i = 0; i < mtp; ++i) { m.mtp_forward(&tok, h, 1, pos[0] + i); m.topk(m.mtp_logits(), 1, 1, ids.data(), vals.data()); tok = ids[0]; out[i] = tok; h = m.mtp_h(); }
        return mtp;
    }
    void verify(const hip::SlotReq* r, int) override { m.forward(r[0].tokens, r[0].T, r[0].pos); }
    void accept(int, int mm, const int* drafts, int pos) override {
        m.accept(mm + 1); if (mm > 0) m.mtp_forward(drafts, m.h_nextn(), mm, pos + 1);
        hipMemcpy(h_pending, m.h_nextn() + (size_t)mm * hip::FnDims::wide, (size_t)hip::FnDims::wide * 4, hipMemcpyDeviceToDevice);
    }
    size_t weight_bytes() override { return m.weight_bytes(); }
};
struct B27Backend : Backend {   // 27B + DFlash2, n slots (rows per pass <= 16)
    hip::Qwen35 m; int nd, ns;
    B27Backend(const std::string& path, const std::string& draft_path, int ctx, int nd_, int slots) : m(path, ctx, slots), nd(nd_), ns(slots) {
        if (!draft_path.empty()) m.load_dflash(draft_path); else { fprintf(stderr, "no --draft: 27B decodes without drafts\n"); nd = 0; }
    }
    int n_slots() override { return ns; }
    void reset_slot(int s) override { m.reset_slot(s); }
    const float* prefill(int s, const int* t, int n, int p, bool spec) override {
        if (n <= 96) {   // short prompt: 16-row GEMV passes (a GEMM-path chunk costs ~1 s of weight dequantisation whatever its length)
            int k = 0;
            for (int c0 = 0; c0 < n; c0 += hip::Qwen35::MAXR) { k = std::min(hip::Qwen35::MAXR, n - c0); hip::SlotReq r{s, t + c0, k, p + c0}; m.forward(&r, 1); m.accept(s, k); if (spec && nd > 0) m.dflash_encode(s, k, p + c0); }
            return m.logits() + (size_t)(k - 1) * N_VOCAB;   // the GEMV path keeps every row's logits
        }
        hip::SlotReq r{s, t, n, p}; m.forward(&r, 1); m.accept(s, n); if (spec && nd > 0) m.dflash_encode(s, n, p);
        return m.logits();
    }
    const float* logits() override { return m.logits(); }
    void topk(const float* l, int T, int k, int* i, float* v) override { m.topk(l, T, k, i, v); }
    int max_draft() override { return nd; }
    int draft(const int* slots, const int* id_last, const int* pos, int S, int* out) override { return m.dflash_draft_b(slots, id_last, pos, S, nd, out); }
    void verify(const hip::SlotReq* r, int S) override { m.forward(r, S); }
    void accept(int s, int mm, const int*, int pos) override { m.accept(s, mm + 1); if (nd > 0) m.dflash_encode(s, mm + 1, pos); }
    size_t weight_bytes() override { return m.weight_bytes(); }
};
struct Params { int max_tokens = 4096; float temperature = 0.f, top_p = 1.f; int top_k = 0; bool stream = false; };
struct Result { std::vector<int> ids; std::string finish; int prompt_tokens = 0; double prefill_s = 0, gen_s = 0; int rounds = 0, accepted = 0, drafted = 0; };
struct Job {   // one request in the scheduler
    std::vector<int> prompt; Params p; bool reasoning; std::function<void(const std::string&, bool)> emit;
    Result r; std::string pending; int pos = 0, id_last = 0, slot = -1; bool spec = false; double t_gen0 = 0;
    std::mutex mu; std::condition_variable cv; bool done = false;
};
struct Server {
    Backend* m; tok::Tokenizer* T; int ctx, chunk, mtp; std::string model_name;
    int eos, im_end, think_open, think_close;
    std::mt19937 rng{42};
    std::mutex qmu; std::condition_variable qcv; std::deque<Job*> queue; std::vector<Job*> active;
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    // greedy top-1 / sampled next token from a logits row on device
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
    bool stop(int id) const { return id == eos || id == im_end; }
    void out(Job& j, int id) {   // emit one token: thinking tags, UTF-8 boundary handling
        j.r.ids.push_back(id);
        if (id == think_close && j.reasoning) { j.reasoning = false; return; }
        if (id == think_open) { j.reasoning = true; return; }
        j.pending += T->piece(id);
        size_t ok = j.pending.size();
        { size_t p = j.pending.size(); int back = 0; while (p > 0 && back < 4 && ((unsigned char)j.pending[p - 1] & 0xC0) == 0x80) { --p; ++back; }
          if (p > 0) { unsigned char c = j.pending[p - 1]; int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1; if (j.pending.size() - (p - 1) < (size_t)need) ok = p - 1; } }
        if (ok > 0) { j.emit(j.pending.substr(0, ok), j.reasoning); j.pending.erase(0, ok); }
    }
    void finish(Job& j) {
        if (!j.pending.empty()) { j.emit(j.pending, j.reasoning); j.pending.clear(); }
        if (j.r.finish.empty()) j.r.finish = "length";
        j.r.gen_s = now() - j.t_gen0;
        m->reset_slot(j.slot);
        { std::lock_guard<std::mutex> lk(j.mu); j.done = true; } j.cv.notify_all();
    }
    // request entry: queue the job and wait for the scheduler thread to finish it
    Result generate(std::vector<int> prompt, const Params& p, bool thinking_open, const std::function<void(const std::string&, bool)>& emit) {
        if ((int)prompt.size() + 8 > ctx) throw std::runtime_error("prompt longer than the context (" + std::to_string(prompt.size()) + " > " + std::to_string(ctx - 8) + ")");
        Job j; j.prompt = std::move(prompt); j.p = p; j.reasoning = thinking_open; j.emit = emit; j.r.prompt_tokens = (int)j.prompt.size();
        j.spec = mtp > 0 && p.temperature <= 0.f;
        { std::lock_guard<std::mutex> lk(qmu); queue.push_back(&j); } qcv.notify_one();
        std::unique_lock<std::mutex> lk(j.mu); j.cv.wait(lk, [&] { return j.done; });
        return j.r;
    }
    // the scheduler: admit queued jobs into free slots (prefill), then lockstep rounds over the active ones
    void run() {
        std::vector<float> host; std::vector<int> ids(16 * 16); std::vector<float> vals(16 * 16);
        const int ns = m->n_slots(), nd = mtp;
        std::vector<bool> used(ns, false);
        while (true) {
            {   std::unique_lock<std::mutex> lk(qmu);
                qcv.wait(lk, [&] { return !queue.empty() || !active.empty(); });
                while (!queue.empty() && (int)active.size() < ns) {
                    Job* j = queue.front(); queue.pop_front(); int s = 0; while (used[s]) ++s; used[s] = true; j->slot = s; active.push_back(j);
                    lk.unlock();
                    const double t0 = now(); m->reset_slot(s); const float* last = nullptr;
                    for (int c0 = 0; c0 < (int)j->prompt.size(); c0 += chunk) { const int n = std::min(chunk, (int)j->prompt.size() - c0); last = m->prefill(s, &j->prompt[c0], n, c0, j->spec); }
                    j->r.prefill_s = now() - t0; j->pos = (int)j->prompt.size(); j->id_last = pick(last, j->p, host); j->t_gen0 = now();
                    lk.lock();
                }
            }
            // one round over the active slots
            std::vector<Job*> spec_jobs; for (Job* j : active) if (j->spec) spec_jobs.push_back(j);
            std::vector<int> drafts((size_t)ns * 8, 0), ndr(ns, 0);
            if (!spec_jobs.empty()) {
                std::vector<int> sl, il, pl; for (Job* j : spec_jobs) { sl.push_back(j->slot); il.push_back(j->id_last); pl.push_back(j->pos); }
                std::vector<int> dr((size_t)spec_jobs.size() * nd);
                const int k = m->draft(sl.data(), il.data(), pl.data(), (int)spec_jobs.size(), dr.data());
                for (size_t q = 0; q < spec_jobs.size(); ++q) { ndr[spec_jobs[q]->slot] = k; for (int i = 0; i < k; ++i) drafts[(size_t)spec_jobs[q]->slot * 8 + i] = dr[q * nd + i]; }
            }
            std::vector<std::vector<int>> batch(active.size()); std::vector<hip::SlotReq> reqs; int rows = 0;
            for (size_t q = 0; q < active.size(); ++q) { Job* j = active[q]; batch[q].push_back(j->id_last); for (int i = 0; i < ndr[j->slot]; ++i) batch[q].push_back(drafts[(size_t)j->slot * 8 + i]);
                reqs.push_back({j->slot, batch[q].data(), (int)batch[q].size(), j->pos}); rows += (int)batch[q].size(); }
            m->verify(reqs.data(), (int)reqs.size());
            bool any_greedy = false; for (Job* j : active) any_greedy |= j->p.temperature <= 0.f;
            if (any_greedy) m->topk(m->logits(), rows, 1, ids.data(), vals.data());
            int r0 = 0; std::vector<Job*> still;
            for (size_t q = 0; q < active.size(); ++q) {
                Job* j = active[q]; const int T = reqs[q].T; const int* dr = &drafts[(size_t)j->slot * 8];
                if (stop(j->id_last)) { j->r.finish = "stop"; m->accept(j->slot, 0, dr, j->pos); finish(*j); r0 += T; continue; }
                out(*j, j->id_last);
                int mm = 0; while (mm < ndr[j->slot] && ids[r0 + mm] == dr[mm] && !stop(dr[mm])) ++mm;
                m->accept(j->slot, mm, dr, j->pos);
                bool full = false;
                for (int i = 0; i < mm; ++i) { if ((int)j->r.ids.size() >= j->p.max_tokens) { full = true; break; } out(*j, dr[i]); }
                j->id_last = j->p.temperature <= 0.f ? ids[r0 + mm] : pick(m->logits() + (size_t)(r0 + mm) * N_VOCAB, j->p, host);
                j->pos += mm + 1; ++j->r.rounds; j->r.drafted += ndr[j->slot]; j->r.accepted += mm; r0 += T;
                if (full || (int)j->r.ids.size() >= j->p.max_tokens) finish(*j); else still.push_back(j);
            }
            { std::lock_guard<std::mutex> lk(qmu); for (Job* j : active) if (std::find(still.begin(), still.end(), j) == still.end()) used[j->slot] = false; active = still; }
        }
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
    std::string model, draft, host = "127.0.0.1", ui_dir = "/models/.work/fn-tree/build/tools/ui/dist"; int port = 8090, ctx = 16384, chunk = 2048, mtp = 2, ndraft = 7, slots = 1;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto next = [&] { return std::string(argv[++i]); };
        if (a == "--model") model = next(); else if (a == "--draft") draft = next(); else if (a == "--port") port = atoi(next().c_str()); else if (a == "--host") host = next(); else if (a == "--ctx") ctx = atoi(next().c_str());
        else if (a == "--chunk") chunk = atoi(next().c_str()); else if (a == "--mtp") mtp = atoi(next().c_str()); else if (a == "--ndraft") ndraft = atoi(next().c_str()); else if (a == "--slots") slots = atoi(next().c_str()); else if (a == "--ui-dir") ui_dir = next(); }
    if (model.empty()) { fprintf(stderr, "usage: hipster-serve --model <gguf> [--draft <dflash2.gguf>] [--port 8090] [--host 127.0.0.1] [--ctx 16384] [--chunk 2048] [--mtp 2] [--ndraft 7] [--slots 2]\n"); return 2; }
    std::string arch; { hip::GGUF g(model); arch = g.arch(); }
    const bool is27b = arch == "qwen35";
    if (is27b) chunk = std::min(chunk, hip::Qwen35Dims::max_prefill);
    if (!is27b && slots != 1) { fprintf(stderr, "Flash-Next serving is one slot for now\n"); slots = 1; }
    Backend* be = is27b ? (Backend*)new B27Backend(model, draft, ctx, std::min(ndraft, 7), std::max(1, std::min(slots, 2))) : (Backend*)new FnBackend(model, ctx, chunk, mtp);
    mtp = be->max_draft();
    tok::Tokenizer* Tp; { hip::GGUF g(model); Tp = new tok::Tokenizer(g); } tok::Tokenizer& T = *Tp;
    Server S{be, &T, ctx, chunk, mtp, is27b ? "qwen3.8-27b" : "qwen3.8-flash-next", T.eos(), T.special("<|im_end|>"), T.special("<think>"), T.special("</think>")};
    {   // warm-up: the GEMM shapes of a short and a full chunk (hipBLASLt autotune), then reset
        std::vector<int> w(chunk, 1); be->reset_slot(0); be->prefill(0, w.data(), 128, 0, false); be->reset_slot(0); be->prefill(0, w.data(), chunk, 0, false); be->reset_slot(0);
        be->prefill(0, w.data(), 40, 0, false); be->reset_slot(0);
        hip::SlotReq r{0, w.data(), 1, 0}; be->verify(&r, 1); be->accept(0, 0, nullptr, 0); be->reset_slot(0);
    }
    std::thread([&S] { S.run(); }).detach();
    fprintf(stderr, "ready: %s, ctx %d, chunk %d, drafts %d, slots %d, weights %.1f GiB\n", S.model_name.c_str(), ctx, chunk, mtp, be->n_slots(), be->weight_bytes() / 1073741824.0);
    http::serve(host, port, [&](const http::Request& req0, http::Response& res) {
        http::Request req = req0; { size_t q = req.path.find('?'); if (q != std::string::npos) req.path = req.path.substr(0, q); }   // route without the query string
        if (req.method == "OPTIONS") { res.send(200, "text/plain", ""); return; }
        if (req.path == "/health") { res.send(200, "application/json", "{\"status\":\"ok\"}"); return; }
        if (req.path == "/props") {   // what llama.cpp's web UI reads
            js::Value v = js::Value::object(); js::Value dgs = js::Value::object(); dgs["params"] = js::Value::object(); dgs["n_ctx"] = ctx; v["default_generation_settings"] = dgs;
            v["total_slots"] = be->n_slots(); v["model_alias"] = S.model_name; v["model_ftype"] = "UD-Q4_K_XL"; v["model_path"] = model;
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
