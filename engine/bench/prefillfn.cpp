// prefillfn <gguf> <ref.json> [n_gen] [chunk] [sizes...] : chunked one-shot prefill of the reference prompt vs the
// token-by-token path (last-row logits, greedy continuation), then steady-state prefill throughput.
#include "../src/qwen4exp.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <hip/hip_runtime.h>

static std::vector<int> parse_ids(const std::string& j, const std::string& key) {
    std::vector<int> v; auto p = j.find("\"" + key + "\""); if (p == std::string::npos) return v;
    p = j.find('[', p); auto e = j.find(']', p);
    std::string s = j.substr(p + 1, e - p - 1); std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(atoi(tok.c_str()));
    return v;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 3) { fprintf(stderr, "usage: %s gguf ref.json [n_gen] [chunk] [sizes...]\n", argv[0]); return 2; }
    std::ifstream f(argv[2]); std::stringstream buf; buf << f.rdbuf();
    auto prompt = parse_ids(buf.str(), "prompt_ids");
    if (getenv("HIPSTER_PROMPT_CUT")) prompt.resize(std::min<size_t>(prompt.size(), atoi(getenv("HIPSTER_PROMPT_CUT"))));   // diff tests on short prompts
    const int n_gen = argc > 3 ? atoi(argv[3]) : 12, chunk = argc > 4 ? atoi(argv[4]) : 2048;
    std::vector<int> sizes; for (int i = 5; i < argc; ++i) sizes.push_back(atoi(argv[i])); if (sizes.empty()) sizes = {512, 1024, 2048};
    const int max_ctx = std::max(8192, (int)((prompt.size() + n_gen + 1023) / 1024 * 1024));
    hip::Qwen4Exp m(argv[1], max_ctx, 1, chunk);
    const int V = hip::FnDims::n_vocab;
    int ids[16]; float vals[16];
    auto greedy = [&](int pos, int n) { std::vector<int> out; m.topk(m.logits(), 1, 1, ids, vals); int cur = ids[0];
        for (int i = 0; i < n; ++i) { out.push_back(cur); m.forward(&cur, 1, pos++); m.accept(1); m.topk(m.logits(), 1, 1, ids, vals); cur = ids[0]; } return out; };

    // reference: token by token (HIPSTER_NO_DECODE_REF=1 skips it: 16K tokens take 10 minutes)
    std::vector<float> lref(V, 0.f); std::vector<int> gref(n_gen, -1);
    if (!getenv("HIPSTER_NO_DECODE_REF")) {
        m.reset(); { m.dbg_arm(true); int pos = 0; for (int t : prompt) { m.forward(&t, 1, pos++); m.accept(1); } m.dbg_arm(false); }
        hipMemcpy(lref.data(), m.logits(), (size_t)V * 4, hipMemcpyDeviceToHost);
        gref = greedy((int)prompt.size(), n_gen);
    }
    // chunked prefill
    m.reset(); double tp = 0;
    for (int c0 = 0; c0 < (int)prompt.size(); c0 += chunk) { const int n = std::min(chunk, (int)prompt.size() - c0); tp += m.prefill(&prompt[c0], n, c0); }
    std::vector<float> lpf(V); hipMemcpy(lpf.data(), m.logits(), (size_t)V * 4, hipMemcpyDeviceToHost);
    {   // vs llama.cpp's top-10 logprobs (docs/ref/*.json "first_top10": [[id, "tok", logprob], ...])
        const std::string& j = buf.str(); std::vector<std::pair<int, float>> ref_top;
        std::regex re(R"(\[\s*(\d+),\s*\"(?:[^\"\\]|\\.)*\",\s*(-?[0-9.eE+-]+)\s*\])"); auto p = j.find("first_top10");
        if (p != std::string::npos) for (auto it = std::sregex_iterator(j.begin() + p, j.end(), re); it != std::sregex_iterator(); ++it) ref_top.push_back({atoi((*it)[1].str().c_str()), (float)atof((*it)[2].str().c_str())});
        float mx = -1e30f; for (float v : lpf) mx = fmaxf(mx, v); double se = 0; for (float v : lpf) se += exp((double)v - mx); const double lse = mx + log(se);
        m.topk(m.logits(), 1, 10, ids, vals); double maxdiff = 0; int nm = 0;
        for (int i = 0; i < 10; ++i) { float rlp = NAN; for (auto& r : ref_top) if (r.first == ids[i]) rlp = r.second; if (!std::isnan(rlp)) { maxdiff = fmax(maxdiff, fabs(vals[i] - lse - rlp)); ++nm; } }
        printf("prefill last-row logits vs llama.cpp: max |logprob diff| over %d matched top-10 ids = %.4f\n", nm, maxdiff);
        float mxr = -1e30f; for (float v : lref) mxr = fmaxf(mxr, v); double ser = 0; for (float v : lref) ser += exp((double)v - mxr); const double lser = mxr + log(ser);
        printf("  id      prefill   decode   llama.cpp (logprob)\n");
        for (int i = 0; i < 10; ++i) { float rlp = NAN; for (auto& r : ref_top) if (r.first == ids[i]) rlp = r.second; printf("  %7d %8.4f %8.4f %8.4f\n", ids[i], vals[i] - lse, lref[ids[i]] - lser, rlp); }
    }
    int a1 = 0, a2 = 0; double md = 0; for (int v = 0; v < V; ++v) { if (lref[v] > lref[a1]) a1 = v; if (lpf[v] > lpf[a2]) a2 = v; md = fmax(md, fabs((double)lref[v] - lpf[v])); }
    m.dbg_report();
    auto gpf = greedy((int)prompt.size(), n_gen);
    int match = 0; for (int i = 0; i < n_gen; ++i) if (gref[i] == gpf[i]) ++match;
    { auto rg = parse_ids(buf.str(), "gen_ids"); int mr = 0; for (size_t i = 0; i < rg.size() && i < gpf.size(); ++i) if (rg[i] == gpf[i]) ++mr; else break;
      printf("vs llama.cpp continuation (%zu ref tokens): first %d identical\n", rg.size(), mr); }
    printf("prefill of the %zu-token prompt in chunks of %d: %.1f ms; last-row top-1 %s (%d vs %d), max |logit diff| %.3f; greedy continuation %d/%d identical\n",
           prompt.size(), chunk, tp, a1 == a2 ? "identical" : "DIFFERENT", a1, a2, md, match, n_gen);
    // throughput: synthetic prompt (the reference prompt repeated), warm-up run per size (hipBLASLt autotune), then 2 timed runs
    for (int P : sizes) {
        if (P > chunk) continue;
        std::vector<int> toks(P); for (int i = 0; i < P; ++i) toks[i] = prompt[i % prompt.size()];
        m.reset(); m.prefill(toks.data(), P, 0);
        double best = 1e30; for (int r = 0; r < 2; ++r) { m.reset(); const double t0 = now_ms(); m.prefill(toks.data(), P, 0); best = fmin(best, now_ms() - t0); }
        printf("[t=%.0f s] prefill %5d tokens: %7.0f ms = %6.0f t/s (PLE host gather %.0f ms cumulative)\n", now_ms() / 1000.0, P, best, P * 1000.0 / best, m.ple_host_ms());
        hip::prefill_timing_report(3.0 * P);
    }
    return match == n_gen ? 0 : 1;
}
