// prefill27b <gguf> <ref.json> [T...] : one-shot prefill correctness vs the reference, then prefill
// throughput on synthetic prompts of T tokens (chunks of <= 2048).
#include "../src/qwen35.h"
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
    if (argc < 3) { fprintf(stderr, "usage: %s gguf ref.json [T...]\n", argv[0]); return 2; }
    std::ifstream f(argv[2]); std::stringstream buf; buf << f.rdbuf(); const std::string j = buf.str();
    auto prompt = parse_ids(j, "prompt_ids"); auto ref_gen = parse_ids(j, "gen_ids");
    std::vector<std::pair<int, float>> ref_top;
    { std::regex re(R"(\[\s*(\d+),\s*\"(?:[^\"\\]|\\.)*\",\s*(-?[0-9.eE+-]+)\s*\])"); auto p = j.find("first_top10");
      for (auto it = std::sregex_iterator(j.begin() + p, j.end(), re); it != std::sregex_iterator(); ++it) ref_top.push_back({atoi((*it)[1].str().c_str()), (float)atof((*it)[2].str().c_str())}); }
    const int max_ctx = getenv("HIPSTER_MAX_CTX") ? atoi(getenv("HIPSTER_MAX_CTX")) : 8192;
    hip::Qwen35 m(argv[1], max_ctx);
    const int V = hip::Qwen35Dims::n_vocab;
    int ids[16]; float vals[16];

    // ---- correctness: whole prompt in one forward (GEMM path), then greedy via T=1 ----
    const int P = (int)prompt.size();
    if (!getenv("HIPSTER_SKIP_CHECK")) {
    m.reset();
    float ms = 0; int lastT = 0; { int p0 = 0; while (p0 < P) { const int n = std::min(hip::Qwen35Dims::max_prefill, P - p0); ms += m.forward(prompt.data() + p0, n, p0); m.accept(n); p0 += n; lastT = n; } }
    // GEMM path (T > max_T) writes the last row's logits into row 0; the GEMV path keeps one row per token
    const float* last_logits = m.logits() + (lastT <= hip::Qwen35Dims::max_T ? (size_t)(lastT - 1) * V : 0);
    std::vector<float> logits(V); hipMemcpy(logits.data(), last_logits, V * 4, hipMemcpyDeviceToHost);
    float mx = -1e30f; for (float v : logits) mx = fmaxf(mx, v); double se = 0; for (float v : logits) se += exp((double)v - mx); const double lse = mx + log(se);
    m.topk(last_logits, 1, 10, ids, vals);
    double maxdiff = 0; for (int i = 0; i < 10; ++i) { float r = NAN; for (auto& t : ref_top) if (t.first == ids[i]) r = t.second; if (!std::isnan(r)) maxdiff = fmax(maxdiff, fabs(vals[i] - lse - r)); }
    printf("one-shot prefill of %d tokens: %.1f ms; top-1 %d (ref %d), max |dlogprob| over top-10 = %.4f\n", P, ms, ids[0], ref_top.empty() ? -1 : ref_top[0].first, maxdiff);
    int cur = ids[0], pos = P, match = 0; std::vector<int> ours;
    for (size_t i = 0; i < ref_gen.size(); ++i) { ours.push_back(cur); if (cur == ref_gen[i]) ++match; m.forward(&cur, 1, pos++); m.accept(1); m.topk(m.logits(), 1, 1, ids, vals); cur = ids[0]; }
    printf("greedy after prefill: %d/%zu match llama.cpp; ours:", match, ref_gen.size()); for (int t : ours) printf(" %d", t); printf("\n");
    }

    // ---- throughput on synthetic prompts ----
    for (int a = 3; a < argc; ++a) {
        const int T = atoi(argv[a]);
        std::vector<int> toks(T); for (int i = 0; i < T; ++i) toks[i] = prompt[i % P];
        auto run = [&] { m.reset(); int p0 = 0; while (p0 < T) { const int n = std::min(hip::Qwen35Dims::max_prefill, T - p0); m.forward(toks.data() + p0, n, p0); m.accept(n); p0 += n; } };
        run();   // warm-up: hipBLASLt per-shape autotune happens here
        m.timing_reset(); double t0 = now_ms(); run(); double dt = now_ms() - t0;
        printf("prefill T=%d: %.0f ms = %.1f t/s\n", T, dt, T * 1000.0 / dt);
        if (getenv("HIPSTER_DEPTH")) {   // decode speed at this depth (plain T=1)
            int cur = 0; float ms = 0; m.topk(m.logits(), 1, 1, ids, vals); cur = ids[0];
            m.timing_reset();
            for (int i = 0; i < 6; ++i) { ms += m.forward(&cur, 1, T + i); m.accept(1); m.topk(m.logits(), 1, 1, ids, vals); cur = ids[0]; }
            printf("   decode at depth %d: %.1f ms/token = %.2f t/s", T, ms / 6, 6000.0 / ms);
            if (getenv("HIPSTER_TIMING")) { const float* tt = m.timing(); printf("  [gemv %.1f attn %.1f gdn %.1f other %.1f ms/token]", tt[0] / 6, tt[4] / 6, tt[5] / 6, (tt[6] + tt[3] + tt[1] + tt[2]) / 6); }
            printf("\n");
        }
        if (getenv("HIPSTER_TIMING")) { const float* tt = m.timing(); const char* nm[] = {"gemv", "dequant", "gemm", "split", "attn", "gdn", "norm/silu"}; float s = 0; for (int i = 0; i < 7; ++i) s += tt[i];
            printf("   GPU ms:"); for (int i = 0; i < 7; ++i) printf(" %s %.0f (%.0f%%)", nm[i], tt[i], 100 * tt[i] / s); printf("  | sum %.0f\n", s); }
    }
    return 0;
}
