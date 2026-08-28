// run27b <gguf> <ref.json> : feed the reference prompt token by token, compare the top-10 of the
// next-token distribution with llama.cpp's, then continue greedily and compare the token ids.
#include "../src/qwen35.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s gguf ref.json [n_gen]\n", argv[0]); return 2; }
    std::ifstream f(argv[2]); std::stringstream buf; buf << f.rdbuf(); const std::string j = buf.str();
    auto prompt = parse_ids(j, "prompt_ids"); auto ref_gen = parse_ids(j, "gen_ids");
    // reference top-10: [[id, "tok", logprob], ...]
    std::vector<std::pair<int, float>> ref_top;
    { std::regex re(R"(\[\s*(\d+),\s*\"(?:[^\"\\]|\\.)*\",\s*(-?[0-9.eE+-]+)\s*\])"); auto p = j.find("first_top10");
      for (auto it = std::sregex_iterator(j.begin() + p, j.end(), re); it != std::sregex_iterator(); ++it) ref_top.push_back({atoi((*it)[1].str().c_str()), (float)atof((*it)[2].str().c_str())}); }
    const int n_gen = argc > 3 ? atoi(argv[3]) : (int)ref_gen.size();
    printf("prompt %zu tokens, ref gen %zu tokens\n", prompt.size(), ref_gen.size());

    if (getenv("HIPSTER_PERSIST")) hip::g_persist_blocks = atoi(getenv("HIPSTER_PERSIST"));
    hip::Qwen35 m(argv[1]);
    printf("loaded %.2f GiB of weights\n%s", m.weight_bytes() / 1073741824.0, m.load_report().c_str());
    float total = 0; int pos = 0;
    for (int t : prompt) { total += m.forward(&t, 1, pos++); m.accept(1); }
    printf("prompt: %.1f ms/token\n", total / prompt.size());
    // compare next-token distribution
    std::vector<float> logits(hip::Qwen35Dims::n_vocab);
    hipMemcpy(logits.data(), m.logits(), logits.size() * 4, hipMemcpyDeviceToHost);
    float mx = -1e30f; for (float v : logits) mx = fmaxf(mx, v);
    double se = 0; for (float v : logits) se += exp((double)v - mx); const double lse = mx + log(se);
    int ids[10]; float vals[10]; m.topk(m.logits(), 1, 10, ids, vals);
    printf("ours top-10 (id logprob) vs ref:\n");
    double maxdiff = 0;
    for (int i = 0; i < 10; ++i) {
        const double lp = vals[i] - lse;
        float rlp = NAN; for (auto& r : ref_top) if (r.first == ids[i]) rlp = r.second;
        if (!std::isnan(rlp)) maxdiff = fmax(maxdiff, fabs(lp - rlp));
        printf("  %7d %8.4f   ref %8.4f %s\n", ids[i], lp, rlp, i < (int)ref_top.size() && ref_top[i].first == ids[i] ? "" : "(order differs)");
    }
    printf("max |logprob diff| over matched ids: %.4f\n", maxdiff);
    // greedy continuation
    int match = 0; int cur = ids[0]; std::vector<int> ours;
    for (int i = 0; i < n_gen; ++i) {
        ours.push_back(cur);
        if (i < (int)ref_gen.size() && cur == ref_gen[i]) ++match;
        total = m.forward(&cur, 1, pos++); m.accept(1);
        int id1[1]; float v1[1]; m.topk(m.logits(), 1, 1, id1, v1); cur = id1[0];
    }
    printf("greedy: %d/%d tokens match llama.cpp; ours:", match, n_gen);
    for (int t : ours) printf(" %d", t); printf("\n  ref:"); for (int t : ref_gen) printf(" %d", t); printf("\n");
    // steady-state timing
    float ms = 0; const int nt = 8; for (int i = 0; i < nt; ++i) { ms += m.forward(&cur, 1, pos++); m.accept(1); }
    printf("decode: %.1f ms/token = %.2f t/s (%.1f GB/s of %.2f GB weights)\n", ms / nt, 1000.0 / (ms / nt), m.weight_bytes() / (ms / nt) / 1e6, m.weight_bytes() / 1e9);
    return 0;
}
