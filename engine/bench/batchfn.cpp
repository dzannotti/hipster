// batchfn <gguf> <ref.json> [n_gen] [S] : S slots decode the same prompt concurrently (plain greedy, one token per
// slot per pass). Contract: every slot's stream equals the single-slot greedy stream. Reports aggregate t/s.
#include "../src/qwen4exp.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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
    if (argc < 3) { fprintf(stderr, "usage: %s gguf ref.json [n_gen] [S]\n", argv[0]); return 2; }
    std::ifstream f(argv[2]); std::stringstream buf; buf << f.rdbuf();
    auto prompt = parse_ids(buf.str(), "prompt_ids");
    const int n_gen = argc > 3 ? atoi(argv[3]) : 64, S = argc > 4 ? atoi(argv[4]) : 4;
    hip::Qwen4Exp m(argv[1], 8192, S);
    std::vector<int> ids(64 * 16); std::vector<float> vals(64 * 16);

    // BATCH_DISTINCT=1: slot s runs the prompt without its first s tokens (distinct sequences -> distinct experts);
    // each slot is checked against its own single-slot run. Default: all slots on the same prompt (experts overlap!).
    const bool distinct = getenv("BATCH_DISTINCT") != nullptr;
    std::vector<std::vector<int>> prompts(S); for (int s = 0; s < S; ++s) prompts[s].assign(prompt.begin() + (distinct ? s : 0), prompt.end());
    std::vector<std::vector<int>> plains(S);
    for (int s = 0; s < (distinct ? S : 1); ++s) {
      m.reset(); int pos = 0; auto& plain = plains[s];
      for (int t : prompts[s]) { m.forward(&t, 1, pos++); m.accept(1); }
      m.topk(m.logits(), 1, 1, ids.data(), vals.data()); int cur = ids[0];
      double t0 = now_ms();
      for (int i = 0; i < n_gen; ++i) { plain.push_back(cur); m.forward(&cur, 1, pos++); m.accept(1); m.topk(m.logits(), 1, 1, ids.data(), vals.data()); cur = ids[0]; }
      if (s == 0) printf("1 slot : %d tokens in %.0f ms = %.2f t/s\n", n_gen, now_ms() - t0, n_gen * 1000.0 / (now_ms() - t0)); }
    if (!distinct) for (int s = 1; s < S; ++s) plains[s] = plains[0];

    // S slots in lockstep: prompts token by token (batched; shorter prompts start later), then decode
    std::vector<std::vector<int>> streams(S); std::vector<int> cur(S); std::vector<hip::SlotReq> reqs(S);
    { m.reset();
      const int plen = (int)prompt.size();
      for (int i = 0; i < plen; ++i) {
          int n = 0;
          for (int s = 0; s < S; ++s) { const int off = plen - (int)prompts[s].size(); if (i >= off) reqs[n++] = {s, &prompts[s][i - off], 1, i - off}; }
          m.forward(reqs.data(), n); for (int k = 0; k < n; ++k) m.accept(reqs[k].slot, 1);
      }
      m.topk(m.logits(), S, 1, ids.data(), vals.data()); for (int s = 0; s < S; ++s) cur[s] = ids[s];   // the last prompt step had all S slots
      std::vector<int> pos(S); for (int s = 0; s < S; ++s) pos[s] = (int)prompts[s].size();
      double t0 = now_ms(), gpu = 0;
      for (int i = 0; i < n_gen; ++i) {
          for (int s = 0; s < S; ++s) { streams[s].push_back(cur[s]); reqs[s] = {s, &cur[s], 1, pos[s]}; }
          gpu += m.forward(reqs.data(), S); for (int s = 0; s < S; ++s) { m.accept(s, 1); ++pos[s]; }
          m.topk(m.logits(), S, 1, ids.data(), vals.data()); for (int s = 0; s < S; ++s) cur[s] = ids[s];
      }
      const double dt = now_ms() - t0;
      printf("%d slots: %d x %d tokens in %.0f ms = %.2f t/s aggregate (%.2f per slot; GPU %.1f ms/pass)\n", S, S, n_gen, dt, S * n_gen * 1000.0 / dt, n_gen * 1000.0 / dt, gpu / n_gen); }
    int bad = 0;
    for (int s = 0; s < S; ++s) { const auto& plain = plains[s]; int match = 0; for (int i = 0; i < n_gen; ++i) if (streams[s][i] == plain[i]) ++match; else { printf("slot %d diverges at %d: plain %d got %d\n", s, i, plain[i], streams[s][i]); break; } if (match != n_gen) ++bad; }
    printf("%s: %d/%d slots identical to their single-slot greedy streams (%s prompts)\n", bad ? "MISMATCH" : "EXACT", S - bad, S, distinct ? "distinct" : "identical");
    return bad ? 1 : 0;
}
