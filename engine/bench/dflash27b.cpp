// dflash27b <gguf> <draft.gguf> <ref.json> [n_gen] [n_draft<=7] [ngram] : greedy generation with the DFlash2 block draft;
// with a 6th argument the n-gram map (llama.cpp "ngram-map-k4v", NGRAM_N / NGRAM_M / NGRAM_MIN_HITS env, defaults 12/48/1)
// is tried first and DFlash2 only drafts when it has nothing (llama.cpp's priority order).
// Contract: the token stream must equal the plain (T=1) greedy stream exactly.
#include "../src/qwen35.h"
#include "../src/ngram_map.h"
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
    if (argc < 4) { fprintf(stderr, "usage: %s gguf draft.gguf ref.json [n_gen] [n_draft]\n", argv[0]); return 2; }
    std::ifstream f(argv[3]); std::stringstream buf; buf << f.rdbuf();
    auto prompt = parse_ids(buf.str(), "prompt_ids");
    const int n_gen = argc > 4 ? atoi(argv[4]) : 64, n_draft = argc > 5 ? atoi(argv[5]) : 7;
    const bool use_ngram = argc > 6;
    auto envi = [](const char* k, int d) { return getenv(k) ? atoi(getenv(k)) : d; };
    hip::NgramMap ngram(envi("NGRAM_N", 12), envi("NGRAM_M", 48), envi("NGRAM_MIN_HITS", 1));
    hip::Qwen35 m(argv[1]); m.load_dflash(argv[2]);
    int ids[16 * 8]; float vals[16 * 8];

    if (getenv("PROF_T")) {   // per-category GPU time of a T-token pass (HIPSTER_TIMING=1 for the categories)
        m.reset(); int pos = 0; for (int t : prompt) { m.forward(&t, 1, pos++); m.accept(1); }
        for (int T : {1, 2, 4, 8}) {
            std::vector<int> toks(T, 1393); m.forward(toks.data(), T, pos); m.accept(T); pos += T; m.timing_reset();
            const int reps = 6; double t0 = now_ms(); float gpu = 0;
            for (int r = 0; r < reps; ++r) { gpu += m.forward(toks.data(), T, pos); m.accept(T); pos += T; }
            hipDeviceSynchronize(); double wall = (now_ms() - t0) / reps; const float* c = m.timing();
            printf("T=%d: %.1f ms wall, %.1f ms gpu | gemv %.1f deq %.1f gemm %.1f split %.1f attn %.1f gdn %.1f other %.1f\n", T, wall, gpu / reps,
                   c[0] / reps, c[1] / reps, c[2] / reps, c[3] / reps, c[4] / reps, c[5] / reps, c[6] / reps);
        }
        return 0;
    }
    std::vector<int> plain;
    { m.reset(); int pos = 0;
      for (int t : prompt) { m.forward(&t, 1, pos++); m.accept(1); }
      m.topk(m.logits(), 1, 1, ids, vals); int cur = ids[0];
      double t0 = now_ms();
      for (int i = 0; i < n_gen; ++i) { plain.push_back(cur); m.forward(&cur, 1, pos++); m.accept(1); m.topk(m.logits(), 1, 1, ids, vals); cur = ids[0]; }
      double dt = now_ms() - t0;
      printf("plain greedy: %d tokens in %.0f ms = %.2f t/s\n", n_gen, dt, n_gen * 1000.0 / dt); }

    std::vector<int> spec;
    { m.reset();
      for (size_t i = 0; i < prompt.size(); ++i) { m.forward(&prompt[i], 1, (int)i); m.accept(1); m.dflash_encode(1, (int)i); }
      int pos = (int)prompt.size();
      m.topk(m.logits(), 1, 1, ids, vals); int id_last = ids[0]; spec.push_back(id_last);
      int rounds = 0, drafted = 0, accepted = 0, ng_rounds = 0, ng_accepted = 0, ng_drafted = 0; std::vector<int> hist(n_draft + 1, 0);
      std::vector<int> history(prompt); if (use_ngram) ngram.begin(history);
      double t0 = now_ms(), t_draft = 0, t_verify = 0, t_enc = 0;
      while ((int)spec.size() < n_gen) {
          int drafts[8]; double ta = now_ms(); int nd = 0; bool from_ngram = false;
          if (use_ngram) { std::vector<int> ng = ngram.draft(history, id_last); if (!ng.empty()) { nd = std::min((int)ng.size(), n_draft); for (int i = 0; i < nd; ++i) drafts[i] = ng[i]; from_ngram = true; } }
          if (nd == 0) nd = m.dflash_draft(id_last, pos, n_draft, drafts);
          t_draft += now_ms() - ta; ta = now_ms();
          std::vector<int> batch; batch.push_back(id_last); for (int i = 0; i < nd; ++i) batch.push_back(drafts[i]);
          const int T = (int)batch.size();
          m.forward(batch.data(), T, pos);
          m.topk(m.logits(), T, 1, ids, vals);
          int mm = 0; while (mm < nd && ids[mm] == drafts[mm]) ++mm;
          m.accept(mm + 1);
          t_verify += now_ms() - ta; ta = now_ms();
          m.dflash_encode(mm + 1, pos);
          t_enc += now_ms() - ta;
          history.push_back(id_last); for (int i = 0; i < mm; ++i) { spec.push_back(drafts[i]); history.push_back(drafts[i]); }
          const int bonus = ids[mm]; spec.push_back(bonus);
          if (from_ngram) { ngram.accept(mm); ++ng_rounds; ng_accepted += mm; ng_drafted += nd; }
          id_last = bonus; pos += mm + 1; ++rounds; drafted += nd; accepted += mm; ++hist[mm];
      }
      double dt = now_ms() - t0; spec.resize(n_gen);
      printf("dflash2 n=%d: %d tokens in %.0f ms = %.2f t/s | %d rounds, acceptance %d/%d = %.0f%%, %.2f tokens/round | draft %.0f ms, verify %.0f ms, encode %.0f ms\n",
             n_draft, n_gen, dt, n_gen * 1000.0 / dt, rounds, accepted, drafted, 100.0 * accepted / drafted, (double)(accepted + rounds) / rounds, t_draft, t_verify, t_enc);
      if (use_ngram) printf("ngram-map-k4v: %d of %d rounds drafted by the map, acceptance %d/%d\n", ng_rounds, rounds, ng_accepted, ng_drafted);
      printf("accepted-per-round histogram:"); for (int i = 0; i <= n_draft; ++i) printf(" %d:%d", i, hist[i]); printf("\n"); }
    int match = 0; for (int i = 0; i < n_gen; ++i) if (plain[i] == spec[i]) ++match; else { printf("first divergence at %d: plain %d spec %d\n", i, plain[i], spec[i]); break; }
    printf("%s: %d/%d tokens identical to plain greedy\n", match == n_gen ? "EXACT" : "MISMATCH", match, n_gen);
    return match == n_gen ? 0 : 1;
}
