// spec27b <gguf> <ref.json> [n_gen] [n_draft] : greedy generation with MTP speculative decoding.
// Correctness contract: the token stream must equal the plain (T=1) greedy stream exactly.
#include "../src/qwen35.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    if (argc < 3) { fprintf(stderr, "usage: %s gguf ref.json [n_gen] [n_draft]\n", argv[0]); return 2; }
    std::ifstream f(argv[2]); std::stringstream buf; buf << f.rdbuf();
    auto prompt = parse_ids(buf.str(), "prompt_ids");
    const int n_gen = argc > 3 ? atoi(argv[3]) : 64, n_draft = argc > 4 ? atoi(argv[4]) : 3;
    hip::Qwen35 m(argv[1]);
    const int V = hip::Qwen35Dims::n_vocab, E = hip::Qwen35Dims::n_embd;
    int ids[16 * 8]; float vals[16 * 8];

    // ---- plain greedy (T = 1) ----
    std::vector<int> plain; std::vector<float> plain_margin; std::vector<float> plain_gap;   // top-1 minus top-2 logit at each step (tie diagnostic)
    { m.reset(); int pos = 0;
      for (int t : prompt) { m.forward(&t, 1, pos++); m.accept(1); }
      m.topk(m.logits(), 1, 1, ids, vals); int cur = ids[0];
      double t0 = now_ms();
      for (int i = 0; i < n_gen; ++i) { plain.push_back(cur); m.forward(&cur, 1, pos++); m.accept(1); m.topk(m.logits(), 1, 2, ids, vals); cur = ids[0]; plain_gap.push_back(vals[0] - vals[1]); }
      double dt = now_ms() - t0;
      printf("plain greedy: %d tokens in %.0f ms = %.2f t/s\n", n_gen, dt, n_gen * 1000.0 / dt); }

    if (getenv("HIPSTER_LOGIT_DIFF")) {   // T=1 vs T>1 consistency on the same token sequence
        // plain logits per step were not stored; recompute both paths from scratch
        std::vector<std::vector<float>> l1;
        { m.reset(); int pos = 0; for (int t : prompt) { m.forward(&t, 1, pos++); m.accept(1); }
          for (int i = 0; i < n_gen; ++i) { std::vector<float> lg(V); hipMemcpy(lg.data(), m.logits(), (size_t)V * 4, hipMemcpyDeviceToHost); l1.push_back(lg); m.forward(&plain[i], 1, pos++); m.accept(1); } }
        m.reset(); { int pos = 0; for (int t : prompt) { m.forward(&t, 1, pos++); m.accept(1); }
          const int B = getenv("HIPSTER_BATCH") ? atoi(getenv("HIPSTER_BATCH")) : 6; double worst = 0; int worst_i = -1;
          for (int i = 0; i < n_gen; i += B) {   // batch: tokens plain[i..i+B) -> logits rows predict plain[i+1..]; row r corresponds to l1[i+r+1]
              const int T = std::min(B, n_gen - i); m.forward(&plain[i], T, pos); m.accept(T); pos += T;
              std::vector<float> lg((size_t)T * V); hipMemcpy(lg.data(), m.logits(), lg.size() * 4, hipMemcpyDeviceToHost);
              for (int r = 0; r < T; ++r) { if (i + r + 1 >= (int)l1.size()) break; double md = 0; for (int v = 0; v < V; ++v) md = fmax(md, fabs((double)lg[(size_t)r * V + v] - l1[i + r + 1][v])); if (md > worst) { worst = md; worst_i = i + r + 1; }
                  if (r == 0 || r == T - 1 || i < 2 * B) printf("  logit diff step %d (row %d of %d): max|T=%d - T=1| = %.4f\n", i + r + 1, r, T, T, md); }
          }
          printf("worst T>1 vs T=1 logit difference: %.4f at step %d\n", worst, worst_i); }
        return 0;
    }
    // ---- MTP speculative greedy ----
    std::vector<int> spec;
    float* h_pending; hipMalloc(&h_pending, E * 4);
    { m.reset(); int pos = 0;
      // prefill token by token (target), then MTP catch-up over the prompt so its KV is filled:
      // MTP at position p takes (token_p, h_target(p-1)); h_target(-1) := zeros.
      std::vector<float> hz(E, 0.f); std::vector<float> hprev(E, 0.f);
      float* d_h; hipMalloc(&d_h, (size_t)8 * E * 4);
      for (size_t i = 0; i < prompt.size(); ++i) {
          hipMemcpy(d_h, hprev.data(), E * 4, hipMemcpyHostToDevice);
          m.forward(&prompt[i], 1, (int)i); m.accept(1);
          m.mtp_forward(&prompt[i], d_h, 1, (int)i);
          hipMemcpy(hprev.data(), m.h_nextn(), E * 4, hipMemcpyDeviceToHost);
      }
      pos = (int)prompt.size();
      m.topk(m.logits(), 1, 1, ids, vals); int id_last = ids[0]; spec.push_back(id_last);
      hipMemcpy(h_pending, m.h_nextn(), E * 4, hipMemcpyDeviceToDevice);
      int rounds = 0, drafted = 0, accepted = 0;
      double t0 = now_ms(), t_draft = 0, t_verify = 0, t_catch = 0;
      while ((int)spec.size() < n_gen) {
          // draft n tokens with the MTP layer, chaining its own h
          std::vector<int> drafts; int tok = id_last; const float* h = h_pending; int p = pos;
          double ta = now_ms();
          for (int i = 0; i < n_draft; ++i) {
              m.mtp_forward(&tok, h, 1, p);
              m.topk(m.mtp_logits(), 1, 1, ids, vals);
              tok = ids[0]; drafts.push_back(tok); h = m.mtp_h(); ++p;
          }
          t_draft += now_ms() - ta; ta = now_ms();
          // verify: target over [id_last, drafts...]
          std::vector<int> batch; batch.push_back(id_last); for (int d : drafts) batch.push_back(d);
          const int T = (int)batch.size();
          m.forward(batch.data(), T, pos);
          m.topk(m.logits(), T, 1, ids, vals);
          int mm = 0; while (mm < (int)drafts.size() && ids[mm] == drafts[mm]) ++mm;
          m.accept(mm + 1);
          t_verify += now_ms() - ta; ta = now_ms();
          // emit: id_last was emitted last round; new tokens = accepted drafts + bonus
          for (int i = 0; i < mm; ++i) spec.push_back(drafts[i]);
          const int bonus = ids[mm]; spec.push_back(bonus);
          // MTP catch-up for the accepted drafts at positions pos+1..pos+mm with h = target h of the previous position
          if (mm > 0) m.mtp_forward(drafts.data(), m.h_nextn(), mm, pos + 1);
          hipMemcpy(h_pending, m.h_nextn() + (size_t)mm * E, E * 4, hipMemcpyDeviceToDevice);
          t_catch += now_ms() - ta;
          id_last = bonus; pos += mm + 1; ++rounds; drafted += n_draft; accepted += mm;
      }
      double dt = now_ms() - t0;
      spec.resize(n_gen);
      printf("mtp n=%d: %d tokens in %.0f ms = %.2f t/s | %d rounds, acceptance %d/%d = %.0f%%, %.2f tokens/round | draft %.0f ms, verify %.0f ms, catch-up %.0f ms\n",
             n_draft, n_gen, dt, n_gen * 1000.0 / dt, rounds, accepted, drafted, 100.0 * accepted / drafted, (double)(accepted + rounds) / rounds, t_draft, t_verify, t_catch); }
    int match = 0; for (int i = 0; i < n_gen; ++i) if (plain[i] == spec[i]) ++match; else { printf("first divergence at %d: plain %d spec %d; plain top1-top2 logit gap at the step that produced it: %.4f\n", i, plain[i], spec[i], i > 0 ? plain_gap[i - 1] : NAN); break; }
    printf("%s: %d/%d tokens identical to plain greedy\n", match == n_gen ? "EXACT" : "MISMATCH", match, n_gen);
    return match == n_gen ? 0 : 1;
}
