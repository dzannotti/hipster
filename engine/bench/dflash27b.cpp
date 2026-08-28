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
    const int n_slots = getenv("SLOTS") ? atoi(getenv("SLOTS")) : 1;
    const int max_ctx = (int)((prompt.size() + (argc > 4 ? atoi(argv[4]) : 64) * 2 + 1023) / 1024 * 1024 + 1024);
    hip::Qwen35 m(argv[1], std::max(8192, max_ctx), n_slots); m.load_dflash(argv[2]);
    int ids[16 * 8]; float vals[16 * 8];

    if (getenv("DIAG") && n_slots >= 2) {   // identical tokens/state in two slots of one batched pass: rows must match bit for bit
        m.reset(); const int V = hip::Qwen35Dims::n_vocab;
        for (int s = 0; s < 2; ++s) for (size_t i = 0; i < prompt.size(); ++i) { hip::SlotReq r{s, &prompt[i], 1, (int)i}; m.forward(&r, 1); m.accept(s, 1); }
        int toks[8] = {1393, 471, 220, 16, 8, 478, 73111, 1393}; const int P = (int)prompt.size();
        for (auto [Ta, Tb] : std::vector<std::pair<int,int>>{{1, 1}, {8, 8}, {8, 1}, {1, 8}}) {
            hip::SlotReq reqs[2] = {{0, toks, Ta, P}, {1, toks, Tb, P}};
            m.forward(reqs, 2);
            std::vector<float> lg((size_t)(Ta + Tb) * V); hipMemcpy(lg.data(), m.logits(), lg.size() * 4, hipMemcpyDeviceToHost);
            const int n = std::min(Ta, Tb); double md = 0; for (int r = 0; r < n; ++r) for (int v = 0; v < V; ++v) md = fmax(md, fabs(lg[(size_t)r * V + v] - lg[(size_t)(Ta + r) * V + v]));
            // and against a single-slot pass of slot 0 with the same tokens (state unchanged: nothing accepted)
            hip::SlotReq r0{0, toks, Ta, P}; m.forward(&r0, 1); std::vector<float> l1((size_t)Ta * V); hipMemcpy(l1.data(), m.logits(), l1.size() * 4, hipMemcpyDeviceToHost);
            double m1 = 0; for (int r = 0; r < Ta; ++r) for (int v = 0; v < V; ++v) m1 = fmax(m1, fabs(lg[(size_t)r * V + v] - l1[(size_t)r * V + v]));
            printf("batched T=(%d,%d): max|slot1 - slot0| over %d rows = %.4f ; max|batched slot0 - single-slot pass| = %.4f\n", Ta, Tb, n, md, m1);
        }
        return 0;
    }
    if (getenv("PROF_T")) {   // per-category GPU time of a T-token pass (HIPSTER_TIMING=1 for the categories)
        m.reset(); int pos = 0;
        if (getenv("PREFILL")) { for (int p0 = 0; p0 < (int)prompt.size(); p0 += 4096) { const int n = std::min(4096, (int)prompt.size() - p0); m.forward(prompt.data() + p0, n, p0); m.accept(n); } pos = (int)prompt.size(); }
        else for (int t : prompt) { m.forward(&t, 1, pos++); m.accept(1); }
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
    const bool prefill = getenv("PREFILL") != nullptr;   // prompt in one GEMM-path pass instead of token by token
    const int P = (int)prompt.size();
    std::vector<int> plain;
    { m.reset(); int pos = 0;
      if (prefill) { for (int p0 = 0; p0 < P; p0 += 4096) { const int n = std::min(4096, P - p0); m.forward(prompt.data() + p0, n, p0); m.accept(n); } pos = P; }
      else for (int t : prompt) { m.forward(&t, 1, pos++); m.accept(1); }
      m.topk(m.logits(), 1, 1, ids, vals); int cur = ids[0];
      double t0 = now_ms();
      for (int i = 0; i < n_gen; ++i) { plain.push_back(cur); m.forward(&cur, 1, pos++); m.accept(1); m.topk(m.logits(), 1, 1, ids, vals); cur = ids[0]; }
      double dt = now_ms() - t0;
      printf("plain greedy: %d tokens in %.0f ms = %.2f t/s\n", n_gen, dt, n_gen * 1000.0 / dt); }

    std::vector<int> spec;
    { m.reset();
      if (prefill) { for (int p0 = 0; p0 < P; p0 += 4096) { const int n = std::min(4096, P - p0); m.forward(prompt.data() + p0, n, p0); m.accept(n); m.dflash_encode(n, p0); } }
      else for (size_t i = 0; i < prompt.size(); ++i) { m.forward(&prompt[i], 1, (int)i); m.accept(1); m.dflash_encode(1, (int)i); }
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
    if (n_slots < 2) return match == n_gen ? 0 : 1;
    // ---- S slots in lockstep, every slot running the same prompt + DFlash2 (aggregate throughput; each stream must stay exact) ----
    { m.reset(); std::vector<int> pos(n_slots, 0), idl(n_slots); std::vector<std::vector<int>> out(n_slots);
      for (int s = 0; s < n_slots; ++s) { if (prefill) { for (int p0 = 0; p0 < P; p0 += 4096) { const int n = std::min(4096, P - p0); hip::SlotReq r{s, prompt.data() + p0, n, p0}; m.forward(&r, 1); m.accept(s, n); m.dflash_encode(s, n, p0); } }
          else for (size_t i = 0; i < prompt.size(); ++i) { hip::SlotReq r{s, &prompt[i], 1, (int)i}; m.forward(&r, 1); m.accept(s, 1); m.dflash_encode(s, 1, (int)i); }
          m.topk(m.logits(), 1, 1, ids, vals); idl[s] = ids[0]; out[s].push_back(ids[0]); pos[s] = (int)prompt.size(); }
      int rounds = 0, drafted = 0, accepted = 0; double t0 = now_ms(), t_draft = 0, t_verify = 0; int total = 0;
      std::vector<std::vector<int>> batch(n_slots); std::vector<int> nd(n_slots);
      while (total < n_gen * n_slots) {
          double ta = now_ms(); hip::SlotReq reqs[8];
          { std::vector<int> sl(n_slots), dr((size_t)n_slots * n_draft); for (int s = 0; s < n_slots; ++s) sl[s] = s;
            static const bool sep = getenv("SEPDRAFT") != nullptr;   // debug: one draft pass per slot
            if (sep) for (int s = 0; s < n_slots; ++s) nd[s] = m.dflash_draft(s, idl[s], pos[s], n_draft, dr.data() + (size_t)s * n_draft);
            else { const int k = m.dflash_draft_b(sl.data(), idl.data(), pos.data(), n_slots, n_draft, dr.data()); for (int s = 0; s < n_slots; ++s) nd[s] = k; }
            for (int s = 0; s < n_slots; ++s) { batch[s].assign(1, idl[s]); for (int i = 0; i < nd[s]; ++i) batch[s].push_back(dr[(size_t)s * n_draft + i]); reqs[s] = {s, batch[s].data(), (int)batch[s].size(), pos[s]}; } }
          t_draft += now_ms() - ta; ta = now_ms();
          int rows = 0; for (int s = 0; s < n_slots; ++s) rows += reqs[s].T;
          static const bool split = getenv("SPLIT") != nullptr;   // debug: one pass per slot instead of one batched pass
          if (split) { int r0 = 0; for (int s = 0; s < n_slots; ++s) { m.forward(&reqs[s], 1); m.topk(m.logits(), reqs[s].T, 1, ids + r0, vals + r0); r0 += reqs[s].T; } }
          else { m.forward(reqs, n_slots); m.topk(m.logits(), rows, 1, ids, vals); }
          int r0 = 0;
          for (int s = 0; s < n_slots; ++s) { int mm = 0; while (mm < nd[s] && ids[r0 + mm] == batch[s][1 + mm]) ++mm;
              m.accept(s, mm + 1); m.dflash_encode(s, mm + 1, pos[s]);
              for (int i = 0; i < mm; ++i) out[s].push_back(batch[s][1 + i]); out[s].push_back(ids[r0 + mm]); idl[s] = ids[r0 + mm]; pos[s] += mm + 1;
              drafted += nd[s]; accepted += mm; total += mm + 1; r0 += reqs[s].T; }
          t_verify += now_ms() - ta; ++rounds;
      }
      double dt = now_ms() - t0;
      printf("dflash2 x %d slots: %d tokens in %.0f ms = %.2f t/s aggregate (%.2f per slot) | %d rounds, acceptance %d/%d, draft %.0f ms, verify %.0f ms\n",
             n_slots, total, dt, total * 1000.0 / dt, total * 1000.0 / dt / n_slots, rounds, accepted, drafted, t_draft, t_verify);
      int bad = 0; for (int s = 0; s < n_slots; ++s) { out[s].resize(n_gen); int mt = 0; for (int i = 0; i < n_gen; ++i) if (out[s][i] == plain[i]) ++mt; else break; if (mt != n_gen) { ++bad; printf("slot %d diverges at %d\n", s, mt); } }
      printf("%s: %d slots identical to plain greedy\n", bad ? "MISMATCH" : "EXACT", n_slots - bad);
      return bad ? 1 : 0; }
}
