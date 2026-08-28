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
    if (getenv("LONG")) {   // LONG=<ref.json with a long prompt>: prefill (GEMM path) per slot, then lockstep decode vs each slot's single-slot run
        std::ifstream lf(getenv("LONG")); std::stringstream lb; lb << lf.rdbuf(); auto lp = parse_ids(lb.str(), "prompt_ids");
        const int P = (int)lp.size(); const int chunk = 2048;
        hip::Qwen4Exp ml(argv[1], (P + 4096 + 1023) / 1024 * 1024, S, chunk);
        std::vector<std::vector<int>> ref(S); std::vector<std::vector<int>> pr(S); for (int s = 0; s < S; ++s) pr[s].assign(lp.begin() + s, lp.end());
        for (int s = 0; s < S; ++s) { ml.reset(); int pos = 0; for (int c0 = 0; c0 < (int)pr[s].size(); c0 += chunk) { const int n = std::min(chunk, (int)pr[s].size() - c0); ml.prefill(&pr[s][c0], n, c0, 0); pos += n; }
            ml.topk(ml.logits(), 1, 1, ids.data(), vals.data()); int cur = ids[0];
            for (int i = 0; i < n_gen; ++i) { ref[s].push_back(cur); ml.forward(&cur, 1, pos++); ml.accept(1); ml.topk(ml.logits(), 1, 1, ids.data(), vals.data()); cur = ids[0]; } }
        ml.reset(); std::vector<int> cur(S), pos(S); std::vector<std::vector<int>> st(S);
        for (int s = 0; s < S; ++s) { for (int c0 = 0; c0 < (int)pr[s].size(); c0 += chunk) { const int n = std::min(chunk, (int)pr[s].size() - c0); ml.prefill(&pr[s][c0], n, c0, s); } ml.topk(ml.logits(), 1, 1, ids.data(), vals.data()); cur[s] = ids[0]; pos[s] = (int)pr[s].size(); }
        std::vector<hip::SlotReq> rq(S); double t0 = now_ms();
        for (int i = 0; i < n_gen; ++i) { for (int s = 0; s < S; ++s) { st[s].push_back(cur[s]); rq[s] = {s, &cur[s], 1, pos[s]}; } ml.forward(rq.data(), S); for (int s = 0; s < S; ++s) { ml.accept(s, 1); ++pos[s]; } ml.topk(ml.logits(), S, 1, ids.data(), vals.data()); for (int s = 0; s < S; ++s) cur[s] = ids[s]; }
        const double dt = now_ms() - t0; int bad = 0;
        for (int s = 0; s < S; ++s) { int mt = 0; for (int i = 0; i < n_gen; ++i) if (st[s][i] == ref[s][i]) ++mt; else break; if (mt != n_gen) { ++bad; printf("slot %d diverges at %d\n", s, mt); } }
        printf("%d slots at %d-token prompts (GEMM prefill per slot): %.2f t/s aggregate; %s: %d/%d slots identical to their single-slot runs\n", S, P, S * n_gen * 1000.0 / dt, bad ? "MISMATCH" : "EXACT", S - bad, S);
        return bad ? 1 : 0;
    }
    if (getenv("ROWS_DIAG")) {   // which rows of a >8-row batched pass differ from the same rows in a single-slot pass?
        const int V = hip::FnDims::n_vocab; std::vector<int> toks(prompt.begin(), prompt.begin() + 9);
        for (auto Ts : std::vector<std::vector<int>>{{8, 1}, {1, 8}, {5, 4}, {4, 5}, {8, 2}, {2, 2, 2, 2, 1}, {7, 1}, {3, 3, 2}}) {
            const int S2 = (int)Ts.size(); if (S2 > S) continue;
            m.reset(); for (int s = 0; s < S2; ++s) for (int i = 0; i < 20; ++i) { hip::SlotReq r{s, &prompt[i], 1, i}; m.forward(&r, 1); m.accept(s, 1); }
            std::vector<hip::SlotReq> rq; int rows = 0; for (int s = 0; s < S2; ++s) { rq.push_back({s, toks.data(), Ts[s], 20}); rows += Ts[s]; }
            m.forward(rq.data(), S2); std::vector<float> lb((size_t)rows * V); hipMemcpy(lb.data(), m.logits(), lb.size() * 4, hipMemcpyDeviceToHost);
            printf("batched T=("); for (int s = 0; s < S2; ++s) printf("%d%s", Ts[s], s + 1 < S2 ? "," : ""); printf(") %d rows: per-row max diff:", rows);
            int r0 = 0;
            for (int s = 0; s < S2; ++s) { hip::SlotReq r{s, toks.data(), Ts[s], 20}; m.forward(&r, 1); std::vector<float> l1((size_t)Ts[s] * V); hipMemcpy(l1.data(), m.logits(), l1.size() * 4, hipMemcpyDeviceToHost);
                for (int t = 0; t < Ts[s]; ++t) { double md = 0; for (int v = 0; v < V; ++v) md = fmax(md, fabs(lb[(size_t)(r0 + t) * V + v] - l1[(size_t)t * V + v])); printf(" %.3f", md); } r0 += Ts[s]; printf(" |"); }
            printf("\n");
        }
        return 0;
    }

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
      const int mtp = getenv("BATCH_MTP") ? atoi(getenv("BATCH_MTP")) : 0;   // BATCH_MTP=n: lockstep MTP rounds over the S slots (the serving loop)
      if (mtp > 0) {
          const int W = hip::FnDims::wide; float *hp, *hb; hipMalloc(&hp, (size_t)S * W * 4); hipMalloc(&hb, (size_t)S * W * 4);
          // the MTP KV over the prompts: token by token per slot, h = the trunk's h of the previous position (zeros at 0)
          for (int s = 0; s < S; ++s) { m.reset_slot(s); hipMemset(hp + (size_t)s * W, 0, (size_t)W * 4);
              for (int i = 0; i < (int)prompts[s].size(); ++i) { hip::SlotReq r{s, &prompts[s][i], 1, i}; m.forward(&r, 1); m.accept(s, 1); m.mtp_forward(&r, 1, hp + (size_t)s * W); hipMemcpy(hp + (size_t)s * W, m.h_nextn(), (size_t)W * 4, hipMemcpyDeviceToDevice); }
              m.topk(m.logits(), 1, 1, ids.data(), vals.data()); cur[s] = ids[0]; }
          int rounds = 0, drafted = 0, accepted = 0; t0 = now_ms(); std::vector<int> r0s(S), mms(S); std::vector<std::vector<int>> dr(S, std::vector<int>(mtp)), batch(S);
          while (true) {
              bool any = false; for (int s = 0; s < S; ++s) any |= (int)streams[s].size() < n_gen; if (!any) break;
              std::vector<int> tok(cur);
              for (int s = 0; s < S; ++s) hipMemcpy(hb + (size_t)s * W, hp + (size_t)s * W, (size_t)W * 4, hipMemcpyDeviceToDevice);
              for (int i = 0; i < mtp; ++i) {
                  for (int s = 0; s < S; ++s) reqs[s] = {s, &tok[s], 1, pos[s] + i};
                  m.mtp_forward(reqs.data(), S, hb); m.topk(m.mtp_logits(), S, 1, ids.data(), vals.data());
                  for (int s = 0; s < S; ++s) { tok[s] = ids[s]; dr[s][i] = ids[s]; }
                  if (i + 1 < mtp) hipMemcpy(hb, m.mtp_h(), (size_t)S * W * 4, hipMemcpyDeviceToDevice);
              }
              int rows = 0; for (int s = 0; s < S; ++s) { batch[s].assign(1, cur[s]); for (int i = 0; i < mtp; ++i) batch[s].push_back(dr[s][i]); reqs[s] = {s, batch[s].data(), 1 + mtp, pos[s]}; r0s[s] = rows; rows += 1 + mtp; }
              gpu += m.forward(reqs.data(), S); m.topk(m.logits(), rows, 1, ids.data(), vals.data());
              for (int s = 0; s < S; ++s) {
                  int mm = 0; while (mm < mtp && ids[r0s[s] + mm] == dr[s][mm]) ++mm;
                  streams[s].push_back(cur[s]); for (int i = 0; i < mm; ++i) streams[s].push_back(dr[s][i]);
                  m.accept(s, mm + 1);
                  if (mm > 0) { hip::SlotReq r{s, dr[s].data(), mm, pos[s] + 1}; m.mtp_forward(&r, 1, m.h_nextn() + (size_t)r0s[s] * W); }
                  hipMemcpy(hp + (size_t)s * W, m.h_nextn() + (size_t)(r0s[s] + mm) * W, (size_t)W * 4, hipMemcpyDeviceToDevice);
                  cur[s] = ids[r0s[s] + mm]; pos[s] += mm + 1; drafted += mtp; accepted += mm;
              }
              ++rounds;
          }
          const double dt = now_ms() - t0; int ntok = 0; for (int s = 0; s < S; ++s) { streams[s].resize(n_gen); ntok += n_gen; }
          printf("%d slots, MTP n=%d: %d tokens in %.0f ms = %.2f t/s aggregate | %d rounds, acceptance %d/%d\n", S, mtp, ntok, dt, ntok * 1000.0 / dt, rounds, accepted, drafted);
      } else
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
