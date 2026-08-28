# Qwen3.8-27B speculative decoding — measurements log

Driver: `./build.sh './build/spec27b <gguf> docs/ref/<prompt>.json <n_gen> <n_draft>'`.
Contract: the speculative token stream must be **identical** to the plain T=1 greedy stream
(the driver diffs them); every row below passed.

## 2026-08-28 · MTP (in-checkpoint `blk.64` draft layer), greedy verify

Mechanism (as in llama.cpp): MTP input at position p = `[rmsnorm(embed(tok_p)) | rmsnorm(h_tgt(p−1))]`
→ eh_proj → attention+FFN layer (own KV) → head norm → shared LM head; drafts chain the MTP's own
output h. Target verifies `[id_last, d_1..d_n]` in one T-token pass; accepted prefix m; bonus =
target argmax at m. Recurrent state is double-buffered: the pass writes state_nxt; `accept(m<T)`
replays the GDN conv+step for m tokens from the saved per-layer inputs (no snapshot copies).
After acceptance the MTP KV for the accepted positions is recomputed with the target's h.

| prompt | n_draft | t/s | acceptance | tokens/round | draft / verify ms per round |
|---|---:|---:|---:|---:|---|
| code (64 tok) | 1 (plain) | 12.4 | | 1 | 80 |
| code | 2 | 25.7 | 93% | 2.86 | 20 / 93 |
| code | 3 | 30.8 | 92% | 3.76 | 27 / 95 |
| code | 4 | 32.0 | 83% | 4.33 | 34 / 99 |
| code | **5** | **35.6** | 85% | 5.25 | 40 / 110 |
| story/chat | 3 | 21.0 | 51% | 2.52 | 27 / 95 |

Halogen (same GPU, its own 6.3-bpw weights): MTP 26.9 mean, DFlash2 31.7 mean (prose 20.8, code 44).

Where the time goes at n=5: each draft step ≈ 8 ms of which the Q6_K LM head (994 MB) is 4.2;
the verify pass at T=6 costs 1.37× a single-token pass (formats other than Q4_K still run the
dot4 path at 6 columns; WMMA decoders for Q5_K/Q6_K/IQ4_XS/Q8_0 would bring it to ~1.2×).

Next cuts, in order of expected gain: (1) DFlash2 drafter (one pass drafts 8 tokens, τ≈4.4–5.5 on
code); (2) draft-only LM head at Q4_K or shortlisted rows (the verify head stays exact);
(3) WMMA decoders for all formats; (4) adaptive n from an acceptance EMA (prose wants n≈2).
