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

## 2026-08-28 · DFlash2 block draft (incoai/Qwen3.8-27B-DFlash2, Q4_K_M) + T-invariant WMMA GEMV

Draft: `models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1.14 GB; `huggingface-cli download incoai/Qwen3.8-27B-DFlash2-GGUF
Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models`). 5 layers, 32 q / 8 kv heads × 128, NEOX rope over all 128 dims,
non-causal windowed (2048) attention, 2-tap grouped (16) dynamic conv around attention and FFN, block of 8 =
`[id_last, MASK × 7]`, selector: top-16 per position, rank-256 predecessor/successor tables gated by the hidden state,
greedy lattice walk on the host. Encoder: `g = rmsnorm(fc · [residual entering target layers 6,20,34,48,62])`; K/V of the
encoded rows are injected per draft layer (`engine/kernels/dflash.hip`, `engine/src/dflash.cpp`, driver `bench/dflash27b.cpp`).
The target captures the five residual rows during its own pass (five 20 KB device copies per token); the encoder runs
after `accept`, so it lags one round like the MTP catch-up (≈0.1 ms/round).

**First run: 38.9 t/s but MISMATCH at token 13 — and so was MTP n=7.** Root cause, pre-existing: the 27B's T>1 pass used
ncol-tuned GEMV kernels (`gemv_q8(…, ncol, 4, 3)`) whose per-column reduction order differs from the T=1 kernel:
`max|logits(T=8) − logits(T=1)| = 0.83` (spec27b `HIPSTER_LOGIT_DIFF=1`). n≤5 only passed because no top-1/top-2 gap on
the test prompts was below the noise. The Flash-Next contract (every T reduced in the same order) now holds for the 27B too:
diff **0.0000** at T=8.

The T-invariant kernel that is also fast at 8 columns is an **int8 WMMA GEMV** (`k_gemv_wmma<F,1>`, `gemv_wmma()`): one lane
= one weight row, 16 rows per wave, the 16 x columns are the B operand (rows ≥ ncol discarded), per-format decoders
(Q4_K, Q5_K, IQ4_XS, Q6_K-SoA, Q8_0-SoA) emit two iu8 fragments + scale (+ min) per 32-sub-block; the int32 dot over 32 is
exact and one float update per sub-block follows, in block order, for every column. `bench_gemv <tensor> 8 q8`, rows
`tpr=32 rpt=1` (old T=1 kernel / old 8-column multi kernel) vs `tpr= 1 rpt=4` (WMMA), GB/s of weights:

| format (tensor) | old ncol=1 | old ncol=8 | WMMA ncol=1 | WMMA ncol=8 |
|---|---:|---:|---:|---:|
| Q4_K (ffn_gate 17408×5120) | 220 | 99 | 190 | 164 |
| Q5_K (attn_gate 6144×5120) | 198 | 79 | 169 | 132 |
| IQ4_XS (ffn_down 5120×17408) | 204 | 68 | 185 | 159 |
| Q6_K-SoA (output, first 17408 rows) | 0.377 ms | 1.060 ms | 0.423 ms | 0.455 ms |
| Q8_0-SoA (ssm_out 6144×5120) | 153 | 82 | 198 | 180 |

One kernel per format (a single kernel with a format switch spilled 87 VGPRs; per format: 137–200 VGPRs, no spills except
Q6_K's 10). `HIPSTER_GEMV=wmma` (default) | `multi` (32-lane, T-invariant, 99 GB/s at T=8) | `fast` (old, not T-invariant).

Results (`dflash27b <27B> <draft> <ref.json> 96 n`, each line EXACT vs plain greedy):

| prompt | drafter | t/s | acceptance | tokens/round | draft / verify ms per round |
|---|---|---:|---:|---:|---|
| code | plain (WMMA, T=1) | 10.3 | | 1 | 97 |
| code | plain (`HIPSTER_GEMV=fast`) | 12.5 | | 1 | 80 |
| code | MTP n=5 (WMMA verify) | 33.8 | 76% | 4.80 | 42 / 100 |
| code | DFlash2 n=5 | 43.5 | 82% | 5.11 | 16 / 100 |
| code | **DFlash2 n=7** | **54.2** | 78% | 6.47 | 17 / 101 |
| story (prose) | DFlash2 n=7 | 21.9 | 26% | 2.83 | 18 / 104 |

Where the ceiling is: a round streams the 16.46 GB of target weights once (68.6 ms at 240 GB/s) plus the 1.14 GB draft
(4.8 ms) → **73 ms/round floor**. At the measured 6.47 tokens/round that is 88 t/s (109 at 8/8 acceptance); we are at
118 ms/round = 62% of it. The gap: the verify pass at T=8 runs the GEMVs at ~150 GB/s weighted (Q5_K is 7.9 GB of the
16.5 at 132 GB/s) → 101 ms instead of 69; the draft's 17 ms is dominated by the Q6_K LM head over 8 rows (~6.5 ms) and
by 8-column GEMVs over the draft's 1.14 GB (~7 ms at the same ~160 GB/s).

Next: (1) 16-row interleaved weight layout for the WMMA kernel (each wave's load instruction currently touches 16 rows
2.9 KB apart; interleaving makes it one contiguous 2.3 KB read) — expected to close the 10–15% gap at ncol=1 and most of
the ncol=8 gap; (2) a Q4_K draft-only LM head (the verify head stays exact); (3) adaptive n on prose (26% acceptance:
n=7 wastes 5 rows per round; n=3 would give ~3 tokens/round at 75 ms).
