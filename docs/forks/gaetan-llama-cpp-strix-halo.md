# gaetan-puleo/llama-cpp-strix-halo — what transfers (sub-agent report, 2026-08-28, condensed)

Fork base upstream a302733; 48 commits, all in ggml-cuda (HIP) + qwen35 models. No docs; claims
come from commit bodies. Patches repo: gaetan-puleo/llama-cpp-strix-halo-patches. Several
commits originate from pedapudi's gist/PR #21344 (closed upstream for dense regressions).

## Facts worth keeping
- Prefill on gfx1151 in llama.cpp is the **MMQ int8-WMMA path** (activations Q8_1, K-quants
  unpacked to int8 in LDS per tile, `wmma_i32_16x16x16_iu8`). Effective ≈ 21 TOPS of the 46 peak:
  Qwen3.6-27B Q8 ≈ 390 t/s, Qwen3.8-27B Q4_K_XL ≈ 417 t/s (pedapudi, pp4096, +13% over stock).
  Upstream's own crossover prefers dequant+hipBLAS only for dense Q6_K at n>256.
- Tile that works: 128 threads (4 waves), I=64 rows × J∈{48,64,128}, K-step 256, occupancy 2,
  ~40–50 KB LDS. 8 waves → split 4 row-groups × 2 col-groups (VGPR 232→136, occupancy 15→36%).
  Dense shapes are sensitive: the July snapshot regressed dense Qwen3.6-27B by up to 44% at long
  depth with the MoE-tuned tiles.
- MoE prefill recipe (Qwen3.6-35B-A3B, 512-expert Ling): one block per expert builds
  `expert_bounds` (sorted routing, no atomics); quantise activations once, shared by gate/up;
  choose J from *average* rows per expert (ub 2048 → ~40 rows → J=32/48), not the max; enqueue
  only non-empty (expert, J-tile) descriptors and grid-stride; fuse silu·up into the down-proj
  input quantisation; fuse weighted expert sum + shared expert + residual. Stacked: +8% (J48),
  +3.8% (paired quant), +3.4% (weighted sum), 8–20% of routed kernel time (compact grid).
  Large ubatch (2048–4096) is worth +30–60% by itself.
- Attention: D=256 with 4 KV heads f16 leaves the 32 MB MALL past ~8k tokens; chunking Q to 256
  columns per launch gave 1.4× at 32k, 1.8× at 64k (pedapudi, not in the fork). Decode with Q8_0
  KV dequantised inside the tile kernel: +18% tg at 30k; the same inside a WMMA D256 prefill
  kernel was 1.7× *slower* than f16 — quantised-KV benefit is kernel-shape dependent. rocWMMA FA
  regresses on gfx1151 (both authors).
- **GDN prefill is a sequential recurrence in every fork** (10–25% of prefill); the only
  improvement is wider blocks / LDS-staged 16-token tiles. A chunked (WY/UT) scan is untouched
  territory and the largest remaining prefill win for Qwen3.8.
- Their decode: MMVQ 1 wave/row, ≤4 columns; a 4-column Q8_0 kernel; BF16 MMVF for MTP batches.

## What we do with it
Prefill v1 here is dequant-to-bf16 + hipBLASLt (52 TFLOPS measured; dequant ≈ 13% overhead at
M=2048 → ~45 TFLOPS effective, i.e. > 2× the MMQ rate) — to be measured against an MMQ-style
int8 kernel before Flash-Next, where per-expert ragged batches need the hand-written path anyway.
