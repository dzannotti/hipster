# Qwen3.8-27B single-token decode — measurements log

Engine: `engine/src/qwen35.cpp` (fixed shapes, f32 activations, int8 GEMV inputs, f16 KV).
Driver: `./build.sh './build/run27b <gguf> docs/ref/<prompt>.json'` — compares the next-token
top-10 logprobs and the greedy continuation with llama.cpp (CPU reference, `tools/ref-server.sh`).
Streamed bytes per token: 16.95 GB (17.67 GB loaded incl. the 0.72 GB embedding gather) → floor
70.6 ms = 14.2 t/s at 240 GB/s.

## 2026-08-28 · first full pass

- Load: 16.46 GiB to GTT; Q3_K/IQ3_S/IQ4_NL tensors (10 of 866) converted to Q8_0-SoA at load
  (+0.55 GB streamed; 3% — a per-format decoder would recover it, low priority).
- Correctness vs llama.cpp, prompt "The capital of France is": top-10 next-token set identical
  (9/10 ids), max |Δlogprob| 0.19 (top-1: −0.488 vs −0.449); greedy continuation identical for 12
  tokens ("Paris.\nThe capital of Germany is Berlin.\nThe").
- **86.0 ms/token = 11.6 t/s** (197 GB/s streamed, 82% of roof). rocprofv3 breakdown per token:

| kernel | calls/token | ms/token | note |
|---|---:|---:|---|
| GEMV Q5_K (7.9 GB) | 191 | 36.4 | 218 GB/s |
| GEMV IQ4_XS (3.1 GB) | 70 | 14.3 | 218 GB/s |
| GEMV Q6_K-SoA (2.9 GB incl. LM head) | 50 | 12.8 | 224 GB/s |
| GEMV Q4_K (2.3 GB) | 68 | 10.7 | 217 GB/s |
| GEMV Q8_0-SoA (1.1 GB) | 118 | 5.7 | 187 GB/s |
| rmsnorm | 129 | 1.1 | 8.8 µs each — 3× too slow for 20 KB |
| gdn_step | 48 | 1.0 | 6 MB state r/w per layer = at its own roof |
| quantize / add / conv / silu / attn / rope | ~440 | 0.7 | |
| launch gaps (wall − kernels) | ~640 launches | 3.0 | |

Gap to floor = 15.4 ms: ~9 ms is GEMV ramp/tail (450 launches × ~15–20 µs), ~3 ms launch gaps,
~2 ms small kernels. Plan, by size: fuse same-input GEMVs into one launch (qkv+z+β+α, q+k+v,
gate+up: 450 → ~250 launches); fuse add+rmsnorm+quantize and silu·mul+quantize; then Q8_0.
- Longer prompts: "code" (29 tokens) and "story" (19-token chat template) both greedy-identical for
  12 generated tokens; top-1 |Δlogprob| 0.002–0.04, worst tail token 0.32 at logprob −8.6.

## 2026-08-28 · fusions, and what did not help

- Fused: same-input GEMVs into one launch (`gemv_multi`: qkv+z+β+α, q+k+v, gate+up), residual
  add + rmsnorm + q8-quantise, silu·mul + quantise, in-kernel quantise of the GDN and attention
  outputs. Launches/token ≈ 640 → ≈ 350. **86.0 → 81.6 ms/token (12.25 t/s, 208 GB/s streamed,
  87% of floor).** Greedy output unchanged (12/12 on all three prompts).
- Did not help (measured, reverted to default): persistent GEMV grids capped at 160–1280 blocks
  (81.5–83.6 ms, noise); one 18 GiB weight arena instead of 866 allocations (81.6 ms) — kept for
  simplicity, but page-table locality is not the loss.
- Remaining ~11 ms: the GEMV bench already shows it — 34–45 MB tensors rotated cold reach 87–89%
  where a 682 MB tensor reaches 98%. Per-launch cold start inside the memory system, not
  scheduling. Ideas not yet tried: prefetch the next layer's weights into MALL (32 MB) during the
  GDN/attention kernels; larger effective launches by fusing the down-projection with the next
  layer's qkv (different inputs — needs a dependency-free formulation). Parked: the 3–4× lever is
  speculation, which is next.
- Non-temporal weight loads (`__builtin_nontemporal_load` on the 16-byte weight loads, per the
  Nathan-fork suggestion): **80.1 → 132.0 ms/token**. On gfx11 the nt bits bypass L2 and break
  the coalescer; reverted (compile switch `HIPSTER_NT_WEIGHTS` kept at 0).
- KV cache moved to head-major `[layer][kv_head][max_ctx][256]` (per-head stride was 2048 B =
  2 of 16 memory channels on this part; see docs/forks/nathan-strix-halo-llamacpp.md). No effect
  at short context, exact output; the gain shows at depth.

## 2026-08-28 (cont.) — decode at depth (prefill driver, `HIPSTER_DEPTH=1`)

| depth | old per-q-head kernel (K/V read 6×) | GQA-packed split-KV WMMA kernel (512-position splits) |
|---:|---:|---:|
| 2048 | 81.4 ms/token | 82.5 |
| 8192 | 83.1 | |
| 32768 | 95.1 (10.5 t/s) | 97.3 |

KV at 32K is 2 GB per token → 8.7 ms at the roof; both kernels lose ~15 ms, the new one for lack
of parallelism (256 single-wave blocks). 128-position splits are being measured. At 262K the f16
KV alone is 17 GB per token (≥ 70 ms) — int8 KV is mandatory for the 262K × 4-slot target.

## 2026-08-28 · WMMA GEMV (the T-invariant verify/decode kernel) — what was tried

`bench_gemv <tensor> 8 q8` rows `rpt=4`: `tpr=1/2/4/8` = split-K waves per 16-row tile, `tpr=3` = 16-row interleaved
layout. GB/s of weights at ncol=1 / ncol=8, one 8-wave block of 16-row tiles:

| variant | Q5_K ffn_up 17408×5120 | Q5_K ffn_down 5120×17408 | Q4_K ffn_gate | Q8_0 ssm_out 6144×5120 |
|---|---|---|---|---|
| old kernels (tpr=32 ncol=1 / ncol-tuned ncol=8) | 222 / 90 | 213 / 87 | 218 / 99 | 153 / 82 |
| WMMA, per-format kernel | 188 / 147 | 190 / 138 | 190 / 164 | 198 / 174 |
| + split-K 4 waves per tile | 197 / 160 | 203 / 167 | 203 / 159 | 194 / 176 |
| 16-row interleaved layout (KS 1) | 203 / 161 | — | 194 / 191 | 213 / 198 |
| + prefetch next block, x fragments per sub-block | 206 / 165 | 203 / 169 | 203 / 161 | 165 / 144 (KS4) |
| scale loads as float4 (8 sub-blocks) | spills (256 VGPRs) → 77–92 | | | |
| scale loads as float4 (4 sub-blocks) | spills → 76–90 | | | |

Neither occupancy (split-K: +5–10%, worse for Q8_0), coalescing (interleaving: +3–15%, −35% for Q6_K), nor latency
(prefetch: ±3%) explains the ~165 GB/s ceiling of the K-quants at 8 columns; per 16-row × 256 block a wave issues
11 weight loads, 16 x-fragment loads and 128 scalar scale loads (`xd`/`xs` per column and sub-block) — the scale loads
dominate the vector-memory instruction count, and holding them in registers spills. Engine (`PROF_T=1 HIPSTER_TIMING=1
dflash27b …`, throttled box): T=1 101 ms (GEMV 85), T=8 120 ms (GEMV 98) with WMMA; old kernels 90 / 152.
Auto mode (`gemv_wmma(..., mode 0)`): split-K 8/4/2/1 by rows (< 4096 / < 12288 / < 65536 / above).

Corrections after a rocprofv3 trace: `HIPSTER_TIMING` events cost ~9 ms per pass, so the real pass times are **T=1 92 ms,
T=2 93, T=4 97, T=8 106 ms** (`PROF_T=1 dflash27b …` without timing); launch gaps are 18 ms per 630 ms of kernels (3%,
median 2 µs). The LM head still ran the old `k_gemv_q8_lds` kernel (`gemv()` member) — now routed through the same
T-invariant selection (T=8: 108.7 → 106.4 ms).

Q8_K activations prototype (`bench_gemv` rows `tpr=7/9 rpt=4`: one scale per 256 + int16 sub-block sums, integer
sub-block accumulation with the int weight scales, one float update per block; llama.cpp's K-quant dot layout): at
8 columns Q4_K 161 → 190 GB/s, IQ4_XS 131 → 155, Q6_K 0.482 → 0.417 ms, Q8_0 143 → 180, but **Q5_K 163 → 176 / 169 → 170**
(the 7.9 GB bulk: its unpack, not the epilogue, is the cost); error vs f32 grows 3.4e-4 → 4.9e-4 (Q5_K). Weighted over
the checkpoint ≈ −12 ms of the 97 ms GEMV at T=8; not adopted (engine-wide quantiser change for ~8% end-to-end, less
precise activations). Kept as a bench option.

## 2026-08-28 · multi-query decode attention (T rows × 6 heads per kv head, split-KV, WMMA f16)

At 16K context the T=8 verify pass cost 265 ms (T=1: 110): `k_attn` launched one block per (row, q head), so the 16-layer
KV was streamed once per row and head — 26 GB per pass. `k_attn_mq` (ops.hip) reads each K/V tile once per wave for a row's
6 heads (rows 0..5 of a 16-row WMMA f16 tile, prefill-kernel tiling), one block per (kv head, slot group, 512-key split),
partials merged in split order (`k_attn_mq_merge`, + gate + q8). `HIPSTER_ATTN=old` restores the scalar kernel.

| context | pass | old kernel | multi-query |
|---|---|---:|---:|
| 29 tokens | T=1 / T=8 | 92 / 106 ms | 93 / 107 |
| 16K | T=1 / T=2 / T=4 / T=8 | 110 / 122 / 188 / 265 | 106 / 106 / 110 / 120 |
| 16K, DFlash2 n=7 (15% acceptance on that prompt) | t/s | 9.65 | **13.5** |

**Exactness finding (hardware).** The first version put the 6 heads × T rows of a group into one 48-row tile and was
T-variant: a row's logits differed by up to 0.6 between a T=1 and a T=8 pass, deterministically, although its softmax
statistics were bit-identical. The cause is `wmma_f32_16x16x16_f16`: `bench/roofline/wmma_pair.hip` shows that with a
single zero in the A operand and only the matching B element changed, **176 of 256 outputs change by an ulp** — the unit's
internal alignment uses operands whose weight is exactly zero. In P·V the V of keys beyond a row's position (masked, P = 0)
therefore leaks into the rounding, and a 16-key V tile shared by rows with different positions cannot be made identical for
each of them. The kernel now gives every batch row its own wave, and each wave zeroes V per key beyond its own position
(and loads no K beyond it). Result: rows of a T=8 pass bit-identical to eight T=1 passes (`DIAG=1 ROWSONLY=1 dflash27b …`),
`HIPSTER_LOGIT_DIFF` 0.0000 at T=8, llama.cpp greedy 12/12, DFlash2 EXACT at 29 tokens, at 16K and with 2 slots. The same
property means the prefill flash-attention kernel's output depends on the (zero) V padding beyond the prompt — consistent
by construction, but worth knowing. Q is f16 in this kernel (f32 before): the greedy stream on the code prompt now
takes the other branch at its near-tie token 13 (gap 0.024), so DFlash2 acceptance statistics on that prompt changed
(78% → 72%): single 48.4 t/s, 2 slots 79.0 aggregate, both exact.
