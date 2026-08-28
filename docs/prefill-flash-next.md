# Qwen3.8-Flash-Next prefill — measurements log

Driver: `./build.sh './build/prefillfn <XL-MTP shard1> docs/ref/fn-code.json 12 2048 512 1024 2048'` — chunked one-shot
prefill of the reference prompt checked against the token-by-token path (last-row top-1, greedy continuation) and
against llama.cpp's top-10 logprobs, then steady-state throughput (warm-up run absorbs hipBLASLt's per-shape autotune).
`HIPSTER_TIMING=1` prints per-phase GPU time and per-shape GEMM min/avg; `HIPSTER_DBG_LAYERS=0,1,3` prints per-layer
activation diffs prefill vs decode for the last token.

Path: bf16 activations; dense linears = dequant to bf16 scratch + hipBLASLt (Q8_0/Q4_K/Q5_1/Q6_K); hyper-connection
read = norm → down GEMM → silu → up GEMM → mix (bf16 in between); GDN = the 27B's tiled scan (prep/scan/out with the
sigmoid gate); attention = the WMMA flash prefill with a 24q/2kv variant; router = f32 hipBLASLt GEMM (exact routing);
MoE = slots sorted by expert on the host (one sync per layer), grouped WMMA GEMM over 16..48-column tiles per expert,
shared expert as its own tile list; PLE = host hash + MADV_WILLNEED gather, key/value GEMMs, sequential conv update.

## 2026-08-28
| step | 512 | 1024 | 2048 | note |
|---|---:|---:|---:|---|
| token-by-token (before) | 27 | 27 | 27 t/s | |
| GEMM path, MoE through the decode gather kernels | 140 | 141 | 141 | MoE re-reads 1.6 GB of experts per token |
| + grouped WMMA MoE GEMM (int8 x, 16-col tiles) | 710 | 860 | 965 | 89% of the time had been the MoE |
| + LDS-staged x (padded: the first version had a 16-way bank conflict), routed/shared split, Q5_K | 829 | 971 | **~1000** | bf16-x variant with 48-col tiles is now the best MoE kernel |

Correctness: last-row top-1 identical to the decode path, 12/12 greedy continuation identical, max |logprob diff| vs
llama.cpp over the top-10 ≈ 1.8–2.6 (decode path: 0.45). Bisected with the layer diffs and synthetic kernel tests:
- GDN prefill scan vs sequential step: 1e-7 relative; 24/2 flash attention vs decode: 0.26% (bf16 output rounding).
- `mixed` after the very first hyper-connection read of layer 0 already differs 4% rms from decode at T=1 — and an
  all-f32 chain (f32 dequantised weights, f32 GEMMs, f32 silu/mix) leaves that unchanged, as does f32 GEMM output.
  So the noise is in **decode's int8 activation quantisation** (q8 per 32), amplified ~10× by the 10240→320 low-rank
  bottleneck of the hyper-connection. Decode and llama.cpp quantise activations identically (per-32 absmax, round
  to nearest) and therefore agree (0.45); the bf16/f32 prefill is the more precise of the three. Not a prefill bug.
  Idea logged: decode's hc chain with bf16 activations (weight-bound, so free) would move decode toward the precise
  answer at the price of greedy agreement with llama.cpp.

Profile at 2048 (per run; the phase timer's first run includes ~1 s of autotune, corrected here): MoE ≈ 1.0 s (50%),
dense GEMMs ≈ 0.3 s (per-shape minima match the isolated hipBLASLt bench within 10%: N=13312/K=2560 3.0 vs 2.8 ms),
hyper-connection chain ≈ 0.25 s, GDN scan 0.18, router f32 GEMM 0.14 (hipBLASLt f32 at ~2 TFLOPS — a custom f32 GEMM
is worth ~0.1 s), PLE 0.13, attention 0.08.
Rejected with numbers: transposed GEMM (weights as the streamed operand: no change), cold-cache autotune (±10%),
blocking host sync (the GPU already runs at 2.8 GHz during prefill; the 700 MHz samples were the load phase),
f32 GEMM outputs, f32 hc chain (no precision change, slower).

MoE grouped GEMM microbench (`bench_moe`, blk.0 gate experts Q4_K, 2048 tokens × 10 experts, 512 experts):
int8-x kernel 4.5 ms (14.8 TOPS), bf16-x kernel 4.9 ms at 48-column tiles (13.7 TOPS), and the same bf16 kernel with
constant weight fragments (no dequant) **2.1 ms = 32 TOPS / 224 GB/s** — the floor of this tiling. The per-fragment
dequant (32 values: cvt + FMA + bf16 round) is the remaining cost → next: f16 fragments via packed converts and wider
column tiles (111 VGPRs at 3 tiles leaves room for 6).

## 2026-08-28 (cont.) — MoE grouped-GEMM kernel: what was tried, with numbers
`bench_moe` on blk.0 gate experts (Q4_K, 2048 tokens, 512 experts; ms per launch, TOPS):
| variant | 16-col tiles | 32 | 48 | 64 | 96 |
|---|---:|---:|---:|---:|---:|
| int8 x + iu8 WMMA, per-column scale fix-ups (LDS-staged, padded) | **4.5 / 14.8** | 4.7 | 5.2 | | |
| bf16 x + bf16 WMMA, scales folded into the fragments | 9.3 | 6.8 | 4.9 / 13.7 | | |
| same with constant fragments (no dequant, no weight loads): the tiling's floor | 2.1 / **32** | 2.5 | 2.2 | | |
| f16 x, packed converts, runtime sub-tile count | 7.3 | 5.2 | 4.5 | 5.4 | 5.3 |
| + raw-block register prefetch (spilled: dynamic register indexing) | 7.9 | 5.9 | 5.7 | | |
| + fully unrolled sub-block loop (no spills) | 5.3 | 4.7 | 5.2 | | |
Hardware counters (f16, 48-col): ~444 M VALU wave-instructions per launch (≈26% of issue capacity), waves waiting
≈50% of their cycles → latency-bound on the per-sub-block weight loads at 3–5 waves/SIMD; prefetching them costs
the registers that the occupancy needed. The two lane-halves of a WMMA fragment dequantise the same row twice
(the layout mirrors them) — a half-split dequant + lane swap is the remaining structural idea (≤2× on the dequant part).
Decision: the int8 kernel with 16-column tiles stays (fastest measured, and its activation quantisation is decode's).
MoE ≈ 1.0 s of the 2.05 s prefill at 2048 tokens; the floor of this tiling would be ≈ 0.45 s.
