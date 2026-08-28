# Qwen3.8-27B prefill — measurements log

Driver: `./build.sh './build/prefill27b <gguf> docs/ref/<prompt>.json 512 1024 2048'` (one-shot
correctness on the reference prompt, then steady-state throughput on synthetic prompts; a warm-up
run absorbs hipBLASLt's per-shape autotune).
Path: activations bf16; each weight dequantised to a bf16 scratch, hipBLASLt bf16×bf16→bf16 GEMM
(f32 accumulate); same-input weights merged into one GEMM; GDN sequential kernel over T tokens
(state in registers); attention = the decode kernel with T causal queries (O(T²), scalar);
logits for the last row only; recurrent state committed with accept(T).

## 2026-08-28

| change | T=512 | T=1024 | T=2048 | note |
|---|---:|---:|---:|---|
| v1: f32 GEMM output, heuristic algo #0 | | 104 | 105 t/s | hipBLASLt ran at 5.7 TFLOPS |
| bf16 GEMM output + per-shape autotune, vectorised dequant | 230 | 258 | 253 | (autotune inside the timed run) |
| + merged same-input GEMMs, warm-up before timing | **418** | **491** | **475 t/s** | steady state |

Correctness: one-shot prefill of the 29-token reference prompt → top-1 identical, max |Δlogprob|
over top-10 = 0.14–0.16 (bf16 activations vs llama.cpp's q8), greedy continuation 12/12.

Roofline facts behind the choices: hipBLASLt bf16→**bf16** 51.8 TFLOPS on all shapes, →f32 5.7 (first
algo) / 30–34 (best); WMMA f16 = bf16 = iu8 = 54.8 TOPS, iu4 109.7 — so int8 GEMMs cannot beat
this path; only W4A4 could (2×). See docs/roofline.md addenda.

Profile at T=2048 (per prefill, 4.3 s): hipBLASLt GEMMs ≈ 2.4 s (~23 TFLOPS effective — see
below), k_attn 0.75 s (O(T²) scalar), gdn_step 0.42 s, dequant ≈ 0.35 s, bf16→f32 splits + norms
+ silu ≈ 0.4 s.

## 2026-08-28 (cont.) — exact attribution and the next cuts

Per-category GPU timing (`HIPSTER_TIMING=1`, events on the stream; no profiler pollution), T=2048:

| step | t/s | gemm | attn | gdn | dequant | split | norm/silu |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar attention (O(T²)) | 486 | 2383 | 743 | 333 | 314 | 212 | 196 |
| + WMMA flash attention (Q/K tiles 32, f16 WMMA, V transposed via LDS, bf16 out) | 563 | 2346 | **209** | 332 | 313 | 212 | 196 |
| + silu reads the GEMM output directly (no split for gate/up) | **582** | 2383 | 210 | 341 | 317 | 87 | 152 |

Facts and rejections:
- GEMM = 2.15 s floor for this model at hipBLASLt's isolated rates (51.8 TFLOPS on K=5120 shapes,
  36.7 on the K=17408 down-projection; **two K=8704 halves run at 52.3** → split-K adopted).
- Dequant is already at the bandwidth roof (16.5 GB read + 54 GB bf16 written ≈ 290 ms) — it is a
  fixed cost per chunk (27% at T=512, 9% at 2048, 4% at 4096).
- One-ahead dequant on a second stream (double-buffered scratch): **net loss** (−90 ms) — the
  GEMM fills every CU, so the second stream only runs in the gaps against the memory-bound
  split/norm kernels; a high-priority stream did not change that.
- GDN scan: 341 ms = 3.3 µs per token per layer, ~10× its arithmetic; no spills (147 VGPRs,
  occupancy 9); staging v/β/decay in LDS: no change; 4 blocks per head: worse (395 ms). Hypothesis
  under test: the GPU downclocks during this 48-block phase.
- 4096-token chunks: no gain (dequant halves, in-chunk attention doubles).
- The sustained-load check: hipBLASLt alone holds 51.5 TFLOPS for 10 s at 2.88 GHz (no power cap);
  the 600 MHz sysfs readings earlier were taken during model load.

## 2026-08-28 (cont.) — depth

| change | T=2048 | T=8192 | T=32768 | attention share @32K |
|---|---:|---:|---:|---:|
| flash v1 (V transposed through LDS per tile) | 627 | 568 | 345 t/s | 51% (48.7 s) |
| **V cache stored transposed** `[kv_head][dim][ctx]` (row stride padded to an odd multiple of 256 B), flash v2: 64-query blocks, V^T fragments from L2, no LDS transpose | **647** | **628** | **475 t/s** | 29% (19.5 s) |
| GDN scan v3: next tile register-prefetched, bank-swizzled LDS | (included) | | | gdn 341 → 240 ms @2K |

Halogen reports 566 t/s @32K on its own weights. Flash v2 still runs at ~10 TFLOPS effective
(243 VGPRs, occupancy 2); the next steps for depth are int8 KV (halves K/V traffic), a larger
Q tile per K/V fragment, and wave64 at hd256 (Nathan's measurement). Decode at depth: pending
(driver path under repair).

## 2026-08-28 — long-context correctness gate (`tools/ref-server-gpu.sh`, `tools/ref-long.py`)
Reference: llama.cpp ROCm build on this GPU, f16 KV, no speculation; haystack prompt with a needle
("amber-falcon-73") at 40% depth and a question at the end; compare the top-10 logprobs at the last
prompt position and the greedy answer.

| prompt | llama.cpp prefill | ours (chunks of 2048) | top-1 | greedy answer | max Δlogprob top-10 |
|---|---:|---:|---|---|---:|
| long16k (16,767 tok) | 67.0 s | 46.6 s | identical | 7/7 identical (needle found) | 0.35 |
| long32k (32,770 tok) | 137.9 s | 76.7 s | identical | 7/7 identical (needle found) | 0.39 |

Also found on the way: RoPE used fast `__cosf/__sinf` (imprecise at angles ~1e4–1e5 rad) → replaced
with `sincosf`; and a driver bug (a final chunk ≤ 8 tokens leaves one logits row per token — the
answer must be read from the last row). The spec-vs-plain "13/16" divergence is a near-tie flip
(top-1/top-2 gap 0.024 at that step) between the 1-column and 4-column GEMV paths, not a defect.
Next depth steps: 128K/262K references (llama.cpp needs ~10 min per 128K prefill), int8 KV, and a
unit test for the GQA split-KV kernel.
