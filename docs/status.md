# hipster — status 2026-08-28 (end of day 1)

## Numbers (Qwen3.8-27B UD-Q4_K_XL, this box, exact vs llama.cpp unless noted)
| | value | floor / reference |
|---|---:|---|
| bare decode | **12.5 t/s** (80.1 ms) | 14.6 t/s floor (16.95 GB/token @240 GB/s); llama.cpp 11.7–14; hipfire 14.8 (Qwen3.5) |
| MTP speculative decode (n=5, code) | **35.6 t/s**, 85% acceptance, exact | halogen 26.9 (MTP) / 31.7 (DFlash2); prose n=3 21.0 |
| prefill T=2048 | **647 t/s** | ~950 t/s GEMM floor at 52 TFLOPS; llama.cpp 300–440; halogen 566 @32K |
| prefill T=32K | **475 t/s** | attention now 29% of the time |
| decode at 32K depth | 10.5 t/s | KV read 2 GB/token |
| long-context gate | 16K and 32K needle: top-1 identical, greedy 7/7, Δlogprob ≤ 0.39 | |

Flash-Next: decode engine runs, **12/12 greedy identical to llama.cpp, 17.5 t/s** (floor 37; llama.cpp
18–27); prefill/QSA/MTP not yet (docs/decode-flash-next.md).


## 2026-08-28 (late) · where the 27B stands after DFlash2, slots and the attention rewrite

| | value | notes |
|---|---|---|
| bare decode (T-invariant WMMA GEMV path) | 10.8 t/s (92 ms) | `HIPSTER_GEMV=fast` (old kernels, not T-invariant) 12.5 |
| verify pass T=8 | 106 ms (short ctx), 120 ms at 16K | was 152 (old kernels) / 265 at 16K (old attention) |
| DFlash2 n=7, code | **48 t/s**, exact | 22 on prose; 13.5 at 16K context (15% acceptance there) |
| DFlash2 × 2 slots | **79 t/s aggregate**, both exact | one 16-row verify + one 16-row draft pass per round |
| served (OpenAI endpoint, DFlash2) | 46 t/s on a 200-token code reply | one slot per server so far |
| MTP n=5 (WMMA verify) | 33.8 t/s, exact | |
| T-invariance | max logit diff 0.0000 at T=8 | GEMV (WMMA, fixed reduction order) + attention (per-row V masking) |
| ngram-map-k4v in front of DFlash2 | neutral at T≤8 | 9 of 78 rounds drafted by the map on 512 tokens of code |
| served, 27B (OpenAI endpoint, DFlash2, `--slots 2`) | 46 t/s alone, 68 aggregate for 2 concurrent code requests | short prompts through 16-row GEMV passes (372 ms) |
| served, Flash-Next (`--slots 4`, MTP policy n=2/1/0) | 43.7 alone, 48 for 2, 51 for 4 concurrent | outputs identical to solo/plain runs |
| multi-query decode attention (27B) | 16K T=8 pass 265 → 120 ms | f16 WMMA rounding depends on zero-weight operands → per-row V masking |

Flash-Next unchanged (36.3 ms/token, MTP 37–41 t/s, 63/80 t/s @4/8 slots, prefill ~1000 t/s @2K).

## What exists
`engine/` C++/HIP: GGUF mmap loader; GEMV for Q4_K/Q5_K/IQ4_XS/Q6_K/Q8_0/Q5_1 (dot4 and WMMA paths,
all validated to 1e-8 vs CPU reference); fused norm/quant kernels; GDN conv+step (state in
registers, replay for speculative rollback); WMMA flash attention; head-major K, transposed V cache
(f16, int8 variant written but untested); dequant-to-bf16 + hipBLASLt prefill with per-shape
autotune, merged GEMMs, split-K; MTP drafting/verify; Flash-Next kernels (gated residual, MoE,
PLE). Drivers: `run27b` (vs llama.cpp), `spec27b` (spec == plain check), `prefill27b` (one-shot +
throughput + depth), `runfn`. Reference tools under `tools/`.

## Decisions with data
HIP over Vulkan (hipBLASLt 52 vs coopmat 15 TFLOPS) · int8 x + dot4 for 1–4 columns, WMMA for
5–16 · f16/bf16 = iu8 WMMA rate, iu4 2× · dequant+hipBLASLt over MMQ · bf16 GEMM output (f32 is 6×
slower) · head-major/transposed KV (memory channels) · rejected: NT loads, persistent grids,
interleaved weights, prefetch streams, 4096 chunks, GQA split-KV v1 (see docs/ideas.md).

## Hard constraint (2026-08-28)
Model weights < 100 GB RAM, no paging (KV may exceed; 2 × 262K acceptable): 84 GB Flash-Next weights;
n-gram table on SSD; weights loaded via pread with the page cache dropped per tensor.

## Open / next
int8 KV validation at 32K → 128K/262K gate · Flash-Next correctness → MoE/expert GEMV speed → QSA
top-k → MTP · W4A4 iu4 prefill with per-256 segment scales · DFlash2 · draft-only LM head ·
multi-slot batching · serving layer (Rust front-end proposed) · own artifact format.
Uncommitted work: everything after the first commit (ask before committing/pushing).
