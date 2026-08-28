# Ideas board — every idea gets a status and a number

Status: `untested` · `measured: <result>` · `adopted` · `rejected: <number>`. Nothing is dismissed
without a row here. Add ideas freely; remove nothing.

## Decode GEMV (bandwidth-bound; roof 240 GB/s; big tensors 98%, real 34–45 MB tensors 87–89%)
| idea | status |
|---|---|
| threads-per-row sweep 4/8/16/32 | measured: tpr=32 best at 1 col, tpr=4 at 3–4 (docs/decode-gemv.md) |
| 2 rows/thread to amortise x | rejected: 5–15% slower (registers) |
| x staged in LDS | adopted for 3–4 columns (95%); no help at 8 |
| int8 x + v_dot4 instead of f32 x | adopted (94→98%) |
| WMMA iu8 for ≥5 columns | adopted (60→87% at 8 cols) |
| persistent grid | rejected: 81.5–83.6 ms, noise |
| one arena vs 866 allocations | measured: no change (kept) |
| non-temporal weight loads | rejected: 80→132 ms |
| **is the small-tensor loss the kernel or the memory system?** plain copy of a cold 34 MB stream vs GEMV | measured: memory system is fine (234 GB/s cold at 34 MB); row-per-wave pattern costs 3–5%; ~12 µs/launch of GEMV ramp on top |
| repacked weight layout (interleave 8 rows at block granularity so each wave streams contiguous memory) | rejected: 209 vs 211 GB/s (Q5_K 34 MB), 233 vs 235 (Q4_K 682 MB) — a full row per wave is already contiguous |
| prefetch next layer's weights into MALL (32 MB) during the GDN/attention kernels | untested |
| two streams: overlap the tail of one GEMV with the head of the next independent one | untested |
| native decoders for Q3_K/IQ3_S/IQ4_NL (−0.55 GB/token, 3%) | untested |
| Q8_0 SoA at 79% → why? (interleaved d/q layout, 2 rows/lane) | untested |
| f16/bf16 x instead of int8 for the verify path (accuracy vs speed) | untested |
| WMMA decoders for Q5_K/Q6_K/IQ4_XS/Q8_0 (verify pass at T=6 is 1.37× a T=1 pass; target 1.2×) | untested |
| bigger effective launches: fuse down-proj GEMV with the next layer's norm+qkv (needs a dependency-free formulation) | untested |
| Vulkan/RADV control GEMV+GEMM | untested (chose HIP on hipBLASLt 52 vs coopmat 15 TFLOPS; the decode side was never compared) |

## Our own artifact format (user idea, 2026-08-28)
A converter (offline, Python) producing a hipster-native file so the runtime loads exactly what its
kernels want. Measured so far: GGUF block layouts themselves are within 2% of roof for the GEMV, so
the value is elsewhere:
| sub-idea | status |
|---|---|
| merged tensors: qkv+z+β+α, q+k+v, gate+up stored as ONE matrix each → one launch, one contiguous stream, no segment table | untested (launch count is the measured remaining decode loss) |
| chosen bit budget: requantise Flash-Next's 4.1 GB of Q8_0 dense tensors to ~5 bits (floor 37 → ~60 t/s) | untested |
| pre-converted odd formats (no Q3_K/IQ3_S/IQ4_NL → Q8 at load; native or better formats) | untested |
| pre-split layouts for the WMMA verify path / per-format SoA | untested |
| weights loaded with `pread` + `posix_fadvise(DONTNEED)` instead of mmap (mmap kept only for on-demand n-gram rows): no page-cache residency, no thrash at 84 GB GTT; the first madvise-only attempt did not drop the cache | adopted (both loaders) |
| zero-copy: mmap the file and register it as GPU-visible memory (unified memory; no 17 GB upload, instant load). Caveat: llama.cpp issue #26209 (host-memory compute corruption on gfx1151) — must validate | untested |
| int8 KV + fused activation formats decided by the file | untested |

## Bit-level tricks (user idea, 2026-08-28: "Quake-3 fast-inverse-sqrt level packing")
| sub-idea | where it would pay | status |
|---|---|---|
| packed f16x2 / bf16x2 math (`v_pk_fma_f16`, 2× the f32 rate) in dequant, GEMV epilogues, softmax | dequant kernel (6% of prefill), WMMA epilogue (13% gap) | untested |
| magic-number int→float: `(0x4B00 \| q)` as f16 = 2048+q, no cvt; 2 nibbles per op with packed subtract | dequant-to-bf16, GDN/attention converts | untested |
| `(q + 0x60) ^ 0x80` byte-wise signed unpack (already used for Q6_K), `v_perm` LUTs (IQ4) | adopted in the dot4 decoders | adopted |
| exp2 via exponent-field bit tricks + short polynomial in softmax / GDN gates (`__expf` is already fast; measure) | k_attn softmax, gdn_step | untested |
| 64-bit packed nibble/bit unpack with `v_alignbit` / `v_bfe` instead of shift+mask chains | Q5_K/Q6_K decoders | untested |
| rsqrt via `v_rsq_f32` (native) vs Quake trick — native is one instruction; nothing to gain | norm kernels | rejected by inspection |

## Prefill GEMM (compute-bound; roofs: hipBLASLt bf16 51.8 TFLOPS, CK int8 46 TOPS, iu4 WMMA ?)
| idea | status |
|---|---|
| dequant to bf16 scratch + hipBLASLt | measured: 253 t/s @2048 (v1); f32 output made hipBLASLt 6× slower — bf16 output + per-shape autotune fixed it (docs/prefill-27b.md) |
| merge same-input weights into one GEMM (qkv+z+β+α, q+k+v, gate+up) — also removes the N=48 GEMMs that take 8 ms each | adopted (part of 253 → 475 t/s with warm-up fix) |
| split-K for the K=17408 down-projection (hipBLASLt 37 → 52 TFLOPS on K=8704 halves) | adopted: 582 → 608 t/s |
| one-ahead dequant on a second stream (double-buffered scratch), also with high priority | rejected: net −90 ms / neutral — GEMM fills all CUs; dequant lands on the memory-bound glue instead |
| 4096-token chunks | rejected: 580 vs 580 t/s (dequant halves, in-chunk attention doubles) |
| bf16 consumers instead of bf16→f32 split kernels | partially adopted (silu reads GEMM output: −125 ms); residual/GDN/attention inputs still split (~90 ms) |
| MMQ-style int8 WMMA tile kernel (llama.cpp gets ~21 TOPS effective on this GPU) | deprioritised by measurement: iu8 = bf16 rate, so it can only match dequant+hipBLASLt, never beat it |
| W4A4 iu4 WMMA (Nathan: private RADV int4 coopmat 108 TOPS = 2× f16; +21.6% pp on the 27B, KLD 0.073; halogen ships it as an option, −0.45 pt) | untested — measure the raw instruction rate first |
| WMMA instruction-rate microbench f16 / bf16 / iu8 / iu4 | measured twice (4/8/16 chains, ISA checked): 54 / 55 / 55 / **109** TOPS — int8 = f16 rate on gfx1151; iu4 is 2× |
| fuse dequant into the GEMM A-tile load (no scratch write/read) | untested |
| activation quant to int8 per row (W8A8 via CK, 46 TOPS) | untested |

## Attention
| idea | status |
|---|---|
| head-major KV (memory-channel camping fix) | adopted (exact; measure at depth) |
| int8 KV (K: per-32-dim f16 scales; V^T: one scale per position) as compile-time kernel variants, `HIPSTER_KV8=1` | implemented (decode + flash prefill + MTP cache), untested — required for 4 × 262K (f16 KV = 17 GB per slot) |
| decode attention: GQA group (6 q-heads × T tokens) as WMMA rows, K/V read once, split-KV with LSE merge | implemented; 512-position splits: no gain at 32K; 128-position splits: slower (attention 20.8 vs 15.3 ms/token at 32K) and NOT exact (13/16 in the spec check) → env-gated `HIPSTER_GQA_ATTN`, old kernel default; needs a unit test against the scalar kernel before more tuning |
| RoPE with fast `__cosf/__sinf` at positions up to 262144 | replaced with `sincosf`; 16K/32K gate green |
| long-context reference harness (llama.cpp GPU, f16 KV, needle at 40% depth, 16K/32K) | adopted: 16K and 32K pass (top-1 + 7/7 greedy identical, Δlogprob ≤ 0.39); run after every depth-affecting change |
| flash-attention prefill kernel (WMMA f16) | adopted: v1 743 → 209 ms @2K; v2 with the V cache stored transposed + 64-query blocks: 102 ms @2K, 48.7 → 19.5 s @32K (still ~10 TFLOPS; bigger Q tile / wave64 / int8 KV next) |
| V cache stored transposed `[kh][dim][ctx]` (padded stride) so V fragments are contiguous for WMMA and decode streams avoid channel camping | adopted |
| Q chunking (256 cols) once KV > MALL | untested |

## GDN
| idea | status |
|---|---|
| state in registers across T tokens (double-buffered, replay on rejection) | adopted |
| GDN scan tile loads were latency-bound (counters: wait/busy 2.5; 24 serialized global loads per 16-token tile) → register-prefetch the next tile, bank-swizzled LDS | adopted: 341 → 240 ms @2K |
| chunked scan for prefill (untouched by every fork) | untested — scan now 7% of prefill @2K |
| 4 blocks per head (more parallelism) | rejected: 332 → 395 ms |

## Flash-Next (from the DeepSeek / b12x / Gaetan / Nathan reports; docs/forks/)
| idea | status |
|---|---|
| expert GEMV: one launch over the 10 routed experts × T tokens, expert ids in device memory (no host sync) | measured: gate\|up + shared expert in one launch 227 GB/s; down (K=640) 152 GB/s at 16 lanes/row; combine folded into the launch (atomics) |
| prefetch expert i+1 while computing expert i; SwiGLU × routing weight fused into the down-proj input | untested |
| requantise the 4.1 GB of Q8_0 dense tensors (attention/GDN/hc) to ~5 bits (floor 37 → ~60 t/s) — needs the artifact converter | untested |
| QSA indexer: index keys on the WMMA M axis, (queries × 4 heads) on N, one score per (q, block); int8 keys with power-of-two scales | untested |
| QSA top-512 blocks: LDS radix select; previous step's threshold as the initial guess; one selection per MTP round (IndexShare) | untested |
| sparse gather attention: 64-key index tiles, −1 sentinels, base-2 online softmax with lazy rescale, split-KV + LSE merge | untested |
| PLE gather: hash on the host at sampling time, issue the 16-row gather + key/value GEMVs so it lands by layer 1; hot-row LRU | partially (host hash + sync gather); prefetch untested |
| gated residual: read/write as two fused kernels; residual streams in FP8 (Qwen: negligible loss) | write folded into the next read's norm (0 launches); read = norm + split-K down + silu + up + mix; FP8 streams untested |
| MoE prefill: sort tokens by expert, per-expert GEMM tiles sized by average rows/expert, shared activation quant for gate/up | untested |
| MTP block (QSA + MoE + hc) with the frozen top-k across draft steps | next: blk.48 in /srv/models/qwen3.8-flash-next/mtp (attention + indexer + MoE + hc + nextn.eh_proj/enorm/hnorm) |
| lanes-per-row policy per tensor shape (short K rows are issue-bound, not bandwidth-bound) | measured, applied: docs/decode-flash-next.md table |
| hc read chain as one persistent kernel (norm → 320-wide down → silu → up → mix; 5 launches + 3.3 MB weights each) | untested |
| `k_topk` over 248K logits: two-stage argmax (270 µs today) | untested |
| router as bf16 (5.25 → 2.6 MB/layer, 0.7 ms/token) — changes routing near ties, needs the exactness gate | untested |

## Quantisation choice (user question 2026-08-28: Q4_K_XS vs UD-Q4_K_XL)
| idea | status |
|---|---|
| Flash-Next: the XL keeps GDN/attention projections at Q8_0 = 1.96 GB of the 6.1 GB streamed per token (32%); a lower-bit variant of just those tensors is worth up to ~4 ms/token | untested — needs the KL/needle gate on the candidate file |
| 27B: XS vs XL — speed is bytes-proportional (decode is bandwidth-bound), quality must be measured with our KL-vs-reference + 16K/32K needle gate | untested — needs the XS GGUF on a non-/srv path |

## Speculation
| idea | status |
|---|---|
| MTP n=1..5 | measured: 35.6 t/s code at n=5, exact |
| DFlash2 drafter (27B; docs/forks/dflash2.md) | researched: 1.8–2.3× on llama.cpp vs our MTP 2.85×; ≈1.3× over MTP if τ=5.4 holds; measure our per-position acceptance first |
| draft-only LM head at Q4_K / shortlisted rows | untested |
| adaptive n from acceptance EMA | untested |
| n-gram self-speculation (ngram-map-k4v style, host, ≤48-token drafts, first non-empty draft wins) stacked in front of MTP | researched: this is where the "6×" comes from (multi-turn code sessions); needs verify T up to 48 |
