# DeepSeek V4 family (V4 report, V3.2/DSA, Engram, mHC, FlashMLA, DeepGEMM, TileKernels, DSpark)
# — what transfers to Flash-Next and the 27B (sub-agent report 2026-08-28, condensed; [V] verified in source)

Corrections to folklore: V4 does **not** use Engram (only a static hash-routing table for the first
3 MoE layers); Flash-Next's residual is Qwen's **Gated Residual**, not mHC — no Sinkhorn, no
branch-mixing matrix ("removes a read of the whole residual state per block").

## QSA / DSA indexer + top-k (Flash-Next: 4 index heads × 128, 1 key head, r=4, budget 2048)
- Score `I_{i,b} = Σ_h ReLU(⟨q_h, k̄_b⟩)` over fully-observed 4-token blocks; `k̄_b` = RMSNorm(mean of 4
  raw keys, pooled in fp32) then RoPE at the block start; queries normed+roped per head. Tail
  (incomplete block, ≤3 tokens) always included; exactly dense below 2051 cached tokens [V].
- DeepGEMM layout [V]: **KV rows on the MMA M axis, (query tokens × heads) on N**, BLOCK_KV=256 →
  the ReLU-weighted head sum is register-local; one scalar per (q, block) is stored — never a
  `[heads × T]` tensor (StreamIndex: the naive tensor is 256 GB at 64K). With 4 heads × ≤4 MTP
  tokens that is N=16 = one WMMA tile.
- Index keys: int8/fp4 with power-of-two (UE8M0) block-32 scales; scores in bf16 → 2× top-k speed at
  99.7% recall [V V4 §5.2.1].
- Top-k: LDS radix select (TileLang selector: 8-bit sign-folded histogram, ≤4096 candidates refined)
  — fully portable. Temporal stability: 35–60% of the set persists between steps → use the previous
  step's threshold as the initial guess (Guess-Verify-Refine, 1.88× vs radix); HiSparse: an LRU of
  2×k turns 87% of selections into hits. **Reuse one selection for the whole MTP draft round**
  (Qwen IndexShare: 4.06 → 4.07 accepted, no loss) but do **not** share across layers (Qwen Fig.5).
- Known gfx1151 hazards: llama.cpp HIP TOP_K fails at 4096-token prefill with k=512; Vulkan top-k
  assert on long generations.

## Sparse gather attention (FlashMLA "MODEL1", portable parts) [V]
64-token index tiles (`page = idx/page_size`, `−1` ⇒ zero row & −inf), next tile's indices prefetched
into registers, per-request `topk_length` skips padded tiles; base-2 online softmax with
`m_init = −1e30` (never −inf), **lazy O rescale only when the running max grows by > 6 log2 units**;
split-KV with fp32 partial O + log2-LSE and a separate combine (≤256 splits); workload balance by
64-token block counts + a fixed per-request overhead. Decode with next_n MTP tokens × 24 heads as
rows (M=96) fills the tensor tiles.

## Engram / PLE gather (Flash-Next: 16 rows × 160 dims per token, 320M rows)
Indices depend only on token ids → compute on the CPU as soon as a token is sampled (before the
forward pass) and issue the gather + W_K/W_V GEMV + gate so it lands by layer 1/2 (Engram's placement
argument; Qwen placed PLE at layer 2 explicitly to overlap layer 0/1). Fused gate kernel
(TileKernels): `x + σ(signed_sqrt(⟨RMSNorm(x), RMSNorm(W_K e)⟩·d^-½))·(W_V e)`, then depthwise
conv(4, dilation 3) + SiLU residual. Engram measured ≤2.8% throughput penalty with a 100B table
entirely in host memory over PCIe; on unified memory the gather is ~5 KB/token — the table can be
int4/int8 in GTT or the page cache; Zipfian hot-row cache optional.

## Gated Residual (Flash-Next hyper-connections) [V Qwen TR eqs 30–34]
Read: group RMSNorm of the 4 branches → `G = σ(W_u SiLU(W_d vec(R̂)/4))`, `x = ¼ Σ_i G_i ⊙ R̂_i`;
write: `s = 2σ(W_w vec(R̂)/4)`, `R_i += s_i·y`. Two fused kernels per sublayer (read, write) so the
4×2560 stream is traversed once per direction; **residual streams in FP8 halve residual traffic with
almost no loss** (Qwen); at small M the [10240×320] GEMV is latency-bound → split-K with the
SiLU/σ/gating in the epilogue (FlashInfer: 2.05×). Sparse 2-of-4 branch reads: rejected by Qwen.

## MoE (V4 DeepSeekMoE; Flash-Next: 512 routed top-10 + shared)
Routed experts MXFP4 with UE8M0 per-32 scales; SwiGLU clamp (`swiglu_limit`) in V4; on one GPU the
MegaMoE lesson reduces to: prefetch expert i+1's weights while computing expert i, fuse
SwiGLU×routing-weight into the epilogue; verified multi-token passes (MTP 4 steps → ~4 accepted)
amortise the expert reads — the biggest lever after quantisation (+26–50% measured on Strix Halo /
DGX Spark for V4-Flash). llama.cpp spends 98.6% of V4-Flash kernel time in expert GEMV.

## MTP / DSpark
V4 MTP accept ≈ 2.5; DSpark (3 extra SWA-only draft layers, block 5, Markov rank 256) 60–85% faster
per user; on Strix Halo V4-Flash 25.3 → 32.0 t/s with a 4-position fused verify. Flash-Next's MTP:
one hybrid block with QSA inside; reuse the top-k across draft steps.

## Copy verbatim
UE8M0 scales for 8-bit KV/index caches · `−1` sentinel indices · 64-token index tiles + `topk_length`
· sink/LSE folding · fixed per-sequence state block (window + pending tokens) · compressed-only
prefix caching · bf16 indexer scores · exactness only within the threshold bin · index reuse across
draft steps · host-side hash indices before the forward pass.

## CUDA-only (do not port)
TMA/`tma_gather4`, wgmma/tcgen05/TMEM, clusters/DSM, 228 KB smem warp-specialised pipelines, PDL,
block-scaled MXFP4/FP8 MMA, NVLink/NCCL symmetric memory, PCIe pinned-host prefetch streams.
