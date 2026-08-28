# Nathanw1014/strix-halo-llamacpp + llama.cpp branches — what transfers (sub-agent report, 2026-08-28, condensed)

Evidence repo (README, 162 KB EXPLORING.md, benchmarks, kv-membench) + kernel branches on
Nathanw1014/llama.cpp. Upstream: PR #25494 merged (transposing dequant of quantised KV into a
per-head-contiguous f16 scratch before FA), PR #27703 open (same for f16 KV).

## The memory-system fact (verified by their microbench, HIP and Vulkan identical)
- 256-byte granule across **16 channels selected by address bits [11:8]**; a stream whose row
  stride is a multiple of 4096 B hits one channel. Predictor: `channels = 16 / gcd(stride/256, 16)`.
  Measured: 256 B runs at 1024 B period 58 GB/s → +16 B pad 160 GB/s; 4096 B runs 228–242 GB/s.
- llama.cpp's token-major KV `[pos][n_head_kv][hd]` gives per-head stride 2048 B for 4 kv × 256
  (our 27B) → **2 of 16 channels**; FA op 2.24× slower than contiguous at hd256 (their probe);
  Qwen3.8-27B pp512 @65k: 64 → 173 t/s from the layout copy alone; +102% pp @64k from a
  head-major allocation with zero shader changes.
- MALL 32 MB: coalesced reads 990 GB/s ≤ 32 MB working set, 245 GB/s above. Per-layer K+V for
  the 27B = 4 KB/token f16 → spills at 8k tokens (16k with int8 KV).
- Decode fixed cost of GDN hybrids in llama.cpp: 3.4–4.1 ms/token from ~913 sync intervals at a
  2.3 µs floor (we are at ~350 launches; keep fusing).

## Kernel facts
- HIP decode with quantised KV: dequantise a KV tile **once into LDS** and share it across the GQA
  group (their tile kernel `flash_attn_tile_load_tile_q`); the fused-dequant-per-head vec kernel
  was 8× redundant. q8_0 KV then beats f16 (+41% tg at 64k on Coder, +14% at hd256); q4 buys
  nothing over q8 (element-bound). q8_0 per-32 scales 43.8 dB SQNR vs fp8 33 dB.
- Prefill FA on gfx1151: WMMA is wave32-native, but at hd256 **keep wave64** (wave32 was 6–18%
  slower from O-accumulator growth). Their contiguous-layout FA reaches only ~10 TFLOPS at hd256:
  issue-bound; a bigger Q tile per CTA (more reuse of each K/V fragment) is the lever a
  from-scratch kernel has. Hoist P fragments out of the head-dim loop (+7–9%), one mask load per
  GQA group, skip PV for all-zero P chunks, `v_dot2_f32_f16` for QK.
- llama.cpp HIP sends hd256 prefill to the scalar tile kernel (WMMA path only for hd ≤ 128) and
  batches GQA only in {2,4,8} → ratio 6 gets 2 heads/block. We do 6.
- Weight stream: use non-temporal loads (`__builtin_nontemporal_load`, HIP-only) so weights do not
  evict KV/activations from MALL; "residency is worth ~3×" (their claim, to be measured here).
- Transposed conv-state concat at 40960 B stride: 13.7 → 139 GB/s with a 32×33 LDS transpose.
- Measurement hygiene: `-r N` is not an error bar; a concurrent compile costs 7%; validate with
  PPL/logits, not only op tests.

## Applied here
- KV cache head-major `[layer][kv_head][max_ctx][256]` (2026-08-28).
- Non-temporal weight loads in the GEMV decoders (experiment, see docs/decode-27b.md).
- Attention prefill kernel design: wave64 at hd256, large Q tile, KV block resident, GQA 6.
