# Kairic Edge / PromptForge (ciru-ai/ROCmFPX branches kairic-edge-qwen38-27b-v1.2, qwen3.8-activefpx-promptforge-v2.3) — condensed sub-agent report 2026-08-28

Branches are siblings, not successors: v2.3 = "ActiveFPX" **W8A8 int8** line; v1.2 = the **IU4** line
(CK `DeviceGemmMultipleD_Wmma_CShuffleV3` with a patched `wmma_i32_16x16x16_iu4`). `mmwq.cuh`, the
shortlisted draft head, adaptive depth and recurrent replay are the user's own mimiron patch stack.

## What it is [V]
- "Dual View": weights resident twice — GGUF (rocmfp4) for M=1 decode via MMVQ, plus `.pfs`
  sidecars (FFN 8.6 GB, GDN-qkvz 2.0 GB, GDN-out 0.8 GB) for prefill. ~26 GiB resident.
- IU4 lane: activations → per-1024-column random-sign Hadamard (10-stage LDS butterfly) → one
  asymmetric u4 scale+zero per token over the whole K; weights s4 with one f32 scale per output
  column over K + int32 column sum; epilogue `(c − zero·colsum)·wscale·rowscale`. Only exact row
  counts 2048/2044/1476 or buckets 128/256/512 (96–512 padded) are routed; everything else falls
  back to MMQ. Keepers: last 6 FFN and 12 GDN-out layers stay higher precision.
- The M=2..5 "decode_rows" IU4 tiles exist but are never enabled ("padding small verification
  batches to M128 regresses decode"); the M65 n-gram verify on IU4 **changed a reproduced greedy
  token** and is shipped off — W4A4 verify breaks target-equivalence.
- Device argmax fast path (only token ids come back): −9.7% TG when disabled.

## Numbers
- Vendor [C]: IU4 harness 104.7 TOPS; FFN op at 512 rows 3.80 ms vs MMQ 13.22 (3.5×); pp 322→529
  t/s at 512 rows; server pp 329/478/450/400 at 2K/4K/8K/16K.
- This machine [M]: pp512 ~694–703 t/s; CK IU4 ≈ 79–86 TOPS effective at M=512 (72–78% of 109.7);
  unrouted MMQ shapes still 700 of 2185 ms; Hadamard pack 340 µs/layer at 512 rows (2× off roof);
  tg 73 ms/token = 226 GB/s. MTP (stock v1.2): code 31 t/s at n-max 4 (τ 4.33), prose 17.7.
- Local mimiron stack [M]: draft head shortlist (131k of 248k rows, Q4_0, −1e30 padding, id remap)
  draft 7.5 → 3.2 ms/token; `mmwq` iu8 WMMA verify: M16 pass = 1.35× the M1 pass; adaptive depth
  (per-position acceptance EMA, α 0.08, maximise expected tokens per ms): prose n≈2, code n≈5.

## Transfers to hipster
1. W4A4 prefill: FFN + GDN-qkvz GEMMs at ~80 TOPS vs hipBLASLt 46–52 → GEMM 2.38 → ~1.4 s at
   T=2048 (upper bound ~880 t/s), minus the pack. Prerequisites: rotated s4 weight copy (or
   rotate+requant Q4_K→s4 into the per-layer scratch: 4× fewer bytes than today's bf16 54 GB),
   keepers, and accepting prefill ≠ decode numerics (prefill is not verified, so this is allowed).
2. Verify width: the honest recipe is iu8 WMMA (ours) + K-split to reach 16 columns; add the missing
   Q5_K/Q6_K/IQ4_XS/Q8_0 WMMA decoders.
3. Draft-only shortlisted head (~+9% at n=5 here), adaptive depth, device-side argmax.
4. Pitfalls: shape rigidity, double residency, fast-math folding −∞ (ROCm 7.2), `MMQ_Y=64` collapse
   on RDNA3.5, thermal (85 °C, 2.45–2.86 GHz within 5 min).
