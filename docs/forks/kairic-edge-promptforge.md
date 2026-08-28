# ciru-ai/ROCmFPX — Kairic Edge (IU4) and ActiveFPX PromptForge (sub-agent report 2026-08-28, condensed)

Branches: `qwen3.8-activefpx-promptforge-v2.3` (2026-08-17) = **W8A8 int8** PromptForge lane, no IU4;
`kairic-edge-qwen38-27b-v1.2` (2026-08-23) = the only **IU4** lane (+ compat mode, M65 verify retired).
"Shortlisted draft head", "recurrent replay", `mmwq.cuh` are this box's own `mimiron-speed` patches,
not upstream.

## Mechanism (verified in code)
- Dual View: the GGUF (ROCmFP4 4.5 bpw) stays resident for M=1 and verify; `.pfs` sidecars (10.6 GiB)
  hold selected weights in a prompt-phase layout; a router by tensor name + exact shape sends admitted
  row counts (2048/2044/1476, small-M buckets 128/256/512, ≥96 rows) to the IU4 lane; the shipped
  profile (`-ub 512`) runs all prefill through the 512 bucket: FFN gate/up/down (58 layers, late-6
  "keepers" stay on GGUF), GDN qkvz (48 layers); attention, GDN-out, lm_head, M1 and M2–5 stay off IU4.
- IU4 lane: `wmma_i32_16x16x16_iu4(neg_a=false, A, neg_b=true, B)` = U4 activations × S4 weights via a
  191-line CK patch (K in bytes, `k_per_wmma=8`); tiles 128×64×64B (gate/up) / 128×256×64B (down)
  4×2 / 4×4 wave repeats; epilogue `e = (c − row_zero·col_wsum)·col_wscale·row_ascale` → bf16 → f32.
- Formats: weights S4 with **one f32 scale + one i32 sum per output row over the full K**; activations:
  block-1024 randomised Hadamard (hash ±1 signs, 10 LDS butterfly stages), then **one asymmetric
  min/max scale per token row over the full K**, zero-point corrected in the epilogue. Gate/up fused
  with SwiGLU + re-pack for down; qkv+z merged into one N=16384 GEMM.
- Tried and not shipped: 256-group segmented scales, int8 keeper corrections, attention IU4 sidecar.

## Measured (their card [C]; this box [V])
- WMMA peaks 104.7 iu4 / 54.3 iu8 / 53.9 f16 TOPS (matches ours). FFN op at 512 rows: 3.80 ms vs
  13.22 control → **72 TOPS effective (65% of iu4 peak)**; prompt sweep ≤512 rows 325–529 t/s vs
  235–322. On this box: pp512 694 t/s (IU4 bucket + `GGML_MMQ_Y_RDNA35=128`), mimiron prefill 580 @7.8K.
- Quality: no PPL/KLD published for IU4; the M65 IU4 verify **changed a greedy token** on a
  low-margin case → retired. The per-row full-K 4-bit activation scale is the cause (predictable).
- Verify cost model fitted here: verify(n) ≈ 82 ms + 2.5 ms·n. MMVQ 8-column ceiling → pp9 cliff
  (78 → 57 t/s) in llama.cpp; `mmwq` (16 tokens × 16 rows per wave, iu8 WMMA) 98 → 139 t/s at pp16.
- Throttling on this box: sclk 2850 → 2530 MHz at 84 °C / 130 W after ~10 min.

## Transfer
1. W4A4 prefill: keep the mechanism (Hadamard pre-rotation offline, S4 per-channel weights + i32
   column sums, U4 activations with zero-point correction) but **scale per 256-K segment** inside
   our own WMMA loop (16 WMMAs then ~8 FMAs per lane), fuse SwiGLU+Hadamard+pack into the gate/up
   epilogue, Hadamard in registers via DPP/`ds_swizzle`. Expect ~2× over bf16 on FFN GEMMs; gate on KLD.
2. Never pad small M into GEMM tiles; verify must run the same numeric path as M1 (ours: the GEMV
   paths differ by accumulation order → near-tie flips only).
3. n-gram-mod long drafts (24/64/64) with a 64-row verify are the source of their 100+ t/s peaks on
   repetitive output; needs an M≈64 verify kernel (our WMMA GEMV: 16 columns; extend).
4. Recurrent-state replay for rollback (we have it) — design the scan kernels with a replay entry.
5. lm_head Q4-shortlist → Q8 rerank two-pass (`output_k8`) if the head shows in the trace (draft
   steps: 4.2 ms of 8 per MTP draft step here — it does).
