# halo-box/llama.cpp PR #7 "Rocm/mmq q8 j128 rdna35" (gaetan-puleo, 2026-08-28; sub-agent report, condensed)

HIP only, all gated on RDNA3.5. Four commits: (A) Q8_0 MMQ tile retune: 256 threads, I=64 × J=128 with
the 8 waves split 4 row-groups × 2 column-groups (32 accumulator VGPRs instead of 64; LDS 57.9 → 38.4 KB);
routed J=48 dispatch hard-wired to 256 experts and 2048×512 slabs (avg 32–64 tokens/expert);
(B) activation Q8_1 quantize done once for adjacent gate/up MUL_MAT_IDs (and dense Q8_0 pairs),
2-chunk quantize blocks with float4 loads; (C) Q8_0 KV dequantised inside the FA *tile* kernel into
the padded half2 LDS tile shared by the GQA group, decode-only (Q->ne[1]==1); (D) tests.

Claimed (self-reported, 8060S, ROCm 7.14): Qwen3.6-35B-A3B Q8 pp2048 1314 → 1515 t/s (+15%), Gemma-4
E2B +8.7%, GPT-OSS +1.9%; decode with Q8 KV +0…1.2%; logits byte-identical (likely measured on f16 KV).
No reviewer discussion.

Transferable: (1) the int8-WMMA tile geometry for gfx1151 (64×128 per WG, 16×64 per wave, K-step 32,
~38 KB LDS; VGPR pressure was the driver) if we ever build an int8 tile GEMM; (2) share activation quant
and routing prep across gate/up — we already merge them into one GEMM/launch; (3) choose the MoE column
tile from tokens-per-expert (mean, plus slack), not from n_tokens — for 512 experts the mean halves
again; (4) Q8 KV in-kernel dequant shared by the GQA group — our int8-KV variant does this.
Correction to the report: it assumes the int8 pipe is ~2× the bf16 rate; measured here iu8 = bf16
= 54.8 TOPS (docs/roofline.md) — MMQ's structural win on this GPU is skipping the dequant pass, not
faster math.
