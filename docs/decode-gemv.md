# Decode GEMV — measurements log

Kernel: `engine/kernels/gemv.hip`. Bench: `./build.sh './build/bench_gemv <gguf> <tensor> <ncol> <f32|q8|both>'`.
Weights stay in the ggml block layout as they sit in the GGUF (no repacking yet — a repack has to
beat these numbers to exist). Roof: 240 GB/s. Validation: 64 random rows vs CPU reference every run.

## 2026-08-28 · Q4_K, `token_embd.weight` [248320 × 5120], 682 MiB

| x format | ncol=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f32 x, tpr=4 | 225 GB/s (94%) | 152 | 131 | 67 | | | | |
| **int8 x + v_dot4, tpr=4** | **230** | **227** | **220** | **201 (84%)** | 176 | 198 | 178 | **178 (74%)** |
| int8 x, tpr=32 | 234 (98%) | 238 (99%) | 207 | 169 | 164 | 148 | 140 | 137 |

- f32 x is L1/ALU-bound past one column (128 B of x per 16 B of weight, 64 flops/col). Int8 x with
  `__builtin_amdgcn_sudot4` fixes it: 4× less x traffic, 8× fewer ops.
- Error vs the f32-x reference is 3.4–4.1e-4 relative (the q8 activation rounding llama.cpp also
  carries); vs the same-math reference it is < 1e-8 — the kernel is exact.
- tpr=4 (8 rows per wave, 5 blocks per thread at K=5120) is best from 3 columns up; tpr=32 wins by
  2–4% at 1–2 columns. Keep both; the engine picks per shape.
- ncol 5–8 at 74–84%: register pressure (8 accumulators + 4 uint4 of x per column per j). Verify
  passes of 4–8 tokens read weights ~1.2× slower than decode — acceptable now; revisit when the
  end-to-end profile says so. Options: x in LDS, or 2 rows/thread with x reuse.

## 2026-08-28 · more columns: why dot4 tapers, and the WMMA fix

Facts measured (`bench/roofline`, compiler `-Rpass-analysis=kernel-resource-usage`):
- `v_dot4_i32_iu8` issues at **2.4 T instr/s — half the fp32 FMA rate (4.3 T)** on gfx1151.
- The dot4 kernel needs ~900 VALU instructions per 256-weight block at 8 columns; no spills,
  occupancy 9–12 waves/SIMD. Hoisting the nibble unpack out of the column loop, 2 rows/thread
  and LDS-staged x all failed to move ncol=8 (132–143 GB/s). It is simply ALU-heavy at 8 columns
  while streaming memory at the roof.
- gfx1151 has no `v_dot4_i32_i8` (`dot1-insts`); signed×signed uses `sudot4(true, a, true, b)`.

Fix: `v_wmma_i32_16x16x16_iu8` (4096 MACs/instr). One wave = 16 rows, two WMMAs per 32-element
sub-block, fp32 epilogue with the per-row Q4_K scale/min and the per-column x scale/sum.

**Probed lane layout (unit test, `bench/roofline`-style probe):** D = 8 int32 per lane where
`lane%16` indexes the *second* operand's row and `2l + lane/16` indexes the *first* operand's row.
ggml's `mma.cuh` documents the mirror image; with x as the first operand and weights as the second,
each lane's 8 outputs share its own weight row so the scales stay local.

| Q4_K `token_embd` 682 MiB | ncol=1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| dot4 direct, tpr=4 | 228 | 224 | 211 | 143 | — | — |
| dot4 LDS-x, tpr=4 | 229 | 227 | **229 (95%)** | 132 | — | — |
| **WMMA, 16 rows/wave** | 216 | 215 | 214 | **209 (87%)** | 206 | **204 (85%)** |

Policy: ncol ≤ 4 → dot4 (LDS-x for 3–4); ncol ≥ 5 → WMMA. Exact to 1e-8 against the same-math
reference in all cases. Remaining WMMA gap: the 16 `xd/xs` global loads per sub-block per lane
(stage in LDS) and the duplicated A loads on mirrored lanes. Q5_K/Q6_K/IQ4_XS/Q8_0 WMMA decoders
still to add (same structure; the decoder yields two 16-byte fragments per sub-block).
