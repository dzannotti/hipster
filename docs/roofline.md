# gfx1151 roofline (measured 2026-08-27)

Box: AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151, 40 CU, wave32), 128 GB LPDDR5X unified,
GTT ≈ 118 GiB visible to the GPU. All numbers below are from `bench/roofline/run.sh`
(HIP, inside `mimiron/rocm:10.0.0`, HIP 7.15 / clang 23) unless stated. Identical on
`mimiron/rocm:7.14.0` to within noise.

## Memory

| test | result |
|---|---:|
| streaming read, 16 B loads, 8–16 GiB, any grid ≥ 1024 blocks | **238–241 GB/s** |
| device-to-device memcpy | 209 GB/s (read+write), 104 GB/s each way |

Consequences (bare decode floors, one token per weight pass):

| model | bytes per token pass | floor @ 240 GB/s |
|---|---:|---:|
| Qwen3.8-27B UD-Q4_K_XL (16.34 GiB total, ~15.2 GiB streamed) | ~16.3 GB | **~14.7 t/s** |
| Qwen3.8-Flash-Next UD-Q4_K_XL, 10/512 experts active | ~5.5–6 GB (est.; the Q8_0 dense/GDN/hc tensors dominate, experts are only ~1.4 GB) | **~40 t/s** |

llama.cpp already sits at the 27B floor (11.7–14 t/s bare). Everything above the floor is
speculation. Flash-Next is the opposite: llama.cpp gets 18–27 t/s against a ~40 t/s floor because
it is launch-bound (~4,800 launches / token).

## Kernel launch

| path | cost |
|---|---:|
| `<<<>>>` on a stream, empty kernel | 1.8–2.2 µs |
| `<<<>>>` on a stream, 256×256 kernel | 2.4 µs |
| HIP graph replay, per node (2000-node graph) | **1.7–2.2 µs** |
| host-side enqueue only (no wait) | 1.1 µs |

HIP graphs do **not** make launches cheap on this part (the GPU-side per-dispatch gap is the
cost). 4,800 launches ≈ 10 ms ≈ a hard cap of ~100 t/s regardless of kernel efficiency.
The lever is fusion: one kernel per (sub)layer, not graph capture.

## Compute (GEMM, prefill regime)

`D[M,N] = A[M,K] · W[N,K]ᵀ`, f16/bf16 inputs, f32 accumulate.

| library | M=2048 K=5120 N=17408 | M=512 | M=64 | M=8 |
|---|---:|---:|---:|---:|
| **hipBLASLt** (heuristic best of 16) | **51.8 TFLOPS** (bf16 51.7) | 49.6 | 14.4 | 1.9 (= 232 GB/s, bandwidth-bound) |
| hipBLASLt, K=17408 N=5120 (down-proj shape) | 37.4 TFLOPS | | | |
| hipBLASLt, K=2560 N=640 (Flash-Next expert shape) | 48.5 TFLOPS | | | |
| CK int8 WMMA (`DeviceGemmMultipleD_Wmma_CShuffleV3`, 128×128 bs256) | 46.0 TOPS | | | |
| rocBLAS via `hipblasGemmEx` | 5.7 TFLOPS | 5.6 | | |
| RADV Vulkan coopmat (llama.cpp `mul_mm`, best LDS pad, m=4096 n=512 k=14336) | 15.4 TFLOPS | | | |

hipBLASLt is 3.4× the best Vulkan number and 9× rocBLAS. Theoretical dense f16 WMMA peak for
40 CU @ ~2.9 GHz is ~59 TFLOPS, so hipBLASLt is at ~88% of peak.

Consequence for prefill: Qwen3.8-27B is ~54 GFLOP/token → ~950 t/s at 51 TFLOPS if the
GEMMs were everything (llama.cpp best: 440 t/s on Vulkan, ~310 on ROCm). Flash-Next is
~12 GFLOP/token active → the prefill ceiling is in the thousands; llama.cpp gets 342.

## Decision: ROCm/HIP, not Vulkan

| axis | HIP | Vulkan (RADV) |
|---|---|---|
| bandwidth | 240 GB/s | same hardware; llama.cpp mat-vec reached the same roof on both |
| launch cost | 1.8–2.4 µs, graphs don't help | command buffers cheaper per dispatch, which is why llama.cpp-Vulkan wins on the launch-bound Flash-Next — irrelevant once we fuse |
| GEMM | **51.8 TFLOPS** via hipBLASLt | 15.4 TFLOPS coopmat |
| ISA access | inline asm, WMMA intrinsics, `ds_read_b128`, LDS control, `__builtin_amdgcn_*` | GLSL/SPIR-V only, no inline asm |
| profiling | rocprofv3 kernel traces, counters | RADV perf counters, thin |
| toolchain | single-source C++ in a 28 GB container | stock Mesa, no container needed |

Vulkan's only measured advantage was per-dispatch cost on an unfused graph, which a
from-scratch engine removes by construction. HIP wins prefill outright.
