# AMD HRX ("hipx") — ROCm/hrx-system + llama.cpp discussion #27219 (investigated 2026-08-27)

Measured follow-up (2026-08-28): `amd-hrx-bench.md` — HRX dispatches at the same ~1.85 µs/packet AQL floor as a HIP graph, 0.7–1.0 µs slower than HIP on real kernels at the commit AMD pins; PM4 not reachable. Not adopted.

Note: there is no "hipx"; the project is **HRX = Hip Runtime Extended** (`ROCm/hrx-system`, HEAD 1425d394).
[V] = verified in code by the agent, [C] = claimed by AMD, unverified. Nothing was run (no numbers exist anywhere).

## What it is [V]
- A from-scratch runtime stack forked from IREE's runtime, *not* a slimmed CLR: native HSA/ROCR HAL driver
  (dlopen `libhsa-runtime64.so.1`; no HIP, no rocBLAS/hipBLASLt/CK/comgr), `libhrx` C ABI (IREE-HAL-shaped:
  `hrx_executable_load_data` -> `hrx_stream_dispatch(exe, ordinal, {wg count, wg size}, constants, bindings)`),
  an optional HIP-compat `libamdhip64.so` (140 `hipErrorNotSupported` stubs, open correctness bugs #156/#421),
  and **Loom**, a new kernel language + JIT (kernel corpus is private: `github.com/rocm/hrx` 404s).
- Early access (created 2026-03, v0.3.0 2026-05-30, dozens of commits/day, "not an official ROCm component");
  builds against TheRock nightlies + ROCm clang; Windows path non-functional; gfx1151 supported via gfx11-generic.
- Dispatch modes: `AQL` (default: pre-templated packets replayed into the HSA queue) and `PM4` (whole graph
  recorded into a resident PM4 indirect buffer, one AQL packet launches it — the "graphics-pipeline dispatch
  overhead" claim). **PM4 is not selectable from libhrx and ggml-hrx never uses it** — the published path is AQL.
- Memory: HSA pools directly; on an APU host-local allocations are also device-local (one copy). Host-pointer
  import is `hsa_amd_memory_lock_to_pool` (= hipHostRegister pinning); **no new map-file-as-GPU-visible primitive**.
  ggml-hrx does *not* zero-copy weights: it uploads them into device-local buffers.

## llama.cpp side [V]
- Draft PR ggml-org/llama.cpp#27218 (+46k lines, 0 reviews). Backend is fail-closed (any op without a fusion
  "recipe" aborts); one model in scope (Qwen3-30B-A3B Q4_K_M) on W7900 + Strix Halo. Whole cgraph -> recipes ->
  Loom JIT -> one recorded HRX graph replayed per token. Kernels use wave64 f16 WMMA on gfx11.
- Published numbers: **none**. Only [C]: prefill +30–50%, decode parity to +15% vs best of Vulkan/HIP.

## Relevance to hipster
- Our decode gap to the bandwidth floor is launch overhead (27B: ~350 launches x ~2 µs ≈ 0.7 ms/token of 80 ms;
  Flash-Next: ~1150 launches, ~16 ms of 57 ms). HRX's only mechanism for that is what we can already do:
  fewer launches (fusion — measured to work) and, if ever needed, writing AQL packets against ROCR ourselves.
  HIP graphs were measured not to help on this box (docs/roofline.md), and HRX's default AQL replay is the same idea.
- Their "+30–50% prefill" is vs llama.cpp's HIP backend; our prefill already beats llama.cpp/halogen at 32K.
- Verdict: **not adopted**. Re-evaluate if (a) PM4 mode becomes reachable and someone publishes a launch-latency
  number below ~1 µs on gfx1151, or (b) the Loom kernel corpus becomes public with a gfx1151 GEMV faster than ours.
  Our kernels are plain HSACO and would load into HRX unchanged if that day comes.
