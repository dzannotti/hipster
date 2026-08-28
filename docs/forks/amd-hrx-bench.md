# AMD HRX — measured launch overhead on gfx1151 (2026-08-28)

Follow-up to `amd-hrx.md` (read-only investigation). This time everything was run. **[M] = measured here,
[C] = claimed by AMD.** GPU was otherwise idle (engine benchmarks paused); every HRX run is bracketed by
`bench/roofline/launch.hip` in the same container session so both sides see the same clocks. All runs < 30 s;
no thermal drift visible (HIP baseline stayed 1.81–2.22 µs across the whole session).

## What was run

| piece | version | how it got here |
|---|---|---|
| libhrx, prebuilt | `hrx-public-linux-x86_64-v0.3.0.tar.zst` (4 MB): `lib/libhrx.so.0.1.0` (2.1 MB, unstripped), `bin/hrx-info`, headers | github release v0.3.0 (2026-05-30) |
| ROCR bundle | `hrx-public-deps-…-v0.3.0.tar.zst`: `libhsa-runtime64.so.1.21.0` | same ROCR version as the container's `/opt/rocm` (1.21.0); libhrx loads and enumerates gfx1151 against **either** [M] |
| libhrx, source | hrx-system `8ef82dbf` (2026-08-14) — the commit pinned by llama.cpp discussion #27219 | built as the ExternalProject of ggml-hrx (below) |
| ggml-hrx | AMD-Ecosystem/llama.cpp branch `users/stella/hrx-rfc-v1` @ `33a1f2b` (2026-08-16) | the build recipe from #27219, inside `mimiron/rocm:10.0.0` with ROCm clang 23: **49 s wall** to build llama.cpp + hrx-system + Loom (582 + 1769 ninja steps) [M]; AMD says "about a minute" on this CPU [C] |

Build (host, then container; scratchpad `$S`):
```
git clone --depth 1 --branch users/stella/hrx-rfc-v1 https://github.com/AMD-Ecosystem/llama.cpp
git init hrx-system && cd hrx-system && git fetch --depth 1 origin 8ef82dbfa0385f7953ceadd884317be7e512ea38 && git checkout FETCH_HEAD
docker run … -v $S:/hrx -w /hrx/ref mimiron/rocm:10.0.0 bash -c 'export CC=/opt/rocm/lib/llvm/bin/clang CXX=/opt/rocm/lib/llvm/bin/clang++;
  cmake -G Ninja -B build -S llama.cpp -DGGML_HRX=ON -DHRX_SOURCE_DIR=/hrx/ref/hrx-system -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DIREE_ROCM_PATH=/opt/rocm -DLLAMA_CURL=OFF && cmake --build build'
build/bin/llama-cli --list-devices   ->   HRX0: AMD Radeon 8060S Graphics (Node 1) (gfx1151) (120832 MiB)   [M]
```
ggml-hrx is fail-closed and only has recipes for Qwen3-30B-A3B Q4_K_M, which is not on this box, so there is
no end-to-end tok/s number. That model's per-token path is one recorded HRX graph replayed per token
(`amd-hrx.md`), so the graph-replay row below is the number that path would see.

Benchmarks (all in `bench/hrx/`, run with `HRX_DIR=<dir with pub/> ./bench/hrx/run.sh` and `./bench/hrx/run_gemv.sh`):
- `kernels.c` — `hrx_empty` (1×64, same as `k_empty`), `hrx_small` (256×256, same as `k_small`), `hrx_store`; plain C
  `[[clang::amdgpu_kernel]]` the way libhrx's CTS does it, compiled with
  `clang -x c -std=c23 --target=amdgcn-amd-amdhsa -mcpu=gfx1151 -nogpulib -O3 -fvisibility=hidden -o kernels.hsaco`.
  Loaded with `hrx_executable_load_data(dev, data, size, "amdgpu", "gfx1151")` (HEAD API) or
  `(…, "amdgcn-amd-amdhsa--gfx1151")` (v0.3.0 API); pointer args become bindings, scalars constants. Both write
  probes verified (`hrx_store` → 0xFEED1234; `hrx_small` converges to 1000.0, element N untouched).
- `launch_hrx.c` — N = 10000 `hrx_stream_dispatch` + one `hrx_stream_synchronize`; the same with
  `hrx_stream_flush` after every dispatch; graphs of 100 and 2000 kernel nodes (chained deps) replayed 100 / 20 ×.
  Min of 3 reps, wall clock around record+submit+wait (same thing `hipEventElapsedTime` brackets in `launch.hip`).
- `gemv_hip.hip` / `gemv_hrx.c` — the engine's `k_gemv_q8<Q4_K,4,1>` from `engine/kernels/gemv.hip` compiled
  `--offload-device-only`, unbundled (`clang-offload-bundler --unbundle --targets=hip-amdgcn-amd-amdhsa--gfx1151`),
  and the **same HSACO** dispatched by `hipModuleLaunchKernel` and by libhrx (5 bindings + 8 B constants; HRX fills
  the HIP hidden args — `gridDim.x` is used by the kernel and the outputs match bit-for-bit, `y[0]` identical).

## How libhrx actually dispatches [V, read + confirmed by the numbers]

`hrx_stream_dispatch` does **not** submit. It records the dispatch plus an execution barrier into a pending
*one-shot* IREE command buffer; nothing reaches the HSA queue until `hrx_stream_flush` / `synchronize` / a
cross-stream wait. So "N dispatches + one wait" on HRX is one queue submission carrying N AQL packets — the
analogue of a HIP graph launch, not of N HIP stream launches. `hrx_stream_flush` after each dispatch gives the
one-submission-per-kernel shape HIP streams have. Graphs (`hrx_graph_*`, HEAD only; v0.3.0 has no graph API —
115 vs 163 API functions) instantiate a reusable command buffer replayed by `hrx_graph_exec_launch`.

PM4 mode: the runtime contains it (`strings libhrx.so`: "PM4 is an experimental gfx1100 dispatch-only
reusable-command-buffer path"), but its only selector is the IREE flag `--amdgpu_command_buffer_mode` and libhrx
never calls `iree_flags_parse`; it applies only to non-one-shot (graph) command buffers anyway. **Not
selectable through the shipped libhrx API, not used by ggml-hrx, not measured** (per guidance: the shipped path
is what counts).

## Numbers — µs per kernel, min of 3 reps, GPU otherwise idle [M]

| path | empty 1×64 | small 256×256 (65536 elems) |
|---|---:|---:|
| **HIP stream launch** (`launch.hip`, 2000 launches) | **1.81–2.22** | **2.43–2.52** |
| HIP graph replay (2000 nodes, 20 replays) | 1.73–1.79 | 2.32–2.33 |
| HIP host enqueue cost only (no wait) | 2.5–6.1 | — |
| HRX v0.3.0 stream, 10000 recorded → 1 submit | 1.88–1.89 | 2.41–2.42 |
| HRX v0.3.0 stream, flush per dispatch (1 submit/kernel) | 14.2–20.5 | 35.4–36.0 |
| HRX 8ef82dbf stream, 10000 recorded → 1 submit | 1.93–1.94 | 3.40–3.41 |
| HRX 8ef82dbf stream, flush per dispatch | 22.2–24.1 | 39.4–45.8 |
| HRX 8ef82dbf graph, 100 nodes × 100 replays | 1.89 | 3.12–3.31 |
| HRX 8ef82dbf graph, 2000 nodes × 20 replays | 1.82–1.83 | 3.02–3.28 |
| HRX host cost to record one dispatch (no submit) | 0.11 | 0.11 |
| HRX graph instantiate, 2000 nodes | 0.2 ms | 0.2 ms |

Ranges are across two full runs each (v0.3.0 and 8ef82dbf), each bracketed by HIP baselines.

Bandwidth, same HSACO, `k_gemv_q8<Q4_K,4,1>`, K = 5120, 50 launches × 3 reps [M]:

| matrix | HIP (`hipModuleLaunchKernel`) | HRX batched | HRX flush per dispatch |
|---|---:|---:|---:|
| 256 MB (N = 93184) — streams from DRAM | 225.3–227.6 GB/s (1179–1191 µs) | 227.2–227.8 GB/s (1178–1181 µs) | 226.9–227.0 GB/s |
| 32 MB (N = 11648) — fits the 32 MB MALL | 545–642 GB/s | 636–649 GB/s | 536–599 GB/s |

Roof is 240 GB/s (`docs/roofline.md`); both runtimes sit at 94–95 % of it on the DRAM-sized matrix. The 32 MB
row is cache-resident and says nothing about DRAM; it is kept only as a warning for anyone who reuses `1<<25`.

## Reading

- The GPU-side floor for one AQL dispatch packet on this box is **~1.75–1.9 µs whoever writes the packet**:
  HIP graph 1.74, HRX batched 1.88, HRX graph 1.82, HIP stream 1.81–2.22. HRX's host side is far cheaper
  (0.11 µs to record vs 2.5–6 µs for a HIP enqueue) but that is hidden behind the packet processor either way.
- On a kernel that actually does something (`small`), HRX v0.3.0 matches HIP (2.41 vs 2.43–2.52) and the
  #27219-pinned commit is **~0.7–1.0 µs slower per dispatch than HIP** (3.0–3.4 vs 2.3–2.5), stream or graph.
  Not chased down; it is the newer runtime, not the older one, that regressed.
- One HSA submission per kernel through libhrx costs 14–46 µs (command-buffer end + `queue_execute` with a
  semaphore wait/signal per submission). Nobody should use HRX that way, but it means HRX cannot replace a HIP
  stream 1:1 — it has to batch, i.e. it is a graph-style runtime.
- Kernel throughput is unaffected: identical HSACO, identical 227 GB/s, identical outputs. HRX changes nothing
  about what a kernel can do; the loader handles HIP-compiled kernels (hidden args included) without changes.
- ggml-hrx builds in 49 s and sees the GPU, but its "+30–50 % prefill / parity–+15 % decode" [C] is vs llama.cpp's
  own HIP backend on one model, with none of the numbers published; the runtime itself does not launch faster.

## Verdict for hipster

Launch overhead is ~1150 launches × ~1.8 µs ≈ 2 ms of a 36 ms Flash-Next decode token and ~350 × ~2 µs ≈ 0.7 ms
of an 80 ms 27B token. Measured, the shipped HRX path (AQL command buffers replayed as one submission per
token, exactly what ggml-hrx does) costs the same ~1.85 µs per packet as a HIP graph and 0.7–1.0 µs *more* than
HIP on real kernels at the commit AMD pins; PM4 — the only mechanism that could go below the AQL floor — is not
reachable from the public API and is marked experimental/gfx1100 in the binary. Switching runtimes would move
the Flash-Next budget by −0 to +1 ms and give up hipBLASLt/rocprofv3; the only thing that lowers the 2 ms is
fewer packets (fusion), which we already do in HIP. **Not adopted.** Re-open if a public HRX release exposes PM4
on gfx11.5 and someone measures < 1 µs per dispatch with it, or if the Loom kernel corpus ships a gfx1151
GEMV above 227 GB/s — our kernels load into HRX as-is, so trying it again is a one-hour job with
`bench/hrx/run.sh`.
