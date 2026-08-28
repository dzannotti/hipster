# hipster — a from-scratch HIP inference engine for Qwen3.8 on AMD Strix Halo (gfx1151)

Two checkpoints, one GPU, nothing generic: **Qwen3.8-27B UD-Q4_K_XL** (dense, hybrid GDN/attention, in-checkpoint MTP)
and **Qwen3.8-Flash-Next UD-Q4_K_XL(-MTP)** (512-expert MoE, hyper-connections, PLE n-gram table streamed from SSD,
QSA sparse attention). Every number below was measured on one box and can be re-measured with the commands given.
Design principles, roofs and per-kernel accounting live in `CLAUDE.md` and `docs/`.

## Hardware and software this was measured on
- AMD Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151, 40 CU wave32), 128 GB unified memory, NVMe (Kingston OM8TAP42048K1).
- Fedora 44 host **without ROCm**; everything GPU-side runs in a docker image, `mimiron/rocm:10.0.0`. **That tag is a
  local build, not on a registry, and nothing in it is custom**: Ubuntu 24.04 + stock ROCm 10.0.0 installed from AMD's
  official apt repository (`stable.repo.amd.com/rocm/core/packages/ubuntu2404`, the `amdrocm-*10.0` packages —
  i.e. exactly what TheRock's install scripts pull; AMD clang 23.0.0git, hipBLASLt, rocprofv3) plus `cmake ninja git`
  and env `GPU_TARGETS=gfx1151 ROCBLAS_USE_HIPBLASLT=1`. `tools/Dockerfile.rocm` rebuilds it
  (`docker build -t mimiron/rocm:10.0.0 -f tools/Dockerfile.rocm .`); any ROCm 10.0.0 image with hipcc, hipBLASLt and
  rocprofv3 for gfx1151 works the same — pass it as `IMG=<image> ./build.sh …`. `build.sh` is the only entry point: it
  runs cmake + ninja inside the image and then your command. The flags to match if your toolchain differs are in
  `build.sh` and `engine/CMakeLists.txt` (HIP via `/opt/rocm/lib/llvm/bin/clang++`, `--offload-arch=gfx1151`, link `libhipblaslt`).
- Measured roofs (`bench/roofline/`): 240 GB/s DRAM read, 1.8–2.4 µs per kernel launch, 51.8 TFLOPS hipBLASLt bf16.

## Models (paths are what `build.sh` mounts as `/models`)
> **Flash-Next speed needs the `-MTP` shards.** The plain `UD-Q4_K_XL` (4 shards) has no `blk.48` draft block: the server
> prints `no MTP block in this GGUF` and decodes plainly at ~27 t/s. The 37–43 t/s numbers use the 5-shard
> `UD-Q4_K_XL-MTP` upload (`…-00001-of-00005.gguf`), with the default `--mtp 2` (3–4 drafts are slower).
| | host path | HF source |
|---|---|---|
| 27B | `/srv/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf` (+ `mmproj-F16.gguf`) | unsloth/Qwen3.8-27B-GGUF, UD-Q4_K_XL |
| Flash-Next, 4 shards | `/srv/models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` | unsloth/Qwen3.8-Flash-Next-GGUF, UD-Q4_K_XL |
| Flash-Next + MTP block, 5 shards | `/srv/models/qwen3.8-flash-next/UD-Q4_K_XL-MTP/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..5}-of-00005.gguf` | same trunk + `blk.48` (the 5-shard "-MTP" upload) |
Pass shard 1; the loader opens the rest. The 28.8 GB n-gram table is never uploaded (rows are gathered from the mmap).
RAM: Flash-Next loads 77–79 GB (pread, page cache dropped per tensor); nothing else GPU-heavy should be running.

## Build
```
git clone git@github.com:dzannotti/hipster.git && cd hipster
./build.sh 'true'            # cmake + ninja in the container; prints BUILD_OK
```
Binaries land in `build/` (root-owned, produced inside the container).

## Verify the claims
Each driver loads the model (60–120 s), checks correctness against stored llama.cpp references (`docs/ref/*.json`,
generated with `tools/ref-server*.sh` + `tools/ref-logits.py` / `tools/ref-long.py`), then measures. Discard the first
timed run of anything (boost clocks); the APU throttles ~25% after ~10 minutes of sustained load. Never run two model
processes at once (memory) and never benchmark while something else uses the GPU.

```
FN=/models/qwen3.8-flash-next/UD-Q4_K_XL-MTP/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00005.gguf
B27=/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
```
| claim | command | what to expect (2026-08-28) |
|---|---|---|
| Flash-Next decode exact, ~28 t/s | `./build.sh "./build/runfn $FN docs/ref/fn-code.json 12"` | `greedy: 12/12 tokens match llama.cpp`, `decode: ~36 ms/token`, max logprob diff vs llama.cpp ≤ ~0.9 |
| MTP exact, 37–41 t/s | `./build.sh "./build/specfn $FN docs/ref/fn-code.json 96 2,3,4"` | plain ~27 t/s, `mtp n=2: … 37–40 t/s`, `EXACT: 96/96` for every n |
| batched decode exact, 63/80 t/s at 4/8 slots | `./build.sh "BATCH_DISTINCT=1 ./build/batchfn $FN docs/ref/fn-code.json 48 4"` (and `… 8`) | `4 slots: … 63 t/s aggregate`, `EXACT: 4/4 slots` |
| prefill ~1000 t/s @2K, exact continuation | `./build.sh "./build/prefillfn $FN docs/ref/fn-code.json 12 2048 512 1024 2048"` | `last-row top-1 identical`, `greedy continuation 12/12`, `prefill 2048 tokens: … ~1000 t/s` |
| QSA sparse attention exact at 4K/16K, 795 t/s @16K | `./build.sh "./build/prefillfn $FN docs/ref/fn-long4k.json 7 2048"` and `./build.sh "HIPSTER_NO_DECODE_REF=1 ./build/prefillfn $FN docs/ref/fn-long16k.json 7 2048"` | `vs llama.cpp continuation: first 7 identical` (the needle `amber-falcon-73`), 16767 tokens in ~21 s |
| kernels vs their references | `./build.sh "./build/test_fnkernels 29"` | GDN scan vs step 1e-7; attention variants ≤ 0.3% |
| tokenizer exact | `./build.sh "./build/tok_test $FN docs/ref/tok-tests.json docs/ref/fn-code.json"` | `EXACT: 45/45 cases identical to llama.cpp` |
| 27B decode exact, 10.3 t/s (12.5 with `HIPSTER_GEMV=fast`); MTP 33.8 t/s | `./build.sh "./build/run27b $B27 docs/ref/code.json"`, `./build.sh "./build/spec27b $B27 docs/ref/code.json 96 5"` | `12/12`, then `EXACT: 96/96`, `mtp n=5: … ~34 t/s` |
| Flash-Next served with 4 slots: 43.7 t/s alone (MTP), 48 aggregate for two, 51 for four concurrent requests, each identical to its solo output | `tools/serve.sh --ctx 8192 --slots 4` then concurrent chat completions (docs/serving.md) | `predicted_per_second` per reply; wall-clock aggregate |
| Flash-Next multi-slot MTP exact | `./build.sh "BATCH_DISTINCT=1 BATCH_MTP=2 ./build/batchfn $FN docs/ref/fn-code.json 64 4"` | `EXACT: 4/4 slots identical to their single-slot greedy streams` |
| 27B served (OpenAI endpoint, DFlash2), 46 t/s on a 200-token code reply; 2 slots: 68 t/s aggregate for two concurrent code requests | `MODEL=$B27 DRAFT=models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf tools/serve.sh --ctx 8192 --slots 2` then chat completions (docs/serving.md) | `timings.predicted_per_second ≈ 46` alone; two concurrent: 600 tokens in ~9 s wall |
| **27B DFlash2 × 2 slots, 79 t/s aggregate, both exact** | `./build.sh "SLOTS=2 ./build/dflash27b $B27 models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf docs/ref/code.json 96 7"` | `dflash2 x 2 slots: … ~79 t/s aggregate`, `EXACT: 2 slots identical to plain greedy` |
| **27B DFlash2 exact, 48 t/s on code** (22 on prose; 13.5 at 16K context) | `./build.sh "./build/dflash27b $B27 models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf docs/ref/code.json 96 7"` (draft GGUF: see docs/speculative-27b.md) | `dflash2 n=7: … ~48 t/s`, `EXACT: 96/96` |
| 27B verify pass T-invariant | `./build.sh "HIPSTER_LOGIT_DIFF=1 HIPSTER_BATCH=8 ./build/spec27b $B27 docs/ref/code.json 28 3"` | `worst T>1 vs T=1 logit difference: 0.0000` |
| 27B prefill 647 @2K / 475 @32K, long-context gate | `./build.sh "./build/prefill27b $B27 docs/ref/code.json 512 1024 2048"`, `./build.sh "HIPSTER_DEPTH=1 ./build/prefill27b $B27 docs/ref/long32k.json 2048"` | throughput table; 32K needle top-1 identical |
| GEMV / MoE / roofline micro-benchmarks | `./build.sh "./build/bench_gemv $FN blk.0.ffn_down_exps.weight 1 q8"`, `./build.sh "./build/bench_moe $FN blk.0.ffn_gate_exps.weight 2048"`, `bench/roofline/run.sh` | GB/s per lane policy (docs/decode-flash-next.md), MoE grouped GEMM ~15 TOPS, the roofs |
| per-kernel profile | `./build.sh "tools/prof-kernels.sh 41 ./build/runfn $FN docs/ref/fn-code.json 4"` | rocprofv3 breakdown per forward |
llama.cpp comparison points on the same box (`docs/decode-flash-next.md`, `docs/prefill-flash-next.md`): Flash-Next decode
18 t/s (ROCm) / 26.7 (Vulkan), prefill 425 t/s @4K and 328 @16K; 27B via halogen 26.9 / 31.7 (MTP).

## Serve (OpenAI-compatible + llama.cpp's web UI)
```
tools/serve.sh --port 8090 --ctx 16384 --mtp 2        # Flash-Next; ~80 s to ready; BIND=127.0.0.1 to keep it local
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write one sentence about the sea."}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false}}'
docker stop hipster-serve
```
Web UI at `/` (llama.cpp's, served from the fn-tree build in `/srv/models/.work/fn-tree/build/tools/ui/dist`; pass
`--ui-dir` for another copy). `docs/serving.md` lists the endpoints, thinking/tool-call handling and the current gaps
(one slot, no prefix cache, no `stop`/`logprobs`, no vision, 27B not served yet).

## Reproducing the references
`tools/ref-server-fn.sh start <ctx>` runs the franken llama.cpp ROCm build (`/srv/models/.work/fn-tree`, `--ngram-on-disk`)
on port 18083; `REF_URL=http://127.0.0.1:18083 python3 tools/ref-logits.py|ref-long.py …` writes `docs/ref/*.json`
(prompt ids, greedy continuation, top-10 logprobs). The 27B references came from `tools/ref-server.sh` (CPU) /
`ref-server-gpu.sh`.

## Layout
`engine/` (kernels, the two model engines, drivers under `engine/bench/`), `front/` (tokenizer, chat template, HTTP,
`hipster-serve`), `bench/` (roofline + HRX dispatch benchmarks), `tools/`, `docs/` (every measurement and decision:
`decode-*.md`, `prefill-*.md`, `speculative-27b.md`, `ideas.md` board, `forks/` reports on the projects investigated).
