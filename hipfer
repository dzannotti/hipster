# References — everything consulted or borrowed from (no outcomes here; see the other docs for measurements)

## Models / checkpoints
- Qwen/Qwen3.8-27B (arch `qwen35` / `qwen3_5`) — unsloth GGUF: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF (UD-Q4_K_XL + mmproj)
- Qwen/Qwen3.8-Flash-Next (arch `qwen4exp` / `qwen4_exp`) — unsloth GGUF: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF (UD-Q4_K_XL, the 5-shard "-MTP" upload with `blk.48`); notes https://unsloth.ai/docs/models/qwen3.8-next
- Qwen3.8-27B-DFlash2 draft (z-lab / Inco AI): https://huggingface.co/incoai/Qwen3.8-27B-DFlash2 , GGUF https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF (Q4_K_M, Q8_0); blog https://inco.ai/blog/dflash2/
- agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF: https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF
- unsloth IQ4_XS / imatrix quant variants of both models (quality tables in `docs/quant-quality.md`)

## Papers
- DFlash (block-diffusion drafter), z-lab: arXiv 2602.06036, code https://github.com/z-lab/dflash ; DFlash2 (selector lattice) via the Inco AI blog above
- FreeToken: "Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution", arXiv 2608.16157, code https://github.com/FlashML-org/FreeToken
- Qwen3.8 / Flash-Next technical report (QSA sparse attention, Gated DeltaNet, PLE n-gram table, hyper-connections, MTP / "IndexShare")
- DeepSeek V4 family: V4 report, V3.2 / DSA sparse attention, Engram, mHC, FlashMLA, DeepGEMM, TileKernels, DSpark (notes in `docs/forks/deepseek-v4.md`)
- Gated DeltaNet (the GDN recurrence used by both models), Medusa / EAGLE-3 / MTP-style speculative decoding background

## Reference implementations (read-only, `/srv/models/.work/`)
- `ninfer/` — from-scratch CUDA engine for the same two models (best written spec of qwen35 GDN / attention / MTP / vision in `docs/maintainer/*.md`)
- `franken/src/models/qwen4exp.cpp` — the only complete Flash-Next forward pass
- `llama.cpp` (ggml-org): `ggml/src/ggml-cuda/` block formats + mmvq, `src/models/dflash.cpp`, `common/speculative.cpp`, `common/ngram-map.cpp`, `tools/ui` web UI, tokenizer (`unicode.cpp`, `llama-vocab.cpp`), `gguf-py`
- `fn-tree/` — llama.cpp build used as the numerical reference server for Flash-Next

## llama.cpp forks, branches, PRs, discussions
- ggml-org/llama.cpp discussion #27219 (AMD HRX on llama.cpp) and PR #25494 (transposing dequant of quantised KV)
- ROCm/hrx-system ("hipx"): https://github.com/ROCm/hrx-system (hrx-rfc-v1, `hrx.c`, `hrx_stream_dispatch`, `hrx_executable_load_data`)
- AMD-Ecosystem/llama.cpp: https://github.com/AMD-Ecosystem/llama.cpp
- halo-box/strix-llama.cpp PR #1 (Vulkan mat-vec column chunking on RDNA3.5) and PR #4 (DFlash draft blocks same width across sequences / `--spec-draft-adaptive`); halo-box/llama.cpp PR #1 (adaptive draft length, cherry-pick of LaurentZuijdwijk's commit), PR #6, PR #7 (gaetan-puleo "Rocm/mmq q8 j128 rdna35")
- Nathanw1014/strix-halo-llamacpp (EXPLORING.md, benchmarks, kv-membench) and Nathanw1014/llama.cpp kernel branches
- gaetan-puleo/llama-cpp-strix-halo and gaetan-puleo/llama-cpp-strix-halo-patches
- ciru-ai/ROCmFPX — branches `kairic-edge-qwen38-27b-v1.2` (IU4 WMMA lane), `qwen3.8-activefpx-promptforge-v2.3` (W8A8 PromptForge lane), `vulkan/qwen4exp-rocmfpx`
- kyuz0/amd-strix-halo-toolboxes (container / ROCm setup notes for Strix Halo)
- Vulkan/RADV (Mesa) as the control backend; llama.cpp's `mul_mat_vec` `NUM_COLS` shader variants, coopmat int4/int8 paths

## Other engines and kernel libraries
- peonist-ai halogen (Strix Halo engine, ~6.3 bpw 27B, MTP + DFlash2 numbers)
- local-inference-lab/b12x (CUDA SM120 kernel library: QSA, GDN decode, PLE, hyper-connections, MTP feedback)
- Kaden-Schutt/hipfire: https://github.com/Kaden-Schutt/hipfire (Rust + HIP via dlopen, ~321 hand-written RDNA kernels)
- SGLang (DFlash2 / MTP acceptance reports), vLLM, lemonade (AMD)
- ROCm: hipBLASLt (bf16 GEMM), rocWMMA / `__builtin_amdgcn_wmma_*` (RDNA3.5 wave32 WMMA f16/bf16/iu8/iu4), Composable Kernel `DeviceGemmMultipleD_Wmma_CShuffleV3`, rocprofv3, HIP graphs
- `mimiron/rocm:10.0.0` container image (host has no ROCm); `luv` (luarocks) only as a stray dependency of a tool

## Ideas taken from those sources (board in `docs/ideas.md`)
- Speculative decoding: in-checkpoint MTP draft layer; DFlash / DFlash2 block draft with selector lattice; n-gram map `ngram-map-k4v` (llama.cpp) in front of the draft model; llama.cpp speculator priority order and `--spec-draft-n-max`; adaptive draft length from an acceptance EMA (`--spec-draft-adaptive`); `p_min` confidence cut; DSpark Markov / confidence head; greedy-exact verification contract
- GEMV / GEMM: per-shape threads-per-row lane policy; 2 rows per thread; x staged in LDS; int8 x + `v_dot4`; WMMA iu8 for multi-column verify passes; persistent grid; one arena; non-temporal weight loads; interleaved / repacked weight layouts; MALL prefetch of the next layer; two streams; native decoders for Q3_K / IQ3_S / IQ4_NL; merged same-input tensors (qkv+z+β+α, q+k+v, gate+up); split-K; one-ahead dequant on a second stream; bf16 consumers; MMQ-style int8 WMMA tiles (halo-box PR #7); W4A4 iu4 WMMA (Nathan / halogen / Kairic Edge); W8A8 via CK (PromptForge); fuse dequant into the GEMM A-tile load; hipBLASLt cold-cache autotune
- Attention / KV: head-major KV (channel camping); int8 KV; GQA-packed WMMA decode attention with split-KV + LSE merge; transposed V cache; flash-attention prefill; QSA indexer top-k with radix select, pooled block keys, "IndexShare" across draft steps; 262K context with 2 slots
- GDN: state in registers across T tokens with replay on rejection; tiled / chunked scan for prefill; register-prefetch of scan tile loads
- MoE (Flash-Next): host-sorted slots → grouped int8 WMMA GEMM; shared expert as a separate format; deterministic down-projection combine (no float atomics); PLE n-gram rows gathered from the SSD mmap with `MADV_WILLNEED`; hyper-connection fusions (norm + inject partials, silu + quant into the up GEMV); Q5_1 requant of Q8_0 dense tensors; FP4 / imatrix variants
- Numerics / bit tricks: `(q + 0x60) ^ 0x80` signed unpack, `v_perm` nibble LUTs, packed f16x2 math, magic-number int→float, `v_alignbit` / `v_bfe` unpacks, native `v_rsq_f32`
- Launch overhead: HIP graphs, HRX/PM4 dispatch, kernel fusion (add+rmsnorm+quant, silu·mul+quant, 1024-thread norms), launch-count accounting at ~2 µs each
- Loading / memory: `pread` + drop page cache per tensor; zero-copy mmap as GPU-visible memory (llama.cpp issue); weights < 100 GB with the 28.8 GB PLE table left on SSD
- Serving: token-id-only engine boundary; llama.cpp web UI for t/s measurement; OpenAI-compatible SSE with `reasoning_content` split and tool-call parsing; multi-slot batching with per-slot double-buffered state; prefix caching
