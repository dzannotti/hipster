# hipster — a from-scratch inference engine for two Qwen3.8 checkpoints on gfx1151

## What this is

A purpose-built inference server for exactly these artifacts, on exactly this GPU:

| target | file | arch | streamed / token |
|---|---|---|---:|
| `qwen3.8-27b` | `/srv/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf` (+ `mmproj-F16.gguf`) | `qwen35`: 64 layers (48 Gated DeltaNet + 16 full attention, every 4th), dense SwiGLU 17408, hidden 5120, 24q/4kv heads × 256, vocab 248320, 1 MTP layer, vision tower | 16.48 GB |
| `qwen3.8-flash-next` | `/srv/models/qwen3.8-flash-next/UD-Q4_K_XL-MTP/*.gguf` (5 shards) | `qwen4exp`: 49 layers (36 GDN + 12 full attention w/ QSA indexer top-k 2048 + compress-ratio 4), 512 experts / 10 active + 1 shared (640 wide), hidden 2560, hyper-connections ×4 (low-rank 320), PLE n-gram table (3-gram, 16 heads, 320M rows, 28.8 GB, IQ4_NL), 1 MTP layer | 6.43 GB (+ 16 PLE rows) |

Hardware: AMD Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151, 40 CU wave32, 128 GB unified.
Measured roofs are in `docs/roofline.md`: **240 GB/s read, 1.8–2.4 µs per kernel launch
(graphs don't help), 51.8 TFLOPS f16 via hipBLASLt.**

Stack: **C++ / HIP (ROCm)**. Decision and data in `docs/roofline.md` §"Decision".

## Hard constraints
- **Model weights stay under 100 GB of RAM, no paging.** Flash-Next: 84 GB in GTT (the 28.8 GB n-gram
  table stays on the SSD: row gathers from the mmap, never uploaded). KV/state may go above the
  weight budget; 2 × 262K slots is an acceptable configuration if 4 × does not fit (int8 KV halves it).
  Loaders read weights with pread and drop the page cache per tensor so loading never thrashes.

## Principles (non-negotiable)

1. **Nothing generic.** Two models, one GPU. Shapes are compile-time constants. No dynamic
   op graph, no backend abstraction, no dtype dispatch at runtime. Shared code is shared by
   template/trait composition of pieces both models genuinely have (GDN layer, GQA attention,
   RMSNorm, sampler, MTP verify), never by a "model description" interpreted at runtime.
2. **Nothing exists until a measurement says it must.** Every kernel, every fusion, every
   cache starts as a number in `docs/` showing what it costs today and what the roof is.
   Optimizations that don't move a measured number are reverted.
3. **Roofline-driven.** Decode: bytes streamed / 240 GB/s is the floor; the gap between the
   floor and the measured pass time is the budget, and it is accounted for kernel by kernel
   (rocprofv3). Prefill: FLOPs / 51.8 TFLOPS. Launches: count × 2 µs.
4. **Bit-exact contracts, then speed.** Every kernel has a reference (CPU or naive HIP) and a
   test with a tolerance stated in the test. Speculative verification is greedy-exact:
   spec-on and spec-off must emit identical token streams at temperature 0.
5. **Serve is thin.** The engine sees token ids only. Tokenizer, chat template, tool-call
   parsing, OpenAI schema, SSE, image decoding live in the front-end; they are not on the
   per-token path.

## Layout

```
bench/roofline/   hardware roofs (bandwidth, launch, GEMM) — the numbers everything is judged against
docs/             measurements and decisions; every optimisation has an entry with before/after
```
(engine/ front-end/ tools/ are added as they earn their place — see docs/roadmap.md)

## Working rules for agents

- Build and run inside `mimiron/rocm:10.0.0` (see `bench/roofline/run.sh` for the docker
  invocation; the host has no ROCm). The GPU is shared with the production llama-swap
  containers on this box — check `docker ps` and `free -g` before loading a model; the
  Flash-Next container alone holds ~84 GB of GTT.
- Report numbers as measured, with the command; prefer `pp2048`/300-token decode runs, discard
  the first run (boost clock), and note thermal state (the APU throttles ~25% after ~10 min).
- Reference implementations to consult (read-only, in `/srv/models/.work/`):
  `ninfer/` (from-scratch CUDA engine for the same models: architecture docs in
  `docs/maintainer/*.md` are the best written spec of qwen35's GDN/attention/MTP/vision),
  `franken/src/models/qwen4exp.cpp` (the only complete Flash-Next forward pass),
  `llama.cpp/ggml/src/ggml-cuda/` (Q4_K/Q5_K/Q6_K/IQ4_XS/IQ4_NL block formats and mmvq).
- Never commit or push without being asked; never touch `/srv/models` or `/homelab`.
