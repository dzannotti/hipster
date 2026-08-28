# Roadmap — measure, resolve, move on

Each phase ends with a number in `docs/` and the next bottleneck named. Order is by what blocks
the next measurement, not by feature.

## Phase 0 — roofs ✅
240 GB/s read · 1.8–2.4 µs/launch (graphs don't help) · 51.8 TFLOPS hipBLASLt · HIP over Vulkan.
Per-token bytes: 27B 16.48 GB (floor 14.6 t/s) · Flash-Next 6.43 GB (floor 37 t/s; 4.1 GB of it
is Q8_0 dense tensors — a requant lever llama.cpp doesn't have).

## Decision 2026-08-28: Flash-Next first
Both engines exist and are exact. Priority goes to the model with the higher ceiling, single-stream and under
concurrency: Flash-Next streams 6.1 GB/token (floor 39 t/s) vs 16.5 GB for the 27B (14.6 t/s); at 4 concurrent
sequences the dense part of Flash-Next (4.3 GB) is shared and only the 1.8 GB of experts scales (mostly disjoint
expert sets) → ~11.5 GB per 4-token pass ≈ 83 t/s aggregate ceiling vs 58 for the 27B, and the gap widens with
MTP because expert reads for the T verify rows overlap. 27B work continues only where it is shared (GEMV,
attention, GDN kernels) or asked for. Order: Flash-Next MTP → multi-slot verify/decode → prefill GEMM path →
QSA beyond 2K → serving.

## Status 2026-08-28: Phase 1 done (12.5 t/s, 87% of floor), Phase 2 MTP done (35.6 t/s code, exact), Phase 3 prefill v1 done (647 t/s @2K, 475 @32K). Depth gate green at 16K/32K (needle + logprobs vs llama.cpp). Phase 4 Flash-Next: decode exact (12/12), 28 t/s bare (floor 39), MTP 37–40 t/s exact, batching 63/80 t/s at 4/8 slots exact, prefill ~1000 t/s @2K and 795 t/s @16K with QSA sparse attention exact vs llama.cpp at 4K/16K (needle). Next: serving layer (tokenizer, chat template, OpenAI endpoints), 262K validation, dense-tensor requant, PLE gather overlap.

## Phase 1 — 27B single-token decode at the wall
Goal: a full forward pass of Qwen3.8-27B at ≥ 13 t/s bare (≥ 90% of floor), bit-comparable to
llama.cpp logits. Nothing else exists until this number exists.
1. GEMV for every block type the file uses: Q4_K ✅ (230 GB/s), Q5_K, Q6_K, IQ4_XS, Q8_0, Q3_K, IQ4_NL, IQ3_S (1 tensor — dequant at load).
2. Layer kernels, each measured against its byte/flop roof: RMSNorm(1+w), GDN step (conv4 + gated delta rule, fp32 state 3 MiB/layer — state r/w is 288 MiB/token ≈ 1.2 ms, must be fused into one kernel), gated GQA attention decode (KV 64 KiB/token), SwiGLU fused into the down-proj GEMV input, embedding gather, LM head (Q6_K 994 MiB = 4.1 ms/token alone — the shortlist trick later).
3. One fused launch per sublayer where the roof says launches matter (64 layers × ~6 launches ≈ 0.8 ms; fine).
4. Correctness harness: dump per-layer activations from llama.cpp for a fixed prompt and compare.
5. Measure: ms/token, GB/s achieved, rocprofv3 breakdown. Gap to 68.7 ms floor accounted kernel by kernel.

## Phase 2 — speculation on the 27B (the only decode lever above the wall)
1. Multi-token verify pass (ncol 2–8 GEMV ✅ at 75–85%; GDN chunk kernel for 8 tokens; attention with 8 queries).
2. GDN recurrent + conv state snapshot/restore for rejected drafts (raw-input replay or snapshot — measure both; state is 144 MiB).
3. MTP (in-checkpoint `blk.64.nextn.*`, 1 attention layer + shared head), greedy-exact verify. Sweep n_max × adaptivity on code/prose/agent-edit corpora (`/srv/models/.work/corpus`).
4. DFlash2 (`incoai/Qwen3.8-27B-DFlash2-GGUF` Q4_K_M): hidden-state capture at layers 5/19/33/47/61, draft KV, 5-layer bidirectional drafter, two-tap conv, top-16 selector.
5. n-gram drafting (draftless; highest precedence when it hits) stacked under MTP/DFlash2.
Target: 35–45 t/s single stream on code; halogen's 31.7 mean is the bar to beat.

## Phase 3 — prefill on the 27B
hipBLASLt bf16 for the big GEMMs (dequant weights to bf16 tiles on the fly or W8A8 via CK — measure), chunked prefill, flash-attention prefill kernel (KV f16 → int8), GDN chunked scan. Target ≥ 550 t/s @32K (halogen), roof ~900.

## Phase 4 — Flash-Next
Fused per-layer kernels: hyper-connection mix/combine, router top-10 + expert GEMV batched over the 10 chosen experts (one launch), shared expert, GDN (36 layers), QSA indexer + top-2048 sparse attention (12 layers), PLE gather (16 rows, NVMe/host-resident, prefetched during layer 0). Bare target ≥ 35 t/s (llama.cpp: 18–27); then requantise the 4.1 GB of Q8_0 dense tensors and re-measure; then MTP (our head GGUF, IndexShare across draft steps). Expert residency: measure LRU hit rate on routing traces before building any paging.

## Phase 5 — serving
OpenAI-compatible front-end (chat/completions with streaming, tools = Qwen XML tool-call parser, vision via the 27B ViT, reasoning split), tokenizer + Jinja template, multi-slot with prefix cache + semantic-anchor checkpoints of GDN state. The engine sees token ids only.
Language: engine C++/HIP. Front-end: recommend Rust (`tokenizers`, `minijinja`, axum) in-process via FFI — the tokenizer is the riskiest correctness piece and the HF crate is exact; decision deferred until Phase 5, nothing in Phases 1–4 depends on it.

## Open questions to settle by measurement
- Repacked weight layout vs ggml blocks (only if a GEMV can't hit ≥ 95% on some type).
- 6-column register pressure in the verify GEMV.
- Whether the PLE table fits in RAM alongside everything else, or stays on NVMe (current: on disk, ~0.5 ms/token).


## Open items (2026-08-28, late)
- 27B verify GEMV at 8 columns: ~165 GB/s (K-quant epilogue); Q8_K activations prototype +8% on Q5_K (bench only); interleaved layout +3–15% on three formats. Both measured, neither adopted.
- 27B with 3 slots: a 24-row verify needs two GEMV launches (≈2.3× a single pass for 3× the tokens) — does not pay; 2 slots is the sweet spot.
- Prefix caching (multi-turn chat re-prefills the whole conversation); `stop` strings; `logprobs`.
- Flash-Next: multi-slot exactness at long context: `LONG=docs/ref/fn-long4k.json batchfn … 32 2` → EXACT, 42 t/s aggregate (QSA path, per-slot GEMM prefill); 262K not validated.
- DFlash2 acceptance on prose is 15–30%: the draft is the limit there (adaptive n gains little because a 16-row verify costs ~the same as 8).
