# Research notes (2026-08-27) — FreeToken, DFlash2, Qwen3.8 architecture, hipfire, halogen, baselines

Condensed from a web-research pass; every claim carries its source. "Verified" = the raw file was fetched.

## FreeToken — NOT speculative decoding
- "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution", arXiv 2608.16157 (2026-08-17), Yang, Fan, Pan, Xi, Wang, Sun, Keutzer, Han, Zaharia, Xu, Stoica. Repo https://github.com/FlashML-org/FreeToken (Apache-2.0, Python + CUDA). No MTP / n-gram / drafting anywhere in it.
- Mechanism: CPU-resident complete expert pool, GPU VRAM as an elastic LRU expert cache, non-expert weights GPU-resident. Per decode step with `m` missing experts, fill `q* ≈ m·B_PCIe/B_host` into the cache and compute the rest on CPU concurrently. Global LRU across layers exploits temporal routing locality. Double-buffered per-layer expert streaming for prefill. CUDA-graph-compatible routing (fixed-shape buffers, device-resident counts). "Semantic anchor" checkpoints of recurrent/KV state at special-token boundaries so prefix reuse survives agent context edits. FTW weight format populated by direct I/O.
- Results: Qwen3.6-35B-A3B BF16 77–83 t/s on a 5090 (1.8–2.3× llama.cpp/KTransformers); GLM-5.2 753B on RTX PRO 6000 14.9 vs 7.3 t/s.
- **What transfers to Strix Halo (unified memory, no PCIe wall):**
  1. Expert residency by routing locality — here the constraint is *fitting* (79 GB experts + 28.8 GB PLE + KV in 128 GB), not PCIe. Worth measuring: LRU hit-rate vs cache size on real Flash-Next routing traces, before deciding whether cold experts can live on NVMe.
  2. Semantic-anchor checkpoints of GDN state + KV at `<think>`/`<tool_call>`/turn boundaries (llama.cpp's `-ctxcp/-cms` is the same idea).
  3. Fixed-shape, device-side routing so MoE fits one fused kernel with no host sync.

## DFlash / DFlash2
- DFlash v1: arXiv 2602.06036 (z-lab, Feb 2026), https://github.com/z-lab/dflash. Block-diffusion drafter: a small non-causal transformer emits a whole block of draft tokens in one pass, conditioned on target hidden states from 5 layers (concat → `W_c` → RMSNorm) injected as extra KV into every draft layer. Chain verification, greedy = longest matching prefix; lossless with rejection sampling.
- **DFlash2** (Inco AI + z-lab, Aug 2026, https://inco.ai/blog/dflash2/): adds (1) grouped two-tap dynamic depthwise conv `Conv(x)_t = k_{t,0}⊙x_t + k_{t,1}⊙x_{t−1}` around every sublayer (group 16), (2) top-16 candidates per position + a path selector `S_t(a,b) = U_t(b) + ⟨A(a)⊙H(h_t), B(b)⟩` (rank 256). Overhead ≈1.3% of cycle.
- **Qwen3.8-27B-DFlash2** (verified config): https://huggingface.co/incoai/Qwen3.8-27B-DFlash2, GGUF https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF (Q4_K_M 1.14 GB, Q8_0 2.06 GB), Apache-2.0. 5 layers, hidden 5120, 32 q / 8 kv heads × 128, FFN 17408, non-causal, sliding window 2048, `block_size 8`, `mask_token_id 248070`, `target_layer_ids [5,19,33,47,61]`, rope θ 1e7, ~2B params. Acceptance length GSM8K 5.46 / MATH 5.28 / HumanEval 4.39 / MBPP 4.79 / MT-Bench 4.10 (Q4_K_M draft ≈ BF16). H200 c=1: 2.67–3.43×.
- llama.cpp support merged 2026-08-27 (PR #27342; Vulkan flagged buggy). **No DFlash/DFlash2 drafter exists for Flash-Next** — it has only its native MTP block.
- Engine implications: capture residual-stream hidden states at layers 5/19/33/47/61 during every verify pass; keep a draft-side KV cache of projected target features (rolled back on rejection); run the 5-layer bidirectional drafter over `[anchor; 7 mask]`; **verify 8 tokens per target pass, which requires snapshot/restore of GDN recurrent + conv state** (KV can be truncated, recurrent state cannot). Same requirement as MTP; it is the central architectural constraint.
- Community head-to-head on Strix Halo is workload-dependent: DFlash2 wins code/structured (halogen 31.7 mean vs MTP 26.9), MTP wins prose (PieBru 33.8 vs 24.6).

## Qwen3.8-27B (verified config, `qwen3_5`)
- Dense; hidden 5120, 64 layers = 16 × [GDN, GDN, GDN, full-attn], FFN 17408 SwiGLU, vocab 248320, untied head, RMSNorm 1e-6, ctx 262144 (1M YaRN×4).
- GDN: 48 V heads × 128, 16 QK heads × 128, conv 4, fp32 state, swish output gate. State = 3 MiB/layer, 144 MiB total, context-independent.
- Attention: 24 q / 4 kv × 256, q/k norm, partial RoPE 64/256, θ 1e7, mRoPE [11,11,10], sigmoid output gate. **KV = 64 KiB/token f16** (16 layers).
- MTP: 1 full-attention layer; `mtp.fc` fuses [norm(embed(next)); norm(h_last)]; shares embeddings + LM head; own small KV.
- Vision: 27-layer ViT hidden 1152, 16 heads, patch 16, merge 2×2, temporal 2, out 5120. Tokens: vision_start 248053, vision_end 248054, image 248056, video 248057; BOS/EOS/pad 248044.
- Chat template (verified): ChatML; tools in system turn as `<tools>[json]</tools>`; **tool calls are XML** `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`; tool results in `<tool_response>…</tool_response>` inside a user turn; `<think>…</think>`; kwargs `enable_thinking`, `preserve_thinking`, `reasoning_effort ∈ {xhigh,medium,low}`.
- Sampling: thinking T=1.0 top_p .95 top_k 20; instruct T=0.7 top_p .8 top_k 20 presence 1.5.

## Qwen3.8-Flash-Next (verified config, `qwen4_exp`, released 2026-08-26)
- 125B MoE + 51B n-gram table + 4B MTP, 6B active. hidden 2560, 48 layers = 12 × [3 GDN, 1 QSA attention]; vocab 248320; θ 1e7; partial rotary 0.25; license qwen-community-1.0.
- GDN as 27B (48 V / 16 QK × 128, conv 4, fp32 state) but **sigmoid** output gate. 36 layers → 108 MiB state.
- QSA attention: 24 q / **2 kv** × 256 → **24 KiB/token KV**. Indexer: 4 q heads × 128, 1 k head, budget 2048 tokens, compress ratio 4 (top-512 blocks + tail). Equations (tech report): `q̂ = RMSNorm(W_Q x)`, `k̂_b = RMSNorm(AvgPool(k_{block b}))`, partial RoPE after pooling, `I_ib = Σ_h ReLU(⟨q_i^h, k̄_b⟩)` block-causal, top-K blocks expanded to tokens. Exactly dense below budget.
- MoE every layer: 512 experts, top-10 + shared expert (640 wide), router `ffn_gate_inp` f32.
- Hyper-connections: 4 residual branches (10240 wide), low-rank 320 mix (`hc_*_up/down/inject/norm`) around every attn and ffn sublayer; SGLang fused them into "Mix"/"Combine" kernels (2× at M≤16).
- PLE n-gram embeddings at layer index 1 in GGUF (`ple.layers=[1]`, HF `ple_layer_ids [2]`): 3-gram (bigram+trigram) × 8 heads = 16 hashed rows of 160 dims per token (GGUF `embedding_length_per_layer_input=160`, `per_layer_token_embd` [160 × 320,001,536] IQ4_NL = 27.5 GiB), through `ple_key`/`ple_value`/conv. Pure gather — designed for host/NVMe residency with prefetch overlapping layer 0.
- MTP: 1 hybrid layer (QSA + MoE + hc); tech report says multi-step MTP reuses top-k indexer results across draft steps ("IndexShare"). SGLang reports 58–90% acceptance.
- llama.cpp: PR #27742 merged 2026-08-27 (host-side splitmix64 hashing for PLE); MTP still WIP upstream — the local MTP head GGUF is ours (`/srv/models/qwen3.8-flash-next/mtp/`).

## hipfire
- https://github.com/Kaden-Schutt/hipfire — Rust, HIP loaded via dlopen (no link-time ROCm), RDNA-first, ~321 hand-written HIP kernels with per-gfx `.hsaco` precompiled + hipcc JIT fallback, own quant formats (MQ4/MQ4R/HFQ…, not GGUF), KV compression modes (q8, asym, FWHT-rotated), fp32 GDN state, DFlash + n-gram drafting. gfx1151: Qwen3.5-27B MQ4 14.8 t/s AR / 104.5 with DFlash on a code prompt. Reference only (formats, kernel build pipeline, KV modes).

## halogen (peonist-ai) — public numbers, Strix Halo, Qwen3.8-27B ~6.3 bpw (23.5 GB/token)
- Prefill **566 t/s @32K** (600+ with better cooling); decode serial 10.58, MTP 26.88, **DFlash2 31.71 mean** (20.8 prose … 44.2 code). Own accounting: 10.58 × 23.51 GB = 249 GB/s vs measured 240 ceiling. 8 slots at 4.87× aggregate; speculation and batching mutually exclusive. Optional W4A4 int4 prefill (+9%, −0.45 pt top-1).

## Strix Halo llama.cpp baselines for the 27B (~Q4)
bare decode 10.5–14 t/s; prefill 300–390 t/s short ctx; MTP 30–36; DFlash2 up to ~42 on structured. Fastest community results all pair ~4-bit weights with MTP/DFlash2 at `-np 1`. Nobody has published Flash-Next numbers on Strix Halo yet.

Targets for this engine, derived from roofs (not yet measured): 27B decode with DFlash2 τ≈4.5 → 35–45 t/s single stream; prefill ≥ 550–800 t/s @32K; Flash-Next bare 40 t/s at current bytes (≥ 60 with the Q8 dense tensors requantised), 100+ with MTP.
