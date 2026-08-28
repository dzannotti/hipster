# DFlash / DFlash2 + ngram-map-k4v in llama.cpp (investigated 2026-08-28)

User report: `-hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M --spec-type draft-dflash,ngram-map-k4v --spec-draft-n-max 5`
"6× the ceiling while MTP was 1.9×". [V] = verified in llama.cpp master d077b4c2 / the GGUF header, [C] = claimed.

## What it is
- **DFlash** (arXiv 2602.06036, z-lab): a 5-layer Qwen3-style draft that predicts a whole block in ONE forward:
  input `[anchor token, MASK × 7]`, bidirectional attention inside the block, conditioned on the target by
  `concat(h_L1..h_L5) → fc → RMSNorm` (5 target layers) injected as K/V into every draft layer's KV cache.
- **DFlash2** (Inco AI, PR #27342 merged 2026-08-27): + two-tap grouped dynamic depthwise conv around attention
  and FFN, + a candidate selector (top-16 per position, rank-256 bilinear transition score, greedy walk on CPU).
- Draft GGUF for Qwen3.8-27B [V]: 1.9B params, Q4_K_M 1.14 GB; `target_layers=[6,20,34,48,62]` (residual stream
  entering those layers = 5×5120 f32 per token exported from the target), block_size 8, no token_embd/output
  (shares the target's), selector tensors 2×[256×248320]. Verification = token match at each position (exact
  at T=0), `k = min(n_max, 7)` drafts per target pass; hybrid GDN needs a state rollback on partial acceptance.
- **Drafting is parallel** (one draft forward = up to 7 tokens), vs our chained MTP (n forwards for n drafts).

## ngram-map-k4v [V] (`common/ngram-map.cpp`)
Draftless self-speculation: key = last 12 tokens, hash → positions in the sequence's own history; drafts the up to
48 tokens that followed the key before, only when the follow-up is consistent (`max_occur ≥ 2·others`). ~1–2 ms
CPU per token, zero GPU. Impl priority order: ngram-* first; **the first non-empty draft wins**, so with
`draft-dflash,ngram-map-k4v` the n-gram draft (≤48 tokens) preempts DFlash2 whenever the text repeats itself.

## Published numbers (all [C])
| setting | plain | spec | note |
|---|---|---|---|
| PR #27342, M5 Pro, Q4_K_M, GSM8K, T=1, n=7 | 10.4 | 18.9 t/s (1.81×), accept 5.03 | |
| H200 BF16 | 57.5 | MTP n=4 90 (1.57×); DFlash2 n=7 105 (1.82×) | |
| RTX PRO 6000, LiveCodeBench | 68 | DFlash2 154 (2.26×); +ngram 156 | single-turn: ngram +1% |
| synthetic repetitive, ctx 4K / 36K | 67 / 59 | DFlash2+ngram 268 (4.0×) / 498 (8.4×) | ngram dominates |
| 18-turn iterative coding session (Qwen3.6-27B, DFlash1) | 53.5 | DFlash 199 (3.7×); +ngram-mod+map-k4v **321 (6.0×)** | **the 6×** |

So: DFlash2 alone is 1.8× (T=1 prose) to ~2.3× (greedy code) on llama.cpp; the 6× comes from 48-token n-gram
drafts on multi-turn code-editing sessions where the model re-emits text it has already seen. Our chained MTP
n=5 already gives 2.85× (12.5 → 35.6 t/s) — above llama.cpp's MTP (1.57×) and at their DFlash2 level.

## What it means for hipster
1. **n-gram self-speculation is the cheap big win** for the repetitive/agentic case: host-side, ~1 ms, stacks in
   front of MTP (first non-empty draft wins), drafts up to 48 tokens verified in one target pass (our verify pass
   is T ≤ 8 today → raise `max_T` for this path; a 48-token verify is a 48-row GEMV pass, ≈ 2–3 ms extra over 80).
2. DFlash2 for the 27B: draft = 1.14 GB streamed per step (≈ 5 ms) + 8-row lm head; gain over MTP hinges on
   acceptance (τ 5.4 claimed vs our MTP at n=5 — measure our per-position acceptance first). It needs: export of
   the residual at layers 6/20/34/48/62, the encoder + draft KV injection, an 8-row bidirectional draft forward,
   the selector gathers, and the same verify loop we have. Sized at ~1.3× over our MTP if their τ holds.
3. No DFlash draft exists for Flash-Next (checked incoai + z-lab): its in-checkpoint MTP block is the only drafter.
Sources: arXiv 2602.06036 · inco.ai/blog/dflash2 · ggml-org/llama.cpp#27342 · #22105 · HF incoai/Qwen3.8-27B-DFlash2-GGUF ·
github.com/lukaLLM/DFlash2_Qwen3.8_3.6_27B_LlamaCPP.
