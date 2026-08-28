# Qwen3.8-Flash-Next (GGUF `qwen4exp`) — forward-pass spec for the engine

Derived from the GGUF metadata (docs/research.md) and llama.cpp's `src/models/qwen4exp.cpp` (the
only complete reference; PR #27742). Own class/binary; shares only compile-time kernels with the 27B.

## Shapes
hidden 2560 · 49 blocks (48 trunk + 1 MTP) · layer types: `(il+1)%4==0` → QSA attention, else GDN
· wide residual hc=4 streams (10240) · GDN: qkv 10240 (16 qk heads ×128 ×2 + 48 v heads ×128), z 6144,
conv 4, **sigmoid** output gate (27B: silu) · attention: q 12288 (q|gate per head ×24), k/v 512 (2 kv
heads ×256), q/k norm, IMRoPE 64 dims · indexer: k_proj 128 (1 head, raw, cached), q_proj 512 (4 heads
×128, normed+roped), compress ratio 4, budget 2048 → top (2048+3) keys · MoE: router 2560→512 f32,
softmax, top-10, renormalised, experts gate/up Q4_K [512][640][2560], down Q5_1 [512][2560][640], shared
expert 640 with its own sigmoid gate (2560→1) · PLE at block 1 · head: final hc_mix then output Q8_0.

## Per-layer dataflow (wide residual R: [4][2560] per token)
```
if il == 1: R = PLE(R, tokens)
m, inject = hc_mix(R; hc_attn_norm/down/up/inject)        # m [2560], inject [4]
b = GDN(m) or QSA_attention(m)
R = hc_combine(R, b, inject)
m, inject = hc_mix(R; hc_ffn_*)
b = MoE(m) + sigmoid(shexp_gate(m)) * shared_ffn(m)
R = hc_combine(R, b, inject)
```
hc_mix: xn[s] = rmsnorm(R[s]) · w_norm[s·2560..] (per stream, eps 1e-6; gamma already (1+w) in the file);
lo = silu(W_down·xn_flat / 4) [320]; gate = sigmoid(W_up·lo) [10240]; mixed = mean_s(xn[s]·gate[s]);
inject = W_inject·xn_flat [4].
hc_combine: w[s] = 2·sigmoid(inject[s]/4); R[s] += b · w[s].
Fusion target: hc_mix = 1 kernel (norm + 10240×320 + 320×10240 + inject 10240×4 are tiny: 6.6 MB
Q8 per layer-half), hc_combine folds into the following norm. SGLang fused these ("Mix/Combine").

## PLE (block 1)
Context per token: (tok, prev1, prev2), EOS (248044) cuts everything at or before it; positions
before the sequence start read as EOS. For n = 2, 3: `mixed = tok·m0 ^ prev1·m1 (^ prev2·m2)`
(uint64, multipliers from `ple.layer_multipliers`); 8 heads per n: row_h = mixed % vocab_h +
offset_h (16 rows total, `ple.head_vocab_sizes/head_offsets`). Gather 16 rows × 160 dims (IQ4_NL,
90 B per row) → emb [2560]. key = W_key·emb [10240], value = W_value·emb [2560];
key[s] = rmsnorm(key[s])·w_key; query[s] = rmsnorm(R[s])·w_query; s_s = ⟨key[s], query[s]⟩/√2560;
gate_s = sigmoid(sgn(s_s)·√max(|s_s|,1e-6)); gated[s] = value·gate_s; n = rmsnorm(gated[s])·w_conv;
conv: depthwise causal, kernel 4, dilation 3 (history 9 tokens per sequence, state to keep):
out[c,t] = Σ_k w[k,c]·n[c, t−(3−k)·3] → silu; R[s] += gated[s] + conv_out[s].
Table: 28.8 GB. The existing llama.cpp fork keeps it on NVMe (`--ngram-on-disk`, 64 threads,
~0.5 ms/token, ~250 ms per 2048-token ubatch). Ideas: host RAM if the budget allows; an LRU of hot
rows (n-grams repeat heavily in code/agent traffic); prefetch the gather for the next token(s)
during layer 0 (the tech report placed the PLE at layer 2 for exactly that overlap).

## QSA attention (12 layers)
k_raw = W_ik·m [128] per token → indexer cache (raw, unnormed). Blocks of r=4 consecutive cache
positions: pooled_b = rope(rmsnorm(mean of the 4 raw keys)·w_kn, pos = block start); q_h =
rope(rmsnorm(W_iq·m)_h·w_qn, pos) for 4 heads; score_b = Σ_h relu(⟨q_h, pooled_b⟩); every token of a
block gets its block's score; +bias (masks future/invalid); top (2048+3) → the attended key set
(dense below 2051 keys). Then standard gated GQA attention (24 q / 2 kv × 256, scale 1/16,
sigmoid(gate)) over the selected keys. KV per token: 12 layers × 2 × 256 × 2 (K,V) × 2 B = 24 KiB f16
+ indexer 256 B. b12x/DeepSeek: radix top-k, ascending-id tie break; at decode the selection is per
token; with MTP drafts the selection can be shared across draft steps ("IndexShare").

## GDN (36 layers): as the 27B kernels (same 48/16/128 shapes, conv 4) with sigmoid gate — a
template parameter, not a runtime flag. State 108 MiB per sequence.

## MoE (every layer)
p = softmax(W_router·m) over 512; top-10; w_i = p_i / Σ_top p; out = Σ_i w_i · down_i(silu(gate_i(m))·up_i(m))
(+ expert_weights_scale if ≠ 1 — check hparams) + sigmoid(W_sg·m) · down_s(silu(gate_s(m))·up_s(m)).
Decode: 10 experts × (2×2560×640 + 640×2560) at ~4.5 bits ≈ 27.6 MB per layer, 1.35 GB per token in
total — one launch per layer over the 10 chosen experts (expert ids in device memory, no host
sync). Prefill: sort tokens by expert, per-expert GEMM tiles (Gaetan/b12x notes).

## Bytes per token (decode, UD-Q4_K_XL): 6.43 GB → floor 37 t/s; 4.1 GB of that is Q8_0 dense
(attention/GDN/hc projections) — a requantisation lever the artifact format can take.

## Memory plan (128 GB box): experts 79 GB + dense 5 GB resident; PLE 28.8 GB on NVMe or partly
cached; KV 24.75 KiB/token → 262K = 6.3 GB per slot f16 (3.2 GB int8); GDN state 108 MiB per slot.

## Implementation state (2026-08-28)
- `engine/src/qwen4exp.{h,cpp}`, `engine/kernels/fn.{h,hip}` (Flash-Next-only kernels: Q8_0 embed,
  gated-residual norm/mix/combine, exact f32 router GEMV, softmax top-10, expert GEMV over routed
  ids, MoE combine with the shared expert, PLE gate + dilated conv), plus compile-time variants of
  shared kernels (GDN step with sigmoid gate, attention 24 q / 2 kv). Q5_1 (expert down-proj) got a
  row-SoA decoder; the SoA decoders handle K = 640 / 320 (partial last 256-block).
- Decode path only (T ≤ 8, dense attention — exact below 2051 cached tokens), no MTP block, no QSA
  top-k, no prefill GEMM path yet. Table rows are gathered from the mmap'd GGUF on the host
  (16 IQ4_NL rows of 90 B per token).
- Reference: `tools/ref-server-fn.sh` (llama.cpp ROCm, n-gram table on disk) → `docs/ref/fn-*.json`.
