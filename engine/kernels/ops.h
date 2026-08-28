// Layer kernels for Qwen3.8 decode (T = 1 token, one sequence). f32 activations.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "gemv.h"

namespace hip {

// out[t][n] = dequant(token_embd row toks[t]) for Q4_K [n_vocab x n] rows, t < T (toks on device).
void embed_q4_K(const void* W, const int* toks, int T, float* out, int n, hipStream_t s);
// rmsnorm over `rows` rows of `n`: out = x * rsqrt(mean(x^2) + eps) * w
void rmsnorm(const float* x, const float* w, float* out, int rows, int n, float eps, hipStream_t s);
// Fused, per row t < T: if y != null: x += y; xn = rmsnorm(x) * w; xq = q8(xn). n multiple of 32.
void add_rmsnorm_quant(float* x, const float* y, const float* w, float* xn, XQ8 xq, int T, int n, float eps, hipStream_t s);
// Two rmsnorms concatenated: out[t] = [rmsnorm(a[t])*wa | rmsnorm(b[t])*wb] (na + nb), quantised into xq.
void norm2_concat_quant(const float* a, const float* wa, int na, const float* b, const float* wb, int nb, float* out, XQ8 xq, int T, float eps, hipStream_t s);
// bf16 variants for the GEMM (prefill) path: if xb != null also write bf16 rows; xq may be {null}.
void add_rmsnorm_bf16(float* x, const float* y, const float* w, uint16_t* xb, int T, int n, float eps, hipStream_t s);
void norm2_concat_bf16(const float* a, const float* wa, int na, const float* b, const float* wb, int nb, uint16_t* out, int T, float eps, hipStream_t s);
void silu_mul_bf16(const float* gate, const float* up, uint16_t* out, size_t n, hipStream_t s);
// gate = columns [off, off+N) and up = [off+N, off+2N) of a bf16 [T][Nt] matrix -> out bf16 [T][N]
void silu_mul_bf16_seg(const uint16_t* gu, int Nt, int off, int N, uint16_t* out, int T, hipStream_t s);
// Fused: xq = q8(silu(gate) * up) (n multiple of 32)
void silu_mul_quant(const float* gate, const float* up, XQ8 xq, int n, hipStream_t s);
// y += x (n)
void add_inplace(float* y, const float* x, int n, hipStream_t s);
// out = silu(gate) * up (n)
void silu_mul(const float* gate, const float* up, float* out, int n, hipStream_t s);

// ---- Gated DeltaNet (per layer) ----
// Causal conv over C channels with 4 taps, T tokens sequentially: st_in [3][C] holds the 3 previous
// inputs (oldest first); x [T][C]. out[t][c] = silu(sum_k w[k][c] * hist); final history -> st_out.
// Batched forms: SeqBatch = up to 8 sequences, each a run of rows [r0, r0+T) with its own state buffers
// (in/out = indices into a state array of stride st_stride floats). RowBatch = per-row position + KV slot for
// attention (n == 0 -> single sequence at pos0+t, slot 0; the prefill path uses that form).
struct SeqDesc { int r0, T, in, out; };
struct SeqBatch { SeqDesc s[8]; int n = 0; };
struct RowBatch { int pos[32]; int kv[32]; int n = 0; };
// One sequence's contribution to a pass: T tokens (continuing that slot's sequence) at positions pos..pos+T-1.
struct SlotReq { int slot; const int* tokens; int T; int pos; };
void gdn_conv(const float* x, int T, const float* st_in, float* st_out, const float* w, float* out, int C, hipStream_t s);
void gdn_conv_b(const float* x, const SeqBatch& sb, size_t st_stride, const float* st_base, float* st_obase, const float* w, float* out, int C, hipStream_t s);
// T recurrent steps for all heads, state kept in registers. qkv [T][2048 q | 2048 k | 6144 v]
// (post-conv), z [T][6144], beta_raw/alpha_raw [T][48], dt_bias [48], a_neg [48] (= -exp(A_log)),
// state_in/out [48][128][128] f32 (M[j][i] = S[i][j]), ssm_norm w [128].
// out [T][6144] = rmsnorm_h(y_h) * silu(z_h), also quantised into xq. out may be null (replay).
void gdn_step(const float* qkv, const float* z, const float* beta_raw, const float* alpha_raw, int T,
              const float* dt_bias, const float* a_neg, const float* state_in, float* state_out,
              const float* norm_w, float* out, XQ8 xq, float eps, hipStream_t s);

// Prefill variant of the GDN step (T >= 16 typical): pre-pass computes per-token l2-normalised
// q/k, beta and decay; the recurrence runs over 16-token tiles staged in LDS (one sync per tile);
// out [T][6144] raw (pre-norm) heads. gdn_out applies rmsnorm_h * silu(z) -> f32 out (+ bf16 if xb).
void gdn_prep(const float* qkv, const float* beta_raw, const float* alpha_raw, const float* dt_bias, const float* a_neg, int T,
              float* qn, float* kn, float* beta, float* decay, float eps, hipStream_t s);
void gdn_scan(const float* qn, const float* kn, const float* qkv, const float* beta, const float* decay, int T,
              const float* state_in, float* state_out, float* raw, hipStream_t s);
void gdn_out(const float* raw, const float* z, const float* norm_w, int T, float* out, uint16_t* xb, float eps, hipStream_t s);

// Flash-Next variants (sigmoid GDN output gate; 24 q / 2 kv heads)
void gdn_step_sig(const float* qkv, const float* z, const float* beta_raw, const float* alpha_raw, int T, const float* dt_bias, const float* a_neg,
                  const float* state_in, float* state_out, const float* norm_w, float* out, XQ8 xq, float eps, hipStream_t s);
void gdn_out_sig(const float* raw, const float* z, const float* norm_w, int T, float* out, uint16_t* xb, float eps, hipStream_t s);
void gdn_step_b(const float* qkv, const float* z, const float* beta_raw, const float* alpha_raw, const SeqBatch& sb, size_t st_stride, const float* dt_bias, const float* a_neg,
                const float* st_base, float* st_obase, const float* norm_w, float* out, XQ8 xq, float eps, hipStream_t s);   // 27B (silu gate)
void gdn_step_sig_b(const float* qkv, const float* z, const float* beta_raw, const float* alpha_raw, const SeqBatch& sb, size_t st_stride, const float* dt_bias, const float* a_neg,
                    const float* st_base, float* st_obase, const float* norm_w, float* out, XQ8 xq, float eps, hipStream_t s);
// 27B geometry (24 q / 4 kv heads, V^T cache with kvt_stride rows per slot), per-row positions / KV slots
void attn_decode_b(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, const RowBatch& rb,
                   uint16_t* kc, uint16_t* vc, size_t kv_stride, size_t kvt_stride, int max_ctx, float* out, XQ8 xq, float eps, hipStream_t s);
void attn_decode_24_2_b(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, const RowBatch& rb,
                        uint16_t* kc, uint16_t* vc, size_t kv_stride, size_t kvt_stride, int max_ctx, float* out, XQ8 xq, float eps, hipStream_t s);
void attn_rope_kv_24_2_rowv(float* q_full, float* k, const float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, const RowBatch& rb,
                            uint16_t* kc, uint16_t* vc, size_t kv_stride, int max_ctx, float eps, hipStream_t s);
void attn_rope_kv_24_2_rowv_pos0(float* q_full, float* k, const float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                                 uint16_t* kc, uint16_t* vc, int max_ctx, float eps, hipStream_t s);
void attn_decode_24_2(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                      uint16_t* kc, uint16_t* vc, int max_ctx, float* out, XQ8 xq, float eps, hipStream_t s);

// int8 KV cache: K [kh][pos][256] int8 + 8 f16 scales per row; V^T [kh][dim][ctx+128] int8 + 1 f16 scale per (kh, pos)
struct KV8 { int8_t* k; uint16_t* ks; int8_t* v; uint16_t* vs; };
void attn_decode_i8(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                    const KV8& c, int max_ctx, float* out, XQ8 xq, float eps, hipStream_t s);
void attn_stage1_i8(float* q_full, float* k, const float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                    const KV8& c, int max_ctx, float eps, hipStream_t s);
void attn_prefill_i8(const float* q_full, const KV8& c, int T, int pos0, int max_ctx, uint16_t* out_bf16, hipStream_t s);

// ---- full attention (per layer) ----
// T queries. q_full [T][24][512] (q | gate per head), k [T][4][256], v [T][4][256]. Applies q/k rmsnorm
// (w [256]), NeoX rope on the first 64 dims, writes k/v at pos0+t into the f16 cache (kc/vc:
// [max_ctx][4][256]) and computes causal gated attention: out [T][24][256] (also quantised into xq).
void attn_decode(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w,
                 float rope_base, int pos0, uint16_t* kc, uint16_t* vc, int max_ctx, float* out, XQ8 xq,
                 float eps, hipStream_t s);

// GQA-packed split-KV variant of attn_decode: K/V read once per kv head; partial buffers
// pO [4][nsplit][48][256] f32, pm/pl [4][nsplit][48] with nsplit = ceil((pos0+T)/512).
void attn_decode_gqa(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                     uint16_t* kc, uint16_t* vc, int max_ctx, float* out, XQ8 xq, float* pO, float* pm, float* pl, float eps, hipStream_t s);
// Prefill attention (T >= 16): q_full [T][24][512] (already normed+roped by stage 1 of attn_decode;
// gate in the second half), KV head-major f16 caches holding positions 0..pos0+T-1. Causal WMMA
// flash attention: out bf16 [T][24*256] (GEMM input) = softmax(QK^T/16)V * sigmoid(gate).
void attn_prefill(const float* q_full, const uint16_t* kc, const uint16_t* vc, int T, int pos0, int max_ctx, uint16_t* out_bf16, hipStream_t s);
// stage 1 only (norm + rope + KV write) for T queries
void attn_stage1(float* q_full, float* k, const float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                 uint16_t* kc, uint16_t* vc, int max_ctx, float eps, hipStream_t s);
void attn_stage1_24_2(float* q_full, float* k, const float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                      uint16_t* kc, uint16_t* vc, int max_ctx, float eps, hipStream_t s);
void attn_prefill_24_2(const float* q_full, const uint16_t* kc, const uint16_t* vc, int T, int pos0, int max_ctx, uint16_t* out_bf16, hipStream_t s);
// per row t < T: top-k (k <= 16) ids/values sorted desc -> out_ids[t*k..], out_vals[t*k..]
void argmax_topk(const float* logits, int T, int n, int k, int* out_ids, float* out_vals, hipStream_t s);

}  // namespace hip
