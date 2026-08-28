// Qwen3.8-Flash-Next-only kernels (hyper-connections, PLE n-gram gate/conv, MoE routing and
// expert GEMV, Q8_0 embedding). Nothing here is shared with the 27B path.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "gemv.h"

namespace hip::fn {

// out[t] = dequant(Q8_0 row toks[t]) of a [n_vocab x n] Q8_0 (GGUF block layout) table
void embed_q8_0(const void* W, const int* toks, int T, float* out, int n, hipStream_t s);

// R[t][s] += y[t] * 2*sigmoid(inject[t][s]/4)  (the write half of the gated residual; inject as [T][20][4] partials)
void hc_combine(float* R, const float* y, const float* inject, int T, hipStream_t s);
// R[t][s] = x[t] for the 4 streams
void init_wide(const float* x, float* R, int T, hipStream_t s);
// Hyper-connection read, part 1, per stream s of R [T][4][2560]: first the pending write half of the previous
// block, R[s] += y * w_s (w_s = 2*sigmoid(inject_s/4), inject given as the [T][20][4] partials hc_mix writes),
// then xn[s] = rmsnorm(R[s]) * w_norm[s*2560..] written f32 [T][10240] and quantised into xq (input of the
// 10240->320 GEMV).
struct HcNormArgs {
    float* R; const float* y = nullptr; const float* inject = nullptr;
    const float* w_norm; float* xn; uint16_t* xb = nullptr;   // xb: bf16 copy of xn (prefill GEMM input)
    int8_t* q = nullptr; float* qd = nullptr; float* qs = nullptr;
};
void hc_norm(const HcNormArgs& a, XQ8 xq, int T, float eps, hipStream_t s);
// part 2: lo = sum of the ksplit split-K partials [T][ksplit][320] (fixed order), silu(lo/4) -> quantised into xq
void hc_silu_quant(const float* part, int ksplit, XQ8 xq, int T, hipStream_t s);
// part 3: gate = sigmoid(up) [T][10240]; mixed[t] = mean_s(xn[s]*gate[s]) [2560] -> f32 out + xq;
// inject = W_inject^T xn (W_inject f32 [10240][4], i.e. GGUF [4 x 10240] ne0=4) as per-block partials [T][20][4]
void hc_mix(const float* xn, const float* up, const float* w_inject, float* mixed, float* inject, XQ8 xq, int T, hipStream_t s);
// prefill variants: up bf16 (GEMM output); mixed also as bf16 rows xb; lo bf16 -> silu -> bf16
void hc_mix_bf16(const float* xn, const uint16_t* up, const float* w_inject, float* mixed, float* inject, XQ8 xq, uint16_t* xb, int T, hipStream_t s);
void hc_silu_bf16(const uint16_t* lo, uint16_t* out, int T, hipStream_t s);

// exact f32 GEMV (router 512x2560, shared-expert gate 1x2560): y [T][N] = W [N][K] . x [T][K]
void f32_gemv(const float* W, const float* x, float* y, int N, int K, int T, hipStream_t s);
// prefill MoE combine over sorted down rows: y[t] = sum_e w[t][e] * down[kpos[t][e]]
void moe_combine_sorted(const float* down, const float* w, const int* kpos, float* y, int T, int nexp, hipStream_t s);
// MTP input rows: xq [T*4][5120] = [rmsnorm(e)*enorm | rmsnorm(h, over the whole 10240 row)*hnorm, stream s] (eh_proj input)
void mtp_prep(const float* e, const float* enorm, const float* h, const float* hnorm, XQ8 xq, int T, float eps, hipStream_t s);
// MoE routing: logits [T][512] f32 -> softmax -> top-10 ids [T][10] and renormalised weights [T][10]
void moe_route(const float* logits, int* ids, float* w, int T, hipStream_t s);
// Expert GEMV for decode (T <= 8): for slot e < 10: y[t][e][N] = W[ids[t][e]] . x[t]  (x quantised in xq,
// weights in GGUF block layout per expert of `bytes` each, format fmt). ncol == T handled as T
// separate rows (routing differs per token).
void moe_gemv(WFmt fmt, WFmt shared_fmt, const MoeSegs& a, int nseg, const int* ids, int nexp, XQ8 xq, int N, int K, int T, hipStream_t s);
// h[t][e][640] = silu(g)*u from the fused gate|up output y [T][10][2*640]? (we run gate and up as two
// segments) -> quantised rows into xq (row index t*10+e)
void moe_silu_quant(const float* gate, const float* up, XQ8 xq, int rows, int n, hipStream_t s);
void moe_silu_bf16(const float* gate, const float* up, uint16_t* out, int rows, int n, hipStream_t s);   // bf16 rows [rows][n]

// PLE gate + conv: emb rows gathered on the host. key [T][10240] (W_key . emb), value [T][2560],
// R [T][4][2560] (the wide residual, updated in place), conv state [9][10240] per sequence.
void ple_apply(float* R, const float* key, const float* value, const float* w_key, const float* w_query, const float* w_conv_norm,
               const float* conv_w, float* conv_state, int T, float eps, hipStream_t s);

}  // namespace hip::fn
