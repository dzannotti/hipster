#include "fn.h"
#include <hip/hip_fp16.h>
#include <cmath>

namespace hip::fn {
namespace {
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ void wave_quant32(float v, int8_t* __restrict__ q, float* __restrict__ d, float* __restrict__ s, int sb, int lane) {
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    const float dd = amax / 127.f, id = amax > 0 ? 127.f / amax : 0.f;
    const int qi = __float2int_rn(v * id);
    int sum = qi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
    q[sb * 32 + lane] = (int8_t)qi;
    if (lane == 0) { d[sb] = dd; s[sb] = dd * sum; }
}
__device__ __forceinline__ uint16_t f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7FFFu + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

constexpr int E = 2560, HC = 4, W = 10240;
constexpr int INJ_PART = 20;   // hc_mix blocks per token (E / 128): inject partials
__device__ __forceinline__ float inject_w(const float* __restrict__ inject, int t, int s) {   // 2*sigmoid(inject_s/4), inject = sum of partials
    float v = 0.f;
    for (int p = 0; p < INJ_PART; ++p) v += inject[((size_t)t * INJ_PART + p) * HC + s];
    return 2.f * sigmoid(v * 0.25f);
}

// ---- Q8_0 embedding: one 256-thread block per token, 10 threads-per-block... block handles all 80 blocks of 32 ----
__global__ void k_embed_q8_0(const uint8_t* __restrict__ tbl, const int* __restrict__ toks, float* __restrict__ out, int n) {
    const uint8_t* row = tbl + (size_t)toks[blockIdx.x] * (n / 32) * 34;
    for (int i = threadIdx.x; i < n; i += 256) {
        const uint8_t* blk = row + (i / 32) * 34;
        out[(size_t)blockIdx.x * n + i] = h2f(*(const uint16_t*)blk) * (float)((const int8_t*)(blk + 2))[i % 32];
    }
}

// ---- hc_norm: block per (t, stream s): 256 threads over 2560. First applies the pending write half of the
// previous block (R[s] += y * 2*sigmoid(inject[s]/4), y either given or = sum_e dw[e]*down[e] (MoE combine)),
// then rmsnorm -> xn (f32) + xq. Stream-0 block also zeroes the buffers the following read chain accumulates
// into (split-K lo, inject atomics) so no memsets are needed.
__global__ void __launch_bounds__(1024) k_hc_norm(HcNormArgs a, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x / HC, s = blockIdx.x % HC, tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    float* r = a.R + ((size_t)t * HC + s) * E;
    const bool comb = a.y != nullptr;
    const float ws = comb ? inject_w(a.inject, t, s) : 0.f;
    // 2560 = 1024 * 2.5: thread handles i = tid and tid + 1024, plus i = tid + 2048 for tid < 512
    float v0 = r[tid], v1 = r[tid + 1024], v2 = tid < 512 ? r[tid + 2048] : 0.f;
    if (comb) { const float* y = a.y + (size_t)t * E; v0 += y[tid] * ws; v1 += y[tid + 1024] * ws; if (tid < 512) v2 += y[tid + 2048] * ws; r[tid] = v0; r[tid + 1024] = v1; if (tid < 512) r[tid + 2048] = v2; }
    float ss = wave_sum(v0 * v0 + v1 * v1 + v2 * v2);
    if (lane == 0) sh[wave] = ss;
    __syncthreads();
    ss = 0.f; for (int i = 0; i < 32; ++i) ss += sh[i];
    const float sc = rsqrtf(ss / E + eps);
    const float* w = a.w_norm + s * E; const size_t ob = ((size_t)t * HC + s) * E;
    auto emit = [&](int i, float v) { const float o = v * sc * w[i]; a.xn[ob + i] = o; if (a.xb) a.xb[ob + i] = f2bf(o); if (a.q) wave_quant32(o, a.q, a.qd, a.qs, (int)((ob + i) / 32), lane); };
    emit(tid, v0); emit(tid + 1024, v1); if (tid < 512) emit(tid + 2048, v2);
}
__global__ void k_hc_silu_quant(const float* __restrict__ part, int ksplit, int8_t* __restrict__ q, float* __restrict__ qd, float* __restrict__ qs, int T) {
    const int sb = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;   // 320 = 10 sub-blocks per token
    if (sb >= T * 10) return;
    const int t = sb / 10, i = (sb % 10) * 32 + lane;
    float v = 0.f;
    for (int ks = 0; ks < ksplit; ++ks) v += part[((size_t)t * ksplit + ks) * 320 + i];
    wave_quant32(silu(v * 0.25f), q, qd, qs, sb, lane);
}
// prefill: lo bf16 [T][320] (GEMM output) -> silu(lo/4) -> bf16 [T][320]
__global__ void k_hc_silu_bf16(const uint16_t* __restrict__ lo, uint16_t* __restrict__ out, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x; if (i < n) out[i] = f2bf(silu(bf2f(lo[i]) * 0.25f));
}
// hc_mix: block per (t, 128 dims): 4 waves; 20 blocks per token. inject written as per-block partials
// [T][INJ_PART][4] (summed by the consumer) instead of contended atomics (4 addresses: 13 us).
template <typename UP>
__global__ void __launch_bounds__(128) k_hc_mix(const float* __restrict__ xn, const UP* __restrict__ up, const float* __restrict__ w_inj,
                                                float* __restrict__ mixed, float* __restrict__ inject, int8_t* __restrict__ q, float* __restrict__ qd, float* __restrict__ qs, uint16_t* __restrict__ xb) {
    __shared__ float sh[4][HC];
    const int t = blockIdx.x / INJ_PART, b = blockIdx.x % INJ_PART;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int d = b * 128 + wave * 32 + lane;   // dim
    float acc = 0.f, inj[HC] = {0.f, 0.f, 0.f, 0.f};
    float x[HC], g[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) { const size_t i = ((size_t)t * HC + s) * E + d; x[s] = xn[i]; if constexpr (sizeof(UP) == 2) g[s] = bf2f(up[i]); else g[s] = up[i]; }
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        acc += x[s] * sigmoid(g[s]);
        if (w_inj) {
#pragma unroll
            for (int n = 0; n < HC; ++n) inj[n] += x[s] * w_inj[(size_t)n * W + s * E + d];
        }
    }
    const float m = acc * 0.25f;
    mixed[(size_t)t * E + d] = m;
    if (xb) xb[(size_t)t * E + d] = f2bf(m);
    if (q) wave_quant32(m, q, qd, qs, t * (E / 32) + b * 4 + wave, lane);
    if (inject) {
#pragma unroll
        for (int n = 0; n < HC; ++n) { const float v = wave_sum(inj[n]); if (lane == 0) sh[wave][n] = v; }
        __syncthreads();
        if (threadIdx.x < HC) inject[((size_t)t * INJ_PART + b) * HC + threadIdx.x] = sh[0][threadIdx.x] + sh[1][threadIdx.x] + sh[2][threadIdx.x] + sh[3][threadIdx.x];
    }
}

// R[t][s] += y[t] * 2*sigmoid(inject[t][s]/4); block per (t,s)
__global__ void k_hc_combine(float* __restrict__ R, const float* __restrict__ y, const float* __restrict__ inject) {
    const int t = blockIdx.x / HC, s = blockIdx.x % HC;
    const float ws = inject_w(inject, t, s);
    float* r = R + ((size_t)t * HC + s) * E; const float* yy = y + (size_t)t * E;
    for (int i = threadIdx.x; i < E; i += 256) r[i] += yy[i] * ws;
}
__global__ void k_init_wide(const float* __restrict__ x, float* __restrict__ R) {
    const int t = blockIdx.x / HC, s = blockIdx.x % HC;
    for (int i = threadIdx.x; i < E; i += 256) R[((size_t)t * HC + s) * E + i] = x[(size_t)t * E + i];
}

// prefill MoE combine: y[t][d] = sum_e w[t*nexp+e] * down[kpos[t*nexp+e]][d] (sorted-position rows, slot order: deterministic)
__global__ void k_moe_combine_sorted(const float* __restrict__ down, const float* __restrict__ w, const int* __restrict__ kpos, float* __restrict__ y, int nexp) {
    const int t = blockIdx.x / (E / 256), d = (blockIdx.x % (E / 256)) * 256 + threadIdx.x;
    float acc = 0.f;
    for (int e = 0; e < nexp; ++e) { const int te = t * nexp + e; acc += w[te] * down[(size_t)kpos[te] * E + d]; }
    y[(size_t)t * E + d] = acc;
}
// ---- MTP input: block per token. e_norm = rmsnorm(e[2560])*enorm; h_norm = rmsnorm(h[10240], whole row)*hnorm;
// per stream s: xq row t*4+s = [e_norm | h_norm[s*2560..]] (5120), quantised (input of eh_proj) ----
__global__ void __launch_bounds__(256) k_mtp_prep(const float* __restrict__ e, const float* __restrict__ enorm, const float* __restrict__ h,
                                                  const float* __restrict__ hnorm, int8_t* __restrict__ q, float* __restrict__ qd, float* __restrict__ qs, float eps) {
    __shared__ float sh[2][8];
    const int t = blockIdx.x, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* et = e + (size_t)t * E; const float* ht = h + (size_t)t * W;
    float se = 0.f, shh = 0.f;
    for (int i = threadIdx.x; i < E; i += 256) se += et[i] * et[i];
    for (int i = threadIdx.x; i < W; i += 256) shh += ht[i] * ht[i];
    se = wave_sum(se); shh = wave_sum(shh);
    if (lane == 0) { sh[0][wave] = se; sh[1][wave] = shh; }
    __syncthreads();
    se = 0.f; shh = 0.f; for (int i = 0; i < 8; ++i) { se += sh[0][i]; shh += sh[1][i]; }
    const float sce = rsqrtf(se / E + eps), sch = rsqrtf(shh / W + eps);
    for (int s = 0; s < HC; ++s) {
        const size_t row = (size_t)t * HC + s;
        for (int sb = wave; sb < 2 * E / 32; sb += 8) {   // 160 sub-blocks of 32 per row
            const int i = sb * 32 + lane;
            const float v = i < E ? et[i] * sce * enorm[i] : ht[s * E + (i - E)] * sch * hnorm[s * E + (i - E)];
            wave_quant32(v, q, qd, qs, (int)(row * (2 * E / 32) + sb), lane);
        }
    }
}
// ---- MoE routing: block per token, 512 logits, 16 waves. softmax, then top-10 by repeated argmax (ties -> smaller
// index): each wave takes its 32 values' top-10 with shuffles, wave 0 merges the 160 candidates. No block-wide
// argmax rounds (the 10 x 9 __syncthreads version took 11.5 us).
__device__ __forceinline__ void wave_argmax(float& v, int& i) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(v, o, 32); const int oi = __shfl_xor(i, o, 32);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
}
__global__ void __launch_bounds__(512) k_moe_route(const float* __restrict__ logits, int* __restrict__ ids, float* __restrict__ w, int nexp) {
    __shared__ float sh[16]; __shared__ float cv[16 * 10]; __shared__ int ci[16 * 10];
    const int t = blockIdx.x, i = threadIdx.x, wave = i / 32, lane = i % 32;
    float v = logits[t * 513 + i];
    float mx = v; for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    if (lane == 0) sh[wave] = mx; __syncthreads();
    mx = sh[0]; for (int k = 1; k < 16; ++k) mx = fmaxf(mx, sh[k]);
    __syncthreads();
    const float e = __expf(v - mx);
    float sum = wave_sum(e); if (lane == 0) sh[wave] = sum; __syncthreads();
    sum = 0.f; for (int k = 0; k < 16; ++k) sum += sh[k];
    float p = e / sum; int idx = i;
    // per-wave top-10
    for (int r = 0; r < nexp; ++r) {
        float bv = p; int bi = idx; wave_argmax(bv, bi);
        if (lane == 0) { cv[wave * 10 + r] = bv; ci[wave * 10 + r] = bi; }
        if (idx == bi) p = -1.f;
    }
    __syncthreads();
    if (wave == 0) {
        // merge 160 candidates: lane holds 5
        float mv[5]; int mi[5];
#pragma unroll
        for (int k = 0; k < 5; ++k) { mv[k] = cv[lane * 5 + k]; mi[k] = ci[lane * 5 + k]; }
        float wsum = 0.f;
        for (int r = 0; r < nexp; ++r) {
            float bv = mv[0]; int bi = mi[0];
#pragma unroll
            for (int k = 1; k < 5; ++k) if (mv[k] > bv || (mv[k] == bv && mi[k] < bi)) { bv = mv[k]; bi = mi[k]; }
            wave_argmax(bv, bi);
            if (lane == 0) { ids[t * (nexp + 1) + r] = bi; w[t * (nexp + 1) + r] = bv; }
            wsum += bv;
#pragma unroll
            for (int k = 0; k < 5; ++k) if (mi[k] == bi) mv[k] = -1.f;
        }
        if (lane < nexp) w[t * (nexp + 1) + lane] /= wsum;
        if (lane == 0) { ids[t * (nexp + 1) + nexp] = -1; w[t * (nexp + 1) + nexp] = sigmoid(logits[t * 513 + 512]); }   // shared expert slot
    }
}
__global__ void k_moe_silu_quant(const float* __restrict__ g, const float* __restrict__ u, int8_t* __restrict__ q, float* __restrict__ qd, float* __restrict__ qs, int total) {
    const int sb = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (sb * 32 >= total) return;
    const int i = sb * 32 + lane;
    wave_quant32(silu(g[i]) * u[i], q, qd, qs, sb, lane);
}

// ---- PLE: block per (t, stream s), 256 threads; conv history in conv_state [9][10240] (oldest first, per stream part) ----
__global__ void __launch_bounds__(256) k_ple_apply(float* __restrict__ R, const float* __restrict__ key, const float* __restrict__ value,
                                                   const float* __restrict__ w_key, const float* __restrict__ w_query, const float* __restrict__ w_cn,
                                                   const float* __restrict__ conv_w, float* __restrict__ st, int T, float eps) {
    __shared__ float sh[8]; __shared__ float sgate;
    const int t = blockIdx.x / HC, s = blockIdx.x % HC, wave = threadIdx.x / 32;
    const float* k = key + (size_t)t * W + s * E; const float* r = R + ((size_t)t * HC + s) * E;
    // norms of key stream and residual stream
    float sk = 0.f, sr = 0.f, dot = 0.f;
    for (int i = threadIdx.x; i < E; i += 256) { sk += k[i] * k[i]; sr += r[i] * r[i]; }
    sk = wave_sum(sk); sr = wave_sum(sr);
    if (threadIdx.x % 32 == 0) { sh[wave] = sk; } __syncthreads(); sk = 0.f; for (int i = 0; i < 8; ++i) sk += sh[i]; __syncthreads();
    if (threadIdx.x % 32 == 0) { sh[wave] = sr; } __syncthreads(); sr = 0.f; for (int i = 0; i < 8; ++i) sr += sh[i]; __syncthreads();
    const float ck = rsqrtf(sk / E + eps), cr = rsqrtf(sr / E + eps);
    for (int i = threadIdx.x; i < E; i += 256) dot += (k[i] * ck * w_key[s * E + i]) * (r[i] * cr * w_query[s * E + i]);
    dot = wave_sum(dot); if (threadIdx.x % 32 == 0) sh[wave] = dot; __syncthreads();
    if (threadIdx.x == 0) { float d = 0.f; for (int i = 0; i < 8; ++i) d += sh[i]; d *= rsqrtf((float)E); const float mag = sqrtf(fmaxf(fabsf(d), 1e-6f)); sgate = sigmoid((d < 0.f ? -mag : mag)); }
    __syncthreads();
    const float gate = sgate;
    // gated value, its grouped norm, then the conv over history. conv state layout: [9][W] with the newest last.
    // Sequential over T inside one block would be needed for the history; decode uses T == 1 (prefill: T tokens processed
    // in order by looping here since blocks are per token -> we restrict this kernel to T == 1 and loop on the host).
    float gsum = 0.f;
    for (int i = threadIdx.x; i < E; i += 256) { const float gv = value[(size_t)t * E + i] * gate; gsum += gv * gv; }
    gsum = wave_sum(gsum); if (threadIdx.x % 32 == 0) sh[wave] = gsum; __syncthreads(); gsum = 0.f; for (int i = 0; i < 8; ++i) gsum += sh[i];
    const float cg = rsqrtf(gsum / E + eps);
    for (int i = threadIdx.x; i < E; i += 256) {
        const int c = s * E + i;
        const float gv = value[(size_t)t * E + i] * gate;
        const float n = gv * cg * w_cn[c];
        // taps k=0..3 read (3-k)*3 positions back: history slots 0..8 hold x[t-9..t-1]; tap 3 is the current n
        const float acc = conv_w[c * 4 + 0] * st[(size_t)0 * W + c] + conv_w[c * 4 + 1] * st[(size_t)3 * W + c] + conv_w[c * 4 + 2] * st[(size_t)6 * W + c] + conv_w[c * 4 + 3] * n;
        R[((size_t)t * HC + s) * E + i] = r[i] + gv + silu(acc);
        // shift history: st[j] = st[j+1]; st[8] = n  (done by the same thread for its channel)
        for (int j = 0; j < 8; ++j) st[(size_t)j * W + c] = st[(size_t)(j + 1) * W + c];
        st[(size_t)8 * W + c] = n;
    }
}
// f32 weights [N][K], f32 x [T][K] -> y [T][N]; one wave per weight row, NT x rows share the read
template <int NT>
__global__ void k_f32_gemv(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int n = (blockIdx.x * 256 + threadIdx.x) / 32, lane = threadIdx.x % 32; if (n >= N) return;
    const float* w = W + (size_t)n * K;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int i = lane * 4; i < K; i += 128) {
        const float4 a = *(const float4*)(w + i);
#pragma unroll
        for (int t = 0; t < NT; ++t) { const float4 b = *(const float4*)(x + (size_t)t * K + i); acc[t] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) { const float v = wave_sum(acc[t]); if (lane == 0) y[(size_t)t * N + n] = v; }
}
}  // namespace

void f32_gemv(const float* W, const float* x, float* y, int N, int K, int T, hipStream_t s) {
    const int grid = (N * 32 + 255) / 256;
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int nt = std::min(8, T - t0); const float* xr = x + (size_t)t0 * K; float* yr = y + (size_t)t0 * N;
        #define KF(NT) k_f32_gemv<NT><<<grid, 256, 0, s>>>(W, xr, yr, N, K)
        switch (nt) { case 1: KF(1); break; case 2: KF(2); break; case 3: KF(3); break; case 4: KF(4); break; case 5: KF(5); break; case 6: KF(6); break; case 7: KF(7); break; default: KF(8); break; }
        #undef KF
    }
}
void embed_q8_0(const void* W, const int* toks, int T, float* out, int n, hipStream_t s) { k_embed_q8_0<<<T, 256, 0, s>>>((const uint8_t*)W, toks, out, n); }
void hc_combine(float* R, const float* y, const float* inject, int T, hipStream_t s) { k_hc_combine<<<T * HC, 256, 0, s>>>(R, y, inject); }
void init_wide(const float* x, float* R, int T, hipStream_t s) { k_init_wide<<<T * HC, 256, 0, s>>>(x, R); }
void hc_norm(const HcNormArgs& a0, XQ8 xq, int T, float eps, hipStream_t s) {
    HcNormArgs a = a0; a.q = xq.q; a.qd = xq.d; a.qs = xq.s;   // xq.q may be null (prefill: bf16 only)
    k_hc_norm<<<T * HC, 1024, 0, s>>>(a, eps);
}
void hc_silu_quant(const float* part, int ksplit, XQ8 xq, int T, hipStream_t s) { k_hc_silu_quant<<<(T * 10 + 7) / 8, 256, 0, s>>>(part, ksplit, xq.q, xq.d, xq.s, T); }
void hc_mix(const float* xn, const float* up, const float* w_inject, float* mixed, float* inject, XQ8 xq, int T, hipStream_t s) {
    k_hc_mix<float><<<T * INJ_PART, 128, 0, s>>>(xn, up, w_inject, mixed, inject, xq.q, xq.d, xq.s, nullptr);
}

void hc_mix_bf16(const float* xn, const uint16_t* up, const float* w_inject, float* mixed, float* inject, XQ8 xq, uint16_t* xb, int T, hipStream_t s) {
    k_hc_mix<uint16_t><<<T * INJ_PART, 128, 0, s>>>(xn, up, w_inject, mixed, inject, xq.q, xq.d, xq.s, xb);
}
void hc_silu_bf16(const uint16_t* lo, uint16_t* out, int T, hipStream_t s) { k_hc_silu_bf16<<<(T * 320 + 255) / 256, 256, 0, s>>>(lo, out, T * 320);
}
void moe_combine_sorted(const float* down, const float* w, const int* kpos, float* y, int T, int nexp, hipStream_t s) {
    k_moe_combine_sorted<<<T * (E / 256), 256, 0, s>>>(down, w, kpos, y, nexp);
}
void mtp_prep(const float* e, const float* enorm, const float* h, const float* hnorm, XQ8 xq, int T, float eps, hipStream_t s) {
    k_mtp_prep<<<T, 256, 0, s>>>(e, enorm, h, hnorm, xq.q, xq.d, xq.s, eps);
}
void moe_route(const float* logits, int* ids, float* w, int T, hipStream_t s) { k_moe_route<<<T, 512, 0, s>>>(logits, ids, w, 10); }
void moe_gemv(WFmt fmt, WFmt shared_fmt, const MoeSegs& a, int nseg, const int* ids, int nexp, XQ8 xq, int N, int K, int T, hipStream_t s) {
    hip::gemv_moe(fmt, shared_fmt, a, nseg, ids, nexp, xq, N, K, T, K == 640, s);   // down-proj (K = 640) reads per-(t,e) rows
}
void moe_silu_quant(const float* gate, const float* up, XQ8 xq, int rows, int n, hipStream_t s) { k_moe_silu_quant<<<(rows * n / 32 + 7) / 8, 256, 0, s>>>(gate, up, xq.q, xq.d, xq.s, rows * n); }
void ple_apply(float* R, const float* key, const float* value, const float* w_key, const float* w_query, const float* w_conv_norm,
               const float* conv_w, float* conv_state, int T, float eps, hipStream_t s) {
    for (int t = 0; t < T; ++t)   // sequential: the conv history is per token
        k_ple_apply<<<HC, 256, 0, s>>>(R + (size_t)t * HC * E, key + (size_t)t * W, value + (size_t)t * E, w_key, w_query, w_conv_norm, conv_w, conv_state, 1, eps);
}
}  // namespace hip::fn
