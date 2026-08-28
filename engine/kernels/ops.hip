#include "ops.h"
#include <hip/hip_fp16.h>
#include <cmath>

namespace hip {
namespace {

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float block_sum(float v, float* sh) {   // 256 threads
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = (threadIdx.x < 8) ? sh[threadIdx.x] : 0.f;
    if (w == 0) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
        if (l == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float block_max(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = (threadIdx.x < 8) ? sh[threadIdx.x] : -INFINITY;
    if (w == 0) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
        if (l == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// one wave: quantise 32 values v (lane-owned) into xq sub-block sb
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

// ---- fused residual add + rmsnorm + q8 quantise: one 1024-thread block per row, n multiple of 32 ----
__device__ __forceinline__ float block_sum1024(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = (threadIdx.x < 32) ? sh[threadIdx.x] : 0.f;
    if (w == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
        if (l == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}
__global__ void __launch_bounds__(1024) k_add_rmsnorm_quant(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ w, float* __restrict__ xn,
                                    int8_t* __restrict__ q, float* __restrict__ qd, float* __restrict__ qs, int n, float eps) {
    __shared__ float sh[32];
    const int row = blockIdx.x;
    x += (size_t)row * n; if (y) y += (size_t)row * n; xn += (size_t)row * n; q += (size_t)row * n; qd += (size_t)row * (n / 32); qs += (size_t)row * (n / 32);
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    // each wave owns sub-blocks wave, wave+32, ... (n/32 sub-blocks); values kept in registers
    constexpr int MAXSB = 3;   // n <= 32*32*3 = 3072 per wave-slot... n = 5120 -> 160 sub-blocks / 32 waves = 5
    float v[6]; float ss = 0.f; int cnt = 0;
    for (int sb = wave; sb < n / 32 && cnt < 6; sb += 32, ++cnt) { float t = x[sb * 32 + lane]; if (y) { t += y[sb * 32 + lane]; x[sb * 32 + lane] = t; } v[cnt] = t; ss += t * t; }
    ss = block_sum1024(ss, sh);
    const float sc = rsqrtf(ss / n + eps);
    cnt = 0;
    for (int sb = wave; sb < n / 32 && cnt < 6; sb += 32, ++cnt) {
        const float t = v[cnt] * sc * w[sb * 32 + lane];
        xn[sb * 32 + lane] = t;
        wave_quant32(t, q, qd, qs, sb, lane);
    }
    (void)MAXSB;
}
__global__ void __launch_bounds__(1024) k_norm2_concat_quant(const float* __restrict__ a, const float* __restrict__ wa, int na, const float* __restrict__ b, const float* __restrict__ wb, int nb,
                                                              float* __restrict__ out, int8_t* __restrict__ q, float* __restrict__ qd, float* __restrict__ qs, float eps) {
    __shared__ float sh[32];
    const int row = blockIdx.x; const int n = na + nb;
    a += (size_t)row * na; b += (size_t)row * nb; out += (size_t)row * n; q += (size_t)row * n; qd += (size_t)row * (n / 32); qs += (size_t)row * (n / 32);
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    float sa = 0.f, sb = 0.f;
    for (int i = threadIdx.x; i < na; i += 1024) sa += a[i] * a[i];
    for (int i = threadIdx.x; i < nb; i += 1024) sb += b[i] * b[i];
    sa = block_sum1024(sa, sh); sb = block_sum1024(sb, sh);
    const float ca = rsqrtf(sa / na + eps), cb = rsqrtf(sb / nb + eps);
    for (int sbk = wave; sbk < n / 32; sbk += 32) {
        const int i = sbk * 32 + lane;
        const float v = i < na ? a[i] * ca * wa[i] : b[i - na] * cb * wb[i - na];
        out[i] = v;
        wave_quant32(v, q, qd, qs, sbk, lane);
    }
}
__device__ __forceinline__ uint16_t f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7FFFu + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
__global__ void __launch_bounds__(1024) k_add_rmsnorm_bf16(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ w, uint16_t* __restrict__ xb, int n, float eps) {
    __shared__ float sh[32];
    const int row = blockIdx.x;
    x += (size_t)row * n; if (y) y += (size_t)row * n; xb += (size_t)row * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 1024) { float t = x[i]; if (y) { t += y[i]; x[i] = t; } ss += t * t; }
    ss = block_sum1024(ss, sh);
    const float sc = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += 1024) xb[i] = f2bf(x[i] * sc * w[i]);
}
__global__ void __launch_bounds__(1024) k_norm2_concat_bf16(const float* __restrict__ a, const float* __restrict__ wa, int na, const float* __restrict__ b, const float* __restrict__ wb, int nb, uint16_t* __restrict__ out, float eps) {
    __shared__ float sh[32];
    const int row = blockIdx.x; const int n = na + nb;
    a += (size_t)row * na; b += (size_t)row * nb; out += (size_t)row * n;
    float sa = 0.f, sb = 0.f;
    for (int i = threadIdx.x; i < na; i += 1024) sa += a[i] * a[i];
    for (int i = threadIdx.x; i < nb; i += 1024) sb += b[i] * b[i];
    sa = block_sum1024(sa, sh); sb = block_sum1024(sb, sh);
    const float ca = rsqrtf(sa / na + eps), cb = rsqrtf(sb / nb + eps);
    for (int i = threadIdx.x; i < n; i += 1024) out[i] = f2bf(i < na ? a[i] * ca * wa[i] : b[i - na] * cb * wb[i - na]);
}
__global__ void k_silu_mul_bf16(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ o, size_t n) { const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; if (i < n) o[i] = f2bf(silu(g[i]) * u[i]); }
__global__ void k_silu_mul_bf16_seg(const uint16_t* __restrict__ gu, int Nt, int off, int N, uint16_t* __restrict__ o, int T) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; if (i >= (size_t)T * N) return;
    const int t = (int)(i / N), c = (int)(i % N);
    const float g = __uint_as_float((uint32_t)gu[(size_t)t * Nt + off + c] << 16), u = __uint_as_float((uint32_t)gu[(size_t)t * Nt + off + N + c] << 16);
    o[i] = f2bf(silu(g) * u);
}
__global__ void k_silu_mul_quant(const float* __restrict__ g, const float* __restrict__ u, int8_t* __restrict__ q, float* __restrict__ qd, float* __restrict__ qs, int n) {
    const int sb = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (sb * 32 >= n) return;
    const int i = sb * 32 + lane;
    wave_quant32(silu(g[i]) * u[i], q, qd, qs, sb, lane);
}

// ---- embedding: one Q4_K row, 256 threads, thread t handles element t of each 256-block ----
__global__ void k_embed_q4_K(const uint8_t* __restrict__ W, const int* __restrict__ toks, float* __restrict__ out, int n) {
    const int nb = n / 256;
    const uint8_t* row = W + (size_t)toks[blockIdx.x] * nb * 144;
    out += (size_t)blockIdx.x * n;
    const int t = threadIdx.x, j = t / 64, l = t % 64;   // sub-block pair j, element l within the 64
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t)b * 144;
        const float d = h2f(*(const uint16_t*)blk), dmin = h2f(*(const uint16_t*)(blk + 2));
        const uint8_t* sc = blk + 4; const uint8_t* qs = blk + 16 + 32 * j;
        const int is = 2 * j + (l >= 32);
        uint8_t s, m;
        if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; }
        else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        const uint8_t q = qs[l % 32];
        const float v = (l < 32) ? (q & 0xF) : (q >> 4);
        out[b * 256 + t] = d * s * v - dmin * m;
    }
}

// ---- rmsnorm: one 256-thread block per row ----
__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int n, float eps) {
    __shared__ float sh[8];
    const float* xr = x + (size_t)blockIdx.x * n; float* o = out + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float sc = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += 256) o[i] = xr[i] * sc * w[i];
}

__global__ void k_add(float* __restrict__ y, const float* __restrict__ x, int n) { const int i = blockIdx.x * 256 + threadIdx.x; if (i < n) y[i] += x[i]; }
__global__ void k_silu_mul(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ o, int n) { const int i = blockIdx.x * 256 + threadIdx.x; if (i < n) o[i] = silu(g[i]) * u[i]; }

// ---- GDN conv: state [3][C] oldest first; weights w [4][C] (tap-major as in GGUF ssm_conv1d [4, C]) ----
__global__ void k_gdn_conv(const float* __restrict__ x, SeqBatch sb, size_t st_stride, const float* __restrict__ st_base, float* __restrict__ st_obase,
                           const float* __restrict__ w, float* __restrict__ out, int C) {
    const int c = blockIdx.x * 256 + threadIdx.x; if (c >= C) return;
    const SeqDesc sd = sb.s[blockIdx.y];
    const float* st_in = st_base + (size_t)sd.in * st_stride; float* st_out = st_obase + (size_t)sd.out * st_stride;
    float h0 = st_in[c], h1 = st_in[C + c], h2 = st_in[2 * C + c];
    // ggml: weight tensor ne = [4, C] -> element (tap, c) at w[c*4 + tap]; taps oldest..newest
    const float* wc = w + (size_t)c * 4;
    for (int t = sd.r0; t < sd.r0 + sd.T; ++t) {
        const float xc = x[(size_t)t * C + c];
        out[(size_t)t * C + c] = silu(wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * xc);
        h0 = h1; h1 = h2; h2 = xc;
    }
    st_out[c] = h0; st_out[C + c] = h1; st_out[2 * C + c] = h2;
}

// ---- GDN recurrent step, T tokens: one block (256 threads) per v-head; thread handles j = tid/2, i-half = tid%2 ----
template <bool SIGMOID_GATE>
__global__ void __launch_bounds__(256) k_gdn_step(const float* __restrict__ qkv, const float* __restrict__ z,
                                                  const float* __restrict__ beta_raw, const float* __restrict__ alpha_raw, SeqBatch sb, size_t st_stride,
                                                  const float* __restrict__ dt_bias, const float* __restrict__ a_neg,
                                                  const float* __restrict__ st_base, float* __restrict__ st_obase, const float* __restrict__ norm_w,
                                                  float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od, float* __restrict__ os, float eps) {
    constexpr int D = 128, QKV = 10240, Z = 6144;
    const int h = blockIdx.x, hk = h % 16;
    const SeqDesc sd = sb.s[blockIdx.y];
    const float* state_in = st_base + (size_t)sd.in * st_stride; float* state_out = st_obase + (size_t)sd.out * st_stride;
    const int tid = threadIdx.x, j = tid / 2, ih = tid % 2;   // this thread: row j of M, i in [64*ih, 64*ih+64)
    __shared__ float sq[D], sk[D], sh[8], sout[D];
    const float* Min = state_in + (size_t)h * D * D + (size_t)j * D + 64 * ih;
    float m[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) m[i] = Min[i];
    for (int t = sd.r0; t < sd.r0 + sd.T; ++t) {
        const float* qh = qkv + (size_t)t * QKV + hk * D; const float* kh = qkv + (size_t)t * QKV + 2048 + hk * D; const float* vh = qkv + (size_t)t * QKV + 4096 + h * D;
        float qq = 0.f, kk = 0.f;
        if (tid < D) { qq = qh[tid] * qh[tid]; kk = kh[tid] * kh[tid]; }
        const float qn = block_sum(qq, sh); const float kn = block_sum(kk, sh);
        if (tid < D) { sq[tid] = qh[tid] / fmaxf(sqrtf(qn), eps); sk[tid] = kh[tid] / fmaxf(sqrtf(kn), eps); }
        __syncthreads();
        const float beta = sigmoid(beta_raw[t * 48 + h]);
        const float decay = __expf(a_neg[h] * softplus(alpha_raw[t * 48 + h] + dt_bias[h]));
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 64; ++i) { m[i] *= decay; dot += m[i] * sk[64 * ih + i]; }
        dot += __shfl_xor(dot, 1, 32);
        const float delta = (vh[j] - dot) * beta;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < 64; ++i) { m[i] += delta * sk[64 * ih + i]; o += m[i] * sq[64 * ih + i]; }
        o += __shfl_xor(o, 1, 32);
        if (ih == 0) sout[j] = o * 0.08838834764831845f;   // 1/sqrt(128)
        __syncthreads();
        if (out) {
            float ss = (tid < D) ? sout[tid] * sout[tid] : 0.f;
            ss = block_sum(ss, sh);
            if (tid < D) {
                const size_t oi = (size_t)t * Z + h * D + tid;
                const float v = sout[tid] * rsqrtf(ss / D + eps) * norm_w[tid] * (SIGMOID_GATE ? sigmoid(z[oi]) : silu(z[oi]));
                out[oi] = v;
                wave_quant32(v, oq, od, os, (int)(oi / 32), tid % 32);
            }
        }
        __syncthreads();
    }
    float* Mout = state_out + (size_t)h * D * D + (size_t)j * D + 64 * ih;
#pragma unroll
    for (int i = 0; i < 64; ++i) Mout[i] = m[i];
}

// ---- GDN prefill: prep (wide, parallel) -> tiled scan (16 tokens per LDS tile, 1 sync per tile) -> out norm ----
// prep: block per (t, 16 k-heads + 48 v-heads gates): 256 threads; q/k l2 norm per (t, kh), beta/decay per (t, h)
__global__ void __launch_bounds__(256) k_gdn_prep(const float* __restrict__ qkv, const float* __restrict__ beta_raw, const float* __restrict__ alpha_raw,
                                                  const float* __restrict__ dt_bias, const float* __restrict__ a_neg,
                                                  float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ beta, float* __restrict__ decay, float eps) {
    constexpr int D = 128, QKV = 10240;
    const int t = blockIdx.x, w = threadIdx.x / 32, lane = threadIdx.x % 32;   // 8 waves: waves 0..7 handle k-heads 2w, 2w+1 for q and k
    for (int kh = w; kh < 16; kh += 8) {
        const float* q = qkv + (size_t)t * QKV + kh * D; const float* kk = qkv + (size_t)t * QKV + 2048 + kh * D;
        float sq = 0.f, sk = 0.f;
        for (int i = lane; i < D; i += 32) { sq += q[i] * q[i]; sk += kk[i] * kk[i]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { sq += __shfl_xor(sq, o, 32); sk += __shfl_xor(sk, o, 32); }
        const float cq = 1.f / fmaxf(sqrtf(sq), eps), ck = 1.f / fmaxf(sqrtf(sk), eps);
        for (int i = lane; i < D; i += 32) { qn[((size_t)t * 16 + kh) * D + i] = q[i] * cq; kn[((size_t)t * 16 + kh) * D + i] = kk[i] * ck; }
    }
    if (threadIdx.x < 48) {
        const int h = threadIdx.x;
        beta[t * 48 + h] = sigmoid(beta_raw[t * 48 + h]);
        decay[t * 48 + h] = __expf(a_neg[h] * softplus(alpha_raw[t * 48 + h] + dt_bias[h]));
    }
}
// scan: block per v-head, 256 threads: thread = (j = tid/2, i-half). q/k tiles of 16 tokens in LDS.
// scan v3: 2 blocks per head, 64 rows x 4 lanes. The next 16-token q/k/v tile is loaded into registers
// while the current tile is computed (the tile load was latency-bound: ~20 us per tile in v1/v2).
// LDS layout [tile][4 quarters][36] keeps the 4 lanes of a row on different banks.
__global__ void __launch_bounds__(256) k_gdn_scan(const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ qkv,
                                                  const float* __restrict__ beta, const float* __restrict__ decay, int T,
                                                  const float* __restrict__ state_in, float* __restrict__ state_out, float* __restrict__ raw) {
    constexpr int D = 128, QKV = 10240, TILE = 16, QW = 36;   // padded quarter width
    const int h = blockIdx.x / 2, part = blockIdx.x % 2, hk = h % 16, tid = threadIdx.x;
    const int j = part * 64 + tid / 4, iq = tid % 4;
    __shared__ __align__(16) float sq[TILE][4][QW], sk[TILE][4][QW], sv[TILE][D];
    __shared__ float sb[TILE], sd[TILE];
    const float* Min = state_in + (size_t)h * D * D + (size_t)j * D + 32 * iq;
    float m[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) m[i] = Min[i];
    // each thread owns 8 q, 8 k, 8 v elements of a tile: element e = tid + 256*r, r < 8 (tile has 16*128 = 2048)
    float pq[8], pk[8], pv[8];
    auto load_tile = [&](int t0) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int e = tid + 256 * r, tt = e / D, d = e % D, t = t0 + tt;
            const bool ok = t < T;
            pq[r] = ok ? qn[((size_t)t * 16 + hk) * D + d] : 0.f; pk[r] = ok ? kn[((size_t)t * 16 + hk) * D + d] : 0.f;
            pv[r] = ok ? qkv[(size_t)t * QKV + 4096 + h * D + d] : 0.f;
        }
    };
    auto store_tile = [&](int t0) {
#pragma unroll
        for (int r = 0; r < 8; ++r) { const int e = tid + 256 * r, tt = e / D, d = e % D; sq[tt][d / 32][d % 32] = pq[r]; sk[tt][d / 32][d % 32] = pk[r]; sv[tt][d] = pv[r]; }
        if (tid < TILE) { const int t = t0 + tid; sb[tid] = t < T ? beta[t * 48 + h] : 0.f; sd[tid] = t < T ? decay[t * 48 + h] : 1.f; }
    };
    load_tile(0);
    for (int t0 = 0; t0 < T; t0 += TILE) {
        __syncthreads();
        store_tile(t0);
        __syncthreads();
        if (t0 + TILE < T) load_tile(t0 + TILE);   // prefetch next tile into registers (latency hidden by the compute below)
        const int nt = min(TILE, T - t0);
        for (int tt = 0; tt < nt; ++tt) {
            float kk[32], qq[32];
#pragma unroll
            for (int i = 0; i < 8; ++i) { const float4 a = *(const float4*)&sk[tt][iq][4 * i]; kk[4 * i] = a.x; kk[4 * i + 1] = a.y; kk[4 * i + 2] = a.z; kk[4 * i + 3] = a.w;
                                          const float4 b = *(const float4*)&sq[tt][iq][4 * i]; qq[4 * i] = b.x; qq[4 * i + 1] = b.y; qq[4 * i + 2] = b.z; qq[4 * i + 3] = b.w; }
            const float b = sb[tt], dc = sd[tt], v = sv[tt][j];
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i += 4) { m[i] *= dc; m[i + 1] *= dc; m[i + 2] *= dc; m[i + 3] *= dc; d0 += m[i] * kk[i]; d1 += m[i + 1] * kk[i + 1]; d2 += m[i + 2] * kk[i + 2]; d3 += m[i + 3] * kk[i + 3]; }
            float dot = (d0 + d1) + (d2 + d3);
            dot += __shfl_xor(dot, 1, 32); dot += __shfl_xor(dot, 2, 32);
            const float delta = (v - dot) * b;
            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i += 4) { m[i] += delta * kk[i]; m[i + 1] += delta * kk[i + 1]; m[i + 2] += delta * kk[i + 2]; m[i + 3] += delta * kk[i + 3];
                                              o0 += m[i] * qq[i]; o1 += m[i + 1] * qq[i + 1]; o2 += m[i + 2] * qq[i + 2]; o3 += m[i + 3] * qq[i + 3]; }
            float o = (o0 + o1) + (o2 + o3);
            o += __shfl_xor(o, 1, 32); o += __shfl_xor(o, 2, 32);
            if (iq == 0) raw[(size_t)(t0 + tt) * 6144 + h * D + j] = o * 0.08838834764831845f;
        }
    }
    float* Mout = state_out + (size_t)h * D * D + (size_t)j * D + 32 * iq;
#pragma unroll
    for (int i = 0; i < 32; ++i) Mout[i] = m[i];
}
// out: block per (t, head): 128 threads
template <bool SIG>
__global__ void __launch_bounds__(128) k_gdn_out(const float* __restrict__ raw, const float* __restrict__ z, const float* __restrict__ norm_w,
                                                 float* __restrict__ out, uint16_t* __restrict__ xb, float eps) {
    constexpr int D = 128;
    const size_t base = (size_t)blockIdx.x * D; const int t = threadIdx.x;
    const float v = raw[base + t];
    float ss = v * v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    __shared__ float sh[4]; if (t % 32 == 0) sh[t / 32] = ss; __syncthreads();
    ss = sh[0] + sh[1] + sh[2] + sh[3];
    const float r = v * rsqrtf(ss / D + eps) * norm_w[t] * (SIG ? sigmoid(z[base + t]) : silu(z[base + t]));
    out[base + t] = r;
    if (xb) xb[base + t] = f2bf(r);
}

// ---- attention decode, T queries ----
// Stage 1: q/k norm + rope + KV write. Block per (t, head in 24 q + 4 k), 256 threads.
template <int NQ, int NKV, bool I8>
__global__ void k_qk_norm_rope_kv(float* __restrict__ qf, float* __restrict__ k, const float* __restrict__ v,
                                  const float* __restrict__ qw, const float* __restrict__ kw, float base, int pos0,
                                  uint16_t* __restrict__ kc, uint16_t* __restrict__ vc, int max_ctx, float eps,
                                  int8_t* __restrict__ kc8, uint16_t* __restrict__ ks, int8_t* __restrict__ vc8, uint16_t* __restrict__ vs,
                                  RowBatch rb, size_t kv_stride, size_t kvt_stride) {
    __shared__ float sh[8]; __shared__ float buf[256];
    const int t = threadIdx.x, hb = blockIdx.x % (NQ + NKV), tt = blockIdx.x / (NQ + NKV);
    const int pos = rb.n ? rb.pos[tt] : pos0 + tt, slot = rb.n ? rb.kv[tt] : 0;
    kc += (size_t)slot * kv_stride; vc += (size_t)slot * kvt_stride;
    if (kc8) { kc8 += (size_t)slot * kv_stride; ks += (size_t)slot * (kv_stride / 32); vc8 += (size_t)slot * kvt_stride; vs += (size_t)slot * (kvt_stride / 256); }
    const bool isq = hb < NQ;
    float* x = isq ? qf + (size_t)tt * NQ * 512 + hb * 512 : k + (size_t)tt * NKV * 256 + (hb - NQ) * 256;
    const float* w = isq ? qw : kw;
    const float xv = x[t];
    const float ss = block_sum(xv * xv, sh);
    float n = xv * rsqrtf(ss / 256 + eps) * w[t];
    buf[t] = n;
    __syncthreads();
    if (t < 64) {   // precise sincos: fast intrinsics lose accuracy for angles ~1e5 rad (positions up to 262144)
        const int i = t % 32; const float theta = (float)pos * powf(base, -2.f * i / 64.f);
        float s, c; sincosf(theta, &s, &c);
        const float x0 = buf[i], x1 = buf[i + 32];
        n = (t < 32) ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    x[t] = n;
    if (!isq) {
        const int kh = hb - NQ;
        const int cp = max_ctx + 128;   // V^T row stride (odd multiple of 256 B: spreads decode streams over all memory channels)
        const float vv = v[(size_t)tt * NKV * 256 + kh * 256 + t];
        if constexpr (!I8) {
            kc[((size_t)kh * max_ctx + pos) * 256 + t] = f2h(n);   // K head-major [kh][pos][256]
            vc[((size_t)kh * 256 + t) * cp + pos] = f2h(vv);       // V^T [kh][dim][pos]
        } else {
            // K: per-32-dim block absmax scale (8 per row); V: one scale per position over 256 dims
            float am = fabsf(n);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor(am, o, 32));
            const float sk = am / 127.f, isk = am > 0.f ? 127.f / am : 0.f;
            kc8[((size_t)kh * max_ctx + pos) * 256 + t] = (int8_t)__float2int_rn(n * isk);
            if (t % 32 == 0) ks[((size_t)kh * max_ctx + pos) * 8 + t / 32] = f2h(sk);
            const float avm = block_max(fabsf(vv), sh);
            const float sv = avm / 127.f, isv = avm > 0.f ? 127.f / avm : 0.f;
            vc8[((size_t)kh * 256 + t) * cp + pos] = (int8_t)__float2int_rn(vv * isv);
            if (t == 0) vs[(size_t)kh * max_ctx + pos] = f2h(sv);
        }
    }
}
// Stage 2: block per (t, q-head), 256 threads; query t attends positions 0..pos0+t.
template <int NQ, int NKV, bool I8>
__global__ void __launch_bounds__(256) k_attn(const float* __restrict__ qf, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                                              const int8_t* __restrict__ kc8, const uint16_t* __restrict__ ks, const int8_t* __restrict__ vc8, const uint16_t* __restrict__ vs,
                                              int pos0, int max_ctx, float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od, float* __restrict__ os,
                                              RowBatch rb, size_t kv_stride, size_t kvt_stride) {
    __shared__ float sh[8]; __shared__ float sq[256]; __shared__ float sc[4096];
    const int h = blockIdx.x % NQ, tt = blockIdx.x / NQ, kh = h / (NQ / NKV), t = threadIdx.x;
    const int cp = max_ctx + 128;
    const int rpos = rb.n ? rb.pos[tt] : pos0 + tt, slot = rb.n ? rb.kv[tt] : 0;
    kc += (size_t)slot * kv_stride; vc += (size_t)slot * kvt_stride;
    if (kc8) { kc8 += (size_t)slot * kv_stride; ks += (size_t)slot * (kv_stride / 32); vc8 += (size_t)slot * kvt_stride; vs += (size_t)slot * (kvt_stride / 256); }
    const uint16_t* kch = kc + (size_t)kh * max_ctx * 256; const uint16_t* vch = vc + ((size_t)kh * 256 + t) * cp;   // this thread's dim row of V^T
    const int8_t* kch8 = kc8 + (size_t)kh * max_ctx * 256; const int8_t* vch8 = vc8 + ((size_t)kh * 256 + t) * cp;
    const uint16_t* ksh = ks + (size_t)kh * max_ctx * 8; const uint16_t* vsh = vs + (size_t)kh * max_ctx;
    const float* q = qf + (size_t)tt * NQ * 512 + h * 512;
    sq[t] = q[t]; __syncthreads();
    const int n = rpos + 1;
    float gmax = -INFINITY, gsum = 0.f; float o = 0.f;
    for (int p0 = 0; p0 < n; p0 += 4096) {
        const int nn = min(4096, n - p0);
        for (int p = t; p < nn; p += 256) {
            float d = 0.f;
            if constexpr (!I8) {
                const uint16_t* kr = kch + (size_t)(p0 + p) * 256;
                for (int i = 0; i < 256; i += 2) { const uint32_t u = *(const uint32_t*)(kr + i); d += sq[i] * h2f(u & 0xFFFF) + sq[i + 1] * h2f(u >> 16); }
            } else {
                const int8_t* kr = kch8 + (size_t)(p0 + p) * 256; const uint16_t* sr = ksh + (size_t)(p0 + p) * 8;
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const float scl = h2f(sr[b]); float acc = 0.f;
                    for (int i = 0; i < 32; i += 4) { const uint32_t u = *(const uint32_t*)(kr + b * 32 + i); const int8_t* q4 = (const int8_t*)&u; acc += sq[b * 32 + i] * q4[0] + sq[b * 32 + i + 1] * q4[1] + sq[b * 32 + i + 2] * q4[2] + sq[b * 32 + i + 3] * q4[3]; }
                    d += acc * scl;
                }
            }
            sc[p] = d * 0.0625f;
        }
        __syncthreads();
        float lm = -INFINITY; for (int p = t; p < nn; p += 256) lm = fmaxf(lm, sc[p]);
        lm = block_max(lm, sh);
        const float nmax = fmaxf(gmax, lm), corr = __expf(gmax - nmax);
        float ls = 0.f; for (int p = t; p < nn; p += 256) { const float e = __expf(sc[p] - nmax); sc[p] = e; ls += e; }
        ls = block_sum(ls, sh);
        gsum = gsum * corr + ls; o *= corr; gmax = nmax;
        __syncthreads();
        if constexpr (!I8) { for (int p = 0; p < nn; ++p) o += sc[p] * h2f(vch[p0 + p]); }
        else { for (int p = 0; p < nn; ++p) o += sc[p] * (float)vch8[p0 + p] * h2f(vsh[p0 + p]); }
        __syncthreads();
    }
    const size_t oi = (size_t)tt * NQ * 256 + h * 256 + t;
    const float v = (o / gsum) * sigmoid(q[256 + t]);
    out[oi] = v;
    wave_quant32(v, oq, od, os, (int)(oi / 32), t % 32);
}

// ---- flash-attention prefill (WMMA f16) ----
// Block = 4 waves = 64 queries of one head; iterate key tiles of 32. LDS: Q tile f16 [64][256] (32 KB), P per wave.
// WMMA f16 operand fragment: lane holds row (lane%16), 16 consecutive k as f16 (mirrored halves).
// D layout (probed): j = lane%16 indexes the SECOND operand's rows, i = 2l + lane/16 the FIRST's.
typedef _Float16 f16x16_t __attribute__((ext_vector_type(16)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));
__device__ __forceinline__ f16x16_t ld_f16x16(const uint16_t* p) {   // 32 contiguous bytes
    const uint4 a = *(const uint4*)p, b = *(const uint4*)(p + 8);
    union { uint4 u[2]; f16x16_t v; } c; c.u[0] = a; c.u[1] = b; return c.v;
}
__device__ __forceinline__ f16x16_t ld_i8x16_scaled(const int8_t* p, float scale) {   // 16 int8 * scale -> f16x16
    const uint4 a = *(const uint4*)p; const int8_t* q = (const int8_t*)&a; f16x16_t r;
#pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = (_Float16)((float)q[i] * scale);
    return r;
}
template <bool I8>
__global__ void __launch_bounds__(128) k_attn_prefill(const float* __restrict__ qf, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                                                      const int8_t* __restrict__ kc8, const uint16_t* __restrict__ ks, const int8_t* __restrict__ vc8, const uint16_t* __restrict__ vs,
                                                      int T, int pos0, int max_ctx, uint16_t* __restrict__ out) {
    constexpr int HD = 256, BQ = 64, BK = 32, NW = 4;
    __shared__ __align__(16) uint16_t sQ[BQ][HD];        // 32 KB
    __shared__ __align__(16) uint16_t sP[NW][16][BK];    // 4 KB
    const int h = blockIdx.x % 24, qt = blockIdx.x / 24, kh = h / 6;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, r16 = lane % 16, half = lane / 16;
    const int q0 = qt * BQ, cp = max_ctx + 128;
    const uint16_t* K = kc + (size_t)kh * max_ctx * HD; const uint16_t* Vt = vc + (size_t)kh * HD * cp;
    const int8_t* K8 = kc8 + (size_t)kh * max_ctx * HD; const int8_t* Vt8 = vc8 + (size_t)kh * HD * cp;
    const uint16_t* KS = ks + (size_t)kh * max_ctx * 8; const uint16_t* VS = vs + (size_t)kh * max_ctx;
    for (int i = threadIdx.x; i < BQ * HD; i += 128) {
        const int q = i / HD, d = i % HD; const int t = q0 + q;
        sQ[q][d] = f2h(t < T ? qf[((size_t)t * 24 + h) * 512 + d] : 0.f);
    }
    __syncthreads();
    f32x8_t O[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) O[i] = (f32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
    float m_row[8], l_row[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) { m_row[l] = -INFINITY; l_row[l] = 0.f; }
    const int qlast = min(q0 + BQ - 1, T - 1);
    const int nk = pos0 + qlast + 1;
    const int qw0 = q0 + wave * 16;                         // this wave's first query
    for (int k0 = 0; k0 < nk; k0 += BK) {
        if (k0 > pos0 + min(qw0 + 15, T - 1)) break;         // fully masked for this wave (causal)
        f32x8_t S[2];
        S[0] = (f32x8_t){0, 0, 0, 0, 0, 0, 0, 0}; S[1] = (f32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
        for (int s = 0; s < HD / 16; ++s) {
            const f16x16_t qa = ld_f16x16(&sQ[wave * 16 + r16][s * 16]);
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int key = k0 + c * 16 + r16;
                f16x16_t kb;
                if (key >= nk) kb = (f16x16_t){};
                else if constexpr (!I8) kb = ld_f16x16(K + (size_t)key * HD + s * 16);
                else kb = ld_i8x16_scaled(K8 + (size_t)key * HD + s * 16, h2f(KS[(size_t)key * 8 + s / 2]));
                S[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa, kb, S[c]);
            }
        }
        float mloc[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int qi = qw0 + 2 * l + half, qpos = pos0 + qi;
#pragma unroll
            for (int c = 0; c < 2; ++c) { const int key = k0 + c * 16 + r16; float v = S[c][l] * 0.0625f; if (key > qpos || qi >= T) v = -INFINITY; S[c][l] = v; }
            float mx = fmaxf(S[0][l], S[1][l]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
            mloc[l] = mx;
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const float mnew = fmaxf(m_row[l], mloc[l]);
            const float corr = (m_row[l] == -INFINITY) ? 0.f : __expf(m_row[l] - mnew);
            float ps = 0.f;
#pragma unroll
            for (int c = 0; c < 2; ++c) { const float p = (mnew == -INFINITY) ? 0.f : __expf(S[c][l] - mnew); S[c][l] = p; ps += p; }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) ps += __shfl_xor(ps, o, 32);
            l_row[l] = l_row[l] * corr + ps; m_row[l] = mnew;
#pragma unroll
            for (int i = 0; i < 16; ++i) O[i][l] *= corr;
        }
#pragma unroll
        for (int l = 0; l < 8; ++l)
#pragma unroll
            for (int c = 0; c < 2; ++c) sP[wave][2 * l + half][c * 16 + r16] = f2h(S[c][l]);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier();
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const f16x16_t pa = ld_f16x16(&sP[wave][r16][c * 16]);
            const int kb0 = k0 + c * 16;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                f16x16_t vb;
                if (kb0 >= nk) vb = (f16x16_t){};   // keys >= nk have P = 0 and finite (zeroed) V
                else if constexpr (!I8) vb = ld_f16x16(Vt + (size_t)(i * 16 + r16) * cp + kb0);
                else {   // 16 keys with 16 per-position scales (contiguous)
                    const uint4 a = *(const uint4*)(Vt8 + (size_t)(i * 16 + r16) * cp + kb0); const int8_t* q8 = (const int8_t*)&a;
                    const uint4 sa = *(const uint4*)(VS + kb0), sb2 = *(const uint4*)(VS + kb0 + 8); const uint16_t* sc16 = (const uint16_t*)&sa; const uint16_t* sc16b = (const uint16_t*)&sb2;
#pragma unroll
                    for (int j = 0; j < 8; ++j) { vb[j] = (_Float16)((float)q8[j] * h2f(sc16[j])); vb[8 + j] = (_Float16)((float)q8[8 + j] * h2f(sc16b[j])); }
                }
                O[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, O[i]);
            }
        }
        __builtin_amdgcn_s_barrier();
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int qi = qw0 + 2 * l + half; if (qi >= T) continue;
        const float inv = l_row[l] > 0.f ? 1.f / l_row[l] : 0.f;
        const float* gate = qf + ((size_t)qi * 24 + h) * 512 + 256;
#pragma unroll
        for (int i = 0; i < 16; ++i) { const int d = i * 16 + r16; out[(size_t)qi * (24 * HD) + h * HD + d] = f2bf(O[i][l] * inv * sigmoid(gate[d])); }
    }
}

// ---- GQA-packed split-KV decode/verify attention (WMMA) ----
// Rows = (q-head g in 0..5, token t in 0..T-1) of one kv head: 6T <= 48 rows -> up to 3 row tiles.
// Block = (kv head, split); each split covers CH positions; K/V read ONCE for the 6 heads.
// Writes unnormalised O (f32), row max m and row sum l per split; k_attn_merge combines splits.
constexpr int GQA_CH = 128;   // positions per split (512 measured: too few waves at depth)
__global__ void __launch_bounds__(32) k_attn_gqa(const float* __restrict__ qf, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                                                 int T, int pos0, int max_ctx, int nsplit, float* __restrict__ pO, float* __restrict__ pm, float* __restrict__ pl) {
    constexpr int HD = 256;
    const int kh = blockIdx.x / nsplit, sp = blockIdx.x % nsplit;
    const int lane = threadIdx.x, r16 = lane % 16, half = lane / 16;
    const int nrows = 6 * T, ntile = (nrows + 15) / 16;
    const int cp = max_ctx + 128;
    const uint16_t* K = kc + (size_t)kh * max_ctx * HD; const uint16_t* Vt = vc + (size_t)kh * HD * cp;
    const int nk_all = pos0 + T;                       // keys 0..pos0+T-1 exist
    const int k_begin = sp * GQA_CH, k_end = min(nk_all, k_begin + GQA_CH);
    __shared__ __align__(16) uint16_t sQ[48][HD];      // 24 KB
    __shared__ __align__(16) uint16_t sP[16][32];
    for (int i = lane; i < 48 * HD; i += 32) {
        const int r = i / HD, d = i % HD; const int g = r / T, t = r % T;
        sQ[r][d] = f2h(r < nrows ? qf[((size_t)t * 24 + kh * 6 + g) * 512 + d] : 0.f);
    }
    __syncthreads();
    for (int rt = 0; rt < ntile; ++rt) {
        f32x8_t O[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) O[i] = (f32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
        float m_row[8], l_row[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) { m_row[l] = -INFINITY; l_row[l] = 0.f; }
        for (int k0 = k_begin; k0 < k_end; k0 += 32) {
            f32x8_t S[2]; S[0] = (f32x8_t){0, 0, 0, 0, 0, 0, 0, 0}; S[1] = (f32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 4
            for (int s = 0; s < HD / 16; ++s) {
                const f16x16_t qa = ld_f16x16(&sQ[rt * 16 + r16][s * 16]);
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int key = k0 + c * 16 + r16;
                    const f16x16_t kb = key < k_end ? ld_f16x16(K + (size_t)key * HD + s * 16) : (f16x16_t){};
                    S[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa, kb, S[c]);
                }
            }
            float mloc[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int row = rt * 16 + 2 * l + half; const int t = row % T; const int qpos = pos0 + t;
#pragma unroll
                for (int c = 0; c < 2; ++c) { const int key = k0 + c * 16 + r16; float v = S[c][l] * 0.0625f; if (key >= k_end || key > qpos || row >= nrows) v = -INFINITY; S[c][l] = v; }
                float mx = fmaxf(S[0][l], S[1][l]);
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
                mloc[l] = mx;
            }
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float mnew = fmaxf(m_row[l], mloc[l]);
                const float corr = (m_row[l] == -INFINITY) ? 0.f : __expf(m_row[l] - mnew);
                float ps = 0.f;
#pragma unroll
                for (int c = 0; c < 2; ++c) { const float p = (mnew == -INFINITY) ? 0.f : __expf(S[c][l] - mnew); S[c][l] = p; ps += p; }
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) ps += __shfl_xor(ps, o, 32);
                l_row[l] = l_row[l] * corr + ps; m_row[l] = mnew;
#pragma unroll
                for (int i = 0; i < 16; ++i) O[i][l] *= corr;
            }
#pragma unroll
            for (int l = 0; l < 8; ++l)
#pragma unroll
                for (int c = 0; c < 2; ++c) sP[2 * l + half][c * 16 + r16] = f2h(S[c][l]);
            __syncthreads();
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const f16x16_t pa = ld_f16x16(&sP[r16][c * 16]);
                const int kb0 = k0 + c * 16;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const f16x16_t vb = (kb0 < k_end) ? ld_f16x16(Vt + (size_t)(i * 16 + r16) * cp + kb0) : (f16x16_t){};
                    O[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, O[i]);
                }
            }
            __syncthreads();
        }
        // write partials: row-major [kh][split][row][256], m/l [kh][split][row]
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int row = rt * 16 + 2 * l + half; if (row >= nrows) continue;
            const size_t pidx = ((size_t)kh * nsplit + sp) * 48 + row;
#pragma unroll
            for (int i = 0; i < 16; ++i) pO[pidx * HD + i * 16 + r16] = O[i][l];
            if (r16 == 0) { pm[pidx] = m_row[l]; pl[pidx] = l_row[l]; }
        }
    }
}
// merge: block per (kv head, row) = 256 threads (dims); out [T][24][256] f32 (gated) + q8 quant
__global__ void __launch_bounds__(256) k_attn_merge(const float* __restrict__ qf, const float* __restrict__ pO, const float* __restrict__ pm, const float* __restrict__ pl,
                                                    int T, int nsplit, float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od, float* __restrict__ os) {
    constexpr int HD = 256;
    const int kh = blockIdx.x / (6 * T), row = blockIdx.x % (6 * T), g = row / T, t = row % T, d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pm[((size_t)kh * nsplit + s) * 48 + row]);
    float L = 0.f, o = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const size_t pidx = ((size_t)kh * nsplit + s) * 48 + row;
        const float ms = pm[pidx]; if (ms == -INFINITY) continue;
        const float w = __expf(ms - M);
        L += w * pl[pidx]; o += w * pO[pidx * HD + d];
    }
    const int h = kh * 6 + g;
    const float gate = qf[((size_t)t * 24 + h) * 512 + 256 + d];
    const float v = (L > 0.f ? o / L : 0.f) * sigmoid(gate);
    const size_t oi = (size_t)t * 24 * HD + h * HD + d;
    out[oi] = v;
    wave_quant32(v, oq, od, os, (int)(oi / 32), d % 32);
}

// ---- argmax / top-k: single block, 1024 threads ----
__global__ void k_topk(const float* __restrict__ x, int n, int k, int* __restrict__ ids, float* __restrict__ vals) {
    __shared__ float sv[1024]; __shared__ int si[1024];
    x += (size_t)blockIdx.x * n; ids += (size_t)blockIdx.x * k; vals += (size_t)blockIdx.x * k;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY; int bi = -1;
        for (int i = threadIdx.x; i < n; i += 1024) {
            bool taken = false; for (int j = 0; j < r; ++j) taken |= (ids[j] == i);
            if (!taken && x[i] > bv) { bv = x[i]; bi = i; }
        }
        sv[threadIdx.x] = bv; si[threadIdx.x] = bi; __syncthreads();
        for (int s = 512; s > 0; s >>= 1) { if (threadIdx.x < s && (sv[threadIdx.x + s] > sv[threadIdx.x] || (sv[threadIdx.x + s] == sv[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x] && si[threadIdx.x + s] >= 0))) { sv[threadIdx.x] = sv[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s]; } __syncthreads(); }
        if (threadIdx.x == 0) { ids[r] = si[0]; vals[r] = sv[0]; }
        __syncthreads();
    }
}
}  // namespace

void embed_q4_K(const void* W, const int* toks, int T, float* out, int n, hipStream_t s) { k_embed_q4_K<<<T, 256, 0, s>>>((const uint8_t*)W, toks, out, n); }
void rmsnorm(const float* x, const float* w, float* out, int rows, int n, float eps, hipStream_t s) { k_rmsnorm<<<rows, 256, 0, s>>>(x, w, out, n, eps); }
void add_inplace(float* y, const float* x, int n, hipStream_t s) { k_add<<<(n + 255) / 256, 256, 0, s>>>(y, x, n); }
void silu_mul(const float* g, const float* u, float* o, int n, hipStream_t s) { k_silu_mul<<<(n + 255) / 256, 256, 0, s>>>(g, u, o, n); }
static SeqBatch one_seq(int T) { SeqBatch sb; sb.n = 1; sb.s[0] = {0, T, 0, 0}; return sb; }
void gdn_conv(const float* x, int T, const float* st_in, float* st_out, const float* w, float* out, int C, hipStream_t s) {
    k_gdn_conv<<<dim3((C + 255) / 256, 1), 256, 0, s>>>(x, one_seq(T), 0, st_in, st_out, w, out, C);
}
void gdn_conv_b(const float* x, const SeqBatch& sb, size_t st_stride, const float* st_base, float* st_obase, const float* w, float* out, int C, hipStream_t s) {
    k_gdn_conv<<<dim3((C + 255) / 256, sb.n), 256, 0, s>>>(x, sb, st_stride, st_base, st_obase, w, out, C);
}
void gdn_step(const float* qkv, const float* z, const float* beta_raw, const float* alpha_raw, int T, const float* dt_bias, const float* a_neg,
              const float* state_in, float* state_out, const float* norm_w, float* out, XQ8 xq, float eps, hipStream_t s) {
    k_gdn_step<false><<<dim3(48, 1), 256, 0, s>>>(qkv, z, beta_raw, alpha_raw, one_seq(T), 0, dt_bias, a_neg, state_in, state_out, norm_w, out, xq.q, xq.d, xq.s, eps);
}
void gdn_step_sig(const float* qkv, const float* z, const float* beta_raw, const float* alpha_raw, int T, const float* dt_bias, const float* a_neg,
                  const float* state_in, float* state_out, const float* norm_w, float* out, XQ8 xq, float eps, hipStream_t s) {
    k_gdn_step<true><<<dim3(48, 1), 256, 0, s>>>(qkv, z, beta_raw, alpha_raw, one_seq(T), 0, dt_bias, a_neg, state_in, state_out, norm_w, out, xq.q, xq.d, xq.s, eps);
}
void gdn_step_sig_b(const float* qkv, const float* z, const float* beta_raw, const float* alpha_raw, const SeqBatch& sb, size_t st_stride, const float* dt_bias, const float* a_neg,
                    const float* st_base, float* st_obase, const float* norm_w, float* out, XQ8 xq, float eps, hipStream_t s) {
    k_gdn_step<true><<<dim3(48, sb.n), 256, 0, s>>>(qkv, z, beta_raw, alpha_raw, sb, st_stride, dt_bias, a_neg, st_base, st_obase, norm_w, out, xq.q, xq.d, xq.s, eps);
}
void gdn_prep(const float* qkv, const float* beta_raw, const float* alpha_raw, const float* dt_bias, const float* a_neg, int T,
              float* qn, float* kn, float* beta, float* decay, float eps, hipStream_t s) {
    k_gdn_prep<<<T, 256, 0, s>>>(qkv, beta_raw, alpha_raw, dt_bias, a_neg, qn, kn, beta, decay, eps);
}
void gdn_scan(const float* qn, const float* kn, const float* qkv, const float* beta, const float* decay, int T,
              const float* state_in, float* state_out, float* raw, hipStream_t s) {
    k_gdn_scan<<<96, 256, 0, s>>>(qn, kn, qkv, beta, decay, T, state_in, state_out, raw);
}
void gdn_out(const float* raw, const float* z, const float* norm_w, int T, float* out, uint16_t* xb, float eps, hipStream_t s) {
    k_gdn_out<false><<<T * 48, 128, 0, s>>>(raw, z, norm_w, out, xb, eps);
}
void gdn_out_sig(const float* raw, const float* z, const float* norm_w, int T, float* out, uint16_t* xb, float eps, hipStream_t s) {
    k_gdn_out<true><<<T * 48, 128, 0, s>>>(raw, z, norm_w, out, xb, eps);
}
void add_rmsnorm_quant(float* x, const float* y, const float* w, float* xn, XQ8 xq, int T, int n, float eps, hipStream_t s) {
    k_add_rmsnorm_quant<<<T, 1024, 0, s>>>(x, y, w, xn, xq.q, xq.d, xq.s, n, eps);
}
void add_rmsnorm_bf16(float* x, const float* y, const float* w, uint16_t* xb, int T, int n, float eps, hipStream_t s) { k_add_rmsnorm_bf16<<<T, 1024, 0, s>>>(x, y, w, xb, n, eps); }
void norm2_concat_bf16(const float* a, const float* wa, int na, const float* b, const float* wb, int nb, uint16_t* out, int T, float eps, hipStream_t s) { k_norm2_concat_bf16<<<T, 1024, 0, s>>>(a, wa, na, b, wb, nb, out, eps); }
void silu_mul_bf16_seg(const uint16_t* gu, int Nt, int off, int N, uint16_t* out, int T, hipStream_t s) { k_silu_mul_bf16_seg<<<(unsigned)(((size_t)T * N + 255) / 256), 256, 0, s>>>(gu, Nt, off, N, out, T); }
void silu_mul_bf16(const float* gate, const float* up, uint16_t* out, size_t n, hipStream_t s) { k_silu_mul_bf16<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(gate, up, out, n); }
void norm2_concat_quant(const float* a, const float* wa, int na, const float* b, const float* wb, int nb, float* out, XQ8 xq, int T, float eps, hipStream_t s) {
    k_norm2_concat_quant<<<T, 1024, 0, s>>>(a, wa, na, b, wb, nb, out, xq.q, xq.d, xq.s, eps);
}
void silu_mul_quant(const float* gate, const float* up, XQ8 xq, int n, hipStream_t s) {
    k_silu_mul_quant<<<(n / 32 + 7) / 8, 256, 0, s>>>(gate, up, xq.q, xq.d, xq.s, n);
}
void attn_decode(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                 uint16_t* kc, uint16_t* vc, int max_ctx, float* out, XQ8 xq, float eps, hipStream_t s) {
    k_qk_norm_rope_kv<24, 4, false><<<28 * T, 256, 0, s>>>(q_full, k, v, q_norm_w, k_norm_w, rope_base, pos0, kc, vc, max_ctx, eps, nullptr, nullptr, nullptr, nullptr, RowBatch{}, 0, 0);
    k_attn<24, 4, false><<<24 * T, 256, 0, s>>>(q_full, kc, vc, nullptr, nullptr, nullptr, nullptr, pos0, max_ctx, out, xq.q, xq.d, xq.s, RowBatch{}, 0, 0);
}
void attn_decode_i8(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                    const KV8& c, int max_ctx, float* out, XQ8 xq, float eps, hipStream_t s) {
    k_qk_norm_rope_kv<24, 4, true><<<28 * T, 256, 0, s>>>(q_full, k, v, q_norm_w, k_norm_w, rope_base, pos0, nullptr, nullptr, max_ctx, eps, c.k, c.ks, c.v, c.vs, RowBatch{}, 0, 0);
    k_attn<24, 4, true><<<24 * T, 256, 0, s>>>(q_full, nullptr, nullptr, c.k, c.ks, c.v, c.vs, pos0, max_ctx, out, xq.q, xq.d, xq.s, RowBatch{}, 0, 0);
}
// Flash-Next geometry: 24 q heads, 2 kv heads (dense; exact below the QSA budget)
void attn_decode_24_2(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                      uint16_t* kc, uint16_t* vc, int max_ctx, float* out, XQ8 xq, float eps, hipStream_t s) {
    k_qk_norm_rope_kv<24, 2, false><<<26 * T, 256, 0, s>>>(q_full, k, v, q_norm_w, k_norm_w, rope_base, pos0, kc, vc, max_ctx, eps, nullptr, nullptr, nullptr, nullptr, RowBatch{}, 0, 0);
    k_attn<24, 2, false><<<24 * T, 256, 0, s>>>(q_full, kc, vc, nullptr, nullptr, nullptr, nullptr, pos0, max_ctx, out, xq.q, xq.d, xq.s, RowBatch{}, 0, 0);
}
void attn_decode_24_2_b(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, const RowBatch& rb,
                        uint16_t* kc, uint16_t* vc, size_t kv_stride, size_t kvt_stride, int max_ctx, float* out, XQ8 xq, float eps, hipStream_t s) {
    k_qk_norm_rope_kv<24, 2, false><<<26 * T, 256, 0, s>>>(q_full, k, v, q_norm_w, k_norm_w, rope_base, 0, kc, vc, max_ctx, eps, nullptr, nullptr, nullptr, nullptr, rb, kv_stride, kvt_stride);
    k_attn<24, 2, false><<<24 * T, 256, 0, s>>>(q_full, kc, vc, nullptr, nullptr, nullptr, nullptr, 0, max_ctx, out, xq.q, xq.d, xq.s, rb, kv_stride, kvt_stride);
}
void attn_decode_gqa(float* q_full, float* k, float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                     uint16_t* kc, uint16_t* vc, int max_ctx, float* out, XQ8 xq, float* pO, float* pm, float* pl, float eps, hipStream_t s) {
    k_qk_norm_rope_kv<24, 4, false><<<28 * T, 256, 0, s>>>(q_full, k, v, q_norm_w, k_norm_w, rope_base, pos0, kc, vc, max_ctx, eps, nullptr, nullptr, nullptr, nullptr, RowBatch{}, 0, 0);
    const int nsplit = (pos0 + T + GQA_CH - 1) / GQA_CH;
    k_attn_gqa<<<4 * nsplit, 32, 0, s>>>(q_full, kc, vc, T, pos0, max_ctx, nsplit, pO, pm, pl);
    k_attn_merge<<<4 * 6 * T, 256, 0, s>>>(q_full, pO, pm, pl, T, nsplit, out, xq.q, xq.d, xq.s);
}
void attn_stage1(float* q_full, float* k, const float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                 uint16_t* kc, uint16_t* vc, int max_ctx, float eps, hipStream_t s) {
    k_qk_norm_rope_kv<24, 4, false><<<28 * T, 256, 0, s>>>(q_full, k, v, q_norm_w, k_norm_w, rope_base, pos0, kc, vc, max_ctx, eps, nullptr, nullptr, nullptr, nullptr, RowBatch{}, 0, 0);
}
void attn_stage1_i8(float* q_full, float* k, const float* v, int T, const float* q_norm_w, const float* k_norm_w, float rope_base, int pos0,
                    const KV8& c, int max_ctx, float eps, hipStream_t s) {
    k_qk_norm_rope_kv<24, 4, true><<<28 * T, 256, 0, s>>>(q_full, k, v, q_norm_w, k_norm_w, rope_base, pos0, nullptr, nullptr, max_ctx, eps, c.k, c.ks, c.v, c.vs, RowBatch{}, 0, 0);
}
void attn_prefill(const float* q_full, const uint16_t* kc, const uint16_t* vc, int T, int pos0, int max_ctx, uint16_t* out_bf16, hipStream_t s) {
    k_attn_prefill<false><<<24 * ((T + 63) / 64), 128, 0, s>>>(q_full, kc, vc, nullptr, nullptr, nullptr, nullptr, T, pos0, max_ctx, out_bf16);
}
void attn_prefill_i8(const float* q_full, const KV8& c, int T, int pos0, int max_ctx, uint16_t* out_bf16, hipStream_t s) {
    k_attn_prefill<true><<<24 * ((T + 63) / 64), 128, 0, s>>>(q_full, nullptr, nullptr, c.k, c.ks, c.v, c.vs, T, pos0, max_ctx, out_bf16);
}
void argmax_topk(const float* logits, int T, int n, int k, int* out_ids, float* out_vals, hipStream_t s) { k_topk<<<T, 1024, 0, s>>>(logits, n, k, out_ids, out_vals); }

}  // namespace hip
