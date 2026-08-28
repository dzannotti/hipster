#include "dflash.h"

namespace hip::dfl {
namespace {
__device__ __forceinline__ float h2f(uint16_t h) { _Float16 x; __builtin_memcpy(&x, &h, 2); return (float)x; }
__device__ __forceinline__ uint16_t f2h(float f) { _Float16 x = (_Float16)f; uint16_t h; __builtin_memcpy(&h, &x, 2); return h; }
__device__ __forceinline__ float wave_sum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32); return v; }
__device__ __forceinline__ float wave_max(float v) { for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32)); return v; }
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = wave_sum(v); __syncthreads(); if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v; __syncthreads();
    float r = 0.f; for (int i = 0; i < (int)(blockDim.x / 32); ++i) r += sh[i]; __syncthreads(); return r;
}
__device__ __forceinline__ void wave_quant32(float v, int8_t* __restrict__ q, float* __restrict__ d, float* __restrict__ s, int sb, int lane) {
    float amax = wave_max(fabsf(v)); const float dd = amax / 127.f, id = amax > 0 ? 127.f / amax : 0.f;
    const int qi = __float2int_rn(v * id); int sum = qi; for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
    q[sb * 32 + lane] = (int8_t)qi; if (lane == 0) { d[sb] = dd; s[sb] = dd * sum; }
}
// Q4_K row of 256 (one block, 144 B) -> f32
__device__ __forceinline__ void dq_q4k_block(const uint8_t* blk, int e, float& out) {
    const float d = h2f(*(const uint16_t*)blk), dmin = h2f(*(const uint16_t*)(blk + 2));
    const uint8_t* sc = blk + 4; const int j = e / 64, l = e % 64, is = 2 * j + (l >= 32);
    uint8_t s, m; if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; } else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
    const uint8_t byte = blk[16 + 32 * j + (l % 32)]; const int nib = l < 32 ? (byte & 0xF) : (byte >> 4);
    out = d * s * nib - dmin * m;
}

// ---- conv: block per (row t, 1024 channels); 256 threads x 4 channels ----
__global__ void __launch_bounds__(256) k_conv(const float* __restrict__ x, const float* __restrict__ delta, const float* __restrict__ base, int side, int T,
                                              float* __restrict__ out, int8_t* __restrict__ q, float* __restrict__ qd, float* __restrict__ qs) {
    const int t = blockIdx.x / (E / 1024), c0 = (blockIdx.x % (E / 1024)) * 1024, lane = threadIdx.x % 32;
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + k * 256 + threadIdx.x, g = c / CONV_G;
        const float* dl = delta + (size_t)t * (2 * 2 * N_GROUPS);
        const float w0 = base[c + E * 0 + 2 * E * side] + dl[g + N_GROUPS * (0 + 2 * side)], w1 = base[c + E * 1 + 2 * E * side] + dl[g + N_GROUPS * (1 + 2 * side)];
        const float v = w0 * x[(size_t)t * E + c] + (t > 0 ? w1 * x[(size_t)(t - 1) * E + c] : 0.f);
        if (out) out[(size_t)t * E + c] = v;
        if (q) wave_quant32(v, q, qd, qs, (int)(((size_t)t * E + c) / 32), lane);
    }
}
// ---- norm + rope (128 dims, NEOX pairs (i, i+64)) + cache write: block per (t, head); heads 0..31 = q, 32..39 = k (+ v copy) ----
__global__ void __launch_bounds__(128) k_qk_rope_kv(float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ qw, const float* __restrict__ kw,
                                                    float base, const int* __restrict__ pos, uint16_t* __restrict__ kc, uint16_t* __restrict__ vc, int max_ctx, float eps) {
    __shared__ float sh[8]; __shared__ float buf[HD];
    const int nh = q ? NQ + NKV : NKV;
    const int t = blockIdx.x / nh, hh = blockIdx.x % nh, d = threadIdx.x;
    const bool isq = q && hh < NQ; const int h = isq ? hh : hh - (q ? NQ : 0);
    float* x = isq ? q + ((size_t)t * NQ + h) * HD : k + ((size_t)t * NKV + h) * HD;
    const float xv = x[d]; const float ss = block_sum(xv * xv, sh);
    float n = xv * rsqrtf(ss / HD + eps) * (isq ? qw : kw)[d];
    buf[d] = n; __syncthreads();
    { const int i = d % 64; const float theta = (float)pos[t] * powf(base, -2.f * i / HD); float s, c; sincosf(theta, &s, &c);
      const float x0 = buf[i], x1 = buf[i + 64]; n = d < 64 ? x0 * c - x1 * s : x0 * s + x1 * c; }
    if (isq) x[d] = n;
    else { kc[((size_t)h * max_ctx + pos[t]) * HD + d] = f2h(n); vc[((size_t)h * max_ctx + pos[t]) * HD + d] = f2h(v[((size_t)t * NKV + h) * HD + d]); }
}
// ---- attention: block per (t, q head), 256 threads; keys [lo, key_end) ----
__global__ void __launch_bounds__(256) k_attend(const float* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc, const int* __restrict__ pos, int key_end, int max_ctx,
                                                float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od, float* __restrict__ os) {
    __shared__ float sh[8]; __shared__ float sq[HD]; __shared__ float sc[2048];
    const int t = blockIdx.x / NQ, h = blockIdx.x % NQ, kh = h / (NQ / NKV), tid = threadIdx.x;
    const int lo = max(0, pos[t] - WINDOW + 1), n = key_end - lo;
    const uint16_t* K = kc + (size_t)kh * max_ctx * HD; const uint16_t* V = vc + (size_t)kh * max_ctx * HD;
    if (tid < HD) sq[tid] = q[((size_t)t * NQ + h) * HD + tid];
    __syncthreads();
    float gmax = -INFINITY, gsum = 0.f, o = 0.f;
    for (int p0 = 0; p0 < n; p0 += 2048) {
        const int nn = min(2048, n - p0);
        for (int p = tid; p < nn; p += 256) { const uint16_t* kr = K + (size_t)(lo + p0 + p) * HD; float d = 0.f; for (int i = 0; i < HD; i += 2) { const uint32_t u = *(const uint32_t*)(kr + i); d += sq[i] * h2f(u & 0xFFFF) + sq[i + 1] * h2f(u >> 16); } sc[p] = d * 0.08838834764831845f; }
        __syncthreads();
        float lm = -INFINITY; for (int p = tid; p < nn; p += 256) lm = fmaxf(lm, sc[p]);
        lm = wave_max(lm); __syncthreads(); if (tid % 32 == 0) sh[tid / 32] = lm; __syncthreads(); lm = sh[0]; for (int i = 1; i < 8; ++i) lm = fmaxf(lm, sh[i]); __syncthreads();
        const float nmax = fmaxf(gmax, lm), corr = __expf(gmax - nmax);
        float ls = 0.f; for (int p = tid; p < nn; p += 256) { const float e = __expf(sc[p] - nmax); sc[p] = e; ls += e; }
        ls = block_sum(ls, sh);
        gsum = gsum * corr + ls; o *= corr; gmax = nmax;
        if (tid < HD) for (int p = 0; p < nn; ++p) o += sc[p] * h2f(V[(size_t)(lo + p0 + p) * HD + tid]);
        __syncthreads();
    }
    if (tid < HD) { const size_t oi = ((size_t)t * NQ + h) * HD + tid; const float v = o / gsum; out[oi] = v; wave_quant32(v, oq, od, os, (int)(oi / 32), tid % 32); }
}
// ---- selector: block per position i in 1..T-1, 256 threads ----
__global__ void __launch_bounds__(256) k_selector(const int* __restrict__ ids, const float* __restrict__ vals, const float* __restrict__ gate, const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                  int anchor, float* __restrict__ lattice) {
    __shared__ float sB[TOPK][RANK]; __shared__ float sA[TOPK][RANK]; __shared__ float sg[RANK];
    const int i = blockIdx.x + 1, tid = threadIdx.x;
    const int npred = i == 1 ? 1 : TOPK;
    for (int e = tid; e < TOPK * RANK; e += 256) { const int c = e / RANK, r = e % RANK; float v; dq_q4k_block(B + (size_t)ids[i * TOPK + c] * 144, r, v); sB[c][r] = v; }
    for (int e = tid; e < npred * RANK; e += 256) { const int p = e / RANK, r = e % RANK; const int pid = i == 1 ? anchor : ids[(i - 1) * TOPK + p]; float v; dq_q4k_block(A + (size_t)pid * 144, r, v); sA[p][r] = v; }
    if (tid < RANK) sg[tid] = gate[(size_t)i * RANK + tid];
    __syncthreads();
    float* row = lattice + (size_t)i * LATTICE;
    if (tid < TOPK) row[tid] = (float)ids[i * TOPK + tid];
    for (int e = tid; e < TOPK * TOPK; e += 256) { const int p = e / TOPK, c = e % TOPK; const float* a = sA[npred == 1 ? 0 : p]; float s = 0.f; for (int r = 0; r < RANK; ++r) s += a[r] * sg[r] * sB[c][r]; row[TOPK + p * TOPK + c] = s + vals[i * TOPK + c]; }
}
}  // namespace

void conv(const float* x, const float* delta, const float* base, int side, int T, float* out, XQ8 xq, hipStream_t s) { k_conv<<<T * (E / 1024), 256, 0, s>>>(x, delta, base, side, T, out, xq.q, xq.d, xq.s); }
void qk_rope_kv(float* q, float* k, const float* v, const float* qw, const float* kw, float base, const int* pos, int T, uint16_t* kc, uint16_t* vc, int max_ctx, float eps, hipStream_t s) {
    k_qk_rope_kv<<<T * (q ? NQ + NKV : NKV), HD, 0, s>>>(q, k, v, qw, kw, base, pos, kc, vc, max_ctx, eps);
}
void attend(const float* q, const uint16_t* kc, const uint16_t* vc, const int* pos, int key_end, int T, int max_ctx, float* out, XQ8 xq, hipStream_t s) {
    k_attend<<<T * NQ, 256, 0, s>>>(q, kc, vc, pos, key_end, max_ctx, out, xq.q, xq.d, xq.s);
}
void selector(const int* ids, const float* vals, const float* gate, const void* A, const void* B, int anchor, int T, float* lattice, hipStream_t s) {
    if (T > 1) k_selector<<<T - 1, 256, 0, s>>>(ids, vals, gate, (const uint8_t*)A, (const uint8_t*)B, anchor, lattice);
}
}  // namespace hip::dfl
