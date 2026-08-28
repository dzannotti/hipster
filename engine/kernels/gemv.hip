#include <cstdio>
#include <cstdlib>
#include "gemv.h"
#include <hip/hip_fp16.h>
#include <cstring>

namespace hip {
int g_moe_tpr = 0;
int g_moe_fake = 0;   // bench: bf16 MoE GEMM with constant weight fragments (WMMA + LDS floor)
int g_persist_blocks = 0;   // 0 = one block per row-block; else cap the grid (persistent loop)
namespace {

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
typedef unsigned int u32x4_t __attribute__((ext_vector_type(4)));
#ifndef HIPSTER_NT_WEIGHTS
#define HIPSTER_NT_WEIGHTS 0
#endif
// weight-stream load: non-temporal so the 17 GB/token stream does not churn MALL/L2 (Nathan's finding)
__device__ __forceinline__ uint4 ldw(const uint4* p) {
#if HIPSTER_NT_WEIGHTS
    u32x4_t v = __builtin_nontemporal_load((const u32x4_t*)p); uint4 r; r.x = v[0]; r.y = v[1]; r.z = v[2]; r.w = v[3]; return r;
#else
    return *p;
#endif
}
// v_dot4_i32_iu8 / v_dot4_i32_i8
__device__ __forceinline__ int dot4_u8s8(uint32_t a, uint32_t b, int c) { return __builtin_amdgcn_sudot4(false, a, true, b, c, false); }
__device__ __forceinline__ int dot4_s8s8(uint32_t a, uint32_t b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }

__device__ __forceinline__ void q4k_scales(const uint8_t* scales, int is, uint8_t& sc0, uint8_t& mn0, uint8_t& sc1, uint8_t& mn1) {
    if (is < 4) { sc0 = scales[is] & 63; mn0 = scales[is + 4] & 63; sc1 = scales[is + 1] & 63; mn1 = scales[is + 5] & 63; }
    else { sc0 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4); mn0 = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
           sc1 = (scales[is + 5] & 0xF) | ((scales[is - 3] >> 6) << 4); mn1 = (scales[is + 5] >> 4) | ((scales[is + 1] >> 6) << 4); }
}

// ---------------- x -> q8 ----------------
__global__ void k_quantize_x_q8(const float* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ d, float* __restrict__ s, int K) {
    const int col = blockIdx.y;
    const int sb = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (sb * 32 >= K) return;
    const float v = x[(size_t)col * K + sb * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    const float dd = amax / 127.f, id = amax > 0 ? 127.f / amax : 0.f;
    const int qi = __float2int_rn(v * id);
    int sum = qi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
    q[(size_t)col * K + sb * 32 + lane] = (int8_t)qi;
    if (lane == 0) { d[(size_t)col * (K / 32) + sb] = dd; s[(size_t)col * (K / 32) + sb] = dd * sum; }
}

// ======================= per-format block decoders =======================
// Each format: ROW_BYTES(K), and `block(row, b, xw, xd, xs)` returning the f32 contribution of
// 256-element block b of one row against one column of x: xw = 64 words of int8 x for that block,
// xd/xs = the 8 per-32 x scales / x sums (d*sum q) for that block. Everything is compile-time.

struct FmtQ4K {
    static __device__ __forceinline__ const uint8_t* row(const uint8_t* W, int r, int K) { return W + (size_t)r * (K / 256) * 144; }
    template <int NCOL>
    static __device__ __forceinline__ void block(const uint8_t* rp, int b, const uint32_t* const* xw, const float* const* xd, const float* const* xs, float* acc) {
        const uint8_t* blk = rp + (size_t)b * 144;
        uint4 v[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) v[i] = ldw((const uint4*)blk + i);
        const uint16_t* hdr = (const uint16_t*)&v[0];
        const float d = h2f(hdr[0]), dmin = h2f(hdr[1]);
        const uint8_t* scales = (const uint8_t*)&v[0] + 4;
        const uint32_t* qw = (const uint32_t*)&v[1];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t sc0, mn0, sc1, mn1; q4k_scales(scales, 2 * j, sc0, mn0, sc1, mn1);
            int i0[NCOL], i1[NCOL];
#pragma unroll
            for (int c = 0; c < NCOL; ++c) { i0[c] = 0; i1[c] = 0; }
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t qq = qw[8 * j + w];
                const uint32_t lo = qq & 0x0F0F0F0Fu, hi = (qq >> 4) & 0x0F0F0F0Fu;
#pragma unroll
                for (int c = 0; c < NCOL; ++c) { i0[c] = dot4_u8s8(lo, xw[c][16 * j + w], i0[c]); i1[c] = dot4_u8s8(hi, xw[c][16 * j + 8 + w], i1[c]); }
            }
            const float d0 = d * sc0, m0 = dmin * mn0, d1 = d * sc1, m1 = dmin * mn1;
#pragma unroll
            for (int c = 0; c < NCOL; ++c) acc[c] += d0 * xd[c][2 * j] * i0[c] - m0 * xs[c][2 * j] + d1 * xd[c][2 * j + 1] * i1[c] - m1 * xs[c][2 * j + 1];
        }
    }
};

struct FmtQ5K {
    static __device__ __forceinline__ const uint8_t* row(const uint8_t* W, int r, int K) { return W + (size_t)r * (K / 256) * 176; }
    template <int NCOL>
    static __device__ __forceinline__ void block(const uint8_t* rp, int b, const uint32_t* const* xw, const float* const* xd, const float* const* xs, float* acc) {
        const uint8_t* blk = rp + (size_t)b * 176;
        uint4 v[11];
#pragma unroll
        for (int i = 0; i < 11; ++i) v[i] = ldw((const uint4*)blk + i);
        const uint16_t* hdr = (const uint16_t*)&v[0];
        const float d = h2f(hdr[0]), dmin = h2f(hdr[1]);
        const uint8_t* scales = (const uint8_t*)&v[0] + 4;
        const uint32_t* qh = (const uint32_t*)&v[1];
        const uint32_t* qw = (const uint32_t*)&v[3];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t sc0, mn0, sc1, mn1; q4k_scales(scales, 2 * j, sc0, mn0, sc1, mn1);
            int i0[NCOL], i1[NCOL];
#pragma unroll
            for (int c = 0; c < NCOL; ++c) { i0[c] = 0; i1[c] = 0; }
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t qq = qw[8 * j + w], hh = qh[w];
                const uint32_t lo = (qq & 0x0F0F0F0Fu) | (((hh >> (2 * j)) & 0x01010101u) << 4);
                const uint32_t hi = ((qq >> 4) & 0x0F0F0F0Fu) | (((hh >> (2 * j + 1)) & 0x01010101u) << 4);
#pragma unroll
                for (int c = 0; c < NCOL; ++c) { i0[c] = dot4_u8s8(lo, xw[c][16 * j + w], i0[c]); i1[c] = dot4_u8s8(hi, xw[c][16 * j + 8 + w], i1[c]); }
            }
            const float d0 = d * sc0, m0 = dmin * mn0, d1 = d * sc1, m1 = dmin * mn1;
#pragma unroll
            for (int c = 0; c < NCOL; ++c) acc[c] += d0 * xd[c][2 * j] * i0[c] - m0 * xs[c][2 * j] + d1 * xd[c][2 * j + 1] * i1[c] - m1 * xs[c][2 * j + 1];
        }
    }
};

// IQ4_XS: nibble -> kvalues_iq4nl via two v_perm_b32 lookups. Per-32 scale (ls-32)*d, no min.
__device__ __forceinline__ uint32_t iq4nl_lut4(uint32_t nib) {
    // kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113} as bytes
    const uint32_t l0 = 0xBFAD9881u, l1 = 0xF6EADDCFu, h0 = 0x26190D01u, h1 = 0x71594535u;
    const uint32_t sel = nib & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(l1, l0, sel), hi = __builtin_amdgcn_perm(h1, h0, sel);
    const uint32_t m = ((nib >> 3) & 0x01010101u) * 0xFFu;
    return (hi & m) | (lo & ~m);
}
struct FmtIQ4XS {
    static __device__ __forceinline__ const uint8_t* row(const uint8_t* W, int r, int K) { return W + (size_t)r * (K / 256) * 136; }
    template <int NCOL>
    static __device__ __forceinline__ void block(const uint8_t* rp, int b, const uint32_t* const* xw, const float* const* xd, const float* const* xs, float* acc) {
        const uint8_t* blk = rp + (size_t)b * 136;
        uint2 h = *(const uint2*)blk;
        uint4 v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = ldw((const uint4*)(blk + 8) + i);
        const float d = h2f((uint16_t)(h.x & 0xFFFF));
        const uint32_t scales_h = h.x >> 16, scales_l = h.y;
        const uint32_t* qw = (const uint32_t*)v;
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l >> (4 * ib)) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
            int i0[NCOL];
#pragma unroll
            for (int c = 0; c < NCOL; ++c) i0[c] = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t qq = qw[4 * ib + w];
                const uint32_t lo = iq4nl_lut4(qq & 0x0F0F0F0Fu), hi = iq4nl_lut4((qq >> 4) & 0x0F0F0F0Fu);
#pragma unroll
                for (int c = 0; c < NCOL; ++c) { i0[c] = dot4_s8s8(lo, xw[c][8 * ib + w], i0[c]); i0[c] = dot4_s8s8(hi, xw[c][8 * ib + 4 + w], i0[c]); }
            }
            const float dl = d * (ls - 32);
#pragma unroll
            for (int c = 0; c < NCOL; ++c) acc[c] += dl * xd[c][ib] * i0[c];
        }
    }
};

template <int NCOL>
__device__ __forceinline__ void q6k_soa_block(const uint8_t* rp, int b, int K, const uint32_t* const* xw, const float* const* xd, float* acc) {
    const int nb = K / 256;
    const uint8_t* ql = rp + (size_t)b * 128;
    const uint8_t* qh = rp + (size_t)nb * 128 + (size_t)b * 64;
    const uint8_t* sc = rp + (size_t)nb * 192 + (size_t)b * 16;
    const uint16_t* dp = (const uint16_t*)(rp + (size_t)nb * 208) + b;
    uint4 vl[8], vh[4], vs;
#pragma unroll
    for (int i = 0; i < 8; ++i) vl[i] = ldw((const uint4*)ql + i);
#pragma unroll
    for (int i = 0; i < 4; ++i) vh[i] = ldw((const uint4*)qh + i);
    vs = *(const uint4*)sc;
    const float d = h2f(*dp);
    const uint32_t* lw = (const uint32_t*)vl;
    const uint32_t* hw = (const uint32_t*)vh;
    const int8_t* s8 = (const int8_t*)&vs;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int i[8][NCOL];
#pragma unroll
        for (int g = 0; g < 8; ++g)
#pragma unroll
            for (int c = 0; c < NCOL; ++c) i[g][c] = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t a = lw[16 * n + w], cc = lw[16 * n + 8 + w], h = hw[8 * n + w];
            const uint32_t q1 = (((a & 0x0F0F0F0Fu) | (((h >> 0) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
            const uint32_t q2 = (((cc & 0x0F0F0F0Fu) | (((h >> 2) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
            const uint32_t q3 = ((((a >> 4) & 0x0F0F0F0Fu) | (((h >> 4) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
            const uint32_t q4 = ((((cc >> 4) & 0x0F0F0F0Fu) | (((h >> 6) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
            const int g = w / 4, xb = 32 * n;
#pragma unroll
            for (int c = 0; c < NCOL; ++c) {
                i[0 + g][c] = dot4_s8s8(q1, xw[c][xb + w], i[0 + g][c]);
                i[2 + g][c] = dot4_s8s8(q2, xw[c][xb + 8 + w], i[2 + g][c]);
                i[4 + g][c] = dot4_s8s8(q3, xw[c][xb + 16 + w], i[4 + g][c]);
                i[6 + g][c] = dot4_s8s8(q4, xw[c][xb + 24 + w], i[6 + g][c]);
            }
        }
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const float ds = d * s8[8 * n + g];
#pragma unroll
            for (int c = 0; c < NCOL; ++c) acc[c] += ds * xd[c][4 * n + g / 2] * i[g][c];
        }
    }
}

// Q8_0 row-SoA: [q K int8][d (K/32)*2]. K multiple of 32: the last 256-block may be partial (guarded per sub-block).
template <int NCOL>
__device__ __forceinline__ void q8_0_soa_block(const uint8_t* rp, int b, int K, const uint32_t* const* xw, const float* const* xd, float* acc) {
    const uint8_t* q = rp + (size_t)b * 256;
    const uint16_t* dp = (const uint16_t*)(rp + K) + b * 8;
    const int nsb_left = K / 32 - b * 8;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        if (ib >= nsb_left) break;
        const uint4 v0 = ldw((const uint4*)q + 2 * ib), v1 = ldw((const uint4*)q + 2 * ib + 1);
        const uint32_t qw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        int i0[NCOL];
#pragma unroll
        for (int c = 0; c < NCOL; ++c) i0[c] = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
#pragma unroll
            for (int c = 0; c < NCOL; ++c) i0[c] = dot4_s8s8(qw[w], xw[c][8 * ib + w], i0[c]);
        }
        const float dd = h2f(dp[ib]);
#pragma unroll
        for (int c = 0; c < NCOL; ++c) acc[c] += dd * xd[c][ib] * i0[c];
    }
}

// Q5_1 row-SoA: [qs K/2][qh K/8][d K/32 f16][m K/32 f16]; w = d*q + m with q = nibble | (hbit<<4).
// ggml block: qh bit j -> element j (0..15: low nibbles), bits 16..31 -> elements 16..31 (high nibbles).
template <int NCOL>
__device__ __forceinline__ void q5_1_soa_block(const uint8_t* rp, int b, int K, const uint32_t* const* xw, const float* const* xd, const float* const* xs, float* acc) {
    const uint8_t* qs = rp + (size_t)b * 128; const uint32_t* qh = (const uint32_t*)(rp + K / 2) + b * 8;
    const uint16_t* dp = (const uint16_t*)(rp + K / 2 + K / 8) + b * 8; const uint16_t* mp = dp + K / 32;
    const int nsb_left = K / 32 - b * 8;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        if (ib >= nsb_left) break;
        const uint4 v = ldw((const uint4*)qs + ib);
        const uint32_t qw[4] = {v.x, v.y, v.z, v.w};
        const uint32_t hb = qh[ib];
        int i0[NCOL];
#pragma unroll
        for (int c = 0; c < NCOL; ++c) i0[c] = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t qq = qw[w];
            const uint32_t hl = ((hb >> (4 * w)) & 1u) | (((hb >> (4 * w + 1)) & 1u) << 8) | (((hb >> (4 * w + 2)) & 1u) << 16) | (((hb >> (4 * w + 3)) & 1u) << 24);
            const uint32_t hh = ((hb >> (16 + 4 * w)) & 1u) | (((hb >> (17 + 4 * w)) & 1u) << 8) | (((hb >> (18 + 4 * w)) & 1u) << 16) | (((hb >> (19 + 4 * w)) & 1u) << 24);
            const uint32_t lo = (qq & 0x0F0F0F0Fu) | (hl << 4), hi = ((qq >> 4) & 0x0F0F0F0Fu) | (hh << 4);
#pragma unroll
            for (int c = 0; c < NCOL; ++c) { i0[c] = dot4_u8s8(lo, xw[c][8 * ib + w], i0[c]); i0[c] = dot4_u8s8(hi, xw[c][8 * ib + 4 + w], i0[c]); }
        }
        const float d = h2f(dp[ib]), m = h2f(mp[ib]);
#pragma unroll
        for (int c = 0; c < NCOL; ++c) acc[c] += d * xd[c][ib] * i0[c] + m * xs[c][ib];
    }
}

// ======================= the kernels =======================
template <WFmt F, int NCOL>
__device__ __forceinline__ void dispatch_block(const uint8_t* rp, int b, int K, const uint32_t* const* xw, const float* const* xd, const float* const* xs, float* acc) {
    if constexpr (F == WFmt::Q4_K) FmtQ4K::block<NCOL>(rp, b, xw, xd, xs, acc);
    else if constexpr (F == WFmt::Q5_K) FmtQ5K::block<NCOL>(rp, b, xw, xd, xs, acc);
    else if constexpr (F == WFmt::IQ4_XS) FmtIQ4XS::block<NCOL>(rp, b, xw, xd, xs, acc);
    else if constexpr (F == WFmt::Q6_K_SOA) q6k_soa_block<NCOL>(rp, b, K, xw, xd, acc);
    else if constexpr (F == WFmt::Q5_1_SOA) q5_1_soa_block<NCOL>(rp, b, K, xw, xd, xs, acc);
    else q8_0_soa_block<NCOL>(rp, b, K, xw, xd, acc);
}
template <WFmt F>
__device__ __forceinline__ const uint8_t* row_ptr(const uint8_t* W, int row, int K) {
    if constexpr (F == WFmt::Q4_K) return FmtQ4K::row(W, row, K);
    else if constexpr (F == WFmt::Q5_K) return FmtQ5K::row(W, row, K);
    else if constexpr (F == WFmt::IQ4_XS) return FmtIQ4XS::row(W, row, K);
    else if constexpr (F == WFmt::Q6_K_SOA) return W + (size_t)row * ((size_t)K * 210 / 256);
    else if constexpr (F == WFmt::Q5_1_SOA) return W + (size_t)row * ((size_t)K * 24 / 32);
    else return W + (size_t)row * ((size_t)K + K / 16);
}

// Direct: x words read from global (L1/L2). Rows in a wave share x addresses.
template <WFmt F, int TPR, int NCOL>
__global__ void __launch_bounds__(256) k_gemv_q8(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                                                 const float* __restrict__ xd, const float* __restrict__ xs,
                                                 float* __restrict__ y, int N, int K) {
    constexpr int ROWS = 256 / TPR;
    const int tid = threadIdx.x, lane = tid % TPR, r = tid / TPR;
    const int nb = (K + 255) / 256, nsb = K / 32, nblk = (N + ROWS - 1) / ROWS;
    for (int vb = blockIdx.x; vb < nblk; vb += gridDim.x) {   // persistent: loop over row-blocks
        const int row = vb * ROWS + r;
        if (row >= N) break;
        const uint8_t* rp = row_ptr<F>(W, row, K);
        float acc[NCOL];
#pragma unroll
        for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;
        for (int b = lane; b < nb; b += TPR) {
            const uint32_t* xw[NCOL]; const float* cxd[NCOL]; const float* cxs[NCOL];
#pragma unroll
            for (int c = 0; c < NCOL; ++c) { xw[c] = (const uint32_t*)(xq + (size_t)c * K + b * 256); cxd[c] = xd + (size_t)c * nsb + b * 8; cxs[c] = xs + (size_t)c * nsb + b * 8; }
            dispatch_block<F, NCOL>(rp, b, K, xw, cxd, cxs, acc);
        }
#pragma unroll
        for (int c = 0; c < NCOL; ++c) {
#pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1) acc[c] += __shfl_xor(acc[c], off, TPR);
        }
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < NCOL; ++c) y[(size_t)c * N + row] = acc[c];
        }
    }
}

// LDS-staged: per iteration the 256 threads cover TPR consecutive blocks (same for every row);
// stage those TPR*NCOL*256 int8 (+ scales) in LDS once, then read x words from LDS.
template <WFmt F, int TPR, int NCOL>
__global__ void __launch_bounds__(256) k_gemv_q8_lds(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                                                     const float* __restrict__ xd, const float* __restrict__ xs,
                                                     float* __restrict__ y, int N, int K) {
    constexpr int ROWS = 256 / TPR;
    __shared__ __align__(16) int8_t sx[NCOL][TPR * 256];
    __shared__ float sd[NCOL][TPR * 8], ss[NCOL][TPR * 8];
    const int tid = threadIdx.x, lane = tid % TPR, r = tid / TPR;
    const int row = min(blockIdx.x * ROWS + r, N - 1);
    const int nb = (K + 255) / 256, nsb = K / 32;
    const uint8_t* rp = row_ptr<F>(W, row, K);
    float acc[NCOL];
#pragma unroll
    for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;
    for (int b0 = 0; b0 < nb; b0 += TPR) {
        __syncthreads();
        for (int i = tid; i < NCOL * TPR * 16; i += 256) {
            const int c = i / (TPR * 16), o = i % (TPR * 16);
            if (b0 * 256 + o * 16 < K) ((uint4*)sx[c])[o] = *(const uint4*)(xq + (size_t)c * K + b0 * 256 + o * 16);
        }
        for (int i = tid; i < NCOL * TPR * 8; i += 256) {
            const int c = i / (TPR * 8), o = i % (TPR * 8);
            if (b0 * 8 + o < nsb) { sd[c][o] = xd[(size_t)c * nsb + b0 * 8 + o]; ss[c][o] = xs[(size_t)c * nsb + b0 * 8 + o]; }
        }
        __syncthreads();
        const int b = b0 + lane;
        if (b < nb) {
            const uint32_t* xw[NCOL]; const float* cxd[NCOL]; const float* cxs[NCOL];
#pragma unroll
            for (int c = 0; c < NCOL; ++c) { xw[c] = (const uint32_t*)(sx[c] + lane * 256); cxd[c] = sd[c] + lane * 8; cxs[c] = ss[c] + lane * 8; }
            dispatch_block<F, NCOL>(rp, b, K, xw, cxd, cxs, acc);
        }
    }
#pragma unroll
    for (int c = 0; c < NCOL; ++c) {
#pragma unroll
        for (int off = TPR / 2; off > 0; off >>= 1) acc[c] += __shfl_xor(acc[c], off, TPR);
    }
    if (lane == 0 && blockIdx.x * ROWS + r < N) {
#pragma unroll
        for (int c = 0; c < NCOL; ++c) y[(size_t)c * N + row] = acc[c];
    }
}

// ======================= WMMA (tensor core) variant for 8..16 columns =======================
// One wave = 16 weight rows; block = 8 waves = 128 rows. x is [16][K] int8 (columns >= ncol zero).
// RDNA3 wave32 iu8 WMMA: each operand fragment = 16 int8 of row (lane%16), lanes 16-31 mirror lanes 0-15;
// D = 8 int32 per lane: index into the 2nd operand = lane%16, index into the 1st operand = 2*l + lane/16 (probed).
typedef int int32x4_t __attribute__((ext_vector_type(4)));
typedef int int32x8_t __attribute__((ext_vector_type(8)));

template <int TILES>   // row tiles (of 16) per wave
__global__ void __launch_bounds__(256) k_gemv_q4_K_wmma(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                                                        const float* __restrict__ xd, const float* __restrict__ xs,
                                                        float* __restrict__ y, int N, int K, int ncol) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int r16 = lane % 16, half = lane / 16;
    const int nb = K / 256, nsb = K / 32;
    const int row_base = (blockIdx.x * 8 + wave) * (16 * TILES);
    float acc[TILES][8];
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
        for (int l = 0; l < 8; ++l) acc[t][l] = 0.f;

    for (int b = 0; b < nb; ++b) {
        // B fragments for the 8 sub-blocks: column (lane%16), 16 bytes per k-step. Shared by all row tiles.
        int32x4_t B[8][2];
        const int8_t* xcol = xq + (size_t)r16 * K + b * 256;
#pragma unroll
        for (int sb = 0; sb < 8; ++sb) { B[sb][0] = *(const int32x4_t*)(xcol + sb * 32); B[sb][1] = *(const int32x4_t*)(xcol + sb * 32 + 16); }
        // column scales for this lane's 8 output columns (2l + half)
        float cxd[8], cxs[8];
#pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const int row = min(row_base + 16 * t + r16, N - 1);
            const uint8_t* blk = W + ((size_t)row * nb + b) * 144;
            uint4 v[9];
#pragma unroll
            for (int i = 0; i < 9; ++i) v[i] = ldw((const uint4*)blk + i);
            const uint16_t* hdr = (const uint16_t*)&v[0];
            const float d = h2f(hdr[0]), dmin = h2f(hdr[1]);
            const uint8_t* scales = (const uint8_t*)&v[0] + 4;
            const uint32_t* qw = (const uint32_t*)&v[1];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t sc0, mn0, sc1, mn1; q4k_scales(scales, 2 * j, sc0, mn0, sc1, mn1);
                int32x4_t A0lo, A1lo, A0hi, A1hi;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t w0 = qw[8 * j + i], w1 = qw[8 * j + 4 + i];
                    A0lo[i] = (int)(w0 & 0x0F0F0F0Fu); A1lo[i] = (int)(w1 & 0x0F0F0F0Fu);
                    A0hi[i] = (int)((w0 >> 4) & 0x0F0F0F0Fu); A1hi[i] = (int)((w1 >> 4) & 0x0F0F0F0Fu);
                }
                int32x8_t C0 = {0, 0, 0, 0, 0, 0, 0, 0}, C1 = {0, 0, 0, 0, 0, 0, 0, 0};
                // Probed on gfx1151: D lane layout is j = lane%16 from the SECOND operand, i = 2l + lane/16 from
                // the first. Weights go second so each lane's 8 outputs share its own row (and scales).
                C0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B[2 * j][0], false, A0lo, C0, false);
                C0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B[2 * j][1], false, A1lo, C0, false);
                C1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B[2 * j + 1][0], false, A0hi, C1, false);
                C1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B[2 * j + 1][1], false, A1hi, C1, false);
                const float d0 = d * sc0, m0 = dmin * mn0, d1 = d * sc1, m1 = dmin * mn1;
                const int sb0 = b * 8 + 2 * j;
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int col = 2 * l + half;
                    const float xd0 = xd[(size_t)col * nsb + sb0], xs0 = xs[(size_t)col * nsb + sb0];
                    const float xd1 = xd[(size_t)col * nsb + sb0 + 1], xs1 = xs[(size_t)col * nsb + sb0 + 1];
                    acc[t][l] += d0 * xd0 * (float)C0[l] - m0 * xs0 + d1 * xd1 * (float)C1[l] - m1 * xs1;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
        const int row = row_base + 16 * t + r16;
        if (row < N) {
#pragma unroll
            for (int l = 0; l < 8; ++l) { const int col = 2 * l + half; if (col < ncol) y[(size_t)col * N + row] = acc[t][l]; }
        }
    }
}


// ======================= MoE grouped GEMM for prefill (WMMA iu8) =======================
// Slots (token, expert) sorted by expert on the host; a tile = up to 16 sorted positions [k0, k0+ncols) of one expert.
// Block = (tile, 128-row group of N): 8 waves x 16 rows; every wave computes its 16 rows x the tile's 16 columns.
// x columns: int8 q8 rows (per 32 scales) at row rowtab[k] (or k itself when rowtab == null); y[k][N] f32.
// Weights decoded per 32-sub-block into iu8 fragments: Q4_K nibbles (+ per-sub-block scale/min), Q5_1 (5-bit + min),
// Q8_0 (signed). Same probed D layout as k_gemv_q4_K_wmma: lane's 8 outputs = columns 2l + lane/16 of its own row.
// x operand of one 256-block for the tile's columns, staged in LDS once per block (all 8 waves share it)
template <int CT> struct MoeLds { int8_t x[16 * CT][256 + 16]; float d[8][16 * CT + 1]; float s[8][16 * CT + 1]; };   // padded: rows land in different LDS banks
template <WFmt F, int CT>
__device__ __forceinline__ void wmma_row_block(const uint8_t* __restrict__ rp, int b, int K, const MoeLds<CT>& L, int r16, int half, float (*acc)[8]) {
    auto loadB = [&](int sb, int32x4_t (*B0), int32x4_t (*B1)) {
#pragma unroll
        for (int ct = 0; ct < CT; ++ct) { B0[ct] = *(const int32x4_t*)&L.x[16 * ct + r16][sb * 32]; B1[ct] = *(const int32x4_t*)&L.x[16 * ct + r16][sb * 32 + 16]; }
    };
    // acc update for sub-block sb of the block with (scale d, min m): acc += d * xd * C + m * xs (m carries the sign)
    auto update = [&](int ct, int sb, float d, float m, const int32x8_t& C) {
#pragma unroll
        for (int l = 0; l < 8; ++l) { const int c = 16 * ct + 2 * l + half; acc[ct][l] += d * L.d[sb][c] * (float)C[l] + m * L.s[sb][c]; }
    };
    if constexpr (F == WFmt::Q4_K || F == WFmt::Q5_K) {
        constexpr int BB = F == WFmt::Q4_K ? 144 : 176, QS = F == WFmt::Q4_K ? 16 : 48;
        const uint8_t* blk = rp + (size_t)b * BB;
        const uint4 h = ldw((const uint4*)blk);
        const uint16_t* hdr = (const uint16_t*)&h;
        const float d = h2f(hdr[0]), dmin = h2f(hdr[1]);
        const uint8_t* scales = (const uint8_t*)&h + 4;
        uint4 qh4[2];
        if constexpr (F == WFmt::Q5_K) { qh4[0] = ldw((const uint4*)(blk + 16)); qh4[1] = ldw((const uint4*)(blk + 32)); }
        const uint32_t* qhw = (const uint32_t*)qh4;
#pragma unroll 1
        for (int j = 0; j < 4; ++j) {
            uint8_t sc0, mn0, sc1, mn1; q4k_scales(scales, 2 * j, sc0, mn0, sc1, mn1);
            const uint4 q0 = ldw((const uint4*)(blk + QS + 32 * j)), q1 = ldw((const uint4*)(blk + QS + 32 * j + 16));
            const uint32_t qw0[4] = {q0.x, q0.y, q0.z, q0.w}, qw1[4] = {q1.x, q1.y, q1.z, q1.w};
            int32x4_t A0lo, A1lo, A0hi, A1hi;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w0 = qw0[i], w1 = qw1[i];
                uint32_t l0 = w0 & 0x0F0F0F0Fu, l1 = w1 & 0x0F0F0F0Fu, h0 = (w0 >> 4) & 0x0F0F0F0Fu, h1 = (w1 >> 4) & 0x0F0F0F0Fu;
                if constexpr (F == WFmt::Q5_K) {
                    const uint32_t x0 = qhw[i], x1 = qhw[4 + i];
                    l0 |= ((x0 >> (2 * j)) & 0x01010101u) << 4; l1 |= ((x1 >> (2 * j)) & 0x01010101u) << 4;
                    h0 |= ((x0 >> (2 * j + 1)) & 0x01010101u) << 4; h1 |= ((x1 >> (2 * j + 1)) & 0x01010101u) << 4;
                }
                A0lo[i] = (int)l0; A1lo[i] = (int)l1; A0hi[i] = (int)h0; A1hi[i] = (int)h1;
            }
            int32x4_t B0[CT], B1[CT];
            loadB(2 * j, B0, B1);
#pragma unroll
            for (int ct = 0; ct < CT; ++ct) {
                int32x8_t C = {0, 0, 0, 0, 0, 0, 0, 0};
                C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B0[ct], false, A0lo, C, false);
                C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B1[ct], false, A1lo, C, false);
                update(ct, 2 * j, d * sc0, -dmin * mn0, C);
            }
            loadB(2 * j + 1, B0, B1);
#pragma unroll
            for (int ct = 0; ct < CT; ++ct) {
                int32x8_t C = {0, 0, 0, 0, 0, 0, 0, 0};
                C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B0[ct], false, A0hi, C, false);
                C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B1[ct], false, A1hi, C, false);
                update(ct, 2 * j + 1, d * sc1, -dmin * mn1, C);
            }
        }
    } else if constexpr (F == WFmt::Q5_1_SOA) {
        const uint8_t* qs = rp + (size_t)b * 128; const uint32_t* qh = (const uint32_t*)(rp + K / 2) + b * 8;
        const uint16_t* dp = (const uint16_t*)(rp + K / 2 + K / 8) + b * 8; const uint16_t* mp = dp + K / 32;
        const int nsb_left = K / 32 - b * 8;
#pragma unroll 2
        for (int sb = 0; sb < 8; ++sb) {
            if (sb >= nsb_left) break;
            const uint4 v = ldw((const uint4*)qs + sb);
            const uint32_t qw[4] = {v.x, v.y, v.z, v.w};
            const uint32_t hb = qh[sb];
            int32x4_t Alo, Ahi;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t qq = qw[w];
                const uint32_t hl = ((hb >> (4 * w)) & 1u) | (((hb >> (4 * w + 1)) & 1u) << 8) | (((hb >> (4 * w + 2)) & 1u) << 16) | (((hb >> (4 * w + 3)) & 1u) << 24);
                const uint32_t hh = ((hb >> (16 + 4 * w)) & 1u) | (((hb >> (17 + 4 * w)) & 1u) << 8) | (((hb >> (18 + 4 * w)) & 1u) << 16) | (((hb >> (19 + 4 * w)) & 1u) << 24);
                Alo[w] = (int)((qq & 0x0F0F0F0Fu) | (hl << 4)); Ahi[w] = (int)(((qq >> 4) & 0x0F0F0F0Fu) | (hh << 4));
            }
            const float d = h2f(dp[sb]), m = h2f(mp[sb]);
            int32x4_t B0[CT], B1[CT];
            loadB(sb, B0, B1);
#pragma unroll
            for (int ct = 0; ct < CT; ++ct) {
                int32x8_t C = {0, 0, 0, 0, 0, 0, 0, 0};
                C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B0[ct], false, Alo, C, false);
                C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B1[ct], false, Ahi, C, false);
                update(ct, sb, d, m, C);
            }
        }
    } else {   // Q8_0 SoA: [q K][d K/32]
        const int8_t* q = (const int8_t*)rp + (size_t)b * 256; const uint16_t* dp = (const uint16_t*)(rp + K) + b * 8;
        const int nsb_left = K / 32 - b * 8;
#pragma unroll 2
        for (int sb = 0; sb < 8; ++sb) {
            if (sb >= nsb_left) break;
            const int32x4_t Alo = *(const int32x4_t*)(q + sb * 32), Ahi = *(const int32x4_t*)(q + sb * 32 + 16);
            const float d = h2f(dp[sb]);
            int32x4_t B0[CT], B1[CT];
            loadB(sb, B0, B1);
#pragma unroll
            for (int ct = 0; ct < CT; ++ct) {
                int32x8_t C = {0, 0, 0, 0, 0, 0, 0, 0};
                C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B0[ct], true, Alo, C, false);
                C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, B1[ct], true, Ahi, C, false);
                update(ct, sb, d, 0.f, C);
            }
        }
    }
}
template <WFmt F, int CT>
__global__ void __launch_bounds__(256) k_moe_gemm(const uint8_t* __restrict__ W, size_t ebytes, const MoeTile* __restrict__ tiles,
                                                  const int* __restrict__ rowtab, const int8_t* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ xs,
                                                  float* __restrict__ y, int N, int K) {
    __shared__ __align__(16) MoeLds<CT> L;
    const int nblkN = (N + 127) / 128;
    const MoeTile tile = tiles[blockIdx.x / nblkN];
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, r16 = lane % 16, half = lane / 16, tid = threadIdx.x;
    const int row_real = (blockIdx.x % nblkN) * 128 + wave * 16 + r16, row = min(row_real, N - 1);
    const int nb = (K + 255) / 256, nsb = K / 32;
    const uint8_t* Wt = tile.expert >= 0 ? W + (size_t)tile.expert * ebytes : W;   // shared launches pass the shared matrix with expert = -1
    const uint8_t* rp = row_ptr<F>(Wt, row, K);
    float acc[CT][8];
#pragma unroll
    for (int ct = 0; ct < CT; ++ct)
#pragma unroll
        for (int l = 0; l < 8; ++l) acc[ct][l] = 0.f;
    for (int b = 0; b < nb; ++b) {
        __syncthreads();   // previous block's LDS reads done
        // stage the 16*CT columns' x block (256 int8 = 16 x uint4 per column) and their 8 sub-block scales
        for (int i = tid; i < 16 * CT * 16; i += 256) {
            const int c = i / 16, part = i % 16;
            const int k = tile.k0 + (c < tile.ncols ? c : 0), xrow = rowtab ? rowtab[k] : k;
            const int kk = b * 256 + part * 16;
            uint4 v = {0, 0, 0, 0};
            if (kk < K) v = *(const uint4*)(xq + (size_t)xrow * K + kk);
            *(uint4*)&L.x[c][part * 16] = v;
        }
        for (int i = tid; i < 16 * CT * 8; i += 256) {
            const int c = i / 8, sb = i % 8;
            const int k = tile.k0 + (c < tile.ncols ? c : 0), xrow = rowtab ? rowtab[k] : k;
            const int sbi = b * 8 + sb;
            const bool ok = sbi < nsb;
            L.d[sb][c] = ok ? xd[(size_t)xrow * nsb + sbi] : 0.f; L.s[sb][c] = ok ? xs[(size_t)xrow * nsb + sbi] : 0.f;
        }
        __syncthreads();
        wmma_row_block<F, CT>(rp, b, K, L, r16, half, acc);
    }
    if (row_real < N) {
#pragma unroll
        for (int ct = 0; ct < CT; ++ct)
#pragma unroll
            for (int l = 0; l < 8; ++l) { const int c = 16 * ct + 2 * l + half; if (c < tile.ncols) y[(size_t)(tile.k0 + c) * N + row] = acc[ct][l]; }
    }
}
template <WFmt F> static void launch_moe_gemm(const void* W, size_t ebytes, const MoeTile* tiles, int ntiles, int ct, const int* rowtab, XQ8 x, float* y, int N, int K, hipStream_t s) {
    if (ntiles <= 0) return;
    const int grid = ntiles * ((N + 127) / 128);
    if (ct == 1) k_moe_gemm<F, 1><<<grid, 256, 0, s>>>((const uint8_t*)W, ebytes, tiles, rowtab, x.q, x.d, x.s, y, N, K);
    else if (ct == 2) k_moe_gemm<F, 2><<<grid, 256, 0, s>>>((const uint8_t*)W, ebytes, tiles, rowtab, x.q, x.d, x.s, y, N, K);
    else k_moe_gemm<F, 3><<<grid, 256, 0, s>>>((const uint8_t*)W, ebytes, tiles, rowtab, x.q, x.d, x.s, y, N, K);
}
static void moe_gemm_fmt(WFmt f, const void* W, size_t ebytes, const MoeTile* tiles, int ntiles, int ct, const int* rowtab, XQ8 x, float* y, int N, int K, hipStream_t s) {
    switch (f) {
        case WFmt::Q4_K: launch_moe_gemm<WFmt::Q4_K>(W, ebytes, tiles, ntiles, ct, rowtab, x, y, N, K, s); break;
        case WFmt::Q5_K: launch_moe_gemm<WFmt::Q5_K>(W, ebytes, tiles, ntiles, ct, rowtab, x, y, N, K, s); break;
        case WFmt::Q5_1_SOA: launch_moe_gemm<WFmt::Q5_1_SOA>(W, ebytes, tiles, ntiles, ct, rowtab, x, y, N, K, s); break;
        case WFmt::Q8_0_SOA: launch_moe_gemm<WFmt::Q8_0_SOA>(W, ebytes, tiles, ntiles, ct, rowtab, x, y, N, K, s); break;
        default: fprintf(stderr, "moe_gemm: unsupported expert format\n"); abort();
    }
}


// ======================= MoE grouped GEMM, bf16 WMMA (prefill) =======================
// Same tiling as k_moe_gemm; the quantisation scales are folded into the weight fragments (bf16), the x operand is
// bf16 rows (no per-32 activation scales) and the WMMA accumulates straight into f32 — no per-column fix-ups.
typedef short bf16x16_t __attribute__((ext_vector_type(16)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));
__device__ __forceinline__ short f2bfs(float f) { uint32_t u = __float_as_uint(f); u += 0x7FFFu + ((u >> 16) & 1); return (short)(u >> 16); }
template <int CT> struct MoeLdsB { uint16_t x[16 * CT][256 + 8]; };   // padded row stride: no 16-way bank conflicts
// 32 dequantised weights of sub-block (b, sb) of row rp -> two bf16x16 fragments (k 0..15, 16..31)
template <WFmt F>
__device__ __forceinline__ void moe_wfrag(const uint8_t* __restrict__ rp, int b, int sb, int K, bf16x16_t& lo, bf16x16_t& hi) {
    float v[32];
    if constexpr (F == WFmt::Q4_K || F == WFmt::Q5_K) {
        constexpr int BB = F == WFmt::Q4_K ? 144 : 176, QS = F == WFmt::Q4_K ? 16 : 48;
        const uint8_t* blk = rp + (size_t)b * BB;
        const uint4 h = ldw((const uint4*)blk); const uint16_t* hdr = (const uint16_t*)&h;
        const float d = h2f(hdr[0]), dmin = h2f(hdr[1]); const uint8_t* scales = (const uint8_t*)&h + 4;
        const int j = sb / 2, hiNib = sb % 2;
        uint8_t sc0, mn0, sc1, mn1; q4k_scales(scales, 2 * j, sc0, mn0, sc1, mn1);
        const float ds = d * (hiNib ? sc1 : sc0), ms = dmin * (hiNib ? mn1 : mn0);
        const uint4 q0 = ldw((const uint4*)(blk + QS + 32 * j)), q1 = ldw((const uint4*)(blk + QS + 32 * j + 16));
        const uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        uint32_t qhw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if constexpr (F == WFmt::Q5_K) { const uint4 a = ldw((const uint4*)(blk + 16)), c = ldw((const uint4*)(blk + 32)); qhw[0] = a.x; qhw[1] = a.y; qhw[2] = a.z; qhw[3] = a.w; qhw[4] = c.x; qhw[5] = c.y; qhw[6] = c.z; qhw[7] = c.w; }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t nib = hiNib ? (qw[i] >> 4) & 0x0F0F0F0Fu : qw[i] & 0x0F0F0F0Fu;
            uint32_t q = nib;
            if constexpr (F == WFmt::Q5_K) q |= ((qhw[i] >> (2 * j + hiNib)) & 0x01010101u) << 4;
#pragma unroll
            for (int e = 0; e < 4; ++e) v[4 * i + e] = (float)((q >> (8 * e)) & 0xFF) * ds - ms;
        }
    } else if constexpr (F == WFmt::Q5_1_SOA) {
        const uint8_t* qs = rp + (size_t)b * 128 + sb * 16; const uint32_t hb = ((const uint32_t*)(rp + K / 2))[b * 8 + sb];
        const uint16_t* dp = (const uint16_t*)(rp + K / 2 + K / 8) + b * 8 + sb; const float d = h2f(dp[0]), m = h2f(dp[K / 32]);
        const uint4 q4 = ldw((const uint4*)qs); const uint32_t qw[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
        for (int w = 0; w < 4; ++w)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int l = 4 * w + e; const uint32_t byte = (qw[w] >> (8 * e)) & 0xFF;
                v[l] = (float)((byte & 0xF) | (((hb >> l) & 1u) << 4)) * d + m;
                v[16 + l] = (float)((byte >> 4) | (((hb >> (16 + l)) & 1u) << 4)) * d + m;
            }
    } else {   // Q8_0 SoA
        const int8_t* q = (const int8_t*)rp + (size_t)b * 256 + sb * 32; const float d = h2f(((const uint16_t*)(rp + K))[b * 8 + sb]);
        const int32x4_t a = *(const int32x4_t*)q, c = *(const int32x4_t*)(q + 16); const int8_t* qa = (const int8_t*)&a; const int8_t* qc = (const int8_t*)&c;
#pragma unroll
        for (int l = 0; l < 16; ++l) { v[l] = (float)qa[l] * d; v[16 + l] = (float)qc[l] * d; }
    }
#pragma unroll
    for (int l = 0; l < 16; ++l) { lo[l] = f2bfs(v[l]); hi[l] = f2bfs(v[16 + l]); }
}
template <WFmt F, int CT>
__global__ void __launch_bounds__(256) k_moe_gemm_bf16(const uint8_t* __restrict__ W, size_t ebytes, const MoeTile* __restrict__ tiles,
                                                       const int* __restrict__ rowtab, const uint16_t* __restrict__ xb, float* __restrict__ y, int N, int K, int fake) {
    __shared__ __align__(16) MoeLdsB<CT> L;
    const int nblkN = (N + 127) / 128;
    const MoeTile tile = tiles[blockIdx.x / nblkN];
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, r16 = lane % 16, half = lane / 16, tid = threadIdx.x;
    const int row_real = (blockIdx.x % nblkN) * 128 + wave * 16 + r16, row = min(row_real, N - 1);
    const int nb = (K + 255) / 256, nsb = K / 32;
    const uint8_t* Wt = tile.expert >= 0 ? W + (size_t)tile.expert * ebytes : W;
    const uint8_t* rp = row_ptr<F>(Wt, row, K);
    f32x8_t C[CT];
#pragma unroll
    for (int ct = 0; ct < CT; ++ct) C[ct] = f32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
    for (int b = 0; b < nb; ++b) {
        __syncthreads();
        for (int i = tid; i < 16 * CT * 32; i += 256) {   // 16*CT columns x 32 uint4 (512 B of bf16) per 256-block
            const int c = i / 32, part = i % 32;
            const int k = tile.k0 + (c < tile.ncols ? c : 0), xrow = rowtab ? rowtab[k] : k;
            const int kk = b * 256 + part * 8;
            uint4 v = {0, 0, 0, 0};
            if (kk < K) v = *(const uint4*)(xb + (size_t)xrow * K + kk);
            *(uint4*)&L.x[c][part * 8] = v;
        }
        __syncthreads();
        const int nsb_left = nsb - b * 8;
        for (int sb = 0; sb < 8; ++sb) {
            if (sb >= nsb_left) break;
            bf16x16_t Alo, Ahi;
            if (fake) { const short v = (short)(0x3F80 + sb + r16); for (int l = 0; l < 16; ++l) { Alo[l] = v; Ahi[l] = v; } } else moe_wfrag<F>(rp, b, sb, K, Alo, Ahi);
#pragma unroll
            for (int ct = 0; ct < CT; ++ct) {
                const bf16x16_t B0 = *(const bf16x16_t*)&L.x[16 * ct + r16][sb * 32], B1 = *(const bf16x16_t*)&L.x[16 * ct + r16][sb * 32 + 16];
                C[ct] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(B0, Alo, C[ct]);
                C[ct] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(B1, Ahi, C[ct]);
            }
        }
    }
    if (row_real < N) {
#pragma unroll
        for (int ct = 0; ct < CT; ++ct)
#pragma unroll
            for (int l = 0; l < 8; ++l) { const int c = 16 * ct + 2 * l + half; if (c < tile.ncols) y[(size_t)(tile.k0 + c) * N + row] = C[ct][l]; }
    }
}
template <WFmt F> static void launch_moe_gemm_bf16(const void* W, size_t ebytes, const MoeTile* tiles, int ntiles, int ct, const int* rowtab, const uint16_t* xb, float* y, int N, int K, hipStream_t s) {
    if (ntiles <= 0) return;
    const int grid = ntiles * ((N + 127) / 128);
    if (ct == 1) k_moe_gemm_bf16<F, 1><<<grid, 256, 0, s>>>((const uint8_t*)W, ebytes, tiles, rowtab, xb, y, N, K, g_moe_fake);
    else if (ct == 2) k_moe_gemm_bf16<F, 2><<<grid, 256, 0, s>>>((const uint8_t*)W, ebytes, tiles, rowtab, xb, y, N, K, g_moe_fake);
    else k_moe_gemm_bf16<F, 3><<<grid, 256, 0, s>>>((const uint8_t*)W, ebytes, tiles, rowtab, xb, y, N, K, g_moe_fake);
}
static void moe_gemm_bf16_fmt(WFmt f, const void* W, size_t ebytes, const MoeTile* tiles, int ntiles, int ct, const int* rowtab, const uint16_t* xb, float* y, int N, int K, hipStream_t s) {
    switch (f) {
        case WFmt::Q4_K: launch_moe_gemm_bf16<WFmt::Q4_K>(W, ebytes, tiles, ntiles, ct, rowtab, xb, y, N, K, s); break;
        case WFmt::Q5_K: launch_moe_gemm_bf16<WFmt::Q5_K>(W, ebytes, tiles, ntiles, ct, rowtab, xb, y, N, K, s); break;
        case WFmt::Q5_1_SOA: launch_moe_gemm_bf16<WFmt::Q5_1_SOA>(W, ebytes, tiles, ntiles, ct, rowtab, xb, y, N, K, s); break;
        case WFmt::Q8_0_SOA: launch_moe_gemm_bf16<WFmt::Q8_0_SOA>(W, ebytes, tiles, ntiles, ct, rowtab, xb, y, N, K, s); break;
        default: fprintf(stderr, "moe_gemm_bf16: unsupported expert format\n"); abort();
    }
}

// ======================= multi-segment GEMV (ncol = 1) =======================
struct SegDev { const uint8_t* w; float* y; int N, K, fmt, block0; };
struct SegArgs { SegDev s[4]; int n; };

template <WFmt F, int TPR>
__device__ __forceinline__ float gemv_row_1(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                            const float* __restrict__ xs, int K, int row, int lane) {
    const int nb = (K + 255) / 256;
    const uint8_t* rp = row_ptr<F>(W, row, K);
    float acc[1] = {0.f};
    for (int b = lane; b < nb; b += TPR) {
        const uint32_t* xw[1] = {(const uint32_t*)(xq + b * 256)}; const float* cxd[1] = {xd + b * 8}; const float* cxs[1] = {xs + b * 8};
        dispatch_block<F, 1>(rp, b, K, xw, cxd, cxs, acc);
    }
#pragma unroll
    for (int off = TPR / 2; off > 0; off >>= 1) acc[0] += __shfl_xor(acc[0], off, TPR);
    return acc[0];
}

template <WFmt F, int TPR, int NCOL>
__device__ __forceinline__ void gemv_row_n(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                           const float* __restrict__ xs, int K, int row, int lane, float* acc) {
    const int nb = (K + 255) / 256, nsb = K / 32;
    const uint8_t* rp = row_ptr<F>(W, row, K);
#pragma unroll
    for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;
    for (int b = lane; b < nb; b += TPR) {
        const uint32_t* xw[NCOL]; const float* cxd[NCOL]; const float* cxs[NCOL];
#pragma unroll
        for (int c = 0; c < NCOL; ++c) { xw[c] = (const uint32_t*)(xq + (size_t)c * K + b * 256); cxd[c] = xd + (size_t)c * nsb + b * 8; cxs[c] = xs + (size_t)c * nsb + b * 8; }
        dispatch_block<F, NCOL>(rp, b, K, xw, cxd, cxs, acc);
    }
#pragma unroll
    for (int c = 0; c < NCOL; ++c) {
#pragma unroll
        for (int off = TPR / 2; off > 0; off >>= 1) acc[c] += __shfl_xor(acc[c], off, TPR);
    }
}
template <int TPR, int NCOL>
__global__ void __launch_bounds__(256) k_gemv_multi(SegArgs a, const int8_t* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ xs) {
    constexpr int ROWS = 256 / TPR;
    const int nblk = a.s[a.n - 1].block0 + (a.s[a.n - 1].N + ROWS - 1) / ROWS;
    for (int vb = blockIdx.x; vb < nblk; vb += gridDim.x) {
    int si = 0;
#pragma unroll
    for (int i = 1; i < 4; ++i) if (i < a.n && vb >= a.s[i].block0) si = i;
    const SegDev sg = a.s[si];
    const int lane = threadIdx.x % TPR, row = (vb - sg.block0) * ROWS + threadIdx.x / TPR;
    if (row >= sg.N) continue;
    float v[NCOL];
    switch (sg.fmt) {
        case (int)WFmt::Q4_K:     gemv_row_n<WFmt::Q4_K, TPR, NCOL>(sg.w, xq, xd, xs, sg.K, row, lane, v); break;
        case (int)WFmt::Q5_K:     gemv_row_n<WFmt::Q5_K, TPR, NCOL>(sg.w, xq, xd, xs, sg.K, row, lane, v); break;
        case (int)WFmt::IQ4_XS:   gemv_row_n<WFmt::IQ4_XS, TPR, NCOL>(sg.w, xq, xd, xs, sg.K, row, lane, v); break;
        case (int)WFmt::Q6_K_SOA: gemv_row_n<WFmt::Q6_K_SOA, TPR, NCOL>(sg.w, xq, xd, xs, sg.K, row, lane, v); break;
        case (int)WFmt::Q5_1_SOA: gemv_row_n<WFmt::Q5_1_SOA, TPR, NCOL>(sg.w, xq, xd, xs, sg.K, row, lane, v); break;
        default:                  gemv_row_n<WFmt::Q8_0_SOA, TPR, NCOL>(sg.w, xq, xd, xs, sg.K, row, lane, v); break;
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < NCOL; ++c) sg.y[(size_t)c * sg.N + row] = v[c];
    }
    }
}

// ======================= MoE expert GEMV: y[(t,e)][N] = W[ids[t*nexp+e]] . x[xrow(t,e)] =======================
// xrow: XROW_T -> x row t (gate/up: input is the token), XROW_TE -> x row t*nexp+e (down: input is per (t,e))
// NSEG weight tensors (e.g. gate|up) of identical shape: segment g uses W[g], writes y[g]. ids[te] < 0 selects a
// "shared" slot: the weights at shared[g] (a single expert-shaped matrix), x row t.
// FS: format of the shared-expert matrices (Q8_0 in the UD-Q4_K_XL checkpoint while the routed experts are Q4_K / Q5_1).
// a.slotw != null (down-projection): y[t][row] += slotw[te] * dot, atomically (the MoE combine; y zeroed by the caller).
template <WFmt F, WFmt FS, int TPR, bool XROW_TE>
__global__ void __launch_bounds__(256) k_gemv_moe(MoeSegs a, int nseg, const int* __restrict__ ids, int nexp,
                                                  const int8_t* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ xs, int N, int K) {
    constexpr int ROWS = 256 / TPR;
    const int nrb = (N + ROWS - 1) / ROWS;
    const int bx = blockIdx.x % (nrb * nseg), te = blockIdx.x / (nrb * nseg);   // 1D grid, te-major
    const int seg = bx / nrb, rb = bx % nrb;
    const int t = te / nexp;
    const int lane = threadIdx.x % TPR, row = rb * ROWS + threadIdx.x / TPR;
    if (row >= N) return;
    const int id = ids[te];
    const int xr = XROW_TE ? te : t, nsb = K / 32;
    const int8_t* xqr = xq + (size_t)xr * K; const float* xdr = xd + (size_t)xr * nsb; const float* xsr = xs + (size_t)xr * nsb;
    float v;
    if (id >= 0) v = gemv_row_1<F, TPR>((const uint8_t*)a.w[seg] + (size_t)id * a.ebytes, xqr, xdr, xsr, K, row, lane);
    else v = gemv_row_1<FS, TPR>((const uint8_t*)a.shared[seg], xqr, xdr, xsr, K, row, lane);
    if (lane == 0) a.y[seg][(size_t)te * N + row] = v;
}
// Down-projection with the combine: y[t][row] = sum_e slotw[t*nexp+e] * W[ids[t*nexp+e]] . x[t*nexp+e], accumulated in
// slot order by the same thread (no atomics: deterministic and T-invariant). Grid = T x row-blocks.
template <WFmt F, WFmt FS, int TPR>
__global__ void __launch_bounds__(256) k_gemv_moe_down(MoeSegs a, const int* __restrict__ ids, int nexp,
                                                       const int8_t* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ xs, int N, int K) {
    constexpr int ROWS = 256 / TPR;
    const int nrb = (N + ROWS - 1) / ROWS;
    const int rb = blockIdx.x % nrb, t = blockIdx.x / nrb;
    const int lane = threadIdx.x % TPR, row = rb * ROWS + threadIdx.x / TPR;
    if (row >= N) return;
    const int nsb = K / 32;
    float acc = 0.f;
    for (int e = 0; e < nexp; ++e) {
        const int te = t * nexp + e, id = ids[te];
        const int8_t* xqr = xq + (size_t)te * K; const float* xdr = xd + (size_t)te * nsb; const float* xsr = xs + (size_t)te * nsb;
        float v;
        if (id >= 0) v = gemv_row_1<F, TPR>((const uint8_t*)a.w[0] + (size_t)id * a.ebytes, xqr, xdr, xsr, K, row, lane);
        else v = gemv_row_1<FS, TPR>((const uint8_t*)a.shared[0], xqr, xdr, xsr, K, row, lane);
        acc += a.slotw[te] * v;
    }
    if (lane == 0) a.y[0][(size_t)t * N + row] = acc;
}
template <WFmt F, WFmt FS, int TPR, bool XT> static void launch_moe2(const MoeSegs& a, int nseg, const int* ids, int nexp, XQ8 x, int N, int K, int T, hipStream_t s) {
    constexpr int ROWS = 256 / TPR;
    if (a.slotw) { k_gemv_moe_down<F, FS, TPR><<<((N + ROWS - 1) / ROWS) * T, 256, 0, s>>>(a, ids, nexp, x.q, x.d, x.s, N, K); return; }
    const int grid = nseg * ((N + ROWS - 1) / ROWS) * T * nexp;
    k_gemv_moe<F, FS, TPR, XT><<<grid, 256, 0, s>>>(a, nseg, ids, nexp, x.q, x.d, x.s, N, K);
}
template <WFmt F, WFmt FS> static void launch_moe(const MoeSegs& a, int nseg, const int* ids, int nexp, XQ8 x, int N, int K, int T, bool xrow_te, hipStream_t s) {
    // bench_gemv on the Flash-Next expert tensors (docs/decode-flash-next.md): K=640 Q5_1 -> 8 lanes/row (224 GB/s;
    // 4 lanes 207, 32 lanes 107); K=2560 Q4_K -> 32 (233; all of 8..32 within 2%).
    int tpr = K / 256 < 8 ? 16 : 32;   // gather bench: K=640 16 lanes 152 GB/s, 8 lanes 144, 4 lanes 97, 32 lanes 95
    if (g_moe_tpr) tpr = g_moe_tpr;
    #define LM(TP) if (xrow_te) launch_moe2<F, FS, TP, true>(a, nseg, ids, nexp, x, N, K, T, s); else launch_moe2<F, FS, TP, false>(a, nseg, ids, nexp, x, N, K, T, s)
    switch (tpr) { case 1: LM(1); break; case 2: LM(2); break; case 4: LM(4); break; case 8: LM(8); break; case 16: LM(16); break; default: LM(32); break; }
    #undef LM
}

// ======================= interleaved layout (experiment) =======================
// Layout: rows grouped by 8; within a group blocks are stored [b][r%8] so that the 32 lanes of a wave
// read 32 consecutive blocks (4 blocks x 8 rows). Lane l -> row l%8, blocks it*4 + l/8.
template <WFmt F> struct IlFmt;   // block-pointer decoders (GGUF block layouts only)
template <> struct IlFmt<WFmt::Q4_K> { static constexpr int BB = 144; };
template <> struct IlFmt<WFmt::Q5_K> { static constexpr int BB = 176; };
template <> struct IlFmt<WFmt::IQ4_XS> { static constexpr int BB = 136; };

template <WFmt F, int NCOL>
__device__ __forceinline__ void il_block(const uint8_t* blk, const uint32_t* const* xw, const float* const* xd, const float* const* xs, float* acc) {
    // reuse the row decoders by pointing them at the block with b = 0
    if constexpr (F == WFmt::Q4_K) FmtQ4K::block<NCOL>(blk, 0, xw, xd, xs, acc);
    else if constexpr (F == WFmt::Q5_K) FmtQ5K::block<NCOL>(blk, 0, xw, xd, xs, acc);
    else FmtIQ4XS::block<NCOL>(blk, 0, xw, xd, xs, acc);
}

template <WFmt F, int NCOL>
__global__ void __launch_bounds__(256) k_gemv_il(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                 const float* __restrict__ xs, float* __restrict__ y, int N, int K) {
    constexpr int BB = IlFmt<F>::BB;
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int group = blockIdx.x * 8 + wave;            // 8 groups (64 rows) per block
    const int row = group * 8 + lane % 8;
    const int nb = K / 256, nsb = K / 32;
    if (group * 8 >= N) return;
    const uint8_t* gb = W + (size_t)group * 8 * nb * BB;
    float acc[NCOL];
#pragma unroll
    for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;
    for (int b = lane / 8; b < nb; b += 4) {
        const uint8_t* blk = gb + ((size_t)b * 8 + lane % 8) * BB;
        const uint32_t* xw[NCOL]; const float* cxd[NCOL]; const float* cxs[NCOL];
#pragma unroll
        for (int c = 0; c < NCOL; ++c) { xw[c] = (const uint32_t*)(xq + (size_t)c * K + b * 256); cxd[c] = xd + (size_t)c * nsb + b * 8; cxs[c] = xs + (size_t)c * nsb + b * 8; }
        il_block<F, NCOL>(blk, xw, cxd, cxs, acc);
    }
    // reduce over the 4 lanes sharing a row: lanes l, l+8, l+16, l+24
#pragma unroll
    for (int c = 0; c < NCOL; ++c) { acc[c] += __shfl_xor(acc[c], 8, 32); acc[c] += __shfl_xor(acc[c], 16, 32); }
    if (lane < 8 && row < N) {
#pragma unroll
        for (int c = 0; c < NCOL; ++c) y[(size_t)c * N + row] = acc[c];
    }
}

// ---------------- Q4_K x f32 (baseline; kept for the record) ----------------
template <int TPR, int NCOL>
__global__ void __launch_bounds__(256) k_gemv_q4_K_f32(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int K) {
    constexpr int ROWS = 256 / TPR;
    const int tid = threadIdx.x, lane = tid % TPR, r = tid / TPR;
    const int row = blockIdx.x * ROWS + r;
    if (row >= N) return;
    const int nb = K / 256;
    const uint8_t* wrow = W + (size_t)row * nb * 144;
    float acc[NCOL];
#pragma unroll
    for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;
    for (int b = lane; b < nb; b += TPR) {
        const uint8_t* blk = wrow + (size_t)b * 144;
        uint4 v[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) v[i] = ldw((const uint4*)blk + i);
        const uint16_t* hdr = (const uint16_t*)&v[0];
        const float d = h2f(hdr[0]), dmin = h2f(hdr[1]);
        const uint8_t* scales = (const uint8_t*)&v[0] + 4;
        const uint8_t* qs = (const uint8_t*)&v[1];
        const float* xb = x + b * 256;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t sc0, mn0, sc1, mn1; q4k_scales(scales, 2 * j, sc0, mn0, sc1, mn1);
            const float d0 = d * sc0, m0 = dmin * mn0, d1 = d * sc1, m1 = dmin * mn1;
            const uint8_t* q = qs + 32 * j;
#pragma unroll
            for (int c = 0; c < NCOL; ++c) {
                const float* xc = xb + (size_t)c * K + 64 * j;
                float s0 = 0.f, s1 = 0.f, sx0 = 0.f, sx1 = 0.f;
#pragma unroll
                for (int l = 0; l < 32; ++l) { const float x0 = xc[l], x1 = xc[32 + l]; s0 += x0 * (q[l] & 0xF); s1 += x1 * (q[l] >> 4); sx0 += x0; sx1 += x1; }
                acc[c] += d0 * s0 - m0 * sx0 + d1 * s1 - m1 * sx1;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NCOL; ++c) { for (int off = TPR / 2; off > 0; off >>= 1) acc[c] += __shfl_xor(acc[c], off, TPR); }
    if (lane == 0) { for (int c = 0; c < NCOL; ++c) y[(size_t)c * N + row] = acc[c]; }
}

#define NCOL_SWITCH(KERNEL, ...) \
    switch (ncol) { \
        case 1: KERNEL<1><<<grid, 256, 0, s>>>(__VA_ARGS__); break; case 2: KERNEL<2><<<grid, 256, 0, s>>>(__VA_ARGS__); break; \
        case 3: KERNEL<3><<<grid, 256, 0, s>>>(__VA_ARGS__); break; case 4: KERNEL<4><<<grid, 256, 0, s>>>(__VA_ARGS__); break; \
        case 5: KERNEL<5><<<grid, 256, 0, s>>>(__VA_ARGS__); break; case 6: KERNEL<6><<<grid, 256, 0, s>>>(__VA_ARGS__); break; \
        case 7: KERNEL<7><<<grid, 256, 0, s>>>(__VA_ARGS__); break; case 8: KERNEL<8><<<grid, 256, 0, s>>>(__VA_ARGS__); break; }

template <WFmt F, int TPR> void launch_fmt(const void* W, XQ8 x, float* y, int N, int K, int ncol, bool lds, hipStream_t s) {
    constexpr int ROWS = 256 / TPR;
    int grid = (N + ROWS - 1) / ROWS;
    if (!lds && g_persist_blocks > 0 && grid > g_persist_blocks) grid = g_persist_blocks;
    #define KD(NC) k_gemv_q8<F, TPR, NC><<<grid, 256, 0, s>>>((const uint8_t*)W, x.q, x.d, x.s, y, N, K)
    #define KL(NC) k_gemv_q8_lds<F, TPR, NC><<<grid, 256, 0, s>>>((const uint8_t*)W, x.q, x.d, x.s, y, N, K)
    if constexpr (TPR <= 8) { if (lds) { switch (ncol) { case 1: KL(1); break; case 2: KL(2); break; case 3: KL(3); break; case 4: KL(4); break; case 5: KL(5); break; case 6: KL(6); break; case 7: KL(7); break; case 8: KL(8); break; } return; } }
    { switch (ncol) { case 1: KD(1); break; case 2: KD(2); break; case 3: KD(3); break; case 4: KD(4); break; case 5: KD(5); break; case 6: KD(6); break; case 7: KD(7); break; case 8: KD(8); break; } }
    #undef KD
    #undef KL
}
template <WFmt F> void launch_tpr(const void* W, XQ8 x, float* y, int N, int K, int ncol, int tpr, int rpt, hipStream_t s) {
    const bool lds = rpt == 3;
    if (lds && tpr > 8) tpr = 8;
    switch (tpr) { case 1: launch_fmt<F, 1>(W, x, y, N, K, ncol, lds, s); break; case 2: launch_fmt<F, 2>(W, x, y, N, K, ncol, lds, s); break;
                   case 4: launch_fmt<F, 4>(W, x, y, N, K, ncol, lds, s); break; case 8: launch_fmt<F, 8>(W, x, y, N, K, ncol, lds, s); break;
                   case 16: launch_fmt<F, 16>(W, x, y, N, K, ncol, lds, s); break; default: launch_fmt<F, 32>(W, x, y, N, K, ncol, lds, s); break; }
}
template <int TPR> void launch_f32(const void* W, const float* x, float* y, int N, int K, int ncol, hipStream_t s) {
    const int grid = (N + 256 / TPR - 1) / (256 / TPR);
    #define KF(NC) k_gemv_q4_K_f32<TPR, NC>
    switch (ncol) {
        case 1: KF(1)<<<grid, 256, 0, s>>>((const uint8_t*)W, x, y, N, K); break; case 2: KF(2)<<<grid, 256, 0, s>>>((const uint8_t*)W, x, y, N, K); break;
        case 3: KF(3)<<<grid, 256, 0, s>>>((const uint8_t*)W, x, y, N, K); break; case 4: KF(4)<<<grid, 256, 0, s>>>((const uint8_t*)W, x, y, N, K); break;
        default: KF(8)<<<grid, 256, 0, s>>>((const uint8_t*)W, x, y, N, K); break;
    }
    #undef KF
}
}  // namespace

void quantize_x_q8(const float* x, XQ8 out, int ncol, int K, hipStream_t s) {
    dim3 grid((K / 32 + 7) / 8, ncol);
    k_quantize_x_q8<<<grid, 256, 0, s>>>(x, out.q, out.d, out.s, K);
}
size_t wfmt_row_bytes(WFmt f, int K) {
    switch (f) { case WFmt::Q4_K: return (size_t)K / 256 * 144; case WFmt::Q5_K: return (size_t)K / 256 * 176; case WFmt::IQ4_XS: return (size_t)K / 256 * 136;
                 case WFmt::Q6_K_SOA: return (size_t)K * 210 / 256; case WFmt::Q5_1_SOA: return (size_t)K * 24 / 32; default: return (size_t)K + K / 16; }
}
void repack_row_q6_K(const uint8_t* src, uint8_t* dst, int K) {
    const int nb = K / 256;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = src + (size_t)b * 210;
        memcpy(dst + (size_t)b * 128, blk, 128);
        memcpy(dst + (size_t)nb * 128 + (size_t)b * 64, blk + 128, 64);
        memcpy(dst + (size_t)nb * 192 + (size_t)b * 16, blk + 192, 16);
        memcpy(dst + (size_t)nb * 208 + (size_t)b * 2, blk + 208, 2);
    }
}
void repack_row_q8_0(const uint8_t* src, uint8_t* dst, int K) {
    const int nb = K / 32;
    for (int b = 0; b < nb; ++b) { memcpy(dst + (size_t)K + b * 2, src + (size_t)b * 34, 2); memcpy(dst + (size_t)b * 32, src + (size_t)b * 34 + 2, 32); }
}
void gemv_multi(const GemvSeg* segs, int nseg, XQ8 x, int ncol, hipStream_t s) {
    constexpr int TPR = 32, ROWS = 256 / TPR;
    for (int c0 = 0; c0 < ncol; c0 += 8) {   // up to 8 columns per launch (all segments share the same x rows)
        const int nc = std::min(8, ncol - c0);
        SegArgs a; a.n = nseg; int blocks = 0;
        for (int i = 0; i < nseg; ++i) { a.s[i] = {(const uint8_t*)segs[i].w, segs[i].y + (size_t)c0 * segs[i].N, segs[i].N, segs[i].K, (int)segs[i].fmt, blocks}; blocks += (segs[i].N + ROWS - 1) / ROWS; }
        for (int i = nseg; i < 4; ++i) a.s[i] = a.s[nseg - 1];
        if (g_persist_blocks > 0 && blocks > g_persist_blocks) blocks = g_persist_blocks;
        const int K = segs[0].K;   // all segments share K (same input)
        const int8_t* xq = x.q + (size_t)c0 * K; const float* xd = x.d + (size_t)c0 * K / 32; const float* xs = x.s + (size_t)c0 * K / 32;
        #define KM(NC) k_gemv_multi<TPR, NC><<<blocks, 256, 0, s>>>(a, xq, xd, xs)
        switch (nc) { case 1: KM(1); break; case 2: KM(2); break; case 3: KM(3); break; case 4: KM(4); break; case 5: KM(5); break; case 6: KM(6); break; case 7: KM(7); break; default: KM(8); break; }
        #undef KM
    }
}
// experiment: interleaved layout (rpt == 5). Weights must have been repacked with repack_interleave8.
template <WFmt F> static void launch_il(const void* W, XQ8 x, float* y, int N, int K, int ncol, hipStream_t s) {
    const int grid = (N + 63) / 64;
    #define KI(NC) k_gemv_il<F, NC><<<grid, 256, 0, s>>>((const uint8_t*)W, x.q, x.d, x.s, y, N, K)
    switch (ncol) { case 1: KI(1); break; case 2: KI(2); break; case 3: KI(3); break; case 4: KI(4); break; default: KI(8); break; }
    #undef KI
}
void moe_gemm(WFmt f, WFmt fs, const void* W, const void* Wsh, size_t ebytes, const MoeTile* tiles, int ntiles, const MoeTile* stiles, int nstiles, int ct,
              const int* rowtab, XQ8 x, float* y, int N, int K, hipStream_t s) {
    moe_gemm_fmt(f, W, ebytes, tiles, ntiles, ct, rowtab, x, y, N, K, s);          // routed experts
    moe_gemm_fmt(fs, Wsh, 0, stiles, nstiles, ct, rowtab, x, y, N, K, s);          // the shared expert (its tiles carry expert = -1)
}
void moe_gemm_bf16(WFmt f, WFmt fs, const void* W, const void* Wsh, size_t ebytes, const MoeTile* tiles, int ntiles, const MoeTile* stiles, int nstiles, int ct,
                   const int* rowtab, const uint16_t* xb, float* y, int N, int K, hipStream_t s) {
    moe_gemm_bf16_fmt(f, W, ebytes, tiles, ntiles, ct, rowtab, xb, y, N, K, s);
    moe_gemm_bf16_fmt(fs, Wsh, 0, stiles, nstiles, ct, rowtab, xb, y, N, K, s);
}
void gemv_moe(WFmt f, WFmt fs, const MoeSegs& a, int nseg, const int* ids, int nexp, XQ8 x, int N, int K, int T, bool xrow_te, hipStream_t s) {
    if (fs != WFmt::Q8_0_SOA && fs != f) { fprintf(stderr, "gemv_moe: unsupported shared-expert format\n"); abort(); }
    const bool s8 = fs == WFmt::Q8_0_SOA;
    switch (f) {
        case WFmt::Q4_K: if (s8) launch_moe<WFmt::Q4_K, WFmt::Q8_0_SOA>(a, nseg, ids, nexp, x, N, K, T, xrow_te, s); else launch_moe<WFmt::Q4_K, WFmt::Q4_K>(a, nseg, ids, nexp, x, N, K, T, xrow_te, s); break;
        case WFmt::Q5_1_SOA: if (s8) launch_moe<WFmt::Q5_1_SOA, WFmt::Q8_0_SOA>(a, nseg, ids, nexp, x, N, K, T, xrow_te, s); else launch_moe<WFmt::Q5_1_SOA, WFmt::Q5_1_SOA>(a, nseg, ids, nexp, x, N, K, T, xrow_te, s); break;
        case WFmt::Q8_0_SOA: launch_moe<WFmt::Q8_0_SOA, WFmt::Q8_0_SOA>(a, nseg, ids, nexp, x, N, K, T, xrow_te, s); break;
        default: if (s8) launch_moe<WFmt::Q5_K, WFmt::Q8_0_SOA>(a, nseg, ids, nexp, x, N, K, T, xrow_te, s); else launch_moe<WFmt::Q5_K, WFmt::Q5_K>(a, nseg, ids, nexp, x, N, K, T, xrow_te, s); break;
    }
}
// split-K GEMV for skinny weights (few rows, long K): grid = row-blocks x ksplit; NCOL columns share one weight read;
// partial sums written to part[(c*ksplit + ks)][N] (summed in a fixed order by the consumer: deterministic, T-invariant)
template <WFmt F, int TPR, int NCOL>
__global__ void __launch_bounds__(256) k_gemv_splitk(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ xs,
                                                     float* __restrict__ part, int N, int K, int ksplit) {
    constexpr int ROWS = 256 / TPR;
    const int nrb = (N + ROWS - 1) / ROWS, rb = blockIdx.x % nrb, ks = blockIdx.x / nrb;
    const int lane = threadIdx.x % TPR, row = rb * ROWS + threadIdx.x / TPR;
    if (row >= N) return;
    const int nb = (K + 255) / 256, nsb = K / 32, b0 = ks * nb / ksplit, b1 = (ks + 1) * nb / ksplit;
    const uint8_t* rp = row_ptr<F>(W, row, K);
    float acc[NCOL];
#pragma unroll
    for (int c = 0; c < NCOL; ++c) acc[c] = 0.f;
    for (int b = b0 + lane; b < b1; b += TPR) {
        const uint32_t* xw[NCOL]; const float* cxd[NCOL]; const float* cxs[NCOL];
#pragma unroll
        for (int c = 0; c < NCOL; ++c) { xw[c] = (const uint32_t*)(xq + (size_t)c * K + b * 256); cxd[c] = xd + (size_t)c * nsb + b * 8; cxs[c] = xs + (size_t)c * nsb + b * 8; }
        dispatch_block<F, NCOL>(rp, b, K, xw, cxd, cxs, acc);
    }
#pragma unroll
    for (int c = 0; c < NCOL; ++c) {
#pragma unroll
        for (int off = TPR / 2; off > 0; off >>= 1) acc[c] += __shfl_xor(acc[c], off, TPR);
        if (lane == 0) part[((size_t)c * ksplit + ks) * N + row] = acc[c];
    }
}
template <WFmt F> static void launch_splitk(const void* W, XQ8 x, float* part, int N, int K, int ksplit, int ncol, hipStream_t s) {
    const int grid = ((N + 7) / 8) * ksplit;   // 8 rows per block
    for (int c0 = 0; c0 < ncol; c0 += 8) {
        const int nc = std::min(8, ncol - c0);
        const int8_t* xq = x.q + (size_t)c0 * K; const float* xd = x.d + (size_t)c0 * K / 32; const float* xs = x.s + (size_t)c0 * K / 32;
        float* p = part + (size_t)c0 * ksplit * N;
        #define KS(NC) k_gemv_splitk<F, 32, NC><<<grid, 256, 0, s>>>((const uint8_t*)W, xq, xd, xs, p, N, K, ksplit)
        switch (nc) { case 1: KS(1); break; case 2: KS(2); break; case 3: KS(3); break; case 4: KS(4); break; case 5: KS(5); break; case 6: KS(6); break; case 7: KS(7); break; default: KS(8); break; }
        #undef KS
    }
}
void gemv_splitk(WFmt f, const void* W, XQ8 x, float* part, int N, int K, int ksplit, int ncol, hipStream_t s) {
    switch (f) {
        case WFmt::Q8_0_SOA: launch_splitk<WFmt::Q8_0_SOA>(W, x, part, N, K, ksplit, ncol, s); break;
        case WFmt::Q4_K: launch_splitk<WFmt::Q4_K>(W, x, part, N, K, ksplit, ncol, s); break;
        default: launch_splitk<WFmt::Q6_K_SOA>(W, x, part, N, K, ksplit, ncol, s); break;
    }
}
void repack_interleave8(WFmt f, const uint8_t* src, uint8_t* dst, int N, int K) {
    const int bb = f == WFmt::Q4_K ? 144 : f == WFmt::Q5_K ? 176 : 136, nb = K / 256;
    for (int r = 0; r < N; ++r) for (int b = 0; b < nb; ++b)
        memcpy(dst + (((size_t)(r / 8) * 8 * nb) + (size_t)b * 8 + r % 8) * bb, src + ((size_t)r * nb + b) * bb, bb);
}
// Q5_0 (d, qh, qs; x = d*(q-16)) is exactly a Q5_1 block with m = -16*d (a power-of-two scale of an f16 is exact).
void repack_row_q5_0(const uint8_t* src, uint8_t* dst, int K) {
    const int nb = K / 32;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = src + (size_t)b * 22;   // d(2) qh(4) qs(16)
        uint16_t dbits; memcpy(&dbits, blk, 2); _Float16 d; memcpy(&d, &dbits, 2);
        const _Float16 m = (_Float16)(-16.0f * (float)d); uint16_t mbits; memcpy(&mbits, &m, 2);
        memcpy(dst + (size_t)K / 2 + K / 8 + b * 2, &dbits, 2); memcpy(dst + (size_t)K / 2 + K / 8 + K / 16 + b * 2, &mbits, 2);
        memcpy(dst + (size_t)K / 2 + b * 4, blk + 2, 4); memcpy(dst + (size_t)b * 16, blk + 6, 16);
    }
}
void repack_row_q5_1(const uint8_t* src, uint8_t* dst, int K) {
    const int nb = K / 32;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = src + (size_t)b * 24;   // d(2) m(2) qh(4) qs(16)
        memcpy(dst + (size_t)K / 2 + K / 8 + b * 2, blk, 2); memcpy(dst + (size_t)K / 2 + K / 8 + K / 16 + b * 2, blk + 2, 2);
        memcpy(dst + (size_t)K / 2 + b * 4, blk + 4, 4); memcpy(dst + (size_t)b * 16, blk + 8, 16);
    }
}
void gemv_q8(WFmt f, const void* W, XQ8 x, float* y, int N, int K, int ncol, int tpr, int rpt, hipStream_t s) {
    if (rpt == 5) {
        if (f == WFmt::Q4_K) launch_il<WFmt::Q4_K>(W, x, y, N, K, ncol, s);
        else if (f == WFmt::Q5_K) launch_il<WFmt::Q5_K>(W, x, y, N, K, ncol, s);
        else launch_il<WFmt::IQ4_XS>(W, x, y, N, K, ncol, s);
        return;
    }
    if (rpt == 4) {   // WMMA path, Q4_K only for now; tpr reused as TILES (1/2)
        if (tpr == 2) k_gemv_q4_K_wmma<2><<<(N + 255) / 256, 256, 0, s>>>((const uint8_t*)W, x.q, x.d, x.s, y, N, K, ncol);
        else          k_gemv_q4_K_wmma<1><<<(N + 127) / 128, 256, 0, s>>>((const uint8_t*)W, x.q, x.d, x.s, y, N, K, ncol);
        return;
    }
    switch (f) {
        case WFmt::Q4_K: launch_tpr<WFmt::Q4_K>(W, x, y, N, K, ncol, tpr, rpt, s); break;
        case WFmt::Q5_K: launch_tpr<WFmt::Q5_K>(W, x, y, N, K, ncol, tpr, rpt, s); break;
        case WFmt::IQ4_XS: launch_tpr<WFmt::IQ4_XS>(W, x, y, N, K, ncol, tpr, rpt, s); break;
        case WFmt::Q6_K_SOA: launch_tpr<WFmt::Q6_K_SOA>(W, x, y, N, K, ncol, tpr, rpt, s); break;
        case WFmt::Q8_0_SOA: launch_tpr<WFmt::Q8_0_SOA>(W, x, y, N, K, ncol, tpr, rpt, s); break;
        case WFmt::Q5_1_SOA: launch_tpr<WFmt::Q5_1_SOA>(W, x, y, N, K, ncol, tpr, rpt, s); break;
    }
}
void gemv_q4_K_f32(const void* W, const float* x, float* y, int N, int K, int ncol, int tpr, hipStream_t s) {
    switch (tpr) { case 4: launch_f32<4>(W, x, y, N, K, ncol, s); break; case 8: launch_f32<8>(W, x, y, N, K, ncol, s); break;
                   case 16: launch_f32<16>(W, x, y, N, K, ncol, s); break; default: launch_f32<32>(W, x, y, N, K, ncol, s); break; }
}
}  // namespace hip
