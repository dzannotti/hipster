// Qwen3.8-Flash-Next QSA kernels (see qsa.h). Semantics follow the reference (llama.cpp franken branch,
// build_qsa_top_k / set_input_qsa): 4-token blocks at absolute positions, only complete blocks are pooled and
// scored, the incomplete tail is always attended, width = min(n_kv, 2048 + 3).
#include "qsa.h"
#include <cstdio>
#include <cstdlib>

namespace hip::qsa {
namespace {
constexpr int HD = 256, NQH = 24, NKV = 2, QPK = NQH / NKV;   // 12 q heads per kv head
__device__ __forceinline__ float h2f(uint16_t h) { _Float16 x; __builtin_memcpy(&x, &h, 2); return (float)x; }
__device__ __forceinline__ uint16_t f2h(float f) { _Float16 x = (_Float16)f; uint16_t h; __builtin_memcpy(&h, &x, 2); return h; }
__device__ __forceinline__ float bf16r(float f) { uint32_t u = __float_as_uint(f); u += 0x7FFFu + ((u >> 16) & 1); return __uint_as_float(u & 0xFFFF0000u); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float wave_sum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32); return v; }
__device__ __forceinline__ float wave_max(float v) { for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32)); return v; }
__device__ __forceinline__ float block_sum(float v, float* sh) {   // sh[8], blockDim 256
    v = wave_sum(v); __syncthreads(); if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v; __syncthreads();
    float r = 0.f; for (int i = 0; i < (int)(blockDim.x / 32); ++i) r += sh[i]; __syncthreads(); return r;
}
__device__ __forceinline__ void wave_quant32(float v, int8_t* __restrict__ q, float* __restrict__ d, float* __restrict__ s, int sb, int lane) {
    float amax = wave_max(fabsf(v));
    const float dd = amax / 127.f, id = amax > 0 ? 127.f / amax : 0.f;
    const int qi = __float2int_rn(v * id); int sum = qi;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
    q[sb * 32 + lane] = (int8_t)qi; if (lane == 0) { d[sb] = dd; s[sb] = dd * sum; }
}
// neox-style rope on the first 64 of 128 dims: pairs (i, i+32), theta = pos * base^(-2i/64); precise sincosf
__device__ __forceinline__ void rope64(float* v, int pos, float base, int i) {   // i < 32: rotates (v[i], v[i+32]) in place (caller holds both)
    const float theta = (float)pos * powf(base, -2.f * i / 64.f); float s, c; sincosf(theta, &s, &c);
    const float x0 = v[0], x1 = v[1]; v[0] = x0 * c - x1 * s; v[1] = x0 * s + x1 * c;
}

__global__ void k_bf16_gemv(const uint16_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int K, int T) {
    const int wave = (blockIdx.x * 256 + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const int t = wave / N, n = wave % N; if (t >= T) return;
    const uint16_t* w = W + (size_t)n * K; const float* xr = x + (size_t)t * K;
    float acc = 0.f;
    for (int i = lane * 4; i < K; i += 128) {
        const uint2 ww = *(const uint2*)(w + i); const float4 xx = *(const float4*)(xr + i);
        acc += __uint_as_float(ww.x << 16) * bf16r(xx.x) + __uint_as_float(ww.x & 0xFFFF0000u) * bf16r(xx.y)
             + __uint_as_float(ww.y << 16) * bf16r(xx.z) + __uint_as_float(ww.y & 0xFFFF0000u) * bf16r(xx.w);
    }
    acc = wave_sum(acc); if (lane == 0) y[(size_t)t * N + n] = acc;
}
__global__ void k_write_k(const float* __restrict__ k, const int* __restrict__ rpos, const int* __restrict__ rkv, uint16_t* __restrict__ cache, size_t slot_stride) {
    const int t = blockIdx.x, d = threadIdx.x;
    cache[(size_t)rkv[t] * slot_stride + (size_t)rpos[t] * IDX_DIM + d] = f2h(k[(size_t)t * IDX_DIM + d]);
}
// block per pooled block b: 128 threads = dims
__global__ void k_pool(const uint16_t* __restrict__ cache, const float* __restrict__ w_kn, float* __restrict__ pooled, int b0, float eps, float base) {
    __shared__ float sh[8]; __shared__ float sv[IDX_DIM];
    const int b = b0 + blockIdx.x, d = threadIdx.x;
    float m = 0.f;
    for (int i = 0; i < R; ++i) m += h2f(cache[((size_t)(R * b + i)) * IDX_DIM + d]);
    m *= 1.f / R;
    const float ss = block_sum(m * m, sh);
    float v = m * rsqrtf(ss / IDX_DIM + eps) * w_kn[d];
    sv[d] = v; __syncthreads();
    if (d < 32) { float pr[2] = {sv[d], sv[d + 32]}; rope64(pr, R * b, base, d); pooled[(size_t)b * IDX_DIM + d] = pr[0]; pooled[(size_t)b * IDX_DIM + d + 32] = pr[1]; }
    else if (d >= 64) pooled[(size_t)b * IDX_DIM + d] = v;
}
// block per (t, head): 128 threads
__global__ void k_query(const float* __restrict__ q_raw, const int* __restrict__ rpos, const float* __restrict__ w_qn, float* __restrict__ q, float eps, float base) {
    __shared__ float sh[8]; __shared__ float sv[IDX_DIM];
    const int t = blockIdx.x / IDX_HEADS, h = blockIdx.x % IDX_HEADS, d = threadIdx.x;
    const float x = q_raw[((size_t)t * IDX_HEADS + h) * IDX_DIM + d];
    const float ss = block_sum(x * x, sh);
    const float v = x * rsqrtf(ss / IDX_DIM + eps) * w_qn[d];
    sv[d] = v; __syncthreads();
    float* o = q + ((size_t)t * IDX_HEADS + h) * IDX_DIM;
    if (d < 32) { float pr[2] = {sv[d], sv[d + 32]}; rope64(pr, rpos[t], base, d); o[d] = pr[0]; o[d + 32] = pr[1]; }
    else if (d >= 64) o[d] = v;
}
// ---- selection: block per query row, 1024 threads ----
// exclusive prefix count of `flag` over the block's 1024 threads (+ carry): returns this thread's rank, total in *tot
__device__ __forceinline__ int block_rank(int flag, int* swarp, int* tot) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m = __ballot(flag);   // wave32 ballot
    const int inwave = __popc(m & ((1u << lane) - 1u)), wsum = __popc(m);
    if (lane == 0) swarp[wave] = wsum;
    __syncthreads();
    int before = 0; for (int w = 0; w < wave; ++w) before += swarp[w];
    int all = 0; for (int w = 0; w < 32; ++w) all += swarp[w];
    __syncthreads();
    *tot = all; return before + inwave;
}
__global__ void __launch_bounds__(1024) k_select(const float* __restrict__ q, const int* __restrict__ rpos, const int* __restrict__ rkv, const float* __restrict__ pooled, size_t pstride, int max_blocks,
                                                 float* __restrict__ scratch, int* __restrict__ idx, int* __restrict__ cnt) {
    __shared__ float sq[IDX_HEADS * IDX_DIM]; __shared__ int hist[256]; __shared__ int swarp[32]; __shared__ int scarry; __shared__ int sprefix, sK, sgt;
    __shared__ float smin; __shared__ int smin_b;
    const int t = blockIdx.x, tid = threadIdx.x, p = rpos[t], n = p + 1;
    int* out = idx + (size_t)t * WIDTH;
    if (n <= WIDTH) { for (int i = tid; i < n; i += 1024) out[i] = i; if (tid == 0) cnt[t] = n; return; }
    const int nvis = n / R, tail = n - R * nvis, want = WIDTH - tail, full = want / R, extra = want % R, K = full + (extra ? 1 : 0);
    for (int i = tid; i < IDX_HEADS * IDX_DIM; i += 1024) sq[i] = q[(size_t)t * IDX_HEADS * IDX_DIM + i];
    __syncthreads();
    // scores of the complete blocks
    const float* pb = pooled + (size_t)rkv[t] * pstride; float* sc = scratch + (size_t)t * max_blocks;
    for (int b = tid; b < nvis; b += 1024) {
        const float* kb = pb + (size_t)b * IDX_DIM; float s = 0.f;
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) { float d = 0.f; for (int i = 0; i < IDX_DIM; ++i) d += sq[h * IDX_DIM + i] * kb[i]; s += fmaxf(d, 0.f); }
        sc[b] = s;
    }
    __syncthreads();
    // radix select of the K-th largest score (keys = float bits, monotone for non-negative floats), MSB digit first
    if (tid == 0) { sprefix = 0; sK = K; sgt = 0; }
    __syncthreads();
    for (int pass = 3; pass >= 0; --pass) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const uint32_t prefix = (uint32_t)sprefix, mask = pass == 3 ? 0u : (0xFFFFFFFFu << (8 * (pass + 1)));
        for (int b = tid; b < nvis; b += 1024) { const uint32_t key = __float_as_uint(sc[b]); if ((key & mask) == prefix) atomicAdd(&hist[(key >> (8 * pass)) & 0xFF], 1); }
        __syncthreads();
        if (tid == 0) {   // walk digits from the top until the cumulative count reaches K
            int k = sK, gt = sgt, d = 255;
            for (; d > 0; --d) { if (hist[d] >= k) break; k -= hist[d]; gt += hist[d]; }
            sprefix = (int)(prefix | ((uint32_t)d << (8 * pass))); sK = k; sgt = gt;
        }
        __syncthreads();
    }
    const float thr = __uint_as_float((uint32_t)sprefix); const int eq_needed = sK;   // K - (#keys > thr)
    // ranks among the == thr blocks in ascending id, and among selected blocks; write full blocks in ascending order
    if (tid == 0) scarry = 0;
    __syncthreads();
    int eq_carry = 0, sel_carry = 0; float mn = INFINITY; int mnb = -1;
    for (int base = 0; base < nvis; base += 1024) {
        const int b = base + tid; const bool in = b < nvis; const float s = in ? sc[b] : -1.f;
        int tot;
        const int eq_rank = block_rank(in && s == thr, swarp, &tot) + eq_carry; eq_carry += tot;
        const bool sel = in && (s > thr || (s == thr && eq_rank < eq_needed));
        const int rank = block_rank(sel, swarp, &tot) + sel_carry; sel_carry += tot;
        if (sel) { if (rank < full) { for (int i = 0; i < R; ++i) out[R * rank + i] = R * b + i; } else { mn = s; mnb = b; } }   // rank == full: the partial block
    }
    if (extra) {   // the lowest-ranked selected block (exactly one has rank == full): its first `extra` positions
        if (tid == 0) { smin_b = -1; }
        __syncthreads();
        if (mnb >= 0) smin_b = mnb;
        __syncthreads();
        if (tid < extra) out[R * full + tid] = R * smin_b + tid;
    }
    for (int i = tid; i < tail; i += 1024) out[R * full + extra + i] = R * nvis + i;
    if (tid == 0) cnt[t] = WIDTH;
}
// ---- gather attention: block per (row, kv head, split), 256 threads ----
constexpr int TK = 32;   // keys per tile
__global__ void __launch_bounds__(256) k_attend(const float* __restrict__ qf, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc, size_t kvs, int max_ctx, const int* __restrict__ rpos, const int* __restrict__ rkv,
                                                const int* __restrict__ idx, const int* __restrict__ cnt, int nsplit, float* __restrict__ partials,
                                                float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od, float* __restrict__ os) {
    __shared__ float sq[QPK][HD]; __shared__ __align__(16) uint16_t sK[TK][HD]; __shared__ __align__(16) uint16_t sV[TK][HD];
    __shared__ float sP[QPK][TK]; __shared__ float sm[QPK], sl[QPK], smax[QPK];
    const int t = blockIdx.x / (NKV * nsplit), kh = (blockIdx.x / nsplit) % NKV, split = blockIdx.x % nsplit, tid = threadIdx.x;
    const int nkeys = idx ? cnt[t] : rpos[t] + 1; const int* list = idx ? idx + (size_t)t * WIDTH : nullptr;
    const int k0 = (int)((long)nkeys * split / nsplit), k1 = (int)((long)nkeys * (split + 1) / nsplit);
    const uint16_t* Kc = kc + (size_t)rkv[t] * kvs + (size_t)kh * max_ctx * HD; const uint16_t* Vc = vc + (size_t)rkv[t] * kvs + (size_t)kh * max_ctx * HD;
    for (int i = tid; i < QPK * HD; i += 256) sq[i / HD][i % HD] = qf[((size_t)t * NQH + kh * QPK + i / HD) * 512 + i % HD];
    if (tid < QPK) { sm[tid] = -INFINITY; sl[tid] = 0.f; }
    float o[QPK];
#pragma unroll
    for (int j = 0; j < QPK; ++j) o[j] = 0.f;
    __syncthreads();
    for (int kb = k0; kb < k1; kb += TK) {
        const int nk = min(TK, k1 - kb);
        // stage K and V rows (each thread: 4 x 16 B of K and of V)
        for (int i = tid; i < TK * 32; i += 256) { const int kk = i / 32, part = i % 32; if (kk < nk) { const int pos = list ? list[kb + kk] : kb + kk; *(uint4*)&sK[kk][part * 8] = *(const uint4*)(Kc + (size_t)pos * HD + part * 8); *(uint4*)&sV[kk][part * 8] = *(const uint4*)(Vc + (size_t)pos * HD + part * 8); } }
        __syncthreads();
        // scores: 12 heads x nk keys; pairs tid and tid + 256
        for (int pr = tid; pr < QPK * TK; pr += 256) {
            const int j = pr / TK, kk = pr % TK; float d = -INFINITY;
            if (kk < nk) { d = 0.f; const uint16_t* kr = sK[kk]; for (int i = 0; i < HD; i += 2) { const uint32_t u = *(const uint32_t*)(kr + i); d += sq[j][i] * h2f(u & 0xFFFF) + sq[j][i + 1] * h2f(u >> 16); } d *= 0.0625f; }
            sP[j][kk] = d;
        }
        __syncthreads();
        // online softmax per head (wave j handles head j; 8 waves cover 12 heads in two rounds)
        for (int j = tid / 32; j < QPK; j += 8) {
            const int lane = tid % 32; const float v = sP[j][lane];
            const float m = fmaxf(sm[j], wave_max(v)); const float e = lane < nk ? __expf(v - m) : 0.f; const float l = wave_sum(e);
            const float corr = __expf(sm[j] - m);
            if (lane == 0) { sl[j] = sl[j] * corr + l; smax[j] = corr; sm[j] = m; }
            sP[j][lane] = e;
        }
        __syncthreads();
        // o[j][d] = o*corr + sum_k P[j][k] V[k][d]   (thread = dim d)
#pragma unroll
        for (int j = 0; j < QPK; ++j) { float acc = 0.f; for (int kk = 0; kk < nk; ++kk) acc += sP[j][kk] * h2f(sV[kk][tid]); o[j] = o[j] * smax[j] + acc; }
        __syncthreads();
    }
    if (nsplit == 1) {
#pragma unroll
        for (int j = 0; j < QPK; ++j) {
            const int h = kh * QPK + j; const size_t oi = ((size_t)t * NQH + h) * HD + tid;
            const float v = (o[j] / sl[j]) * sigmoid(qf[((size_t)t * NQH + h) * 512 + HD + tid]);
            out[oi] = v; if (oq) wave_quant32(v, oq, od, os, (int)(oi / 32), tid % 32);
        }
    } else {
        float* part = partials + (((size_t)t * NKV + kh) * nsplit + split) * QPK * (HD + 2);
#pragma unroll
        for (int j = 0; j < QPK; ++j) { part[j * (HD + 2) + tid] = o[j]; if (tid == 0) { part[j * (HD + 2) + HD] = sm[j]; part[j * (HD + 2) + HD + 1] = sl[j]; } }
    }
}
__global__ void k_merge(const float* __restrict__ qf, const float* __restrict__ partials, int nsplit, float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od, float* __restrict__ os) {
    const int t = blockIdx.x / NQH, h = blockIdx.x % NQH, kh = h / QPK, j = h % QPK, d = threadIdx.x;
    const float* base = partials + ((size_t)t * NKV + kh) * nsplit * QPK * (HD + 2) + j * (HD + 2);
    float m = -INFINITY; for (int s = 0; s < nsplit; ++s) m = fmaxf(m, base[(size_t)s * QPK * (HD + 2) + HD]);
    float o = 0.f, l = 0.f;
    for (int s = 0; s < nsplit; ++s) { const float* p = base + (size_t)s * QPK * (HD + 2); const float w = __expf(p[HD] - m); o += p[d] * w; l += p[HD + 1] * w; }
    const size_t oi = ((size_t)t * NQH + h) * HD + d;
    const float v = (o / l) * sigmoid(qf[((size_t)t * NQH + h) * 512 + HD + d]);
    out[oi] = v; if (oq) wave_quant32(v, oq, od, os, (int)(oi / 32), d % 32);
}

// ---- gather attention, WMMA f16 (prefill rows): block per (row, kv head), 4 waves; keys in 32-key tiles ----
// S[16 heads (12 + pad)][32 keys] = Q . K^T (2 key tiles x 16 k-steps), online softmax per head in LDS, O[16][256] += P . V
// (16 N-tiles x 2 k-steps, 4 per wave). f16 operands, f32 accumulate — the precision of the flash prefill kernel.
typedef _Float16 f16x16_t __attribute__((ext_vector_type(16)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));
constexpr int TKW = 32, PADH = HD + 8;   // padded LDS row (bank spread)
__global__ void __launch_bounds__(128) k_attend_wmma(const float* __restrict__ qf, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc, size_t kvs, int max_ctx,
                                                     const int* __restrict__ rpos, const int* __restrict__ rkv, const int* __restrict__ idx, const int* __restrict__ cnt,
                                                     float* __restrict__ out) {
    __shared__ __align__(16) uint16_t sQ[16][PADH]; __shared__ __align__(16) uint16_t sK[TKW][PADH]; __shared__ __align__(16) uint16_t sV[TKW][PADH];
    __shared__ float sS[16][TKW]; __shared__ __align__(16) uint16_t sP[16][TKW + 8]; __shared__ float sm[16], sl[16], scorr[16];
    const int t = blockIdx.x / NKV, kh = blockIdx.x % NKV, tid = threadIdx.x, wave = tid / 32, lane = tid % 32, r16 = lane % 16, half = lane / 16;
    const int nkeys = idx ? cnt[t] : rpos[t] + 1; const int* list = idx ? idx + (size_t)t * WIDTH : nullptr;
    const uint16_t* Kc = kc + (size_t)rkv[t] * kvs + (size_t)kh * max_ctx * HD; const uint16_t* Vc = vc + (size_t)rkv[t] * kvs + (size_t)kh * max_ctx * HD;
    for (int i = tid; i < 16 * HD; i += 128) { const int h = i / HD, d = i % HD; sQ[h][d] = h < QPK ? f2h(qf[((size_t)t * NQH + kh * QPK + h) * 512 + d] * 0.0625f) : (uint16_t)0; }
    if (tid < 16) { sm[tid] = -INFINITY; sl[tid] = 0.f; }
    f32x8_t O[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) O[i] = f32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
    __syncthreads();
    for (int kb = 0; kb < nkeys; kb += TKW) {
        const int nk = min(TKW, nkeys - kb);
        for (int i = tid; i < TKW * 32; i += 128) { const int kk = i / 32, part = i % 32;
            uint4 kv4 = {0, 0, 0, 0}, vv4 = {0, 0, 0, 0};
            if (kk < nk) { const int pos = list ? list[kb + kk] : kb + kk; kv4 = *(const uint4*)(Kc + (size_t)pos * HD + part * 8); vv4 = *(const uint4*)(Vc + (size_t)pos * HD + part * 8); }
            *(uint4*)&sK[kk][part * 8] = kv4; *(uint4*)&sV[kk][part * 8] = vv4; }
        __syncthreads();
        // S = Q . K^T : waves 0,1 -> key tiles 0,1 (16 keys each); A = Q rows (heads), B = K rows (keys), 16 k-steps of 16 dims
        if (wave < 2) {
            f32x8_t C = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < HD / 16; ++ks) {
                const f16x16_t A = *(const f16x16_t*)&sQ[r16][ks * 16], B = *(const f16x16_t*)&sK[wave * 16 + r16][ks * 16];
                C = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, C);   // D[i = 2l + half (head)][j = r16 (key)]
            }
#pragma unroll
            for (int l = 0; l < 8; ++l) { const int h = 2 * l + half, kk = wave * 16 + r16; sS[h][kk] = kk < nk ? C[l] : -INFINITY; }
        }
        __syncthreads();
        // online softmax per head: 16 heads x 32 keys; wave w handles heads 4w..4w+3 (lane -> key)
        for (int h = wave * 4; h < wave * 4 + 4; ++h) {
            const float v = sS[h][lane];
            const float m = fmaxf(sm[h], wave_max(v)); const float e = lane < nk ? __expf(v - m) : 0.f; const float l = wave_sum(e);
            if (lane == 0) { scorr[h] = __expf(sm[h] - m); sl[h] = sl[h] * scorr[h] + l; sm[h] = m; }
            sP[h][lane] = f2h(e);
        }
        __syncthreads();
        // O[h][n] = O * corr + P . V : wave w -> N-tiles 4w..4w+3 (16 dims each); A = P (16 heads x 32 keys: 2 k-steps), B = V[k][n] column fragments
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int nt = wave * 4 + i;
#pragma unroll
            for (int l = 0; l < 8; ++l) O[i][l] *= scorr[2 * l + half];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const f16x16_t A = *(const f16x16_t*)&sP[r16][ks * 16];
                f16x16_t B;
#pragma unroll
                for (int e = 0; e < 16; ++e) { uint16_t u = sV[ks * 16 + e][nt * 16 + r16]; _Float16 x; __builtin_memcpy(&x, &u, 2); B[e] = x; }
                O[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, O[i]);   // D[i = head 2l+half][j = r16 (dim within the N-tile)]
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int nt = wave * 4 + i, d = nt * 16 + r16;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int h = 2 * l + half; if (h >= QPK) continue;
            const int hh = kh * QPK + h;
            out[((size_t)t * NQH + hh) * HD + d] = (O[i][l] / sl[h]) * sigmoid(qf[((size_t)t * NQH + hh) * 512 + HD + d]);
        }
    }
}
}  // namespace

void bf16_gemv(const uint16_t* W, const float* x, float* y, int N, int K, int T, hipStream_t s) { k_bf16_gemv<<<(T * N * 32 + 255) / 256, 256, 0, s>>>(W, x, y, N, K, T); }
void write_k(const float* k, const int* rpos, const int* rkv, uint16_t* cache, size_t slot_stride, int T, hipStream_t s) { k_write_k<<<T, IDX_DIM, 0, s>>>(k, rpos, rkv, cache, slot_stride); }
void pool(const uint16_t* cache, const float* w_kn, float* pooled, int b0, int b1, float eps, float rope_base, hipStream_t s) {
    if (b1 > b0) k_pool<<<b1 - b0, IDX_DIM, 0, s>>>(cache, w_kn, pooled, b0, eps, rope_base);
}
void query(const float* q_raw, const int* rpos, const float* w_qn, float* q, int T, float eps, float rope_base, hipStream_t s) { k_query<<<T * IDX_HEADS, IDX_DIM, 0, s>>>(q_raw, rpos, w_qn, q, eps, rope_base); }
void select(const float* q, const int* rpos, const int* rkv, const float* pooled, size_t pooled_slot_stride, int max_blocks, float* scratch, int* idx, int* cnt, int T, hipStream_t s) {
    k_select<<<T, 1024, 0, s>>>(q, rpos, rkv, pooled, pooled_slot_stride, max_blocks, scratch, idx, cnt);
}
void attend(const float* qf, const uint16_t* kc, const uint16_t* vc, size_t kv_slot_stride, int max_ctx, const int* rpos, const int* rkv,
            const int* idx, const int* cnt, int T, int nsplit, float* partials, float* out, XQ8 xq, hipStream_t s) {
    if (T > 8 && nsplit == 1 && !xq.q) { k_attend_wmma<<<T * NKV, 128, 0, s>>>(qf, kc, vc, kv_slot_stride, max_ctx, rpos, rkv, idx, cnt, out); return; }   // prefill: f16 WMMA tiles
    k_attend<<<T * NKV * nsplit, 256, 0, s>>>(qf, kc, vc, kv_slot_stride, max_ctx, rpos, rkv, idx, cnt, nsplit, partials, out, xq.q, xq.d, xq.s);
    if (nsplit > 1) k_merge<<<T * NQH, HD, 0, s>>>(qf, partials, nsplit, out, xq.q, xq.d, xq.s);
}
}  // namespace hip::qsa
