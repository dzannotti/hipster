#include "gemm.h"
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_fp16.h>
#include <stdexcept>
#include <string>

#define CKB(x) do { auto e = (x); if (e != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error("hipblaslt error " + std::to_string((int)e) + " @" + std::to_string(__LINE__)); } while (0)

namespace hip {
namespace {
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7FFFu + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
__device__ __forceinline__ int iq4nl_val(int i) { const int8_t k[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}; return k[i]; }

// scalar decode of element e (0..255) of block b of row `row`
template <WFmt F>
__device__ __forceinline__ float dq_elem(const uint8_t* __restrict__ W, int row, int b, int e, int nb, int K) {
    if constexpr (F == WFmt::Q4_K) {
        const uint8_t* blk = W + ((size_t)row * nb + b) * 144;
        const float d = h2f(*(const uint16_t*)blk), dmin = h2f(*(const uint16_t*)(blk + 2));
        const uint8_t* sc = blk + 4; const int j = e / 64, l = e % 64, is = 2 * j + (l >= 32);
        uint8_t s, m; if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; } else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        const uint8_t q = blk[16 + 32 * j + (l % 32)];
        return d * s * ((l < 32) ? (q & 0xF) : (q >> 4)) - dmin * m;
    } else if constexpr (F == WFmt::Q5_K) {
        const uint8_t* blk = W + ((size_t)row * nb + b) * 176;
        const float d = h2f(*(const uint16_t*)blk), dmin = h2f(*(const uint16_t*)(blk + 2));
        const uint8_t* sc = blk + 4; const uint8_t* qh = blk + 16; const uint8_t* qs = blk + 48;
        const int j = e / 64, l = e % 64, is = 2 * j + (l >= 32);
        uint8_t s, m; if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; } else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        const uint8_t q = qs[32 * j + (l % 32)], hb = (qh[l % 32] >> (2 * j + (l >= 32))) & 1;
        return d * s * (((l < 32) ? (q & 0xF) : (q >> 4)) + 16 * hb) - dmin * m;
    } else if constexpr (F == WFmt::IQ4_XS) {
        const uint8_t* blk = W + ((size_t)row * nb + b) * 136;
        const float d = h2f(*(const uint16_t*)blk); const uint16_t sh = *(const uint16_t*)(blk + 2); const uint8_t* sl = blk + 4; const uint8_t* qs = blk + 8;
        const int ib = e / 32, l = e % 32;
        const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const uint8_t q = qs[16 * ib + (l % 16)];
        return d * (ls - 32) * iq4nl_val((l < 16) ? (q & 0xF) : (q >> 4));
    } else if constexpr (F == WFmt::Q6_K_SOA) {
        const uint8_t* rp = W + (size_t)row * ((size_t)K * 210 / 256);
        const uint8_t* ql = rp + (size_t)b * 128; const uint8_t* qh = rp + (size_t)nb * 128 + (size_t)b * 64;
        const int8_t* sc = (const int8_t*)(rp + (size_t)nb * 192 + (size_t)b * 16); const float d = h2f(((const uint16_t*)(rp + (size_t)nb * 208))[b]);
        const int n = e / 128, r = e % 128, grp = r / 32, l = r % 32;
        int q;
        if (grp == 0) q = (ql[64 * n + l] & 0xF) | (((qh[32 * n + l] >> 0) & 3) << 4);
        else if (grp == 1) q = (ql[64 * n + 32 + l] & 0xF) | (((qh[32 * n + l] >> 2) & 3) << 4);
        else if (grp == 2) q = (ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4);
        else q = (ql[64 * n + 32 + l] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4);
        return d * sc[8 * n + r / 16] * (q - 32);
    } else if constexpr (F == WFmt::Q5_1_SOA) {
        const uint8_t* rp = W + (size_t)row * ((size_t)K * 24 / 32);
        const int ib = b * 8 + e / 32, l = e % 32;
        const uint8_t q4 = rp[(size_t)ib * 16 + (l % 16)]; const uint32_t hb = ((const uint32_t*)(rp + K / 2))[ib];
        const int q = ((l < 16) ? (q4 & 0xF) : (q4 >> 4)) | (((hb >> l) & 1) << 4);
        const uint16_t* dp = (const uint16_t*)(rp + K / 2 + K / 8); const uint16_t* mp = dp + K / 32;
        return h2f(dp[ib]) * q + h2f(mp[ib]);
    } else {
        const uint8_t* rp = W + (size_t)row * ((size_t)K + K / 16);
        const int8_t* q = (const int8_t*)rp + (size_t)b * 256; const uint16_t* dp = (const uint16_t*)(rp + K) + b * 8;
        return h2f(dp[e / 32]) * q[e];
    }
}
// 256 threads = 8 blocks of 256 elements; thread t: block (t/32), elements 8*(t%32)..+8 -> one 16-byte store
template <WFmt F>
__global__ void k_dequant_bf16(const uint8_t* __restrict__ W, uint16_t* __restrict__ out, int N, int K) {
    const int nb = K / 256;
    const size_t gb = (size_t)blockIdx.x * 8 + threadIdx.x / 32;   // global block index over rows*nb
    if (gb >= (size_t)N * nb) return;
    const int row = (int)(gb / nb), b = (int)(gb % nb), e0 = 8 * (threadIdx.x % 32);
    uint16_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = f2bf(dq_elem<F>(W, row, b, e0 + i, nb, K));
    uint4 pk; pk.x = v[0] | ((uint32_t)v[1] << 16); pk.y = v[2] | ((uint32_t)v[3] << 16); pk.z = v[4] | ((uint32_t)v[5] << 16); pk.w = v[6] | ((uint32_t)v[7] << 16);
    *(uint4*)(out + (size_t)row * K + b * 256 + e0) = pk;
}
template <WFmt F>
__global__ void k_dequant_f32(const uint8_t* __restrict__ W, float* __restrict__ out, int N, int K) {
    const int nb = K / 256;
    const size_t gb = (size_t)blockIdx.x * 8 + threadIdx.x / 32;
    if (gb >= (size_t)N * nb) return;
    const int row = (int)(gb / nb), b = (int)(gb % nb), e0 = 8 * (threadIdx.x % 32);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[(size_t)row * K + b * 256 + e0 + i] = dq_elem<F>(W, row, b, e0 + i, nb, K);
}
// rows m < M of a [M][Nt] bf16 matrix, columns off..off+N -> f32 [M][N]
__global__ void k_bf16_to_f32_seg(const uint16_t* __restrict__ x, int Nt, int off, int N, float* __restrict__ o, int M) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), c = (int)(i % N);
    o[i] = __uint_as_float((uint32_t)x[(size_t)m * Nt + off + c] << 16);
}
__global__ void k_bf16_to_f32_seg2(const uint16_t* __restrict__ x, const uint16_t* __restrict__ x2, int Nt, int off, int N, float* __restrict__ o, int M) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), c = (int)(i % N); const size_t idx = (size_t)m * Nt + off + c;
    float v = __uint_as_float((uint32_t)x[idx] << 16); if (x2) v += __uint_as_float((uint32_t)x2[idx] << 16);
    o[i] = v;
}
__global__ void k_bf16_to_f32(const uint16_t* __restrict__ x, float* __restrict__ o, size_t n) { const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; if (i < n) o[i] = __uint_as_float((uint32_t)x[i] << 16); }
__global__ void k_f32_to_bf16(const float* __restrict__ x, uint16_t* __restrict__ o, size_t n) { const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; if (i < n) o[i] = f2bf(x[i]); }
}  // namespace

void bf16_to_f32_seg2(const uint16_t* x, const uint16_t* x2, int Nt, int off, int N, float* out, int M, hipStream_t s) { k_bf16_to_f32_seg2<<<(unsigned)(((size_t)M * N + 255) / 256), 256, 0, s>>>(x, x2, Nt, off, N, out, M); }
void bf16_to_f32_seg(const uint16_t* x, int Nt, int off, int N, float* out, int M, hipStream_t s) { k_bf16_to_f32_seg<<<(unsigned)(((size_t)M * N + 255) / 256), 256, 0, s>>>(x, Nt, off, N, out, M); }
void bf16_to_f32(const uint16_t* x, float* out, size_t n, hipStream_t s) { k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, out, n); }
void dequant_f32(WFmt f, const void* W, float* out, int N, int K, hipStream_t s) {
    const int grid = (int)(((size_t)N * (K / 256) + 7) / 8);
    switch (f) {
        case WFmt::Q4_K: k_dequant_f32<WFmt::Q4_K><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        case WFmt::Q6_K_SOA: k_dequant_f32<WFmt::Q6_K_SOA><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        case WFmt::Q5_1_SOA: k_dequant_f32<WFmt::Q5_1_SOA><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        case WFmt::Q5_K: k_dequant_f32<WFmt::Q5_K><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        default: k_dequant_f32<WFmt::Q8_0_SOA><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
    }
}
void dequant_bf16(WFmt f, const void* W, uint16_t* out, int N, int K, hipStream_t s) {
    const int grid = (N * (K / 256) + 7) / 8;
    switch (f) {
        case WFmt::Q4_K: k_dequant_bf16<WFmt::Q4_K><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        case WFmt::Q5_K: k_dequant_bf16<WFmt::Q5_K><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        case WFmt::IQ4_XS: k_dequant_bf16<WFmt::IQ4_XS><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        case WFmt::Q6_K_SOA: k_dequant_bf16<WFmt::Q6_K_SOA><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        case WFmt::Q5_1_SOA: k_dequant_bf16<WFmt::Q5_1_SOA><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        case WFmt::Q8_0_SOA: k_dequant_bf16<WFmt::Q8_0_SOA><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
        default: k_dequant_bf16<WFmt::Q8_0_SOA><<<grid, 256, 0, s>>>((const uint8_t*)W, out, N, K); break;
    }
}
void f32_to_bf16(const float* x, uint16_t* out, size_t n, hipStream_t s) { k_f32_to_bf16<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, out, n); }

struct BlasLt::Impl {
    hipblasLtHandle_t h; void* ws; size_t wsz = 64u << 20; void* flush = nullptr;   // 64 MB scratch: evicts L2 + MALL between autotune reps
    hipblasLtMatmulDesc_t op; hipblasLtMatmulPreference_t pref;
    std::map<std::tuple<int, int, int>, hipblasLtMatmulAlgo_t> algos;
};
BlasLt::BlasLt() : p_(new Impl) {
    CKB(hipblasLtCreate(&p_->h)); hipMalloc(&p_->ws, p_->wsz); hipMalloc(&p_->flush, 64u << 20);
    CKB(hipblasLtMatmulDescCreate(&p_->op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    CKB(hipblasLtMatmulDescSetAttribute(p_->op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    CKB(hipblasLtMatmulDescSetAttribute(p_->op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    CKB(hipblasLtMatmulPreferenceCreate(&p_->pref));
    CKB(hipblasLtMatmulPreferenceSetAttribute(p_->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p_->wsz, sizeof p_->wsz));
}
BlasLt::~BlasLt() {}
void BlasLt::gemm(const uint16_t* A, const uint16_t* W, uint16_t* D, int M, int N, int K, hipStream_t s) { gemm_ld(A, K, W, K, D, M, N, K, s); }
void BlasLt::gemm_ld(const uint16_t* A, int lda, const uint16_t* W, int ldw, uint16_t* D, int M, int N, int K, hipStream_t s) {
    // column-major view: C(N x M) = W^T-as-(K x N)^T · A(K x M)
    hipblasLtMatrixLayout_t la, lb, ld;
    CKB(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, K, N, ldw));
    CKB(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, M, lda));
    CKB(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, N, M, N));
    auto key = std::make_tuple(M, N, K * 1000003 + (lda != K));
    const float alpha = 1.f, beta = 0.f;
    auto it = p_->algos.find(key);
    if (it == p_->algos.end()) {
        // The heuristic's first choice measured ~6 TFLOPS on this GPU where the best of its candidates
        // does ~50: time every candidate once per shape (a few ms) and keep the fastest.
        hipblasLtMatmulHeuristicResult_t res[16]; int n = 0;
        CKB(hipblasLtMatmulAlgoGetHeuristic(p_->h, p_->op, la, lb, ld, ld, p_->pref, 16, res, &n));
        if (!n) throw std::runtime_error("hipblaslt: no algo");
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        float best = 1e30f; int bi = 0;
        for (int i = 0; i < n; ++i) {
            if (hipblasLtMatmul(p_->h, p_->op, &alpha, W, la, A, lb, &beta, D, ld, D, ld, &res[i].algo, p_->ws, p_->wsz, s) != HIPBLAS_STATUS_SUCCESS) continue;
            // time COLD: the engine always feeds freshly dequantised weights; warm-timed candidates picked algorithms that
            // only perform from cache (measured 2-4x slower in the engine than in the warm autotune)
            float tot = 0.f;
            for (int r = 0; r < 2; ++r) {
                hipMemsetAsync(p_->flush, r, 64u << 20, s);
                hipEventRecord(e0, s); hipblasLtMatmul(p_->h, p_->op, &alpha, W, la, A, lb, &beta, D, ld, D, ld, &res[i].algo, p_->ws, p_->wsz, s);
                hipEventRecord(e1, s); hipEventSynchronize(e1); float ms; hipEventElapsedTime(&ms, e0, e1); tot += ms;
            }
            if (tot < best) { best = tot; bi = i; }
        }
        hipEventDestroy(e0); hipEventDestroy(e1);
        it = p_->algos.emplace(key, res[bi].algo).first;
    }
    CKB(hipblasLtMatmul(p_->h, p_->op, &alpha, W, la, A, lb, &beta, D, ld, D, ld, &it->second, p_->ws, p_->wsz, s));
    hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(ld);
}
// f32 in / f32 out (the MoE router: 513 x 2560, exact routing logits); same autotune, separate cache key
void BlasLt::gemm_f32(const float* A, const float* W, float* D, int M, int N, int K, hipStream_t s) {
    hipblasLtMatrixLayout_t la, lb, ld;
    CKB(hipblasLtMatrixLayoutCreate(&la, HIP_R_32F, K, N, K));
    CKB(hipblasLtMatrixLayoutCreate(&lb, HIP_R_32F, K, M, K));
    CKB(hipblasLtMatrixLayoutCreate(&ld, HIP_R_32F, N, M, N));
    auto key = std::make_tuple(M, N, K * 1000003 + 7);
    const float alpha = 1.f, beta = 0.f;
    auto it = p_->algos.find(key);
    if (it == p_->algos.end()) {
        hipblasLtMatmulHeuristicResult_t res[16]; int n = 0;
        CKB(hipblasLtMatmulAlgoGetHeuristic(p_->h, p_->op, la, lb, ld, ld, p_->pref, 16, res, &n));
        if (!n) throw std::runtime_error("hipblaslt: no f32 algo");
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        float best = 1e30f; int bi = 0;
        for (int i = 0; i < n; ++i) {
            if (hipblasLtMatmul(p_->h, p_->op, &alpha, W, la, A, lb, &beta, D, ld, D, ld, &res[i].algo, p_->ws, p_->wsz, s) != HIPBLAS_STATUS_SUCCESS) continue;
            // time COLD: the engine always feeds freshly dequantised weights; warm-timed candidates picked algorithms that
            // only perform from cache (measured 2-4x slower in the engine than in the warm autotune)
            float tot = 0.f;
            for (int r = 0; r < 2; ++r) {
                hipMemsetAsync(p_->flush, r, 64u << 20, s);
                hipEventRecord(e0, s); hipblasLtMatmul(p_->h, p_->op, &alpha, W, la, A, lb, &beta, D, ld, D, ld, &res[i].algo, p_->ws, p_->wsz, s);
                hipEventRecord(e1, s); hipEventSynchronize(e1); float ms; hipEventElapsedTime(&ms, e0, e1); tot += ms;
            }
            if (tot < best) { best = tot; bi = i; }
        }
        hipEventDestroy(e0); hipEventDestroy(e1);
        it = p_->algos.emplace(key, res[bi].algo).first;
    }
    CKB(hipblasLtMatmul(p_->h, p_->op, &alpha, W, la, A, lb, &beta, D, ld, D, ld, &it->second, p_->ws, p_->wsz, s));
    hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(ld);
}
}  // namespace hip
