// Prefill GEMM: quantised weights dequantised to bf16 [N][K] in a scratch buffer, then hipBLASLt
// D[M][N] (f32) = A[M][K] (bf16) · W[N][K]^T. Measured roof 51.8 TFLOPS (docs/roofline.md).
#pragma once
#include "gemv.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <map>
#include <tuple>

namespace hip {
// dequantise a whole weight into bf16 [N][K]
void dequant_bf16(WFmt f, const void* W, uint16_t* out, int N, int K, hipStream_t s);
void dequant_f32(WFmt f, const void* W, float* out, int N, int K, hipStream_t s);
// f32 <-> bf16
void f32_to_bf16(const float* x, uint16_t* out, size_t n, hipStream_t s);
void bf16_to_f32(const uint16_t* x, float* out, size_t n, hipStream_t s);
void bf16_to_f32_seg(const uint16_t* x, int Nt, int off, int N, float* out, int M, hipStream_t s);
void f32_seg(const float* x, int Nt, int off, int N, float* out, int M, hipStream_t s);
// same, summing a second bf16 matrix (split-K partial) when x2 != null
void bf16_to_f32_seg2(const uint16_t* x, const uint16_t* x2, int Nt, int off, int N, float* out, int M, hipStream_t s);

class BlasLt {
public:
    BlasLt();
    ~BlasLt();
    // D[M][N] bf16 = A[M][K] bf16 · W[N][K]^T bf16 (f32 accumulate; f32 output measured 6× slower on this GPU).
    // Algorithms are timed once per shape and cached.
    void gemm(const uint16_t* A, const uint16_t* W, uint16_t* D, int M, int N, int K, hipStream_t s);
    // same with explicit leading dimensions (row strides in elements) for A and W: enables split-K on column halves
    void gemm_ld(const uint16_t* A, int lda, const uint16_t* W, int ldw, uint16_t* D, int M, int N, int K, hipStream_t s);
    // D[M][N] f32 = A[M][K] f32 . W[N][K]^T f32 (small N: the MoE router)
    void gemm_f32(const float* A, const float* W, float* D, int M, int N, int K, hipStream_t s);
private:
    struct Impl; Impl* p_;
};
}  // namespace hip
