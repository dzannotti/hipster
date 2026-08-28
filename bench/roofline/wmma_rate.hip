// WMMA instruction throughput on gfx1151: f16, bf16, iu8, iu4 (register-only, 4 independent chains)
#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
template <int OP> __global__ void __launch_bounds__(256) k(float* out, int iters) {
    f16x16 a16, b16; bf16x16 ab, bb; i32x4 ai, bi; i32x2 a4, b4;
    for (int i = 0; i < 16; ++i) { a16[i] = (_Float16)(0.001f * (threadIdx.x + i)); b16[i] = (_Float16)(0.002f * i); ab[i] = (__bf16)(0.001f * i); bb[i] = (__bf16)(0.003f * i); }
    for (int i = 0; i < 4; ++i) { ai[i] = 0x01020304 * (threadIdx.x + i); bi[i] = 0x05060708 + i; }
    a4[0] = 0x12345678; a4[1] = threadIdx.x; b4[0] = 0x0f0f0f0f; b4[1] = 0x33333333;
    f32x8 c[4]; i32x8 ci[4];
    for (int j = 0; j < 4; ++j) for (int i = 0; i < 8; ++i) { c[j][i] = 0.f; ci[j][i] = 0; }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (OP == 0) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a16, b16, c[j]);
            else if (OP == 1) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ab, bb, c[j]);
            else if (OP == 2) ci[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ai, true, bi, ci[j], false);
            else ci[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a4, true, b4, ci[j], false);
        }
    }
    float s = 0; for (int j = 0; j < 4; ++j) s += c[j][0] + ci[j][0];
    if (s == 12345.f) out[0] = s;
}
int main() {
    float* o; hipMalloc(&o, 4); hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    const int iters = 2048, blocks = 40 * 16;
    const char* n[4] = {"wmma f32.16x16x16.f16", "wmma f32.16x16x16.bf16", "wmma i32.16x16x16.iu8", "wmma i32.16x16x16.iu4"};
    for (int op = 0; op < 4; ++op) {
        auto run = [&] { if (op == 0) k<0><<<blocks, 256>>>(o, iters); else if (op == 1) k<1><<<blocks, 256>>>(o, iters); else if (op == 2) k<2><<<blocks, 256>>>(o, iters); else k<3><<<blocks, 256>>>(o, iters); };
        run(); hipDeviceSynchronize();
        hipEventRecord(a); for (int r = 0; r < 5; ++r) run(); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); ms /= 5;
        const double instr = (double)blocks * 8 /*waves*/ * iters * 4, flops = instr * 16 * 16 * 16 * 2;
        printf("%-26s: %.3f ms  %.1f TFLOPS/TOPS (%.0f M wave-instr/s)\n", n[op], ms, flops / ms / 1e9, instr / ms / 1e3);
    }
    return 0;
}
