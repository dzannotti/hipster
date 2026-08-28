// Does wmma_f32_16x16x16_f16 give bit-identical D when B entries multiplied by A == 0 change? (denormal / normal / zero B)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
typedef _Float16 f16x16_t __attribute__((ext_vector_type(16)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));
__global__ void k(const uint16_t* A, const uint16_t* B, float* D) {   // A, B: [16][16] f16 (row-major, row = lane%16), D [8][32]
    const int lane = threadIdx.x, r = lane % 16;
    f16x16_t a, b; memcpy(&a, A + r * 16, 32); memcpy(&b, B + r * 16, 32);
    f32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    for (int i = 0; i < 8; ++i) D[i * 32 + lane] = acc[i];
}
static uint16_t h(float f) { _Float16 x = (_Float16)f; uint16_t u; memcpy(&u, &x, 2); return u; }
int main() {
    uint16_t A[256], B0[256], B1[256], B2[256]; float D[3][256];
    srand(1);
    for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
        A[r * 16 + c] = (c < 8) ? h((rand() % 2000 - 1000) / 700.f) : 0;   // k >= 8: A is exactly zero
        const float v = (rand() % 2000 - 1000) / 900.f;
        B0[r * 16 + c] = h(v); B1[r * 16 + c] = B0[r * 16 + c]; B2[r * 16 + c] = B0[r * 16 + c];
        if (c >= 8) { B1[r * 16 + c] = 0; B2[r * 16 + c] = (uint16_t)(rand() % 1024) | ((rand() & 1) << 15); }   // B1: zeros, B2: denormals (exp field 0)
    }
    uint16_t *dA, *dB; float* dD; hipMalloc(&dA, 512); hipMalloc(&dB, 512); hipMalloc(&dD, 1024);
    hipMemcpy(dA, A, 512, hipMemcpyHostToDevice);
    const uint16_t* Bs[3] = {B0, B1, B2};
    for (int t = 0; t < 3; ++t) { hipMemcpy(dB, Bs[t], 512, hipMemcpyHostToDevice); k<<<1, 32>>>(dA, dB, dD); hipMemcpy(D[t], dD, 1024, hipMemcpyDeviceToHost); }
    // B3: only key row 15 changed (the B fragment of lane 15): D columns j != 15 must not change
    uint16_t B3[256]; memcpy(B3, B0, 512); for (int c = 0; c < 16; ++c) B3[15 * 16 + c] = h((rand() % 2000 - 1000) / 300.f);
    float D3[256]; hipMemcpy(dB, B3, 512, hipMemcpyHostToDevice); k<<<1, 32>>>(dA, dB, dD); hipMemcpy(D3, dD, 1024, hipMemcpyDeviceToHost);
    int d03 = 0; for (int i = 0; i < 8; ++i) for (int lane = 0; lane < 32; ++lane) if (lane % 16 != 15 && memcmp(&D[0][i * 32 + lane], &D3[i * 32 + lane], 4) != 0) ++d03;
    printf("B row 15 changed: D elements of other columns that changed: %d/240\n", d03);
    int d01 = 0, d02 = 0, d12 = 0; for (int i = 0; i < 256; ++i) { d01 += memcmp(&D[0][i], &D[1][i], 4) != 0; d02 += memcmp(&D[0][i], &D[2][i], 4) != 0; d12 += memcmp(&D[1][i], &D[2][i], 4) != 0; }
    printf("A has zeros at k>=8. D differs: B(normal) vs B(zero): %d/256 elements; B(normal) vs B(denormal): %d; B(zero) vs B(denormal): %d\n", d01, d02, d12);
    for (int i = 0; i < 4; ++i) printf("  D0 %.9g D1 %.9g D2 %.9g\n", D[0][i], D[1][i], D[2][i]);
    return 0;
}
