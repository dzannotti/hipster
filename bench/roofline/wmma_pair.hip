// Which A element pairs with which B element in wmma_f32_16x16x16_f16_w32? Probe: A = unit vector e_k (all rows), B random;
// D[i][j] = B[j][pair(k)] if the hardware pairs A element k with B element pair(k).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
typedef _Float16 f16x16_t __attribute__((ext_vector_type(16)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));
__global__ void k(const uint16_t* A, const uint16_t* B, float* D) {
    const int lane = threadIdx.x, r = lane % 16;
    f16x16_t a, b; memcpy(&a, A + r * 16, 32); memcpy(&b, B + r * 16, 32);
    f32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    for (int i = 0; i < 8; ++i) D[i * 32 + lane] = acc[i];
}
static uint16_t h(float f) { _Float16 x = (_Float16)f; uint16_t u; memcpy(&u, &x, 2); return u; }
int main() {
    uint16_t A[256], B[256]; float D[256];
    for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) B[r * 16 + c] = h(100.f * r + c);   // B[j][k] = 100 j + k, exact in f16
    uint16_t *dA, *dB; float* dD; hipMalloc(&dA, 512); hipMalloc(&dB, 512); hipMalloc(&dD, 1024);
    hipMemcpy(dB, B, 512, hipMemcpyHostToDevice);
    printf("A = e_k for all rows -> D[i][j] should be B[j][k] = 100 j + k. D element at lane 0 (i=0, j=0) and lane 5 (j=5):\n");
    for (int kk = 0; kk < 16; ++kk) {
        for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) A[r * 16 + c] = h(c == kk ? 1.f : 0.f);
        hipMemcpy(dA, A, 512, hipMemcpyHostToDevice); k<<<1, 32>>>(dA, dB, dD); hipMemcpy(D, dD, 1024, hipMemcpyDeviceToHost);
        printf("  k=%2d: D[0][0] = %6.1f (expect %d)  D[0][5] = %6.1f (expect %d)\n", kk, D[0], kk, D[5], 500 + kk);
    }
    // the real case: A row with a single zero at k=15, B element 15 changed
    for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) { A[r * 16 + c] = h(c == 15 ? 0.f : (0.3f + 0.05f * c)); B[r * 16 + c] = h(0.7f - 0.09f * c + 0.01f * r); }
    hipMemcpy(dA, A, 512, hipMemcpyHostToDevice); hipMemcpy(dB, B, 512, hipMemcpyHostToDevice); k<<<1, 32>>>(dA, dB, dD); float D1[256]; hipMemcpy(D1, dD, 1024, hipMemcpyDeviceToHost);
    for (int r = 0; r < 16; ++r) B[r * 16 + 15] = h(5.5f);
    hipMemcpy(dB, B, 512, hipMemcpyHostToDevice); k<<<1, 32>>>(dA, dB, dD); float D2[256]; hipMemcpy(D2, dD, 1024, hipMemcpyDeviceToHost);
    int nd = 0; for (int i = 0; i < 256; ++i) nd += memcmp(&D1[i], &D2[i], 4) != 0;
    printf("single zero at A[k=15], B[.][15] changed: D elements changed = %d/256 (D1[0]=%.9g D2[0]=%.9g)\n", nd, D1[0], D2[0]);
    return 0;
}
