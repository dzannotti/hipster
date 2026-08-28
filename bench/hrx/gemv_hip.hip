// HIP side of the gemv-through-HRX comparison (bench/hrx/gemv_hrx.c is the HRX
// side). Loads the SAME unbundled HSACO (engine/kernels/gemv.hip compiled
// device-only) with hipModuleLoad and launches k_gemv_q8<Q4_K,4,1> the way
// engine launch_fmt does (grid = N/64 row-blocks, 256 threads), with the same
// LCG-generated data as the HRX side; both print a checksum of y.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint32_t g_lcg = 12345u;
static uint32_t lcg(void) { g_lcg = g_lcg * 1664525u + 1013904223u; return g_lcg; }
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s line %d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)
static const char* KNAME = "_ZN3hip12_GLOBAL__N_19k_gemv_q8ILNS_4WFmtE0ELi4ELi1EEEvPKhPKaPKfS8_Pfii";

int main(int argc, char** argv) {
    const char* hsaco = argc > 1 ? argv[1] : "build/hrx/gemv_dev.hsaco";
    const int N = argc > 2 ? atoi(argv[2]) : 11648, K = argc > 3 ? atoi(argv[3]) : 5120, iters = 50;
    const size_t rb = (size_t)K / 256 * 144, wbytes = rb * N, nsb = K / 32;  // Q4_K: 144 B per 256 weights
    printf("HIP  k_gemv_q8<Q4_K,4,1> N=%d K=%d row_bytes=%zu W=%.1f MB grid=%d (hipModuleLaunchKernel on %s)\n", N, K, rb, wbytes / 1048576.0, (N + 63) / 64, hsaco);
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, hsaco)); CK(hipModuleGetFunction(&fn, mod, KNAME));
    std::vector<uint8_t> hw(wbytes); for (auto& b : hw) b = lcg() >> 24;
    std::vector<int8_t> hq(K); for (auto& b : hq) b = (int8_t)(lcg() >> 24);
    std::vector<float> hd(nsb), hs(nsb); for (size_t i = 0; i < nsb; ++i) { hd[i] = ((lcg() >> 8) & 0xFFFF) / 65536.f * 0.01f; hs[i] = ((lcg() >> 8) & 0xFFFF) / 65536.f * 0.01f; }
    uint8_t* W; int8_t* xq; float *xd, *xs, *y;
    CK(hipMalloc(&W, wbytes)); CK(hipMalloc(&xq, K)); CK(hipMalloc(&xd, nsb * 4)); CK(hipMalloc(&xs, nsb * 4)); CK(hipMalloc(&y, N * 4));
    CK(hipMemcpy(W, hw.data(), wbytes, hipMemcpyHostToDevice)); CK(hipMemcpy(xq, hq.data(), K, hipMemcpyHostToDevice));
    CK(hipMemcpy(xd, hd.data(), nsb * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(xs, hs.data(), nsb * 4, hipMemcpyHostToDevice));
    CK(hipMemset(y, 0, N * 4));
    const int grid = (N + 63) / 64;
    int n = N, k = K;
    void* args[] = {&W, &xq, &xd, &xs, &y, &n, &k};
    auto launch = [&] { return hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr); };
    for (int i = 0; i < 5; ++i) CK(launch());
    CK(hipDeviceSynchronize());
    for (int rep = 0; rep < 3; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) CK(launch());
        CK(hipDeviceSynchronize());
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        printf("HIP  rep %d: %.1f us/launch, %.1f GB/s\n", rep, us / iters, wbytes * (double)iters / us / 1e3);
    }
    std::vector<float> hy(N); CK(hipMemcpy(hy.data(), y, N * 4, hipMemcpyDeviceToHost));
    double sum = 0; for (float v : hy) sum += v;
    printf("HIP  checksum: sum=%.6e y[0]=%.6e y[N-1]=%.6e\n", sum, hy[0], hy[N - 1]);
    return 0;
}
