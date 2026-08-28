// Is the small-tensor GEMV loss the kernel or the memory system?
// Stream N bytes with a plain 16-byte-load reduction kernel, rotating over C copies so every launch is
// cold, for sizes 8..682 MiB. Compare with bench_gemv's per-size numbers.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
__global__ void rd(const u32x4* __restrict__ p, size_t n, unsigned* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (; i < n; i += stride) { u32x4 v = p[i]; acc ^= v[0] ^ v[1] ^ v[2] ^ v[3]; }
    if (acc == 0x7fffffff) out[0] = acc;
}
// row-structured variant: each 32-lane wave streams one row of `row` bytes then jumps (like the GEMV at tpr=32)
__global__ void rd_rows(const u32x4* __restrict__ p, int nrows, int row_u4, unsigned* out) {
    const int wave = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (wave >= nrows) return;
    const u32x4* r = p + (size_t)wave * row_u4;
    unsigned acc = 0;
    for (int i = lane; i < row_u4; i += 32) { u32x4 v = r[i]; acc ^= v[0] ^ v[1] ^ v[2] ^ v[3]; }
    if (acc == 0x7fffffff) out[0] = acc;
}
int main() {
    unsigned* o; hipMalloc(&o, 4);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    const size_t total = (size_t)768 << 20;
    u32x4* buf; hipMalloc(&buf, total); hipMemset(buf, 1, total);
    printf("%-10s %10s %10s %10s\n", "size", "flat GB/s", "rows GB/s", "launch us");
    for (size_t mb : {4, 8, 16, 34, 45, 90, 180, 360, 682}) {
        const size_t bytes = mb << 20, n = bytes / 16; const int copies = (int)(total / bytes);
        const int row_u4 = 2880 / 16 * 2;   // 5760-byte rows (Q5_K K=5120 x2 -> sim), 
        const int nrows = (int)(bytes / (row_u4 * 16));
        auto run_flat = [&](int it) { rd<<<4096, 256>>>(buf + (size_t)(it % copies) * n, n, o); };
        auto run_rows = [&](int it) { rd_rows<<<(nrows * 32 + 255) / 256, 256>>>(buf + (size_t)(it % copies) * n, nrows, row_u4, o); };
        for (int i = 0; i < 3; ++i) { run_flat(i); run_rows(i); } hipDeviceSynchronize();
        const int iters = 20; float ms;
        hipEventRecord(a); for (int i = 0; i < iters; ++i) run_flat(i); hipEventRecord(b); hipEventSynchronize(b); hipEventElapsedTime(&ms, a, b); const double gf = bytes / (ms / iters) / 1e6;
        hipEventRecord(a); for (int i = 0; i < iters; ++i) run_rows(i); hipEventRecord(b); hipEventSynchronize(b); hipEventElapsedTime(&ms, a, b); const double gr = bytes / (ms / iters) / 1e6;
        printf("%6zu MiB %10.1f %10.1f %10.1f\n", mb, gf, gr, ms / iters * 1000);
    }
    return 0;
}
