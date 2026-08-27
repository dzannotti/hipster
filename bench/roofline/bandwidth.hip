#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
// read-bandwidth test: each thread streams 16-byte loads over a large buffer and accumulates
__global__ void rd(const int4* __restrict__ p, size_t n, int* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    int acc = 0;
    for (; i < n; i += stride) { int4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x7fffffff) out[0] = acc;
}
int main(int argc, char** argv) {
    size_t gb = argc > 1 ? atoi(argv[1]) : 8;
    size_t bytes = gb << 30, n = bytes / 16;
    int4* p; int* o; hipMalloc(&p, bytes); hipMalloc(&o, 4); hipMemset(p, 1, bytes);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int blocks : {1024, 4096, 16384, 65536}) {
        rd<<<blocks, 256>>>(p, n, o); hipDeviceSynchronize();
        hipEventRecord(a); for (int i = 0; i < 3; ++i) rd<<<blocks, 256>>>(p, n, o); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b); ms /= 3;
        printf("read %zu GiB, %d blocks x 256: %.1f ms -> %.1f GB/s\n", gb, blocks, ms, bytes / ms / 1e6);
    }
    // device-to-device copy (read+write)
    int4* q; hipMalloc(&q, bytes);
    hipMemcpy(q, p, bytes, hipMemcpyDeviceToDevice); hipDeviceSynchronize();
    hipEventRecord(a); hipMemcpy(q, p, bytes, hipMemcpyDeviceToDevice); hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    printf("d2d memcpy %zu GiB: %.1f ms -> %.1f GB/s read+write (%.1f GB/s each way)\n", gb, ms, 2.0 * bytes / ms / 1e6, bytes / ms / 1e6);
    return 0;
}
