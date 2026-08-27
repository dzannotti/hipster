#include <chrono>
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k_empty(float* p) { if (threadIdx.x == 0 && blockIdx.x == 0 && p[0] == 12345.f) p[1] = 1.f; }
__global__ void k_small(float* p, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if (i < n) p[i] = p[i]*0.999f + 1.f; }
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); return 1;} }while(0)
int main() {
    float* d; CK(hipMalloc(&d, 1<<24)); CK(hipMemset(d, 0, 1<<24));
    hipStream_t s; CK(hipStreamCreate(&s));
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); float ms;
    const int N = 2000;
    // warm
    for (int i = 0; i < 100; ++i) k_empty<<<1,64,0,s>>>(d);
    CK(hipStreamSynchronize(s));
    // 1. stream launches, empty kernel
    hipEventRecord(a, s); for (int i = 0; i < N; ++i) k_empty<<<1,64,0,s>>>(d); hipEventRecord(b, s); hipEventSynchronize(b);
    hipEventElapsedTime(&ms, a, b); printf("stream  empty   1x64      : %.2f us/launch\n", 1000*ms/N);
    // 2. stream launches, small kernel (grid 256x256)
    hipEventRecord(a, s); for (int i = 0; i < N; ++i) k_small<<<256,256,0,s>>>(d, 65536); hipEventRecord(b, s); hipEventSynchronize(b);
    hipEventElapsedTime(&ms, a, b); printf("stream  small   256x256   : %.2f us/launch\n", 1000*ms/N);
    // 3. graph of N nodes, replay
    for (int variant = 0; variant < 2; ++variant) {
        hipGraph_t g; hipGraphExec_t ge;
        CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        for (int i = 0; i < N; ++i) { if (variant==0) k_empty<<<1,64,0,s>>>(d); else k_small<<<256,256,0,s>>>(d, 65536); }
        CK(hipStreamEndCapture(s, &g));
        CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        CK(hipGraphLaunch(ge, s)); CK(hipStreamSynchronize(s));
        const int R = 20;
        hipEventRecord(a, s); for (int r = 0; r < R; ++r) hipGraphLaunch(ge, s); hipEventRecord(b, s); hipEventSynchronize(b);
        hipEventElapsedTime(&ms, a, b); printf("graph   %s %d nodes: %.2f us/node (%.3f ms per replay)\n", variant==0?"empty":"small", N, 1000*ms/R/N, ms/R);
        hipGraphExecDestroy(ge); hipGraphDestroy(g);
    }
    // 4. host-side launch cost (no sync) - how long does the CPU spend enqueuing
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) k_empty<<<1,64,0,s>>>(d);
    auto t1 = std::chrono::steady_clock::now();
    CK(hipStreamSynchronize(s));
    printf("host enqueue cost       : %.2f us/launch\n", std::chrono::duration<double,std::micro>(t1-t0).count()/N);
    return 0;
}
