// hipBLASLt f16 GEMM roof: D[M,N] = A[M,K] * B[N,K]^T, heuristic best algo
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{ auto e=(x); if(e!=HIPBLAS_STATUS_SUCCESS){printf("hipblaslt err %d line %d\n",(int)e,__LINE__); return 1;} }while(0)
int main(int argc, char** argv) {
    int M = argc>1?atoi(argv[1]):2048, K = argc>2?atoi(argv[2]):5120, N = argc>3?atoi(argv[3]):17408;
    hipDataType tin = HIP_R_16F; if (argc>4 && argv[4][0]=='b') tin = HIP_R_16BF;
    void *A,*B,*D; hipMalloc(&A,(size_t)M*K*2); hipMalloc(&B,(size_t)N*K*2); hipMalloc(&D,(size_t)M*N*2);
    hipMemset(A,0,(size_t)M*K*2); hipMemset(B,0,(size_t)N*K*2);
    void* ws; size_t wsz = 64<<20; hipMalloc(&ws, wsz);
    hipblasLtHandle_t h; CK(hipblasLtCreate(&h));
    hipblasLtMatmulDesc_t op; CK(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    // column-major view: D^T[N,M] = B[K-major as N x K]^T ... use: C(N x M) = op(Bmat)(N x K) * op(Amat)(K x M)
    CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    hipblasLtMatrixLayout_t la, lb, ld;
    CK(hipblasLtMatrixLayoutCreate(&la, tin, K, N, K));   // B weights [N,K] row-major == col-major K x N, lda=K, transposed
    CK(hipblasLtMatrixLayoutCreate(&lb, tin, K, M, K));   // A acts    [M,K] row-major == col-major K x M
    CK(hipblasLtMatrixLayoutCreate(&ld, tin, N, M, N));   // D [M,N] row-major == col-major N x M
    hipblasLtMatmulPreference_t pref; CK(hipblasLtMatmulPreferenceCreate(&pref));
    CK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
    hipblasLtMatmulHeuristicResult_t res[16]; int nres=0;
    CK(hipblasLtMatmulAlgoGetHeuristic(h, op, la, lb, ld, ld, pref, 16, res, &nres));
    if (!nres) { printf("no algos\n"); return 1; }
    float alpha=1, beta=0; hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
    float best = 1e9; int bi=-1;
    for (int i = 0; i < nres; ++i) {
        auto run=[&]{ return hipblasLtMatmul(h, op, &alpha, B, la, A, lb, &beta, D, ld, D, ld, &res[i].algo, ws, wsz, 0); };
        if (run()!=HIPBLAS_STATUS_SUCCESS) continue; hipDeviceSynchronize();
        hipEventRecord(a); for(int r=0;r<10;++r) run(); hipEventRecord(b); hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms,a,b); ms/=10; if (ms<best){best=ms;bi=i;}
    }
    printf("hipblaslt %s M=%d K=%d N=%d: best of %d algos %.3f ms = %.1f TFLOPS\n", tin==HIP_R_16F?"f16":"bf16", M,K,N,nres,best,2.0*M*K*N/best/1e9);
    return 0;
}
