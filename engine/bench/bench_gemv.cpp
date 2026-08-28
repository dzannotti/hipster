// bench_gemv <gguf> <tensor> [ncol_max] [variant f32|q8|both]
// Validates a quantised GEMV against the CPU reference on sampled rows and reports achieved
// bandwidth vs the 240 GB/s roof. For the q8 variant the reference quantises x the same way.
#include "../src/gguf.h"
#include "../src/quant.h"
#include "../kernels/gemv.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

static void quantize_ref(const float* x, int K, std::vector<int8_t>& q, std::vector<float>& d, std::vector<float>& s) {
    q.resize(K); d.resize(K / 32); s.resize(K / 32);
    for (int b = 0; b < K / 32; ++b) {
        float amax = 0; for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(x[b * 32 + l]));
        float dd = amax / 127.f, id = amax > 0 ? 127.f / amax : 0.f; int sum = 0;
        for (int l = 0; l < 32; ++l) { int qi = (int)nearbyintf(x[b * 32 + l] * id); q[b * 32 + l] = (int8_t)qi; sum += qi; }
        d[b] = dd; s[b] = dd * sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s gguf tensor [ncol] [f32|q8|both]\n", argv[0]); return 2; }
    hip::GGUF g(argv[1]);
    const auto& t = g.get(argv[2]);
    const int ncol_max = std::min(16, argc > 3 ? atoi(argv[3]) : 1);
    const char* variant = argc > 4 ? argv[4] : "both";
    const int K = (int)t.ne[0]; int N = (int)(t.ne[1] * t.ne[2]); if (getenv("BENCH_ROWS")) N = std::min(N, atoi(getenv("BENCH_ROWS")));
    printf("%s %s [%d x %d] %.1f MiB\n", t.name.c_str(), hip::gtype_name(t.type), N, K, t.nbytes / 1048576.0);
    hip::WFmt fmt; bool repack = false;
    switch (t.type) {
        case hip::GType::Q4_K: fmt = hip::WFmt::Q4_K; break; case hip::GType::Q5_K: fmt = hip::WFmt::Q5_K; break;
        case hip::GType::IQ4_XS: fmt = hip::WFmt::IQ4_XS; break;
        case hip::GType::Q6_K: fmt = hip::WFmt::Q6_K_SOA; repack = true; break; case hip::GType::Q8_0: fmt = hip::WFmt::Q8_0_SOA; repack = true; break;
        case hip::GType::Q5_1: fmt = hip::WFmt::Q5_1_SOA; repack = true; break;
        default: fprintf(stderr, "type not wired\n"); return 1;
    }
    if (t.type != hip::GType::Q4_K && !strcmp(variant, "both")) variant = "q8";
    // Rotate through enough copies that the working set exceeds the 32 MB MALL + L2 (else small
    // tensors measure cache bandwidth, not DRAM).
    const int ncopy = (int)std::max<size_t>(1, (512u << 20) / t.nbytes + 1);
    std::vector<uint8_t*> dWs(ncopy), dWil, dWil16;
    {
        std::vector<uint8_t> tmp;
        const uint8_t* src = t.data;
        if (repack) {
            tmp.resize(t.nbytes); const size_t rb = hip::gtype_row_bytes(t.type, K);
            for (int r = 0; r < N; ++r) (t.type == hip::GType::Q6_K ? hip::repack_row_q6_K : t.type == hip::GType::Q5_1 ? hip::repack_row_q5_1 : hip::repack_row_q8_0)(t.data + (size_t)r * rb, tmp.data() + (size_t)r * rb, K);
            src = tmp.data();
        }
        for (int i = 0; i < ncopy; ++i) { CK(hipMalloc(&dWs[i], t.nbytes)); CK(hipMemcpy(dWs[i], src, t.nbytes, hipMemcpyHostToDevice)); }
        // interleaved copies for the layout experiment (block formats only)
        if (fmt == hip::WFmt::Q4_K || fmt == hip::WFmt::Q5_K || fmt == hip::WFmt::IQ4_XS) {
            std::vector<uint8_t> il(t.nbytes); hip::repack_interleave8(fmt, src, il.data(), N, K);
            dWil.resize(ncopy); for (int i = 0; i < ncopy; ++i) { CK(hipMalloc(&dWil[i], t.nbytes)); CK(hipMemcpy(dWil[i], il.data(), t.nbytes, hipMemcpyHostToDevice)); }
        }
        if (K % 256 == 0 && fmt != hip::WFmt::Q5_1_SOA) {   // 16-row interleaved copies for the WMMA kernel (cfg 303)
            const size_t ib = hip::wmma_il_bytes(fmt, N, K); std::vector<uint8_t> il(ib); hip::repack_wmma_il(fmt, src, il.data(), N, K);
            dWil16.resize(ncopy); for (int i = 0; i < ncopy; ++i) { CK(hipMalloc(&dWil16[i], ib)); CK(hipMemcpy(dWil16[i], il.data(), ib, hipMemcpyHostToDevice)); }
        }
    }
    uint8_t* dW = dWs[0];
    printf("  (%d copies rotated to defeat the cache)\n", ncopy);
    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t)16 * K); for (auto& v : hx) v = nd(rng);
    float *dx, *dy; CK(hipMalloc(&dx, hx.size() * 4)); CK(hipMalloc(&dy, (size_t)16 * N * 4));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    hip::XQ8 xq; CK(hipMalloc(&xq.q, hx.size())); CK(hipMalloc(&xq.d, hx.size() / 32 * 4)); CK(hipMalloc(&xq.s, hx.size() / 32 * 4));
    CK(hipMemset(xq.q, 0, hx.size())); CK(hipMemset(xq.d, 0, hx.size() / 32 * 4)); CK(hipMemset(xq.s, 0, hx.size() / 32 * 4));

    std::vector<int> rows; std::uniform_int_distribution<int> rd(0, N - 1); for (int i = 0; i < 64; ++i) rows.push_back(rd(rng));
    std::vector<float> wrow(K);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int vi = 0; vi < 2; ++vi) {
        const bool q8 = vi == 1;
        if (!strcmp(variant, "f32") && q8) continue; if (!strcmp(variant, "q8") && !q8) continue;
        printf("-- variant %s\n", q8 ? "q8 x (int dot4)" : "f32 x");
        for (int ncol = 1; ncol <= ncol_max; ncol = ncol < 4 ? ncol + 1 : ncol + 4) {
            // reference x (per column) for q8: dequantised int8
            std::vector<std::vector<float>> xr(ncol, std::vector<float>(K));
            for (int c = 0; c < ncol; ++c) {
                if (!q8) memcpy(xr[c].data(), hx.data() + (size_t)c * K, K * 4);
                else { std::vector<int8_t> q; std::vector<float> d, s; quantize_ref(hx.data() + (size_t)c * K, K, q, d, s); for (int k = 0; k < K; ++k) xr[c][k] = q[k] * d[k / 32]; }
            }
            for (int cfg : {32, 16, 8, 4, 2, 1, 208, 204, 301, 302, 304, 308, 303, 404}) { if (ncol > 1 && (cfg == 16 || cfg == 8 || cfg == 2 || cfg == 1 || cfg == 208)) continue; if (cfg / 100 == 3 && (K % 256 != 0)) continue; if (cfg == 303 && dWil16.empty()) continue; if (cfg == 404 && dWil.empty()) continue; const int tpr = cfg % 100, rpt = cfg / 100 + 1;
                int it = 0; auto run = [&] { uint8_t* w = (rpt == 5 ? dWil : (rpt == 4 && tpr == 3) ? dWil16 : dWs)[it++ % ncopy]; if (q8) { hip::quantize_x_q8(dx, xq, ncol, K, 0); if (rpt == 4 && ncol > 16) return; hip::gemv_q8(fmt, w, xq, dy, N, K, ncol, tpr, rpt, 0); } else hip::gemv_q4_K_f32(w, dx, dy, N, K, ncol, tpr, 0); };
                it = 0; run(); CK(hipDeviceSynchronize());
                std::vector<float> hy((size_t)ncol * N); CK(hipMemcpy(hy.data(), dy, hy.size() * 4, hipMemcpyDeviceToHost));
                double maxrel = 0, maxrel_f32 = 0;
                for (int r : rows) {
                    hip::dequant_row((int)t.type, t.data + (size_t)r * hip::gtype_row_bytes(t.type, K), wrow.data(), K);
                    for (int c = 0; c < ncol; ++c) {
                        double ref = 0, mag = 0, ref32 = 0;
                        for (int k = 0; k < K; ++k) { ref += (double)wrow[k] * xr[c][k]; mag += fabs((double)wrow[k] * xr[c][k]); ref32 += (double)wrow[k] * hx[(size_t)c * K + k]; }
                        double got = hy[(size_t)c * N + r];
                        maxrel = fmax(maxrel, fabs(ref - got) / (mag + 1e-20)); maxrel_f32 = fmax(maxrel_f32, fabs(ref32 - got) / (mag + 1e-20));
                    }
                }
                const int iters = 20;
                hipEventRecord(a); for (int i = 0; i < iters; ++i) run(); hipEventRecord(b); hipEventSynchronize(b);
                float ms; hipEventElapsedTime(&ms, a, b); ms /= iters;
                const double bytes = (double)t.nbytes + (double)ncol * K * (q8 ? 1 : 4);
                printf("  ncol=%d tpr=%2d rpt=%d: %8.3f ms  %6.1f GB/s  (%3.0f%% of 240)  err_vs_ref=%.1e %s  err_vs_f32=%.1e\n", ncol, tpr, rpt, ms, bytes / ms / 1e6, bytes / ms / 1e6 / 240 * 100,
                       maxrel, maxrel < 1e-5 ? "OK" : "MISMATCH", maxrel_f32);
            }
        }
    }
    if (t.ne[2] > 1) {   // expert tensor: reproduce the engine's decode launch (T=1, 11 gathered experts, x rows per slot for K==640)
        const int E = (int)t.ne[2], Ne = (int)t.ne[1]; const size_t eb = t.nbytes / E; const int nexp = getenv("BENCH_NEXP") ? atoi(getenv("BENCH_NEXP")) : 11;
        const char* idmode = getenv("BENCH_IDS") ? getenv("BENCH_IDS") : "rand";
        std::vector<int> hid(nexp); int* dids; CK(hipMalloc(&dids, nexp * 4)); float* dy2; CK(hipMalloc(&dy2, (size_t)nexp * Ne * 4));
        std::uniform_int_distribution<int> ed(0, E - 1);
        hip::quantize_x_q8(dx, xq, 16, K, 0);
        printf("-- MoE gather: %d experts of [%d x %d] (%.1f MiB per launch)\n", nexp, Ne, K, nexp * eb / 1048576.0);
        for (int tpr : {32, 16, 8, 4, 2, 1}) {
            hip::g_moe_tpr = tpr;
            int it = 0; auto run = [&] { for (int e = 0; e < nexp; ++e) hid[e] = !strcmp(idmode, "same") ? 0 : !strcmp(idmode, "seq") ? e : ed(rng);
                if (strcmp(idmode, "same") || it == 0) CK(hipMemcpyAsync(dids, hid.data(), nexp * 4, hipMemcpyHostToDevice, 0));
                hip::MoeSegs a = {{dWs[it++ % ncopy], nullptr}, {nullptr, nullptr}, {dy2, nullptr}, eb};
                hip::gemv_moe(fmt, fmt, a, 1, dids, nexp, xq, Ne, K, 1, K == 640, 0); };
            run(); CK(hipDeviceSynchronize());
            const int iters = 50; hipEventRecord(a); for (int i = 0; i < iters; ++i) run(); hipEventRecord(b); hipEventSynchronize(b);
            float ms; hipEventElapsedTime(&ms, a, b); ms /= iters;
            printf("  moe tpr=%2d: %8.3f ms  %6.1f GB/s  (%3.0f%% of 240)\n", tpr, ms, nexp * eb / ms / 1e6, nexp * eb / ms / 1e6 / 240 * 100);
        }
    }
    return 0;
}
