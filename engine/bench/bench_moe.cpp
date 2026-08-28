// bench_moe <gguf> <expert tensor> [T] : isolated MoE grouped-GEMM benchmark. T tokens x 10 random experts (+ the
// shared slot omitted), slots sorted by expert into tiles; times moe_gemm (int8 x) and moe_gemm_bf16 for CT = 1..3.
// Validates a few outputs against a CPU reference (dequantised weights . x).
#include "../src/gguf.h"
#include "../src/quant.h"
#include "../kernels/gemv.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFFu + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s gguf tensor [T]\n", argv[0]); return 2; }
    hip::GGUF g(argv[1]); const auto& t = g.get(argv[2]);
    const int K = (int)t.ne[0], N = (int)t.ne[1], E = (int)t.ne[2], T = argc > 3 ? atoi(argv[3]) : 2048, NEXP = 10;
    printf("%s %s: %d experts of [%d x %d], T=%d\n", t.name.c_str(), hip::gtype_name(t.type), E, N, K, T);
    // upload in the engine's layout (block formats as-is, Q5_1/Q8_0 rows repacked)
    hip::WFmt fmt; const size_t rb = hip::gtype_row_bytes(t.type, K); std::vector<uint8_t> tmp; const uint8_t* src = t.data;
    switch (t.type) {
        case hip::GType::Q4_K: fmt = hip::WFmt::Q4_K; break; case hip::GType::Q5_K: fmt = hip::WFmt::Q5_K; break;
        case hip::GType::Q5_1: fmt = hip::WFmt::Q5_1_SOA; tmp.resize(t.nbytes); for (int r = 0; r < N * E; ++r) hip::repack_row_q5_1(t.data + (size_t)r * rb, tmp.data() + (size_t)r * rb, K); src = tmp.data(); break;
        case hip::GType::Q8_0: fmt = hip::WFmt::Q8_0_SOA; tmp.resize(t.nbytes); for (int r = 0; r < N * E; ++r) hip::repack_row_q8_0(t.data + (size_t)r * rb, tmp.data() + (size_t)r * rb, K); src = tmp.data(); break;
        default: fprintf(stderr, "format not wired\n"); return 1;
    }
    uint8_t* dW; CK(hipMalloc(&dW, t.nbytes)); CK(hipMemcpy(dW, src, t.nbytes, hipMemcpyHostToDevice));
    const size_t ebytes = t.nbytes / E;
    // x rows [T][K] f32 -> bf16 rows and q8 rows
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t)T * K); for (auto& v : hx) v = nd(rng);
    std::vector<uint16_t> hxb(hx.size()); for (size_t i = 0; i < hx.size(); ++i) hxb[i] = f2bf(hx[i]);
    float* dx; CK(hipMalloc(&dx, hx.size() * 4)); CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    uint16_t* dxb; CK(hipMalloc(&dxb, hxb.size() * 2)); CK(hipMemcpy(dxb, hxb.data(), hxb.size() * 2, hipMemcpyHostToDevice));
    hip::XQ8 xq; CK(hipMalloc(&xq.q, hx.size())); CK(hipMalloc(&xq.d, hx.size() / 32 * 4)); CK(hipMalloc(&xq.s, hx.size() / 32 * 4));
    hip::quantize_x_q8(dx, xq, T, K, 0);
    // routing: 10 distinct random experts per token; sorted slots -> tiles (as the engine)
    const int NS = T * NEXP; std::vector<int> eid(NS); std::uniform_int_distribution<int> ed(0, E - 1);
    for (int tk = 0; tk < T; ++tk) { std::vector<int> pick; while ((int)pick.size() < NEXP) { int e = ed(rng); if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); } for (int i = 0; i < NEXP; ++i) eid[tk * NEXP + i] = pick[i]; }
    std::vector<int> order(NS), rowtok(NS); for (int i = 0; i < NS; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return eid[a] != eid[b] ? eid[a] < eid[b] : a < b; });
    int* d_rowtok; CK(hipMalloc(&d_rowtok, NS * 4)); float* dy; CK(hipMalloc(&dy, (size_t)NS * N * 4));
    hip::MoeTile* d_tiles; CK(hipMalloc(&d_tiles, (NS + E) * sizeof(hip::MoeTile)));
    std::vector<float> ref; std::vector<int> ref_k;   // reference for 8 sorted positions
    std::vector<float> wrow(K);
    for (int ct = 1; ct <= 3; ++ct) {
        const int maxc = 16 * ct; std::vector<hip::MoeTile> tiles;
        for (int k = 0; k < NS; ++k) { const int te = order[k]; rowtok[k] = te / NEXP; if (k == 0 || eid[te] != eid[order[k - 1]] || tiles.back().ncols == maxc) tiles.push_back({eid[te], k, 1}); else ++tiles.back().ncols; }
        CK(hipMemcpy(d_rowtok, rowtok.data(), NS * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(hip::MoeTile), hipMemcpyHostToDevice));
        const double flop = 2.0 * NS * N * K, bytes = (double)std::min<size_t>(E, tiles.size()) * ebytes;
        for (int variant = 0; variant < 3; ++variant) {
            hip::g_moe_fake = variant == 2;
            auto run = [&] { if (variant == 0) hip::moe_gemm(fmt, hip::WFmt::Q8_0_SOA, dW, dW, ebytes, d_tiles, (int)tiles.size(), nullptr, 0, ct, d_rowtok, xq, dy, N, K, 0);
                             else if (variant >= 1) hip::moe_gemm_bf16(fmt, hip::WFmt::Q8_0_SOA, dW, dW, ebytes, d_tiles, (int)tiles.size(), nullptr, 0, ct, d_rowtok, dxb, dy, N, K, 0); };
            run(); CK(hipDeviceSynchronize());
            // check 8 outputs vs CPU (dequantised weights . bf16-rounded / q8-rounded x)
            std::vector<float> hy((size_t)NS * N); CK(hipMemcpy(hy.data(), dy, hy.size() * 4, hipMemcpyDeviceToHost));
            double maxrel = 0;
            for (int i = 0; i < 8; ++i) { const int k = (int)((size_t)i * 977 % NS), te = order[k], e = eid[te], tk = te / NEXP, row = (int)((size_t)i * 131 % N);
                hip::dequant_row((int)t.type, t.data + ((size_t)e * N + row) * rb, wrow.data(), K);
                double r = 0, mag = 0; for (int kk = 0; kk < K; ++kk) { const double xv = variant ? bf2f(hxb[(size_t)tk * K + kk]) : hx[(size_t)tk * K + kk]; r += (double)wrow[kk] * xv; mag += fabs((double)wrow[kk] * xv); }
                maxrel = fmax(maxrel, fabs(r - hy[(size_t)k * N + row]) / (mag + 1e-20)); }
            hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); const int it = 5;
            hipEventRecord(a); for (int r = 0; r < it; ++r) run(); hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms, a, b); ms /= it;
            printf("  %s CT=%d: %4zu tiles  %8.2f ms  %6.1f TOPS  weights %.2f GB -> %5.0f GB/s per pass  maxrel %.1e %s\n", variant == 0 ? "int8     " : variant == 1 ? "bf16     " : "bf16 FAKE", ct, tiles.size(), ms, flop / ms / 1e9, bytes / 1e9, bytes / ms / 1e6, maxrel, variant == 2 ? "(floor: no dequant)" : maxrel < (variant ? 2e-2 : 1e-2) ? "OK" : "MISMATCH");
        }
    }
    return 0;
}
