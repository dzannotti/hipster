// test_fnkernels: synthetic-input checks of the Flash-Next prefill kernels against the decode kernels.
//  1. GDN: gdn_prep/gdn_scan/gdn_out_sig over T tokens vs gdn_step_sig (sequential, T tokens): outputs + state.
//  2. attention 24q/2kv: attn_stage1_24_2 + attn_prefill_24_2 (bf16 out) vs attn_decode_24_2 (f32 out) on the same KV.
#include "../kernels/ops.h"
#include "../kernels/gemv.h"
#include "../kernels/qsa.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
template <class T> static T* up(const std::vector<T>& v) { T* d; CK(hipMalloc(&d, v.size() * sizeof(T))); CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }
template <class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> v(n); CK(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }
static void report(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    double md = 0, mx = 0, se = 0, sa = 0; for (size_t i = 0; i < a.size(); ++i) { md = fmax(md, fabs((double)a[i] - b[i])); mx = fmax(mx, fabs((double)b[i])); se += ((double)a[i] - b[i]) * ((double)a[i] - b[i]); sa += (double)b[i] * b[i]; }
    printf("  %-40s max|diff| %.3e / max|ref| %.3e = %.3f%%   rel rms %.3e\n", what, md, mx, 100 * md / mx, sqrt(se / sa));
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 29; const float eps = 1e-6f;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    auto randv = [&](size_t n, float sc) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sc; return v; };
    hipStream_t s = 0;
    // ---------- GDN ----------
    {
        const int QKV = 10240, Z = 6144, H = 48, D = 128;
        auto qkv = randv((size_t)T * QKV, 1.f), z = randv((size_t)T * Z, 1.f), b = randv((size_t)T * 48, 1.f), a = randv((size_t)T * 48, 1.f);
        auto dt = randv(48, 0.5f), an = randv(48, 1.f); for (auto& v : an) v = -fabsf(v); auto nw = randv(D, 0.3f); for (auto& v : nw) v += 1.f;
        auto st0 = randv((size_t)H * D * D, 0.1f);
        float *dqkv = up(qkv), *dz = up(z), *db = up(b), *da = up(a), *ddt = up(dt), *dan = up(an), *dnw = up(nw), *dst_in = up(st0);
        float *st_a, *st_b, *out_a, *out_b, *qn, *raw; CK(hipMalloc(&st_a, st0.size() * 4)); CK(hipMalloc(&st_b, st0.size() * 4));
        CK(hipMalloc(&out_a, (size_t)T * Z * 4)); CK(hipMalloc(&out_b, (size_t)T * Z * 4)); CK(hipMalloc(&qn, (size_t)T * (4096 + 96) * 4)); CK(hipMalloc(&raw, (size_t)T * Z * 4));
        uint16_t* xb; CK(hipMalloc(&xb, (size_t)T * Z * 2));
        hip::XQ8 xq; CK(hipMalloc(&xq.q, (size_t)T * Z)); CK(hipMalloc(&xq.d, (size_t)T * Z / 32 * 4)); CK(hipMalloc(&xq.s, (size_t)T * Z / 32 * 4));
        hip::gdn_step_sig(dqkv, dz, db, da, T, ddt, dan, dst_in, st_a, dnw, out_a, xq, eps, s);
        float* qnp = qn; float* knp = qn + (size_t)T * 2048; float* bt = qn + (size_t)T * 4096; float* dc = bt + (size_t)T * 48;
        hip::gdn_prep(dqkv, db, da, ddt, dan, T, qnp, knp, bt, dc, eps, s);
        hip::gdn_scan(qnp, knp, dqkv, bt, dc, T, dst_in, st_b, raw, s);
        hip::gdn_out_sig(raw, dz, dnw, T, out_b, xb, eps, s);
        CK(hipDeviceSynchronize());
        auto oa = down(out_a, (size_t)T * Z), ob = down(out_b, (size_t)T * Z), sa = down(st_a, st0.size()), sb = down(st_b, st0.size());
        printf("GDN (T=%d):\n", T);
        report("out: prefill scan vs sequential step", ob, oa);
        report("state after T tokens", sb, sa);
        auto xbh = down(xb, (size_t)T * Z); std::vector<float> xbf(xbh.size()); for (size_t i = 0; i < xbh.size(); ++i) xbf[i] = bf2f(xbh[i]);
        report("out bf16 copy vs f32 (rounding only)", xbf, ob);
        // per-token: where does the divergence start?
        for (int t : {0, 1, 2, T / 2, T - 1}) { double md = 0, mx = 0; for (int i = 0; i < Z; ++i) { md = fmax(md, fabs((double)oa[(size_t)t * Z + i] - ob[(size_t)t * Z + i])); mx = fmax(mx, fabs((double)oa[(size_t)t * Z + i])); } printf("    token %2d: max|diff| %.3e (%.3f%%)\n", t, md, 100 * md / mx); }
    }
    // ---------- attention 24/2 ----------
    {
        const int NQ = 24, NKV = 2, HD = 256, max_ctx = 512, pos0 = 0;
        auto q = randv((size_t)T * NQ * 512, 1.f), k = randv((size_t)T * NKV * HD, 1.f), v = randv((size_t)T * NKV * HD, 1.f);
        auto qw = randv(HD, 0.2f), kw = randv(HD, 0.2f); for (auto& x : qw) x += 1.f; for (auto& x : kw) x += 1.f;
        float *dq1 = up(q), *dq2 = up(q), *dk1 = up(k), *dk2 = up(k), *dv = up(v), *dqw = up(qw), *dkw = up(kw);
        const size_t kvn = (size_t)NKV * max_ctx * HD, kvtn = (size_t)NKV * HD * (max_ctx + 128);
        uint16_t *kc1, *vc1, *kc2, *vc2; CK(hipMalloc(&kc1, kvn * 2)); CK(hipMalloc(&vc1, kvtn * 2)); CK(hipMalloc(&kc2, kvn * 2)); CK(hipMalloc(&vc2, kvtn * 2));
        CK(hipMemset(vc1, 0, kvtn * 2)); CK(hipMemset(vc2, 0, kvtn * 2));
        float* out1; CK(hipMalloc(&out1, (size_t)T * NQ * HD * 4)); uint16_t* out2; CK(hipMalloc(&out2, (size_t)T * NQ * HD * 2));
        hip::XQ8 xq; CK(hipMalloc(&xq.q, (size_t)T * NQ * HD)); CK(hipMalloc(&xq.d, (size_t)T * NQ * HD / 32 * 4)); CK(hipMalloc(&xq.s, (size_t)T * NQ * HD / 32 * 4));
        hip::attn_decode_24_2(dq1, dk1, dv, T, dqw, dkw, 1e7f, pos0, kc1, vc1, max_ctx, out1, xq, eps, s);
        hip::attn_stage1_24_2(dq2, dk2, dv, T, dqw, dkw, 1e7f, pos0, kc2, vc2, max_ctx, eps, s);
        hip::attn_prefill_24_2(dq2, kc2, vc2, T, pos0, max_ctx, out2, s);
        CK(hipDeviceSynchronize());
        auto o1 = down(out1, (size_t)T * NQ * HD); auto o2h = down(out2, (size_t)T * NQ * HD); std::vector<float> o2(o2h.size()); for (size_t i = 0; i < o2.size(); ++i) o2[i] = bf2f(o2h[i]);
        printf("attention 24/2 (T=%d):\n", T);
        report("out: flash prefill (bf16) vs decode", o2, o1);
        auto k1 = down(kc1, kvn), k2 = down(kc2, kvn); size_t nd_ = 0; for (size_t i = 0; i < kvn; ++i) nd_ += k1[i] != k2[i]; printf("  KV cache K entries differing: %zu\n", nd_);
        for (int t : {0, 1, T / 2, T - 1}) { double md = 0, mx = 0; for (int i = 0; i < NQ * HD; ++i) { md = fmax(md, fabs((double)o1[(size_t)t * NQ * HD + i] - o2[(size_t)t * NQ * HD + i])); mx = fmax(mx, fabs((double)o1[(size_t)t * NQ * HD + i])); } printf("    query %2d: max|diff| %.3e (%.3f%%)\n", t, md, 100 * md / mx); }
        // 3. gather attention (dense list, row-major V) vs the dense decode kernel, nsplit 1 and 16
        float *dq3 = up(q), *dk3 = up(k); uint16_t *kc3, *vc3; CK(hipMalloc(&kc3, kvn * 2)); CK(hipMalloc(&vc3, kvn * 2));
        std::vector<int> pos(T), kvs(T, 0); for (int i = 0; i < T; ++i) pos[i] = i; int *dpos = up(pos), *dkv = up(kvs);
        hip::attn_rope_kv_24_2_rowv_pos0(dq3, dk3, dv, T, dqw, dkw, 1e7f, pos0, kc3, vc3, max_ctx, eps, s);
        float* part; CK(hipMalloc(&part, (size_t)T * 2 * 16 * 12 * 258 * 4)); float* out3; CK(hipMalloc(&out3, (size_t)T * NQ * HD * 4));
        for (int nsplit : {1, 16}) {
            hip::qsa::attend(dq3, kc3, vc3, kvn, max_ctx, dpos, dkv, nullptr, nullptr, T, nsplit, part, out3, xq, s);
            CK(hipDeviceSynchronize()); auto o3 = down(out3, (size_t)T * NQ * HD);
            char what[64]; snprintf(what, sizeof what, "gather attention (dense, nsplit %d) vs decode", nsplit); report(what, o3, o1);
        }
    }
    return 0;
}
