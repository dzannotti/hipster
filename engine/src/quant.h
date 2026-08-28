// ggml block formats as they sit in the GGUF, plus CPU reference dequantisation.
// The reference is the contract every GPU kernel is tested against.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace hip {

using f16_t = uint16_t;
inline float f16_to_f32(f16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff; uint32_t o;
    if (e == 0) { if (m == 0) o = s << 31; else { e = 1; while (!(m & 0x400)) { m <<= 1; --e; } m &= 0x3ff; o = (s << 31) | ((e + 112) << 23) | (m << 13); } }
    else if (e == 31) o = (s << 31) | 0x7f800000 | (m << 13);
    else o = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &o, 4); return f;
}

#include "iq3s_grid.h"
#define QK_K 256

struct block_q4_K { f16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };   // 144 B
struct block_q5_K { f16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };  // 176 B
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; f16_t d; };  // 210 B
struct block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; f16_t d; }; // 110 B
struct block_q8_0 { f16_t d; int8_t qs[32]; };                                // 34 B
struct block_q5_1 { f16_t d, m; uint8_t qh[4]; uint8_t qs[16]; };             // 24 B
struct block_iq4_nl { f16_t d; uint8_t qs[16]; };                             // 18 B
struct block_iq3_s { f16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; };  // 110 B
struct block_iq4_xs { f16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };  // 136 B
static_assert(sizeof(block_iq3_s) == 110 && sizeof(block_q4_K) == 144 && sizeof(block_q5_K) == 176 && sizeof(block_q6_K) == 210 &&
              sizeof(block_q3_K) == 110 && sizeof(block_q8_0) == 34 && sizeof(block_q5_1) == 24 &&
              sizeof(block_iq4_nl) == 18 && sizeof(block_iq4_xs) == 136, "block sizes");

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// ---- reference dequant of one block into y[block_elems] (mirrors ggml-quants.c) ----
inline void dequant_q4_K(const block_q4_K* x, float* y) {
    const float d = f16_to_f32(x->d), min = f16_to_f32(x->dmin);
    const uint8_t* q = x->qs; int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m; get_scale_min_k4(is + 0, x->scales, &sc, &m); const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m); const float d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}
inline void dequant_q5_K(const block_q5_K* x, float* y) {
    const float d = f16_to_f32(x->d), min = f16_to_f32(x->dmin);
    const uint8_t* ql = x->qs; const uint8_t* qh = x->qh; int is = 0; uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m; get_scale_min_k4(is + 0, x->scales, &sc, &m); const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m); const float d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
inline void dequant_q6_K(const block_q6_K* x, float* y) {
    const float d = f16_to_f32(x->d);
    const uint8_t* ql = x->ql; const uint8_t* qh = x->qh; const int8_t* sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1; y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}
inline void dequant_q8_0(const block_q8_0* x, float* y) { const float d = f16_to_f32(x->d); for (int j = 0; j < 32; ++j) y[j] = x->qs[j] * d; }
inline void dequant_q5_1(const block_q5_1* x, float* y) {
    const float d = f16_to_f32(x->d), m = f16_to_f32(x->m); uint32_t qh; memcpy(&qh, x->qh, 4);
    for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10, xh_1 = ((qh >> (j + 12))) & 0x10;
        y[j] = ((x->qs[j] & 0x0F) | xh_0) * d + m; y[j + 16] = ((x->qs[j] >> 4) | xh_1) * d + m;
    }
}
inline void dequant_iq4_nl(const block_iq4_nl* x, float* y) {
    const float d = f16_to_f32(x->d);
    for (int j = 0; j < 16; ++j) { y[j] = d * kvalues_iq4nl[x->qs[j] & 0xf]; y[j + 16] = d * kvalues_iq4nl[x->qs[j] >> 4]; }
}
inline void dequant_iq4_xs(const block_iq4_xs* x, float* y) {
    const float d = f16_to_f32(x->d); const uint8_t* qs = x->qs;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) { y[j] = dl * kvalues_iq4nl[qs[j] & 0xf]; y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4]; }
        y += 32; qs += 16;
    }
}
inline void dequant_q3_K(const block_q3_K* x, float* y) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const float d_all = f16_to_f32(x->d);
    const uint8_t* q = x->qs; const uint8_t* hm = x->hmask; uint8_t m = 1;
    uint32_t aux[4]; memcpy(aux, x->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = (const int8_t*)aux;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2; m <<= 1;
        }
        q += 32;
    }
}

inline void dequant_iq3_s(const block_iq3_s* x, float* y) {
    const float d = f16_to_f32(x->d);
    const uint8_t* qs = x->qs; const uint8_t* qh = x->qh; const uint8_t* signs = x->signs;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        const float db1 = d * (1 + 2 * (x->scales[ib32 / 2] & 0xf));
        const float db2 = d * (1 + 2 * (x->scales[ib32 / 2] >> 4));
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
            const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; ++j) { y[j + 0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f); y[j + 4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f); }
            y += 8;
        }
        qs += 8; signs += 4;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
            const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; ++j) { y[j + 0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f); y[j + 4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f); }
            y += 8;
        }
        qs += 8; signs += 4; qh += 2;
    }
}

inline uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000; int e = ((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) { if (e < -10) return (uint16_t)s; m |= 0x800000; uint32_t t = 14 - e; uint32_t r = m >> t; if ((m >> (t - 1)) & 1) r++; return (uint16_t)(s | r); }
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    uint32_t r = s | (e << 10) | (m >> 13); if (m & 0x1000) r++; return (uint16_t)r;
}
// Quantise a float row to Q8_0 row-SoA ([q K int8][d K/32 f16]) — load-time conversion of rare formats.
inline void quantize_row_q8_0_soa(const float* x, uint8_t* dst, uint64_t K) {
    int8_t* q = (int8_t*)dst; uint16_t* dd = (uint16_t*)(dst + K);
    for (uint64_t b = 0; b < K / 32; ++b) {
        float amax = 0; for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        dd[b] = f32_to_f16(amax / 127.f);
        const float dq = f16_to_f32(dd[b]), idq = dq ? 1.f / dq : 0.f;
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t)nearbyintf(x[b * 32 + j] * idq);
    }
}

// Dequantise one whole row of `ne0` elements of the given type.
inline void dequant_row(int type, const uint8_t* src, float* dst, uint64_t ne0) {
    switch (type) {
        case 12: for (uint64_t b = 0; b < ne0 / 256; ++b) dequant_q4_K((const block_q4_K*)src + b, dst + b * 256); break;
        case 13: for (uint64_t b = 0; b < ne0 / 256; ++b) dequant_q5_K((const block_q5_K*)src + b, dst + b * 256); break;
        case 14: for (uint64_t b = 0; b < ne0 / 256; ++b) dequant_q6_K((const block_q6_K*)src + b, dst + b * 256); break;
        case 11: for (uint64_t b = 0; b < ne0 / 256; ++b) dequant_q3_K((const block_q3_K*)src + b, dst + b * 256); break;
        case 21: for (uint64_t b = 0; b < ne0 / 256; ++b) dequant_iq3_s((const block_iq3_s*)src + b, dst + b * 256); break;
        case 23: for (uint64_t b = 0; b < ne0 / 256; ++b) dequant_iq4_xs((const block_iq4_xs*)src + b, dst + b * 256); break;
        case 8:  for (uint64_t b = 0; b < ne0 / 32; ++b) dequant_q8_0((const block_q8_0*)src + b, dst + b * 32); break;
        case 7:  for (uint64_t b = 0; b < ne0 / 32; ++b) dequant_q5_1((const block_q5_1*)src + b, dst + b * 32); break;
        case 20: for (uint64_t b = 0; b < ne0 / 32; ++b) dequant_iq4_nl((const block_iq4_nl*)src + b, dst + b * 32); break;
        case 0:  memcpy(dst, src, ne0 * 4); break;
        case 1:  for (uint64_t i = 0; i < ne0; ++i) dst[i] = f16_to_f32(((const f16_t*)src)[i]); break;
        case 30: for (uint64_t i = 0; i < ne0; ++i) { uint32_t u = (uint32_t)((const uint16_t*)src)[i] << 16; memcpy(dst + i, &u, 4); } break;
        default: for (uint64_t i = 0; i < ne0; ++i) dst[i] = NAN;
    }
}

}  // namespace hip
