// DFlash2 draft kernels for the 27B (incoai/Qwen3.8-27B-DFlash2): 5-layer block-diffusion draft, head 128, 32 q / 8 kv
// heads, NEOX rope over the full 128 dims, 2-tap grouped dynamic conv, windowed non-causal block attention, selector.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "gemv.h"

namespace hip::dfl {
constexpr int E = 5120, HD = 128, NQ = 32, NKV = 8, CONV_G = 16, N_GROUPS = E / CONV_G, WINDOW = 2048, TOPK = 16, RANK = 256, LATTICE = TOPK + TOPK * TOPK;
// two-tap grouped dynamic conv inside one block of T rows: out[i][c] = w0*x[i][c] + w1*x[i-1][c] (x[-1] = 0),
// w_tap = base[side][tap][c] + delta[i][group(c) + 320*(tap + 2*side)]. out f32 (may be null) and/or q8 rows of E.
// rows are blocks of B consecutive rows (one draft block per sequence); the conv never crosses a block boundary
struct Blk { int slot, key_end, anchor; };
void conv(const float* x, const float* delta, const float* base, int side, int T, int B, float* out, XQ8 xq, hipStream_t s);
// per (row, head): rmsnorm(128)*w then NEOX rope at pos[t]; q in place (q may be null: injection), k -> cache K[kh][pos][128],
// v -> cache V (row-major f16). Caches of one layer.
// blk (device, one per block of B rows) selects the slot's caches: cache + blk[t/B].slot * slot_stride. blk may be null (slot 0).
void qk_rope_kv(float* q, float* k, const float* v, const float* qw, const float* kw, float base, const int* pos, int T, int B, const Blk* blk, size_t slot_stride,
                uint16_t* kc, uint16_t* vc, int max_ctx, float eps, hipStream_t s);
// attention for T block rows: query t at pos[t] attends keys [max(0, pos[t]-WINDOW+1), key_end) of one layer's cache
// (non-causal inside the block), scale 1/sqrt(128); out f32 [T][32][128] + q8 rows of 4096
void attend(const float* q, const uint16_t* kc, const uint16_t* vc, const int* pos, int T, int B, const Blk* blk, size_t slot_stride, int max_ctx, float* out, XQ8 xq, hipStream_t s);
// selector lattice for positions 1..T-1: cand ids/logits (top-16 per row), gate [T][256]; A = predecessor, B = successor
// (Q4_K [V][256]); lattice[i][0..15] = ids (as float), [16 + p*16 + c] = <A[pred_p] * gate_i, B[cand_c]> + logit_c
void selector(const int* ids, const float* vals, const float* gate, const void* A, const void* Bm, const Blk* blk, int S, int B, float* lattice, hipStream_t s);
}  // namespace hip::dfl
