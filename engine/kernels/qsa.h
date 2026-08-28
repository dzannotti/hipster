// Qwen3.8-Flash-Next QSA (quantised sparse attention) kernels: indexer keys, block pooling, top-k selection over
// 4-token blocks, and gather attention over a per-query key list. Nothing here is shared with the 27B.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "gemv.h"
#include "ops.h"

namespace hip::qsa {
constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4, BUDGET = 2048, WIDTH = BUDGET + R - 1;   // 2051 keys per query

// y[t][N] = sum_k bf16(x[t][k]) * W[n][k]  (W bf16 [N][K], f32 accumulate; x rounded to bf16 as ggml does)
void bf16_gemv(const uint16_t* W, const float* x, float* y, int N, int K, int T, hipStream_t s);
// raw indexer keys k[t][128] f32 -> cache f16 at (slot rb.kv[t], position rb.pos[t]); cache layout [slot][pos][128]
void write_k(const float* k, const int* rpos, const int* rkv, uint16_t* cache, size_t slot_stride, int T, hipStream_t s);   // rpos/rkv: device arrays [T]
// pooled keys for blocks [b0, b1) of one slot: mean of the 4 raw keys -> rmsnorm * w_kn -> rope(pos = 4b) -> pooled[b][128] f32
void pool(const uint16_t* cache, const float* w_kn, float* pooled, int b0, int b1, float eps, float rope_base, hipStream_t s);
// query heads: q_raw[t][4][128] f32 -> rmsnorm * w_qn -> rope(pos[t]) -> q[t][4][128]
void query(const float* q_raw, const int* rpos, const float* w_qn, float* q, int T, float eps, float rope_base, hipStream_t s);
// per query row t (position p = rb.pos[t], slot rb.kv[t]): the attended key set idx[t][WIDTH], count cnt[t]:
// all keys when p+1 <= WIDTH; otherwise the top-scoring complete 4-token blocks (score = sum_h relu(q_h . pooled_b),
// ties -> lower block id) up to WIDTH - tail keys, plus the incomplete tail (positions >= 4*floor((p+1)/4)).
// scratch: f32 [T][max_blocks]. pooled: [slot][max_blocks][128].
void select(const float* q, const int* rpos, const int* rkv, const float* pooled, size_t pooled_slot_stride, int max_blocks, float* scratch, int* idx, int* cnt, int T, hipStream_t s);   // T <= 256 per call
// gather attention over idx lists: qf [T][24][512] (normed + roped, gate in the second half), K/V caches row-major
// [slot][kh][pos][256] f16 (slot strides in elements), idx [T][WIDTH] + cnt[T] (idx == null: dense 0..pos).
// nsplit key-range splits per (row, kv head) -> partials [T][2][nsplit][12][258], merged into out f32 [T][24][256] + xq.
void attend(const float* qf, const uint16_t* kc, const uint16_t* vc, size_t kv_slot_stride, int max_ctx, const int* rpos, const int* rkv,
            const int* idx, const int* cnt, int T, int nsplit, float* partials, float* out, XQ8 xq, hipStream_t s);   // xq.q may be null
}  // namespace hip::qsa
