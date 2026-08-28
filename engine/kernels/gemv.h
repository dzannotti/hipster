#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace hip {
extern int g_persist_blocks;   // experiment knob: cap GEMV grids (persistent row loop); 0 = off
// x quantised to int8 per 32-block: q8 [ncol][K] int8, d [ncol][K/32] f32 (scale), s [ncol][K/32] f32 (d*sum(q)).
struct XQ8 { int8_t* q; float* d; float* s; };
void quantize_x_q8(const float* x, XQ8 out, int ncol, int K, hipStream_t s);

// Weight layouts the decode GEMV understands. GGUF-native where the block is 16-byte friendly,
// row-SoA repacks where it is not (Q6_K: 210-byte blocks).
enum class WFmt { Q4_K, Q5_K, IQ4_XS, Q6_K_SOA, Q8_0_SOA, Q5_1_SOA };
// Repack one GGUF row into the SoA layout (same byte count). K multiple of 256.
void repack_row_q6_K(const uint8_t* src, uint8_t* dst, int K);   // [ql K/2][qh K/4][scales K/16][d K/256*2]
void repack_row_q8_0(const uint8_t* src, uint8_t* dst, int K);   // [q K][d K/32*2]
void repack_row_q5_1(const uint8_t* src, uint8_t* dst, int K);
void repack_row_q5_0(const uint8_t* src, uint8_t* dst, int K);   // -> the Q5_1 SoA layout (exact), row bytes = K*24/32   // [qs K/2][qh K/8][d K/32*2][m K/32*2]
size_t wfmt_row_bytes(WFmt f, int K);
// experiment: block-interleaved layout in groups of 8 rows (Q4_K/Q5_K/IQ4_XS). N must be a multiple of 8.
void repack_interleave8(WFmt f, const uint8_t* src, uint8_t* dst, int N, int K);

// y[col][n] = sum_k W[n][k] * x[col][k]   for col < ncol (ncol <= 8). y f32 [ncol][N]. K multiple of 256.
// tpr = threads per row (4/8/16/32) — chosen by benchmark.
void gemv_q8(WFmt f, const void* W, XQ8 x, float* y, int N, int K, int ncol, int tpr, int rpt, hipStream_t s);  // rpt = rows per thread (1/2)
// Several weights sharing the same x in ONE launch (fewer launches, bigger grids). ncol == 1 only.
struct GemvSeg { WFmt fmt; const void* w; float* y; int N, K; };
void gemv_multi(const GemvSeg* segs, int nseg, XQ8 x, int ncol, hipStream_t s);   // ncol x rows (spaced by K) in one launch per 8 columns
// int8-WMMA GEMV for 1..8 columns: same cost for every ncol, every column reduced in the same order (T-invariant).
// x must have 16 readable rows. Formats Q4_K/Q5_K/IQ4_XS/Q6_K_SOA/Q8_0_SOA, K % 256 == 0 (gemv_wmma_ok).
bool gemv_wmma_ok(const GemvSeg* segs, int nseg);
void gemv_wmma(const GemvSeg* segs, int nseg, XQ8 x, int ncol, int mode, hipStream_t s);   // mode: 0 auto | 1/2/4/8 split-K waves per tile | 3 interleaved layout
size_t wmma_il_bytes(WFmt f, int N, int K);
void repack_wmma_il(WFmt f, const uint8_t* src, uint8_t* dst, int N, int K);
// MoE: for token t < T and slot e < nexp, y[g][(t*nexp+e)][N] = W[g][expert ids[t*nexp+e]] . x[row] for segment g < nseg
// (gate|up share one launch). ids < 0 selects the shared-expert matrices `shared[g]`. xrow_te: x row = t*nexp+e (down) else t.
// slotw: per-slot combine weights [T*nexp]; when set (down-projection), y[0][t][N] = sum_e slotw[te] * W_e . x[te], one
// thread group per (t, row) looping over the slots in order (deterministic, no atomics)
struct MoeSegs { const void* w[2]; const void* shared[2]; float* y[2]; size_t ebytes; const float* slotw = nullptr; };
void gemv_moe(WFmt f, WFmt shared_fmt, const MoeSegs& a, int nseg, const int* ids, int nexp, XQ8 x, int N, int K, int T, bool xrow_te, hipStream_t s);
// split-K GEMV for skinny weights (N small, K long): partial sums part[ncol][ksplit][N], summed by the consumer in a fixed order
// MoE grouped GEMM (prefill): slots sorted by expert into tiles of <= 16 columns; x columns at rows rowtab[k] (or k);
// y[k][N]. expert < 0 in a tile selects the shared matrix Wsh (Q8_0). Weights decoded to iu8 WMMA fragments.
struct MoeTile { int expert, k0, ncols; };
void moe_gemm(WFmt f, WFmt fs, const void* W, const void* Wsh, size_t ebytes, const MoeTile* tiles, int ntiles, const MoeTile* stiles, int nstiles, int ct,
              const int* rowtab, XQ8 x, float* y, int N, int K, hipStream_t s);   // routed tiles + shared-expert tiles (separate launches, one format each); ct = 16-column tiles per block
// hyper-connection up-projection (Q8_0 [N][320]) with the silu(sum of split-K partials / 4) + q8 prologue in LDS: y[T][N]
void hc_up_fused(const void* W, const float* part, int ksplit, float* y, int N, int T, hipStream_t s);
extern int g_moe_tpr;
extern int g_moe_fake;   // bench knob: bf16 MoE GEMM with constant weight fragments (WMMA + LDS floor)   // bench knob: force lanes/row in gemv_moe (0 = policy)
void gemv_splitk(WFmt f, const void* W, XQ8 x, float* part, int N, int K, int ksplit, int ncol, hipStream_t s);
// f32-x baseline, Q4_K only (kept for the record).
void gemv_q4_K_f32(const void* W, const float* x, float* y, int N, int K, int ncol, int tpr, hipStream_t s);
}
