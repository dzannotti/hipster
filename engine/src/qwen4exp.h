// Qwen3.8-Flash-Next (GGUF arch qwen4exp) — its own engine. Decode path first (T <= 8 tokens per
// pass, dense attention, which is exact below the QSA budget of 2051 cached tokens). Shares only
// compile-time kernels with the 27B (GEMV formats, GDN step with the sigmoid gate, attention 24/2).
#pragma once
#include "gguf.h"
#include "../kernels/gemv.h"
#include "../kernels/ops.h"
#include "../kernels/gemm.h"
#include "../kernels/qsa.h"
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <set>
#include <map>

namespace hip {

struct FnDims {
    static constexpr int n_embd = 2560, hc = 4, wide = 10240, hc_lr = 320, n_layer = 48, n_vocab = 248320;
    static constexpr int n_head = 24, n_head_kv = 2, head_dim = 256;
    static constexpr int gdn_qkv = 10240, gdn_z = 6144, gdn_v_heads = 48, gdn_dim = 128, n_gdn = 36, n_attn = 12;
    static constexpr int n_exp = 512, n_used = 10, exp_ff = 640;
    static constexpr int ple_layer = 1, ple_heads = 16, ple_dim = 160, ple_hist = 9;
    static constexpr int qsa_dense_limit = 2048 + 3;   // dense attention is exact below this many cached tokens
    static constexpr int max_T = 8;
    static constexpr float rms_eps = 1e-6f, rope_base = 1e7f;
    static constexpr bool is_attn(int il) { return (il + 1) % 4 == 0; }
};

struct FnLin { WFmt fmt; uint8_t* w = nullptr; int N = 0, K = 0; size_t bytes = 0; };

struct FnLayer {
    float *hc_attn_norm, *hc_attn_inject, *hc_ffn_norm, *hc_ffn_inject;   // [10240], [4][10240]
    FnLin hc_attn_down, hc_attn_up, hc_ffn_down, hc_ffn_up;
    // GDN
    FnLin qkv, zgate, beta, alpha, ssm_out; float *conv_w, *dt_bias, *a_neg, *ssm_norm;
    // attention (+ QSA indexer: bf16 projections, raw keys cached per token, 4 query heads)
    FnLin wq, wk, wv, wo; float *q_norm, *k_norm;
    uint16_t *idx_k = nullptr, *idx_q = nullptr; float *idx_kn = nullptr, *idx_qn = nullptr;
    // MoE
    float* router;        // f32 [512][2560]
    float* shexp_gate;    // f32 [2560]
    uint8_t *gate_exps, *up_exps, *down_exps; WFmt gate_fmt, up_fmt, down_fmt; size_t gate_eb, up_eb, down_eb;
    FnLin sh_gate, sh_up, sh_down;
    // PLE (layer 1)
    FnLin ple_key, ple_value; float *ple_nk, *ple_nq, *ple_nc, *ple_conv;
};

struct FnMtp { FnLayer L; FnLin eh_proj; float *enorm, *hnorm; };   // blk.48: nextn.* + one attention layer (dense here) + MoE + hc

// One sequence's contribution to a pass: T tokens (continuing that slot's sequence) at positions pos..pos+T-1.
void prefill_timing_report(double tokens);   // HIPSTER_TIMING=1: per-phase GPU ms accumulated over prefill() calls (stderr), then reset

class Qwen4Exp {
public:
    static constexpr int MAXR = 32, MAXS = 8;   // rows per pass, slots per pass
    explicit Qwen4Exp(const std::string& gguf_path, int max_ctx = 8192, int n_slots = 1, int max_prefill = 0);
    // Prefill (slot 0): T tokens (<= max_prefill) at positions pos0.. through the GEMM path (bf16 activations,
    // hipBLASLt); logits for the LAST row only in logits(); state committed (no accept). h_nextn() = the last row.
    float prefill(const int* tokens, int T, int pos0, int slot = 0);
    // A pass over S slots (rows = sum of T, <= MAXR, in request order). Logits [rows][n_vocab] on device.
    // Commit per slot with accept(slot, m): m == T swaps that slot's state buffers, m < T replays the recurrent
    // state (GDN, conv, PLE conv history) for the first m tokens from the saved inputs.
    float forward(const SlotReq* reqs, int S);
    float forward(const int* tokens, int T, int pos0) { SlotReq r{0, tokens, T, pos0}; return forward(&r, 1); }
    void accept(int slot, int m);
    void accept(int m) { accept(0, m); }
    // MTP draft block over the same request shape: h [rows][10240] = the wide residual the trunk (h_nextn) or the
    // previous draft (mtp_h) hands over. Logits in mtp_logits(), h_out in mtp_h().
    void mtp_forward(const SlotReq* reqs, int S, const float* h);
    void mtp_forward(const int* tokens, const float* h, int T, int pos0) { SlotReq r{0, tokens, T, pos0}; mtp_forward(&r, 1, h); }
    int n_slots() const { return n_slots_; }
    const GGUF& gguf() const { return *gguf_; }
    // after prefill(tokens, T, pos0): run the MTP block over the same rows (h for row p = the trunk residual after row p-1,
    // the previous chunk's last row for the first one) so the draft KV covers the prompt
    void mtp_catchup(const int* tokens, int T, int pos0);
    bool has_mtp() const { return has_mtp_; }
    double ple_host_ms() const { return ple_host_ms_; }   // host time spent hashing + gathering n-gram rows from the mmap (SSD)
    double gpu_ms() const { return gpu_ms_; }             // accumulated forward() GPU time
    void dbg_report() const;                              // HIPSTER_DBG_LAYERS: prefill vs decode activation diffs (stderr)
    void dbg_arm(bool on) { dbg_armed_ = on; }            // decode-side captures only while armed
    const float* h_nextn() const { return R_; }        // [T][4][2560] after the last combine
    const float* mtp_h() const { return R_mtp_; }
    const float* mtp_logits() const { return mlogits_; }
    const float* logits() const { return logits_; }
    void topk(const float* logits, int T, int k, int* ids, float* vals);
    void reset();
    void reset_slot(int slot);   // zero one slot's recurrent/PLE state and n-gram history
    size_t weight_bytes() const { return weight_bytes_; }
    std::string load_report() const { return report_; }

private:
    using D = FnDims;
    FnLin upload_lin(const GGUF& g, const std::string& name);
    float* upload_f32(const GGUF& g, const std::string& name, size_t expect, bool from_any = false);
    void upload_experts(const GGUF& g, const std::string& name, uint8_t** dst, WFmt* fmt, size_t* eb);
    // the write half of a block (R += y * w(inject)) is deferred and folded into the next read's norm
    struct Pending { const float* y = nullptr; const float* inject = nullptr; };
    void load_layer(const GGUF& g, int il, FnLayer& L, bool attn);
    void gemv(const FnLin& l, float* y, int ncol);
    void attn_layer(const FnLayer& L, int T, int pos0, uint16_t* kc, uint16_t* vc);
    void moe_layer(const FnLayer& L, int T);
    void head(int T, const Pending& p, float* R, float* logits_out);
    void hc_block(const FnLayer& L, bool ffn, int T, const Pending& p, float* mixed, float* inject_out, float* R);
    void ple(const SlotReq* reqs, int S, const int* r0s, int rows);
    // indexer + top-k selection (qsa) or dense, norm/rope/KV write, gather attention -> ao f32 [rows][24][256] (+ xq)
    void qsa_attention(const FnLayer& L, int rows, const std::vector<int>& pos, const std::vector<int>& kv, const RowBatch* rb, const float* mixed,
                       float* qf, float* k, float* v, float* ao, XQ8 xq, uint16_t* kc, uint16_t* vc, size_t kvs, int ai, bool qsa);
    void attn_layer_b(const FnLayer& L, int rows, const RowBatch& rb, uint16_t* kc, uint16_t* vc, size_t kvs, int ai, bool qsa);
    void gemv_cols(WFmt fmt, const void* w, int N, int K, int tpr, float* y, int ncol);
    void ple_rows(const std::vector<int>& hist, const int* tokens, int T, float* emb_out);   // host hash + gather -> emb_out [T][2560]
    void lin_gemm(const GemvSeg* segs, int n, int M, const uint16_t* A);                      // dequant segments -> one GEMM -> gout_ bf16 [M][sum N]
    void gemm_seg(int Nt, int off, int N, float* dst, int M);
    void hc_read_pf(const FnLayer& L, bool ffn, int T, const Pending& p, float* inject_out);

    GGUF* gguf_ = nullptr;             // kept open: the n-gram table is read from the mmap on demand
    const GTensor* ple_table_ = nullptr;
    std::vector<uint64_t> ple_mult_; std::vector<int64_t> ple_off_, ple_vocab_; int eos_ = 248044;
    std::vector<std::vector<int>> hist_;   // per slot: last 2 tokens of the sequence (for the n-gram context)
    int max_ctx_, n_slots_;
    std::vector<FnLayer> layers_;
    uint8_t* tok_embd_ = nullptr; float* head_norm_ = nullptr; FnLin head_down_, head_up_, output_;
    // per-slot state, double-buffered: index slot*2 + buf, cur_[slot] = committed buffer
    float *gdn_state_, *conv_state_, *ple_state_; size_t st_stride_, cs_stride_, ple_stride_; std::vector<int> cur_;
    uint16_t *kc_, *vc_, *mkc_ = nullptr, *mvc_ = nullptr; size_t kv_stride_, mkv_stride_;   // per-slot strides (elements); K and V both [kh][pos][256]
    // QSA: raw indexer keys f16 [slot][attn layer][max_ctx][128], pooled block keys f32 [slot][layer][max_ctx/4][128]
    uint16_t* idxc_ = nullptr; float* pooled_ = nullptr; size_t idx_slot_stride_, pooled_slot_stride_; int max_blocks_;
    int *d_idx_ = nullptr, *d_cnt_, *d_pos_, *d_kv_; float *q_raw_, *q_idx_, *k_idx_, *sel_scratch_, *part_; int idx_rows_;
    FnMtp mtp_; bool has_mtp_ = false;
    // saved inputs of the last pass (by row) for replay on partial accept, and each slot's rows in it
    float *sv_qkv_, *sv_beta_, *sv_alpha_, *sv_plekey_, *sv_pleval_, *sv_pleR_, *ple_scratch_;
    std::vector<std::vector<int>> pend_tokens_; std::vector<int> lr0_, lT_;
    // work
    float *R_, *R_mtp_, *xn_, *lo_, *up_, *mixed_, *inject_, *inject2_, *y_, *big0_, *big1_, *big2_, *logits_, *mlogits_, *emb_;
    float *rlogits_, *rw_, *eg_, *eu_, *ymoe_, *shx_, *sg_; int* eid_;
    int* d_tok_; XQ8 xq_; int* d_ids_; float* d_vals_;
    hipStream_t s_; hipEvent_t ev0_, ev1_;
    // prefill (GEMM path) buffers, allocated when max_prefill > 0
    int max_prefill_ = 0; BlasLt blas_;
    uint16_t *xb_ = nullptr, *xb2_, *gout_, *wscratch_;
    float *hprev_, *mtp_h_; float *pR_, *pxn_, *pmixed_, *py_, *pymoe_, *pqkv_, *pqkvc_, *pz_, *pba_, *pqn_, *praw_, *pao_, *pq_, *pk_, *pv_, *prl_, *prw_, *peg_, *peu_, *pinj_, *pinj2_, *pkey_, *pval_, *pemb_;
    int *peid_, *pd_tok_, *d_kpos_, *d_rowtok_; MoeTile* d_tiles_; float* pdown_; XQ8 pxq_;
    std::set<int> dbg_layers_; std::map<int, std::vector<float>> dbg_pf_, dbg_dec_; bool dbg_armed_ = false;
    void dbg_capture(int il, const float* y, const float* ymoe, bool pf); void dbg_capture_mix(int il, const float* mixed, bool pf);
    size_t weight_bytes_ = 0; std::string report_; double ple_host_ms_ = 0, gpu_ms_ = 0;
};

}  // namespace hip
