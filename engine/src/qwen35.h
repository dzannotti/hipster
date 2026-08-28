// Qwen3.8-27B (GGUF arch qwen35) — fixed-shape decode engine with a T-token verify pass and the
// in-checkpoint MTP draft layer. Shapes are constants; only tensor formats are read from the file.
#pragma once
#include "gguf.h"
#include "../kernels/gemv.h"
#include "../kernels/gemm.h"
#include "../kernels/ops.h"
#include "../kernels/dflash.h"
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace hip {

struct Qwen35Dims {
    static constexpr int n_embd = 5120, n_layer = 64, n_ff = 17408, n_vocab = 248320;
    static constexpr int n_head = 24, n_head_kv = 4, head_dim = 256, n_rot = 64;
    static constexpr int gdn_qk_heads = 16, gdn_v_heads = 48, gdn_dim = 128;
    static constexpr int gdn_qkv = 2 * gdn_qk_heads * gdn_dim + gdn_v_heads * gdn_dim;   // 10240
    static constexpr int gdn_z = gdn_v_heads * gdn_dim;                                   // 6144
    static constexpr int conv_k = 4, n_gdn = 48, n_attn = 16;
    static constexpr int max_T = 8;   // tokens per verify pass (GEMV path, replayable)
    static constexpr int max_prefill = 4096;   // tokens per prefill chunk (GEMM path)
    static constexpr float rms_eps = 1e-6f, rope_base = 1e7f;
    static constexpr bool is_attn(int il) { return (il + 1) % 4 == 0; }
};

struct Lin { WFmt fmt; uint8_t* w = nullptr; int N = 0, K = 0; size_t bytes = 0; };

struct AttnBlock {   // one full-attention layer (target layers 3,7,... and the MTP layer)
    Lin wq, wk, wv, wo; float *q_norm, *k_norm;
};
struct Qwen35Layer {
    float *attn_norm, *post_norm;
    Lin ffn_gate, ffn_up, ffn_down;
    Lin qkv, zgate, beta, alpha, ssm_out;                 // GDN
    float *conv_w, *dt_bias, *a_neg, *ssm_norm;
    AttnBlock attn;                                       // attention layers
};
struct MtpLayer {
    Lin eh_proj; float *enorm, *hnorm, *attn_norm, *post_norm, *head_norm;
    AttnBlock attn; Lin ffn_gate, ffn_up, ffn_down;
};

struct DflashLayer { float *attn_norm, *ffn_norm, *q_norm, *k_norm, *attn_conv_base, *ffn_conv_base; Lin wq, wk, wv, wo, attn_conv_proj, ffn_conv_proj, ffn_gate, ffn_up, ffn_down; };
struct DflashDraft {   // incoai DFlash2 draft (target layers 6/20/34/48/62, block 8, selector top-16 rank 256)
    Lin fc, selector_hidden; float *enc_norm, *out_norm; const void *sel_pred, *sel_succ; std::vector<DflashLayer> layers; int target_layers[5]; int mask_id;
};

class Qwen35 {
public:
    static constexpr int MAXR = 16;   // rows per pass (2 slots x max_T; the WMMA GEMV computes 16 columns per launch)
    explicit Qwen35(const std::string& gguf_path, int max_ctx = 8192, int n_slots = 1);
    ~Qwen35();
    using D = Qwen35Dims;

    // Target: T tokens at positions pos0..pos0+T-1. T <= max_T: GEMV path, logits for all T rows,
    // replayable. T > max_T: prefill (GEMM path), logits for the LAST row only, accept(T) required.
    // h_nextn [T][n_embd] stays on device either way.
    float forward(const int* tokens, int T, int pos0) { SlotReq r{0, tokens, T, pos0}; return forward(&r, 1); }
    // A pass over S slots (rows = sum of T <= MAXR, GEMV path; a single request may be a prefill). Logits [rows][n_vocab].
    float forward(const SlotReq* reqs, int S);
    // Commit the first m (1..T) tokens of that slot's rows of the last forward: m == T swaps the state buffers, m < T
    // replays the recurrent state for m tokens from the saved per-layer inputs.
    void accept(int slot, int m);
    void accept(int m) { accept(0, m); }
    int n_slots() const { return n_slots_; }
    // MTP draft layer over T (token, h) pairs at positions pos0..: logits_mtp [T][n_vocab],
    // h_out [T][n_embd]. h [T][n_embd] on device (target h_nextn of the previous position).
    void mtp_forward(const int* tokens, const float* h, int T, int pos0);

    // DFlash2: load the draft GGUF; then forward() captures the residual at its target layers, encode() injects the
    // captured rows (positions pos0..) into the draft KV, draft() returns <= nd draft tokens after id_last at position n.
    void load_dflash(const std::string& gguf_path);
    bool has_dflash() const { return dfl_ != nullptr; }
    void dflash_encode(int slot, int T, int pos0);                       // the slot's first T rows of the last forward
    void dflash_encode(int T, int pos0) { dflash_encode(0, T, pos0); }
    // S sequences in one draft pass (S*(1+nd) rows <= MAXR): out[s*nd + i]; returns nd (greedy selector walk per block)
    int dflash_draft_b(const int* slots, const int* id_last, const int* n, int S, int nd, int* out);
    int dflash_draft(int slot, int id_last, int n, int nd, int* out) { return dflash_draft_b(&slot, &id_last, &n, 1, nd, out); }
    int dflash_draft(int id_last, int n, int nd, int* out) { return dflash_draft(0, id_last, n, nd, out); }
    const float* logits() const { return logits_; }
    const float* h_nextn() const { return h_; }
    const float* mtp_logits() const { return mlogits_; }
    const float* mtp_h() const { return mh_; }
    // top-k per row of the given logits buffer -> host arrays [T][k]
    void topk(const float* logits, int T, int k, int* ids, float* vals);
    void reset();
    size_t weight_bytes() const { return weight_bytes_; }
    std::string load_report() const { return report_; }
    float last_ms() const { return last_ms_; }
    // per-category GPU time (ms) accumulated when HIPSTER_TIMING is set: gemv, dequant, gemm, split, attention, gdn, other
    enum { T_GEMV, T_DEQ, T_GEMM, T_SPLIT, T_ATTN, T_GDN, T_OTHER, T_N };
    const float* timing() const { return tacc_; }
    void timing_reset() { for (float& t : tacc_) t = 0; }

private:
    Lin upload_lin(const GGUF& g, const std::string& name);
    float* upload_f32(const GGUF& g, const std::string& name, size_t expect);
    AttnBlock upload_attn(const GGUF& g, const std::string& p);
    void gemv(const Lin& l, float* y, int ncol);
    void lin(const Lin& l, float* y, int M);   // GEMV (M <= max_T, input xq_) or GEMM (input xb_)
    void lin_multi(const GemvSeg* segs, int n, int M);
    // shared attention-layer body: x [T] (residual stream) after attn_norm already quantised in xq_
    void attn_block(const AttnBlock& a, int T, int pos0, uint16_t* kc, uint16_t* vc, float* y, const KV8* c8, const RowBatch* rb = nullptr);
    KV8 layer_kv8(int ai) const;
    void ffn_block(const Lin& g, const Lin& u, const Lin& d, int T, float* y);

    int max_ctx_, n_slots_;
    std::vector<Qwen35Layer> layers_;
    MtpLayer mtp_;
    uint8_t* tok_embd_ = nullptr;
    float* output_norm_ = nullptr;
    Lin output_;
    // recurrent state per slot, double-buffered: index slot*2 + buf, cur_[slot] = committed buffer
    float *gdn_state_, *conv_state_; size_t st_stride_, cs_stride_; std::vector<int> cur_;
    uint16_t *kc_ = nullptr, *vc_ = nullptr;   // target attention KV per slot: [slot][16][max_ctx][4][256] (V^T rows padded: kvt_slot_)
    size_t kv_slot_ = 0, kvt_slot_ = 0;
    uint16_t *mkc_ = nullptr, *mvc_ = nullptr; // MTP KV [max_ctx][4][256] (slot 0)
    bool kv8_ = false; KV8 kv8c_{}, mkv8c_{};    // int8 KV caches (HIPSTER_KV8=1)
    // saved per-layer inputs of the last pass for replay: qkv_raw [48][MAXR][10240], beta/alpha [48][MAXR][64]; each slot's rows in it
    float *sv_qkv_, *sv_beta_, *sv_alpha_; int last_T_ = 0; std::vector<int> lr0_, lT_;
    // work
    float *x_, *xn_, *big0_, *big1_, *big2_, *y_, *logits_, *h_, *mlogits_, *mh_, *cat_, *pf_qkv_, *pf_qn_, *pf_raw_, *pf_ao_, *pO_, *pm_, *pl_;
    uint16_t *xb_, *wscratch_, *gout_, *gout2_;   // bf16 activations / dequantised weight / GEMM output (GEMM path)
    // one-ahead weight prefetch (dequant on s2_ into the other scratch buffer)
    uint16_t* wscratch2_[2]; hipStream_t s2_; hipEvent_t scratch_free_[2], pf_ready_;
    bool pf_pending_ = false; int pf_buf_ = 0; const void* pf_w_ = nullptr; int pf_n_ = 0;
    GemvSeg next_pf_[4], next2_pf_[4]; int next_pfn_ = 0, next2_pfn_ = 0;
    DflashDraft* dfl_ = nullptr; GGUF* dfl_gguf_ = nullptr;
    float *dfl_feat_ = nullptr, *dfl_g_, *dfl_x_, *dfl_y_, *dfl_q_, *dfl_k_, *dfl_v_, *dfl_o_, *dfl_delta_, *dfl_logits_, *dfl_gate_, *dfl_lattice_, *dfl_vals_; int *dfl_ids_, *dfl_pos_, *dfl_tok_;
    uint16_t *dfl_kc_, *dfl_vc_; size_t dfl_layer_stride_, dfl_slot_stride_; dfl::Blk* dfl_blk_;
    void prefetch(const GemvSeg* segs, int n);
    BlasLt blas_;
    int* d_tok_;
    XQ8 xq_;
    int* d_ids_; float* d_vals_;
    hipStream_t s_;
    hipEvent_t ev0_, ev1_;
    size_t weight_bytes_ = 0;
    std::string report_;
    float last_ms_ = 0;
    bool gm_ = false, last_gm_ = false;   // the pass in flight is a prefill (GEMM path); rows > max_T in a batched pass stay on the GEMV path
    bool timing_ = false, prefetch_on_ = false; float tacc_[T_N]; hipEvent_t tev_[2];
    void tmark(int cat);
};

}  // namespace hip
