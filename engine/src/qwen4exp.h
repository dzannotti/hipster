// Qwen3.8-Flash-Next (GGUF arch qwen4exp) — its own engine. Decode path first (T <= 8 tokens per
// pass, dense attention, which is exact below the QSA budget of 2051 cached tokens). Shares only
// compile-time kernels with the 27B (GEMV formats, GDN step with the sigmoid gate, attention 24/2).
#pragma once
#include "gguf.h"
#include "../kernels/gemv.h"
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

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
    // attention
    FnLin wq, wk, wv, wo; float *q_norm, *k_norm;
    // MoE
    float* router;        // f32 [512][2560]
    float* shexp_gate;    // f32 [2560]
    uint8_t *gate_exps, *up_exps, *down_exps; WFmt gate_fmt, up_fmt, down_fmt; size_t gate_eb, up_eb, down_eb;
    FnLin sh_gate, sh_up, sh_down;
    // PLE (layer 1)
    FnLin ple_key, ple_value; float *ple_nk, *ple_nq, *ple_nc, *ple_conv;
};

class Qwen4Exp {
public:
    explicit Qwen4Exp(const std::string& gguf_path, int max_ctx = 8192);
    // T tokens (<= max_T) at positions pos0..; tokens must continue the sequence (PLE history).
    // Logits [T][n_vocab] on device. State committed with accept(m) (m == T only for now).
    float forward(const int* tokens, int T, int pos0);
    void accept(int m);
    const float* logits() const { return logits_; }
    void topk(const float* logits, int T, int k, int* ids, float* vals);
    void reset();
    size_t weight_bytes() const { return weight_bytes_; }
    std::string load_report() const { return report_; }

private:
    using D = FnDims;
    FnLin upload_lin(const GGUF& g, const std::string& name);
    float* upload_f32(const GGUF& g, const std::string& name, size_t expect, bool from_any = false);
    void upload_experts(const GGUF& g, const std::string& name, uint8_t** dst, WFmt* fmt, size_t* eb);
    void gemv(const FnLin& l, float* y, int ncol);
    // the write half of a block (R += y * w(inject)) is deferred and folded into the next read's norm
    struct Pending { const float* y = nullptr; const float* inject = nullptr; };
    void hc_block(const FnLayer& L, bool ffn, int T, const Pending& p, float* mixed, float* inject_out);
    void ple(const int* tokens, int T);

    GGUF* gguf_ = nullptr;             // kept open: the n-gram table is read from the mmap on demand
    const GTensor* ple_table_ = nullptr;
    std::vector<uint64_t> ple_mult_; std::vector<int64_t> ple_off_, ple_vocab_; int eos_ = 248044;
    std::vector<int> hist_;             // last 2 tokens of the sequence (for the n-gram context)
    int max_ctx_;
    std::vector<FnLayer> layers_;
    uint8_t* tok_embd_ = nullptr; float* head_norm_ = nullptr; FnLin head_down_, head_up_, output_;
    float *gdn_state_[2], *conv_state_[2], *ple_state_; int cur_ = 0;
    uint16_t *kc_, *vc_;
    // work
    float *R_, *xn_, *lo_, *up_, *mixed_, *inject_, *inject2_, *y_, *big0_, *big1_, *big2_, *logits_, *emb_;
    float *rlogits_, *rw_, *eg_, *eu_, *ymoe_, *shx_, *sg_; int* eid_;
    int* d_tok_; XQ8 xq_; int* d_ids_; float* d_vals_;
    hipStream_t s_; hipEvent_t ev0_, ev1_;
    size_t weight_bytes_ = 0; std::string report_; int last_T_ = 0;
};

}  // namespace hip
