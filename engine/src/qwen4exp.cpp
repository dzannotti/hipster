#include "qwen4exp.h"
#include <chrono>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <sys/mman.h>
#include "quant.h"
#include "../kernels/ops.h"
#include "../kernels/fn.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) throw std::runtime_error(std::string("HIP: ") + hipGetErrorString(e) + " @" + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace hip {

static constexpr int KSPLIT = 8;   // split-K partials of the 320-row hc down GEMVs
using D = FnDims;

static int gdn_index(int il) { return il - (il + 1) / 4; }
static int attn_index(int il) { return (il + 1) / 4 - 1; }

template <class F> static void parallel_rows(int N, F fn) {
    const int nt = 24; std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i) th.emplace_back([=] { for (int r = i; r < N; r += nt) fn(r); });
    for (auto& t : th) t.join();
}

// GGUF tensor -> device weight in a format the GEMV understands (row-SoA repacks where needed)
static FnLin to_lin(const GTensor& t, int K, int N, const uint8_t* src_rows, size_t& bytes_acc, std::string& report) {
    FnLin l; l.K = K; l.N = N;
    const size_t rb_src = gtype_row_bytes(t.type, K);
    std::vector<uint8_t> tmp; const uint8_t* src = src_rows;
    switch (t.type) {
        case GType::Q4_K: l.fmt = WFmt::Q4_K; l.bytes = rb_src * N; break;
        case GType::Q5_K: l.fmt = WFmt::Q5_K; l.bytes = rb_src * N; break;
        case GType::IQ4_XS: l.fmt = WFmt::IQ4_XS; l.bytes = rb_src * N; break;
        case GType::Q6_K: l.fmt = WFmt::Q6_K_SOA; l.bytes = rb_src * N; tmp.resize(l.bytes);
            parallel_rows(N, [&](int r) { repack_row_q6_K(src_rows + (size_t)r * rb_src, tmp.data() + (size_t)r * rb_src, K); }); src = tmp.data(); break;
        case GType::Q8_0: l.fmt = WFmt::Q8_0_SOA; l.bytes = rb_src * N; tmp.resize(l.bytes);
            parallel_rows(N, [&](int r) { repack_row_q8_0(src_rows + (size_t)r * rb_src, tmp.data() + (size_t)r * rb_src, K); }); src = tmp.data(); break;
        case GType::Q5_0: l.fmt = WFmt::Q5_1_SOA; l.bytes = wfmt_row_bytes(WFmt::Q5_1_SOA, K) * N; tmp.resize(l.bytes);   // exact: m = -16 d
            parallel_rows(N, [&](int r) { repack_row_q5_0(src_rows + (size_t)r * rb_src, tmp.data() + (size_t)r * (l.bytes / N), K); }); src = tmp.data(); break;
        case GType::Q5_1: l.fmt = WFmt::Q5_1_SOA; l.bytes = rb_src * N; tmp.resize(l.bytes);
            parallel_rows(N, [&](int r) { repack_row_q5_1(src_rows + (size_t)r * rb_src, tmp.data() + (size_t)r * rb_src, K); }); src = tmp.data(); break;
        default: {   // F32/BF16/Q5_0/... -> Q8_0 SoA
            l.fmt = WFmt::Q8_0_SOA; const size_t rb = wfmt_row_bytes(WFmt::Q8_0_SOA, K); l.bytes = rb * N; tmp.resize(l.bytes);
            parallel_rows(N, [&](int r) { std::vector<float> row(K); dequant_row((int)t.type, src_rows + (size_t)r * rb_src, row.data(), K); quantize_row_q8_0_soa(row.data(), tmp.data() + (size_t)r * rb, K); });
            src = tmp.data(); report += "  " + t.name + ": " + gtype_name(t.type) + " -> Q8_0\n";
        }
    }
    CK(hipMalloc(&l.w, l.bytes)); CK(hipMemcpy(l.w, src, l.bytes, hipMemcpyHostToDevice));
    bytes_acc += l.bytes;
    return l;
}
static const GGUF* g_release_gguf = nullptr;   // set by the loader so uploads can drop their page cache
void Qwen4Exp::load_layer(const GGUF& g, int il, FnLayer& L, bool attn) {
    const std::string p = "blk." + std::to_string(il) + ".";
        L.hc_attn_norm = upload_f32(g, p + "hc_attn_norm.weight", D::wide); L.hc_attn_inject = upload_f32(g, p + "hc_attn_inject.weight", (size_t)D::wide * 4, true);
        L.hc_ffn_norm = upload_f32(g, p + "hc_ffn_norm.weight", D::wide); L.hc_ffn_inject = upload_f32(g, p + "hc_ffn_inject.weight", (size_t)D::wide * 4, true);
        L.hc_attn_down = upload_lin(g, p + "hc_attn_down.weight"); L.hc_attn_up = upload_lin(g, p + "hc_attn_up.weight");
        L.hc_ffn_down = upload_lin(g, p + "hc_ffn_down.weight"); L.hc_ffn_up = upload_lin(g, p + "hc_ffn_up.weight");
        if (attn) {
            L.wq = upload_lin(g, p + "attn_q.weight"); L.wk = upload_lin(g, p + "attn_k.weight"); L.wv = upload_lin(g, p + "attn_v.weight"); L.wo = upload_lin(g, p + "attn_output.weight");
            L.q_norm = upload_f32(g, p + "attn_q_norm.weight", D::head_dim); L.k_norm = upload_f32(g, p + "attn_k_norm.weight", D::head_dim);
        } else {
            L.qkv = upload_lin(g, p + "attn_qkv.weight"); L.zgate = upload_lin(g, p + "attn_gate.weight");
            L.beta = upload_lin(g, p + "ssm_beta.weight"); L.alpha = upload_lin(g, p + "ssm_alpha.weight"); L.ssm_out = upload_lin(g, p + "ssm_out.weight");
            L.conv_w = upload_f32(g, p + "ssm_conv1d.weight", (size_t)4 * D::gdn_qkv); L.dt_bias = upload_f32(g, p + "ssm_dt.bias", D::gdn_v_heads);
            L.a_neg = upload_f32(g, p + "ssm_a", D::gdn_v_heads); L.ssm_norm = upload_f32(g, p + "ssm_norm.weight", D::gdn_dim);
        }
        {   // router [512][2560] + shared-expert gate [1][2560] concatenated -> one f32 GEMV of 513 rows
            const GTensor& tr = g.get(p + "ffn_gate_inp.weight"); const GTensor& ts = g.get(p + "ffn_gate_inp_shexp.weight");
            if (tr.type != GType::F32 || ts.type != GType::F32) throw std::runtime_error("router must be f32");
            std::vector<uint8_t> a(tr.nbytes + ts.nbytes); g.read_tensor(tr, a.data()); g.read_tensor(ts, a.data() + tr.nbytes);
            CK(hipMalloc(&L.router, a.size())); CK(hipMemcpy(L.router, a.data(), a.size(), hipMemcpyHostToDevice)); weight_bytes_ += a.size(); L.shexp_gate = nullptr;
        }
        upload_experts(g, p + "ffn_gate_exps.weight", &L.gate_exps, &L.gate_fmt, &L.gate_eb);
        upload_experts(g, p + "ffn_up_exps.weight", &L.up_exps, &L.up_fmt, &L.up_eb);
        upload_experts(g, p + "ffn_down_exps.weight", &L.down_exps, &L.down_fmt, &L.down_eb);
        L.sh_gate = upload_lin(g, p + "ffn_gate_shexp.weight"); L.sh_up = upload_lin(g, p + "ffn_up_shexp.weight"); L.sh_down = upload_lin(g, p + "ffn_down_shexp.weight");
        if (il == D::ple_layer) {
            L.ple_key = upload_lin(g, p + "ple_key.weight"); L.ple_value = upload_lin(g, p + "ple_value.weight");
            L.ple_nk = upload_f32(g, p + "ple_norm_key.weight", D::wide); L.ple_nq = upload_f32(g, p + "ple_norm_query.weight", D::wide);
            L.ple_nc = upload_f32(g, p + "ple_norm_conv.weight", D::wide); L.ple_conv = upload_f32(g, p + "ple_conv1d.weight", (size_t)4 * D::wide);
        }
}

FnLin Qwen4Exp::upload_lin(const GGUF& g, const std::string& name) {
    const GTensor& t = g.get(name);
    std::vector<uint8_t> stage(t.nbytes); g.read_tensor(t, stage.data());   // pread + drop cache: no mmap residency
    return to_lin(t, (int)t.ne[0], (int)t.ne[1], stage.data(), weight_bytes_, report_);
}
float* Qwen4Exp::upload_f32(const GGUF& g, const std::string& name, size_t expect, bool from_any) {
    const GTensor& t = g.get(name);
    std::vector<uint8_t> stage(t.nbytes); g.read_tensor(t, stage.data());
    std::vector<float> buf;
    const float* src;
    if (t.type == GType::F32) { if (t.nbytes != expect * 4) throw std::runtime_error("size mismatch " + name); src = (const float*)stage.data(); }
    else if (from_any) { buf.resize(expect); const int K = (int)t.ne[0]; const size_t rb = gtype_row_bytes(t.type, K); for (size_t r = 0; r < expect / K; ++r) dequant_row((int)t.type, stage.data() + r * rb, buf.data() + r * K, K); src = buf.data(); }
    else throw std::runtime_error("expected f32 " + name);
    float* d; CK(hipMalloc(&d, expect * 4)); CK(hipMemcpy(d, src, expect * 4, hipMemcpyHostToDevice));
    weight_bytes_ += expect * 4;
    return d;
}
void Qwen4Exp::upload_experts(const GGUF& g, const std::string& name, uint8_t** dst, WFmt* fmt, size_t* eb) {
    const GTensor& t = g.get(name);   // [K, N, 512]
    const int K = (int)t.ne[0], N = (int)t.ne[1], E = (int)t.ne[2];
    std::vector<uint8_t> stage(t.nbytes); g.read_tensor(t, stage.data());
    FnLin l = to_lin(t, K, N * E, stage.data(), weight_bytes_, report_);   // rows of all experts are contiguous
    *dst = l.w; *fmt = l.fmt; *eb = l.bytes / E;
}

Qwen4Exp::Qwen4Exp(const std::string& path, int max_ctx, int n_slots, int max_prefill) : max_ctx_(max_ctx), n_slots_(n_slots), max_prefill_(max_prefill) {
    if (n_slots < 1 || n_slots > MAXS) throw std::runtime_error("n_slots must be 1..8");
    gguf_ = new GGUF(path);
    if (const char* e = getenv("HIPSTER_DBG_LAYERS")) { std::stringstream ss(e); std::string t; while (std::getline(ss, t, ',')) dbg_layers_.insert(atoi(t.c_str())); }
    GGUF& g = *gguf_;
    if (g.arch() != "qwen4exp") throw std::runtime_error("not a qwen4exp GGUF: " + g.arch());
    if (getenv("HIPSTER_BLOCKING_SYNC")) CK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));   // experiment: a spinning host core may steal the APU's power budget from the GPU
    CK(hipStreamCreate(&s_)); CK(hipEventCreate(&ev0_)); CK(hipEventCreate(&ev1_));
    // PLE constants
    { auto* m = g.kv("qwen4exp.ple.layer_multipliers"); auto* o = g.kv("qwen4exp.ple.head_offsets"); auto* v = g.kv("qwen4exp.ple.head_vocab_sizes");
      for (auto& x : m->arr) ple_mult_.push_back((uint64_t)x.i ? (uint64_t)x.i : x.u); for (auto& x : o->arr) ple_off_.push_back(x.i ? x.i : (int64_t)x.u); for (auto& x : v->arr) ple_vocab_.push_back(x.i ? x.i : (int64_t)x.u);
      eos_ = (int)g.kv_u32("qwen4exp.ple.eos_token_id");
      ple_table_ = &g.get("per_layer_token_embd.weight"); }
    layers_.resize(D::n_layer);
    for (int il = 0; il < D::n_layer; ++il) {
        load_layer(g, il, layers_[il], D::is_attn(il));
        if (il % 8 == 7) fprintf(stderr, "  loaded %d layers, %.1f GiB\n", il + 1, weight_bytes_ / 1073741824.0);
    }
    if (g.has("blk.48.nextn.eh_proj.weight")) {   // the UD-Q4_K_XL-MTP shards carry the draft block; the 4-shard file does not
        const std::string p = "blk.48.";
        load_layer(g, 48, mtp_.L, true);
        mtp_.eh_proj = upload_lin(g, p + "nextn.eh_proj.weight");
        mtp_.enorm = upload_f32(g, p + "nextn.enorm.weight", D::n_embd); mtp_.hnorm = upload_f32(g, p + "nextn.hnorm.weight", D::wide);
        has_mtp_ = true; fprintf(stderr, "  loaded the MTP block, %.1f GiB\n", weight_bytes_ / 1073741824.0);
    }
    { const GTensor& t = g.get("token_embd.weight"); if (t.type != GType::Q8_0) throw std::runtime_error("token_embd must be Q8_0");
      std::vector<uint8_t> stage(t.nbytes); g.read_tensor(t, stage.data());
      CK(hipMalloc(&tok_embd_, t.nbytes)); CK(hipMemcpy(tok_embd_, stage.data(), t.nbytes, hipMemcpyHostToDevice)); weight_bytes_ += t.nbytes; }
    head_norm_ = upload_f32(g, "output_hc_norm.weight", D::wide); head_down_ = upload_lin(g, "output_hc_down.weight"); head_up_ = upload_lin(g, "output_hc_up.weight");
    output_ = upload_lin(g, "output.weight");
    // state: per slot, double-buffered
    st_stride_ = (size_t)D::n_gdn * D::gdn_v_heads * D::gdn_dim * D::gdn_dim; cs_stride_ = (size_t)D::n_gdn * 3 * D::gdn_qkv; ple_stride_ = (size_t)D::ple_hist * D::wide;
    CK(hipMalloc(&gdn_state_, (size_t)n_slots_ * 2 * st_stride_ * 4)); CK(hipMalloc(&conv_state_, (size_t)n_slots_ * 2 * cs_stride_ * 4)); CK(hipMalloc(&ple_state_, (size_t)n_slots_ * 2 * ple_stride_ * 4));
    cur_.assign(n_slots_, 0); hist_.resize(n_slots_); pend_tokens_.resize(n_slots_); lr0_.assign(n_slots_, 0); lT_.assign(n_slots_, 0);
    CK(hipMalloc(&sv_qkv_, (size_t)D::n_gdn * MAXR * D::gdn_qkv * 4)); CK(hipMalloc(&sv_beta_, (size_t)D::n_gdn * MAXR * 64 * 4)); CK(hipMalloc(&sv_alpha_, (size_t)D::n_gdn * MAXR * 64 * 4));
    CK(hipMalloc(&sv_plekey_, (size_t)MAXR * D::wide * 4)); CK(hipMalloc(&sv_pleval_, (size_t)MAXR * D::n_embd * 4));
    CK(hipMalloc(&sv_pleR_, (size_t)MAXR * D::wide * 4)); CK(hipMalloc(&ple_scratch_, (size_t)MAXR * D::wide * 4));
    kv_stride_ = (size_t)D::n_attn * max_ctx_ * D::n_head_kv * D::head_dim; kvt_stride_ = (size_t)D::n_attn * (max_ctx_ + 128) * D::n_head_kv * D::head_dim;
    CK(hipMalloc(&kc_, (size_t)n_slots_ * kv_stride_ * 2)); CK(hipMalloc(&vc_, (size_t)n_slots_ * kvt_stride_ * 2)); CK(hipMemset(vc_, 0, (size_t)n_slots_ * kvt_stride_ * 2));
    const int T = MAXR;
    CK(hipMalloc(&R_, (size_t)T * D::wide * 4)); CK(hipMalloc(&R_mtp_, (size_t)T * D::wide * 4)); CK(hipMalloc(&mlogits_, (size_t)T * D::n_vocab * 4));
    if (has_mtp_) { mkv_stride_ = (size_t)max_ctx_ * D::n_head_kv * D::head_dim; mkvt_stride_ = (size_t)(max_ctx_ + 128) * D::n_head_kv * D::head_dim;
                    CK(hipMalloc(&mkc_, (size_t)n_slots_ * mkv_stride_ * 2)); CK(hipMalloc(&mvc_, (size_t)n_slots_ * mkvt_stride_ * 2)); CK(hipMemset(mvc_, 0, (size_t)n_slots_ * mkvt_stride_ * 2)); }
    CK(hipMalloc(&xn_, (size_t)T * D::wide * 4)); CK(hipMalloc(&lo_, (size_t)T * KSPLIT * D::hc_lr * 4)); CK(hipMalloc(&up_, (size_t)T * D::wide * 4));
    CK(hipMalloc(&mixed_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&inject_, (size_t)T * 20 * 4 * 4)); CK(hipMalloc(&inject2_, (size_t)T * 20 * 4 * 4)); CK(hipMalloc(&ymoe_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&y_, (size_t)T * D::n_embd * 4));
    const size_t bigw = (size_t)T * 12288;   // largest per-token row: attention q (24 x 512); qkv is 10240
    CK(hipMalloc(&big0_, bigw * 4)); CK(hipMalloc(&big1_, bigw * 4)); CK(hipMalloc(&big2_, bigw * 4));
    CK(hipMalloc(&logits_, (size_t)T * D::n_vocab * 4)); CK(hipMalloc(&emb_, (size_t)T * D::n_embd * 4));
    CK(hipMalloc(&rlogits_, (size_t)T * (D::n_exp + 1) * 4)); CK(hipMalloc(&rw_, (size_t)T * (D::n_used + 1) * 4)); CK(hipMalloc(&eid_, (size_t)T * (D::n_used + 1) * 4));
    CK(hipMalloc(&eg_, (size_t)T * (D::n_used + 1) * D::exp_ff * 4)); CK(hipMalloc(&eu_, (size_t)T * (D::n_used + 1) * D::exp_ff * 4));
    CK(hipMalloc(&shx_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&sg_, (size_t)T * 4));
    const size_t xk = (size_t)T * 4 * 5120;   // widest quantised rows: 4 x 5120 per token (eh_proj input); hc norm 10240
    CK(hipMalloc(&xq_.q, xk)); CK(hipMalloc(&xq_.d, xk / 32 * 4)); CK(hipMalloc(&xq_.s, xk / 32 * 4));
    CK(hipMemset(xq_.q, 0, xk)); CK(hipMemset(xq_.d, 0, xk / 32 * 4)); CK(hipMemset(xq_.s, 0, xk / 32 * 4));
    CK(hipMalloc(&d_tok_, T * 4)); CK(hipMalloc(&d_ids_, T * 16 * 4)); CK(hipMalloc(&d_vals_, T * 16 * 4));
    if (max_prefill_ > 0) {
        const size_t P = max_prefill_;
        CK(hipMalloc(&xb_, P * D::wide * 2)); CK(hipMalloc(&xb2_, P * D::n_embd * 2)); CK(hipMalloc(&gout_, P * 16480 * 2)); CK(hipMalloc(&wscratch_, (size_t)16480 * D::n_embd * 2));
        auto f32 = [&](float** p, size_t n) { CK(hipMalloc(p, n * 4)); };
        f32(&pR_, P * D::wide); f32(&pxn_, P * D::wide); f32(&pmixed_, P * D::n_embd); f32(&py_, P * D::n_embd); f32(&pymoe_, P * D::n_embd);
        f32(&pqkv_, P * D::gdn_qkv); f32(&pqkvc_, P * D::gdn_qkv); f32(&pz_, P * D::gdn_z); f32(&pba_, P * 96); f32(&pqn_, P * (4096 + 96)); f32(&praw_, P * D::gdn_z); f32(&pao_, P * D::gdn_z);
        f32(&pq_, P * 12288); f32(&pk_, P * 512); f32(&pv_, P * 512); f32(&prl_, P * (D::n_exp + 1)); f32(&prw_, P * (D::n_used + 1));
        f32(&peg_, P * (D::n_used + 1) * D::exp_ff); f32(&peu_, P * (D::n_used + 1) * D::exp_ff); f32(&pinj_, P * 20 * 4); f32(&pinj2_, P * 20 * 4);
        f32(&pkey_, P * D::wide); f32(&pval_, P * D::n_embd); f32(&pemb_, P * D::n_embd);
        CK(hipMalloc(&peid_, P * (D::n_used + 1) * 4)); CK(hipMalloc(&pd_tok_, P * 4));
        CK(hipMalloc(&d_kpos_, P * (D::n_used + 1) * 4)); CK(hipMalloc(&d_rowtok_, P * (D::n_used + 1) * 4)); CK(hipMalloc(&d_tiles_, (P * (D::n_used + 1) / 16 + D::n_exp + 2) * sizeof(MoeTile)));
        f32(&pdown_, P * (D::n_used + 1) * D::n_embd);
        const size_t xk = P * (size_t)(D::n_used + 1) * D::exp_ff;   // widest int8 rows: the MoE down input [P*11][640] (> [P][2560])
        CK(hipMalloc(&pxq_.q, xk)); CK(hipMalloc(&pxq_.d, xk / 32 * 4)); CK(hipMalloc(&pxq_.s, xk / 32 * 4));
    }
    reset();
}

void Qwen4Exp::reset() {
    CK(hipMemset(gdn_state_, 0, (size_t)n_slots_ * 2 * st_stride_ * 4)); CK(hipMemset(conv_state_, 0, (size_t)n_slots_ * 2 * cs_stride_ * 4));
    CK(hipMemset(ple_state_, 0, (size_t)n_slots_ * 2 * ple_stride_ * 4));
    for (int s = 0; s < n_slots_; ++s) { hist_[s].clear(); cur_[s] = 0; lT_[s] = 0; }
}

// lanes per row for the T == 1 GEMV, from bench_gemv on this checkpoint's tensors (docs/decode-flash-next.md):
// Q8_0 K=6144 (ssm_out, wo): 2 lanes 199 GB/s vs 32 lanes 142; Q8_0 K=320 (hc up): 8 lanes 150 vs 4 lanes 104;
// Q8_0 K=2560 (qkv, gate, attn_q): 32 lanes 190-197, everything else worse.
static int fn_tpr(const FnLin& l) {
    if (l.K == 320) return 8;
    if (l.K == 6144) return 2;
    return 32;
}
// The lane count depends on the tensor only, never on ncol: a row's reduction order is then identical for every T,
// so a T-token verify pass produces bit-identical logits to T single-token passes (the exactness contract).
void Qwen4Exp::gemv_cols(WFmt fmt, const void* w, int N, int K, int tpr, float* y, int ncol) {
    for (int c0 = 0; c0 < ncol; c0 += 8) {   // the direct kernel takes up to 8 columns; more rows = more launches of the same shape
        const int nc = std::min(8, ncol - c0);
        XQ8 xs = {xq_.q + (size_t)c0 * K, xq_.d + (size_t)c0 * K / 32, xq_.s + (size_t)c0 * K / 32};
        gemv_q8(fmt, w, xs, y + (size_t)c0 * N, N, K, nc, tpr, 1, s_);
    }
}
void Qwen4Exp::gemv(const FnLin& l, float* y, int ncol) { gemv_cols(l.fmt, l.w, l.N, l.K, fn_tpr(l), y, ncol); }

// gated-residual read: (pending write of the previous block folded into the norm) R -> mixed (f32 + xq) and inject partials
void Qwen4Exp::hc_block(const FnLayer& L, bool ffn, int T, const Pending& p, float* mixed, float* inject_out, float* R) {
    fn::HcNormArgs na; na.R = R; na.y = p.y; na.inject = p.inject;
    na.w_norm = ffn ? L.hc_ffn_norm : L.hc_attn_norm; na.xn = xn_;
    fn::hc_norm(na, xq_, T, D::rms_eps, s_);
    const FnLin& dn = ffn ? L.hc_ffn_down : L.hc_attn_down; const FnLin& up = ffn ? L.hc_ffn_up : L.hc_attn_up;
    gemv_splitk(dn.fmt, dn.w, xq_, lo_, dn.N, dn.K, KSPLIT, T, s_);   // 320 rows x 10240: 8 K-splits fill the GPU
    fn::hc_silu_quant(lo_, KSPLIT, xq_, T, s_);
    gemv(up, up_, T);
    fn::hc_mix(xn_, up_, ffn ? L.hc_ffn_inject : L.hc_attn_inject, mixed, inject_out, xq_, T, s_);
}

// PLE (layer 1): n-gram rows hashed on the host and gathered from the mmap; key/value GEMVs over all rows land in the
// replay buffers; the conv history is advanced per slot, token by token.
// n-gram rows for T tokens continuing `hist`: 16 rows per token, random over a 28.8 GB mmap on NVMe; hash them all,
// tell the kernel to fetch the pages in parallel (MADV_WILLNEED), then decode -> emb_out [T][2560] (host memory).
void Qwen4Exp::ple_rows(const std::vector<int>& hist, const int* tokens, int T, float* emb_out) {
    std::vector<int> seq = hist; for (int i = 0; i < T; ++i) seq.push_back(tokens[i]);
    const int base = (int)hist.size();
    const size_t rb = gtype_row_bytes(ple_table_->type, D::ple_dim);
    std::vector<const uint8_t*> rowp((size_t)T * D::ple_heads);
    for (int i = 0; i < T; ++i) {
        const int pos = base + i;
        int64_t ctx[3]; ctx[0] = seq[pos]; bool cut = false;
        for (int s = 1; s < 3; ++s) { const int64_t prev = pos - s >= 0 ? seq[pos - s] : eos_; ctx[s] = cut ? eos_ : prev; if (ctx[s] == eos_) cut = true; }
        for (int n = 2; n <= 3; ++n) {
            uint64_t mixed = (uint64_t)ctx[0] * ple_mult_[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t)ctx[j] * ple_mult_[j];
            for (int h8 = 0; h8 < 8; ++h8) {
                const int h = (n - 2) * 8 + h8;
                const int64_t row = (int64_t)(mixed % (uint64_t)ple_vocab_[h]) + ple_off_[h];
                rowp[(size_t)i * D::ple_heads + h] = ple_table_->data + (size_t)row * rb;
            }
        }
    }
    for (const uint8_t* r : rowp) { const uintptr_t a = (uintptr_t)r & ~(uintptr_t)4095; madvise((void*)a, ((uintptr_t)r + rb) - a, MADV_WILLNEED); }
    for (int i = 0; i < T; ++i)
        for (int h = 0; h < D::ple_heads; ++h)
            dequant_row((int)ple_table_->type, rowp[(size_t)i * D::ple_heads + h], emb_out + (size_t)i * D::n_embd + h * D::ple_dim, D::ple_dim);
}

// PLE (layer 1, decode passes): key/value GEMVs over all rows land in the replay buffers; the conv history is advanced per
// slot, token by token.
void Qwen4Exp::ple(const SlotReq* reqs, int S, const int* r0s, int rows) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto& L = layers_[D::ple_layer];
    std::vector<float> emb((size_t)rows * D::n_embd);
    for (int si = 0; si < S; ++si) ple_rows(hist_[reqs[si].slot], reqs[si].tokens, reqs[si].T, emb.data() + (size_t)r0s[si] * D::n_embd);
    ple_host_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CK(hipMemcpyAsync(emb_, emb.data(), emb.size() * 4, hipMemcpyHostToDevice, s_));
    quantize_x_q8(emb_, xq_, rows, D::n_embd, s_);
    gemv(L.ple_key, sv_plekey_, rows);       // key [rows][10240]
    gemv(L.ple_value, sv_pleval_, rows);     // value [rows][2560]
    CK(hipMemcpyAsync(sv_pleR_, R_, (size_t)rows * D::wide * 4, hipMemcpyDeviceToDevice, s_));   // the gate reads R: replay needs this R
    for (int si = 0; si < S; ++si) {
        const int sl = reqs[si].slot, cur = cur_[sl], nxt = cur ^ 1, r0 = r0s[si];
        float* stc = ple_state_ + ((size_t)sl * 2 + cur) * ple_stride_; float* stn = ple_state_ + ((size_t)sl * 2 + nxt) * ple_stride_;
        CK(hipMemcpyAsync(stn, stc, ple_stride_ * 4, hipMemcpyDeviceToDevice, s_));
        fn::ple_apply(R_ + (size_t)r0 * D::wide, sv_plekey_ + (size_t)r0 * D::wide, sv_pleval_ + (size_t)r0 * D::n_embd, L.ple_nk, L.ple_nq, L.ple_nc, L.ple_conv, stn, reqs[si].T, D::rms_eps, s_);
    }
}

void Qwen4Exp::attn_layer_b(const FnLayer& L, int rows, const RowBatch& rb, uint16_t* kc, uint16_t* vc, size_t kvs, size_t kvts) {
    float* qf = big0_; float* k = big1_; float* v = big1_ + (size_t)rows * 512; float* ao = big2_;
    GemvSeg segs[3] = {{L.wq.fmt, L.wq.w, qf, L.wq.N, L.wq.K}, {L.wk.fmt, L.wk.w, k, L.wk.N, L.wk.K}, {L.wv.fmt, L.wv.w, v, L.wv.N, L.wv.K}};
    gemv_multi(segs, 3, xq_, rows, s_);
    attn_decode_24_2_b(qf, k, v, rows, L.q_norm, L.k_norm, D::rope_base, rb, kc, vc, kvs, kvts, max_ctx_, ao, xq_, D::rms_eps, s_);
    gemv(L.wo, y_, rows);
}

// MoE: router (exact f32, row 512 = shared-expert gate) -> top-10 -> gate|up (+ shared) in one launch -> silu -> down with the
// weighted combine accumulated into ymoe_ (zeroed by the ffn read's norm)
void Qwen4Exp::moe_layer(const FnLayer& L, int T) {
    fn::f32_gemv(L.router, mixed_, rlogits_, D::n_exp + 1, D::n_embd, T, s_);
    fn::moe_route(rlogits_, eid_, rw_, T, s_);                                   // 11 slots per token: 10 routed + shared (-1)
    MoeSegs gu = {{L.gate_exps, L.up_exps}, {L.sh_gate.w, L.sh_up.w}, {eg_, eu_}, L.gate_eb};
    if (L.sh_gate.fmt != L.sh_up.fmt || L.up_fmt != L.gate_fmt) throw std::runtime_error("gate/up format mismatch");
    fn::moe_gemv(L.gate_fmt, L.sh_gate.fmt, gu, 2, eid_, D::n_used + 1, xq_, D::exp_ff, D::n_embd, T, s_);
    fn::moe_silu_quant(eg_, eu_, xq_, T * (D::n_used + 1), D::exp_ff, s_);
    MoeSegs dn = {{L.down_exps, nullptr}, {L.sh_down.w, nullptr}, {ymoe_, nullptr}, L.down_eb, rw_};
    fn::moe_gemv(L.down_fmt, L.sh_down.fmt, dn, 1, eid_, D::n_used + 1, xq_, D::n_embd, D::exp_ff, T, s_);
}

// head: final gated-residual read (the pending write folded in, so R ends as the wide residual after the last combine), LM head
void Qwen4Exp::head(int T, const Pending& p, float* R, float* logits_out) {
    { fn::HcNormArgs na; na.R = R; na.y = p.y; na.inject = p.inject; na.w_norm = head_norm_; na.xn = xn_;
      fn::hc_norm(na, xq_, T, D::rms_eps, s_); }
    gemv_splitk(head_down_.fmt, head_down_.w, xq_, lo_, head_down_.N, head_down_.K, KSPLIT, T, s_);
    fn::hc_silu_quant(lo_, KSPLIT, xq_, T, s_); gemv(head_up_, up_, T);
    fn::hc_mix(xn_, up_, nullptr, mixed_, nullptr, xq_, T, s_);
    gemv(output_, logits_out, T);
}

// Row bookkeeping for a pass: rows in request order; RowBatch (attention) and SeqBatch (GDN/conv) descriptors
struct PassRows { int rows = 0, r0[Qwen4Exp::MAXS]; std::vector<int> tokens; RowBatch rb; SeqBatch sb_in_out; };
static PassRows pass_rows(const SlotReq* reqs, int S, const std::vector<int>& cur, int max_ctx) {
    if (S < 1 || S > Qwen4Exp::MAXS) throw std::runtime_error("bad S");
    PassRows p; p.rb.n = 0;
    for (int i = 0; i < S; ++i) {
        const SlotReq& q = reqs[i];
        if (q.T < 1 || p.rows + q.T > Qwen4Exp::MAXR) throw std::runtime_error("too many rows");
        if (q.pos + q.T > max_ctx) throw std::runtime_error("pos >= max_ctx");
        if (q.pos + q.T > FnDims::qsa_dense_limit) fprintf(stderr, "warning: %d cached tokens > QSA budget; dense attention is no longer exact\n", q.pos + q.T);
        p.r0[i] = p.rows;
        for (int t = 0; t < q.T; ++t) { p.rb.pos[p.rows] = q.pos + t; p.rb.kv[p.rows] = q.slot; p.tokens.push_back(q.tokens[t]); ++p.rows; }
        p.sb_in_out.s[i] = {p.r0[i], q.T, q.slot * 2 + cur[q.slot], q.slot * 2 + (cur[q.slot] ^ 1)};
    }
    p.rb.n = p.rows; p.sb_in_out.n = S;
    return p;
}

float Qwen4Exp::forward(const SlotReq* reqs, int S) {
    PassRows P = pass_rows(reqs, S, cur_, max_ctx_);
    const int rows = P.rows;
    CK(hipEventRecord(ev0_, s_));
    CK(hipMemcpyAsync(d_tok_, P.tokens.data(), rows * 4, hipMemcpyHostToDevice, s_));
    fn::embed_q8_0(tok_embd_, d_tok_, rows, emb_, D::n_embd, s_);
    fn::init_wide(emb_, R_, rows, s_);
    Pending pend;
    for (int il = 0; il < D::n_layer; ++il) {
        auto& L = layers_[il];
        if (il == D::ple_layer) {   // PLE edits R itself: flush the pending MoE write of the previous layer first
            fn::hc_combine(R_, ymoe_, inject2_, rows, s_);
            pend = Pending{};
            ple(reqs, S, P.r0, rows);
        }
        hc_block(L, false, rows, pend, mixed_, inject_, R_);
        if (dbg_armed_ && dbg_layers_.count(il)) dbg_capture_mix(il, mixed_ + (size_t)(rows - 1) * D::n_embd, false);
        if (!D::is_attn(il)) {
            const int gi = gdn_index(il);
            // raw qkv / beta / alpha land in the replay buffers (accept(slot, m < T) recomputes the state from them)
            float* qkv_raw = sv_qkv_ + (size_t)gi * MAXR * D::gdn_qkv; float* b = sv_beta_ + (size_t)gi * MAXR * 64; float* a = sv_alpha_ + (size_t)gi * MAXR * 64;
            float* z = big1_; float* qkv = big2_; float* ao = big0_;   // z [rows][6144], qkv [rows][10240], ao [rows][6144]
            GemvSeg segs[4] = {{L.qkv.fmt, L.qkv.w, qkv_raw, L.qkv.N, L.qkv.K}, {L.zgate.fmt, L.zgate.w, z, L.zgate.N, L.zgate.K},
                               {L.beta.fmt, L.beta.w, b, L.beta.N, L.beta.K}, {L.alpha.fmt, L.alpha.w, a, L.alpha.N, L.alpha.K}};
            gemv_multi(segs, 4, xq_, rows, s_);
            const size_t cso = (size_t)gi * 3 * D::gdn_qkv, gso = (size_t)gi * D::gdn_v_heads * D::gdn_dim * D::gdn_dim;
            gdn_conv_b(qkv_raw, P.sb_in_out, cs_stride_, conv_state_ + cso, conv_state_ + cso, L.conv_w, qkv, D::gdn_qkv, s_);
            gdn_step_sig_b(qkv, z, b, a, P.sb_in_out, st_stride_, L.dt_bias, L.a_neg, gdn_state_ + gso, gdn_state_ + gso, L.ssm_norm, ao, xq_, D::rms_eps, s_);
            gemv(L.ssm_out, y_, rows);
        } else {
            const int ai = attn_index(il);
            const size_t off = (size_t)ai * max_ctx_ * D::n_head_kv * D::head_dim, offv = (size_t)ai * (max_ctx_ + 128) * D::n_head_kv * D::head_dim;
            attn_layer_b(L, rows, P.rb, kc_ + off, vc_ + offv, kv_stride_, kvt_stride_);
        }
        pend = Pending{}; pend.y = y_; pend.inject = inject_;
        hc_block(L, true, rows, pend, mixed_, inject2_, R_);
        moe_layer(L, rows);
        pend = Pending{}; pend.y = ymoe_; pend.inject = inject2_;   // the write half is folded into the next read's norm
        if (dbg_armed_ && dbg_layers_.count(il)) dbg_capture(il, y_ + (size_t)(rows - 1) * D::n_embd, ymoe_ + (size_t)(rows - 1) * D::n_embd, false);
    }
    head(rows, pend, R_, logits_);
    for (int i = 0; i < S; ++i) { const int sl = reqs[i].slot; pend_tokens_[sl].assign(reqs[i].tokens, reqs[i].tokens + reqs[i].T); lr0_[sl] = P.r0[i]; lT_[sl] = reqs[i].T; }
    CK(hipEventRecord(ev1_, s_)); CK(hipEventSynchronize(ev1_));
    float ms; CK(hipEventElapsedTime(&ms, ev0_, ev1_));
    gpu_ms_ += ms;
    return ms;
}

void Qwen4Exp::mtp_forward(const SlotReq* reqs, int S, const float* h) {
    if (!has_mtp_) throw std::runtime_error("no MTP block in this GGUF (use the UD-Q4_K_XL-MTP shards)");
    PassRows P = pass_rows(reqs, S, cur_, max_ctx_);
    const int rows = P.rows;
    const auto& ML = mtp_.L;
    CK(hipMemcpyAsync(d_tok_, P.tokens.data(), rows * 4, hipMemcpyHostToDevice, s_));
    fn::embed_q8_0(tok_embd_, d_tok_, rows, emb_, D::n_embd, s_);
    fn::mtp_prep(emb_, mtp_.enorm, h, mtp_.hnorm, xq_, rows, D::rms_eps, s_);              // xq rows [rows*4][5120]
    gemv_cols(mtp_.eh_proj.fmt, mtp_.eh_proj.w, mtp_.eh_proj.N, mtp_.eh_proj.K, 32, R_mtp_, 4 * rows);   // eh_proj per stream: R_mtp[t][s] = W . [e_norm | h_norm_s]
    Pending pend;
    hc_block(ML, false, rows, pend, mixed_, inject_, R_mtp_);
    attn_layer_b(ML, rows, P.rb, mkc_, mvc_, mkv_stride_, mkvt_stride_);
    pend = Pending{}; pend.y = y_; pend.inject = inject_;
    hc_block(ML, true, rows, pend, mixed_, inject2_, R_mtp_);
    moe_layer(ML, rows);
    pend = Pending{}; pend.y = ymoe_; pend.inject = inject2_;
    head(rows, pend, R_mtp_, mlogits_);
}

void Qwen4Exp::accept(int slot, int m) {
    if (slot < 0 || slot >= n_slots_ || m < 1 || m > lT_[slot]) throw std::runtime_error("bad accept");
    const int cur = cur_[slot], nxt = cur ^ 1, r0 = lr0_[slot];
    if (m < lT_[slot]) {   // replay the recurrent state for m tokens from the saved inputs (cur is intact)
        SeqBatch sb; sb.n = 1; sb.s[0] = {r0, m, slot * 2 + cur, slot * 2 + nxt};
        for (int il = 0; il < D::n_layer; ++il) {
            if (D::is_attn(il)) continue;
            const auto& L = layers_[il]; const int gi = gdn_index(il);
            const size_t cso = (size_t)gi * 3 * D::gdn_qkv, gso = (size_t)gi * D::gdn_v_heads * D::gdn_dim * D::gdn_dim;
            const float* qkv_raw = sv_qkv_ + (size_t)gi * MAXR * D::gdn_qkv;
            gdn_conv_b(qkv_raw, sb, cs_stride_, conv_state_ + cso, conv_state_ + cso, L.conv_w, big2_, D::gdn_qkv, s_);
            gdn_step_sig_b(big2_, nullptr, sv_beta_ + (size_t)gi * MAXR * 64, sv_alpha_ + (size_t)gi * MAXR * 64, sb, st_stride_, L.dt_bias, L.a_neg,
                           gdn_state_ + gso, gdn_state_ + gso, L.ssm_norm, nullptr, xq_, D::rms_eps, s_);
        }
        const auto& P = layers_[D::ple_layer];
        float* stc = ple_state_ + ((size_t)slot * 2 + cur) * ple_stride_; float* stn = ple_state_ + ((size_t)slot * 2 + nxt) * ple_stride_;
        CK(hipMemcpyAsync(stn, stc, ple_stride_ * 4, hipMemcpyDeviceToDevice, s_));
        CK(hipMemcpyAsync(ple_scratch_, sv_pleR_ + (size_t)r0 * D::wide, (size_t)m * D::wide * 4, hipMemcpyDeviceToDevice, s_));   // R_ (= h_nextn) must stay intact
        fn::ple_apply(ple_scratch_, sv_plekey_ + (size_t)r0 * D::wide, sv_pleval_ + (size_t)r0 * D::n_embd, P.ple_nk, P.ple_nq, P.ple_nc, P.ple_conv, stn, m, D::rms_eps, s_);
    }
    auto& hist = hist_[slot];
    for (int i = 0; i < m; ++i) hist.push_back(pend_tokens_[slot][i]);
    if (hist.size() > 2) hist.erase(hist.begin(), hist.end() - 2);
    cur_[slot] = nxt; lT_[slot] = 0;
}

// ======================= prefill: GEMM path (bf16 activations, hipBLASLt) =======================
// per-phase GPU timing for the prefill path (HIPSTER_TIMING=1): events after each phase on the stream
static const bool g_timing = getenv("HIPSTER_TIMING") != nullptr;
double g_ptime[8]; static const char* g_pname[8] = {"hc(norm+down+silu+up+mix)", "dense gemm+dequant+split", "gdn", "attention", "router+route+sort", "moe gemm+silu+combine", "ple", "other"};
struct PhaseTimer {
    hipStream_t s; hipEvent_t e0, e1; bool on;
    PhaseTimer(hipStream_t st) : s(st), on(g_timing) { if (on) { hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0, s); } }
    void mark(int cat) { if (!on) return; hipEventRecord(e1, s); hipEventSynchronize(e1); float ms; hipEventElapsedTime(&ms, e0, e1); g_ptime[cat] += ms; std::swap(e0, e1); }
    ~PhaseTimer() { if (on) { hipEventDestroy(e0); hipEventDestroy(e1); } }
};
static std::map<std::pair<int, int>, std::pair<double, int>> g_gemm_ms;   // (N,K) -> (sum ms, calls), HIPSTER_TIMING
static std::map<std::pair<int, int>, double> g_gemm_min;
void prefill_timing_report(double denom_tokens) {
    if (!g_timing) return;
    double tot = 0; for (double v : g_ptime) tot += v;
    fprintf(stderr, "prefill phases (%.0f tokens): ", denom_tokens);
    for (int i = 0; i < 8; ++i) if (g_ptime[i] > 0) fprintf(stderr, "%s %.0f ms (%.0f%%) | ", g_pname[i], g_ptime[i], 100 * g_ptime[i] / tot);
    fprintf(stderr, "total %.0f ms\n", tot);
    for (double& v : g_ptime) v = 0;
    for (auto& kv : g_gemm_ms) if (kv.first.first % 8192 > 64) fprintf(stderr, "  gemm M=%d N=%5d K=%5d: %4d calls, avg %.3f ms, min %.3f ms (avg includes the first call's autotune)\n", kv.first.first % 8192, kv.first.first / 8192, kv.first.second / 10, kv.second.second, kv.second.first / kv.second.second, g_gemm_min[kv.first]);
    g_gemm_ms.clear(); g_gemm_min.clear();
}
void Qwen4Exp::lin_gemm(const GemvSeg* segs, int n, int M, const uint16_t* A) {
    int Nt = 0; const int K = segs[0].K;
    for (int i = 0; i < n; ++i) { dequant_bf16(segs[i].fmt, segs[i].w, wscratch_ + (size_t)Nt * K, segs[i].N, K, s_); Nt += segs[i].N; }
    auto run = [&] { blas_.gemm(A, wscratch_, gout_, M, Nt, K, s_); };
    if (g_timing) { static const bool twice = getenv("HIPSTER_GEMM_TWICE") != nullptr;
                    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
                    for (int rep = 0; rep < (twice ? 2 : 1); ++rep) {
                        hipEventRecord(a, s_); run(); hipEventRecord(b, s_); hipEventSynchronize(b);
                        float ms; hipEventElapsedTime(&ms, a, b); auto& e = g_gemm_ms[{Nt * 8192 + M, K * 10 + rep}]; e.first += ms; ++e.second; auto& mn = g_gemm_min[{Nt * 8192 + M, K * 10 + rep}]; mn = (e.second == 1 || ms < mn) ? ms : mn; }
                    hipEventDestroy(a); hipEventDestroy(b); return; }
    run();
}
// columns [off, off+N) of the last GEMM output -> f32 [M][N]
void Qwen4Exp::gemm_seg(int Nt, int off, int N, float* dst, int M) { bf16_to_f32_seg(gout_, Nt, off, N, dst, M, s_); }
// hyper-connection read over T rows: norm (pending write folded) -> down GEMM -> silu -> up GEMM -> mix.
// Leaves: pmixed_ f32, xb_ = bf16 mixed [T][2560] (next GEMM input), pxq_ int8 rows when ffn (MoE input), inject partials.
void Qwen4Exp::hc_read_pf(const FnLayer& L, bool ffn, int T, const Pending& p, float* inject_out) {
    fn::HcNormArgs na; na.R = pR_; na.y = p.y; na.inject = p.inject; na.w_norm = ffn ? L.hc_ffn_norm : L.hc_attn_norm; na.xn = pxn_; na.xb = xb_;
    fn::hc_norm(na, XQ8{nullptr, nullptr, nullptr}, T, D::rms_eps, s_);
    const FnLin& dn = ffn ? L.hc_ffn_down : L.hc_attn_down; const FnLin& up = ffn ? L.hc_ffn_up : L.hc_attn_up;
    GemvSeg sd = {dn.fmt, dn.w, nullptr, dn.N, dn.K}; lin_gemm(&sd, 1, T, xb_);         // gout_ [T][320] bf16
    fn::hc_silu_bf16(gout_, xb2_, T, s_);
    GemvSeg su = {up.fmt, up.w, nullptr, up.N, up.K}; lin_gemm(&su, 1, T, xb2_);        // gout_ [T][10240] bf16
    fn::hc_mix_bf16(pxn_, gout_, ffn ? L.hc_ffn_inject : L.hc_attn_inject, pmixed_, inject_out, ffn ? pxq_ : XQ8{nullptr, nullptr, nullptr}, xb_, T, s_);
}

// HIPSTER_DBG_LAYERS="0,1,3,47": capture y (attention/GDN output) and ymoe (MoE output) of the last row after those layers,
// on both paths, so prefill vs decode can be bisected layer by layer
void Qwen4Exp::dbg_capture(int il, const float* y, const float* ymoe, bool pf) {
    auto& d = pf ? dbg_pf_[il] : dbg_dec_[il];
    d.resize(3 * D::n_embd);
    CK(hipMemcpy(d.data(), y, D::n_embd * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(d.data() + D::n_embd, ymoe, D::n_embd * 4, hipMemcpyDeviceToHost));
}
void Qwen4Exp::dbg_capture_mix(int il, const float* mixed, bool pf) {   // mixed after the attention-side hc read of layer il
    auto& d = pf ? dbg_pf_[il] : dbg_dec_[il]; d.resize(3 * D::n_embd);
    CK(hipMemcpy(d.data() + 2 * D::n_embd, mixed, D::n_embd * 4, hipMemcpyDeviceToHost));
}
void Qwen4Exp::dbg_report() const {
    for (auto& kv : dbg_pf_) {
        auto it = dbg_dec_.find(kv.first); if (it == dbg_dec_.end()) continue;
        const auto& a = kv.second; const auto& b = it->second;
        for (int part = 0; part < 3; ++part) {
            double md = 0, mx = 0, se = 0, sz = 0; for (int i = 0; i < D::n_embd; ++i) { const double x = a[part * D::n_embd + i], z = b[part * D::n_embd + i]; md = fmax(md, fabs(x - z)); mx = fmax(mx, fabs(z)); se += (x - z) * (x - z); sz += z * z; }
            fprintf(stderr, "  layer %2d %-5s: max|diff| %.3g / max %.3g = %.2f%%   rel rms %.4f\n", kv.first, part == 0 ? "y" : part == 1 ? "moe" : "mix0", md, mx, 100 * md / mx, sqrt(se / sz));
        }
    }
}

float Qwen4Exp::prefill(const int* tokens, int T, int pos0) {
    if (!max_prefill_) throw std::runtime_error("engine built without prefill buffers");
    if (T < 1 || T > max_prefill_) throw std::runtime_error("bad prefill T");
    if (pos0 + T > max_ctx_) throw std::runtime_error("pos >= max_ctx");
    if (pos0 + T > D::qsa_dense_limit) fprintf(stderr, "warning: %d cached tokens > QSA budget; dense attention is no longer exact\n", pos0 + T);
    const int sl = 0, cur = cur_[sl], nxt = cur ^ 1;
    CK(hipEventRecord(ev0_, s_));
    CK(hipMemcpyAsync(pd_tok_, tokens, T * 4, hipMemcpyHostToDevice, s_));
    fn::embed_q8_0(tok_embd_, pd_tok_, T, pemb_, D::n_embd, s_);
    fn::init_wide(pemb_, pR_, T, s_);
    PhaseTimer pt(s_);
    Pending pend;
    for (int il = 0; il < D::n_layer; ++il) {
        auto& L = layers_[il];
        if (il == D::ple_layer) {
            fn::hc_combine(pR_, pymoe_, pinj2_, T, s_); pend = Pending{};
            std::vector<float> emb((size_t)T * D::n_embd);
            const auto t0 = std::chrono::steady_clock::now();
            ple_rows(hist_[sl], tokens, T, emb.data());
            ple_host_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            CK(hipMemcpyAsync(pemb_, emb.data(), emb.size() * 4, hipMemcpyHostToDevice, s_));
            f32_to_bf16(pemb_, xb2_, (size_t)T * D::n_embd, s_);
            GemvSeg sk = {L.ple_key.fmt, L.ple_key.w, nullptr, L.ple_key.N, L.ple_key.K}; lin_gemm(&sk, 1, T, xb2_); gemm_seg(D::wide, 0, D::wide, pkey_, T);
            GemvSeg sv = {L.ple_value.fmt, L.ple_value.w, nullptr, L.ple_value.N, L.ple_value.K}; lin_gemm(&sv, 1, T, xb2_); gemm_seg(D::n_embd, 0, D::n_embd, pval_, T);
            float* stc = ple_state_ + ((size_t)sl * 2 + cur) * ple_stride_; float* stn = ple_state_ + ((size_t)sl * 2 + nxt) * ple_stride_;
            CK(hipMemcpyAsync(stn, stc, ple_stride_ * 4, hipMemcpyDeviceToDevice, s_));
            fn::ple_apply(pR_, pkey_, pval_, L.ple_nk, L.ple_nq, L.ple_nc, L.ple_conv, stn, T, D::rms_eps, s_);
            pt.mark(6);
        }
        hc_read_pf(L, false, T, pend, pinj_); pt.mark(0);
        if (dbg_layers_.count(il)) dbg_capture_mix(il, pmixed_ + (size_t)(T - 1) * D::n_embd, true);
        if (!D::is_attn(il)) {
            const int gi = gdn_index(il);
            GemvSeg segs[4] = {{L.qkv.fmt, L.qkv.w, nullptr, L.qkv.N, L.qkv.K}, {L.zgate.fmt, L.zgate.w, nullptr, L.zgate.N, L.zgate.K},
                               {L.beta.fmt, L.beta.w, nullptr, L.beta.N, L.beta.K}, {L.alpha.fmt, L.alpha.w, nullptr, L.alpha.N, L.alpha.K}};
            lin_gemm(segs, 4, T, xb_);   // gout_ [T][10240 | 6144 | 48 | 48]
            const int Nt = D::gdn_qkv + D::gdn_z + 96;
            gemm_seg(Nt, 0, D::gdn_qkv, pqkv_, T); gemm_seg(Nt, D::gdn_qkv, D::gdn_z, pz_, T);
            gemm_seg(Nt, D::gdn_qkv + D::gdn_z, 48, pba_, T); gemm_seg(Nt, D::gdn_qkv + D::gdn_z + 48, 48, pba_ + (size_t)T * 48, T);
            pt.mark(1);
            const size_t cso = (size_t)gi * 3 * D::gdn_qkv, gso = (size_t)gi * D::gdn_v_heads * D::gdn_dim * D::gdn_dim;
            gdn_conv(pqkv_, T, conv_state_ + ((size_t)sl * 2 + cur) * cs_stride_ + cso, conv_state_ + ((size_t)sl * 2 + nxt) * cs_stride_ + cso, L.conv_w, pqkvc_, D::gdn_qkv, s_);
            float* qn = pqn_; float* kn = pqn_ + (size_t)T * 2048; float* bt = pqn_ + (size_t)T * 4096; float* dc = bt + (size_t)T * 48;
            gdn_prep(pqkvc_, pba_, pba_ + (size_t)T * 48, L.dt_bias, L.a_neg, T, qn, kn, bt, dc, D::rms_eps, s_);
            gdn_scan(qn, kn, pqkvc_, bt, dc, T, gdn_state_ + ((size_t)sl * 2 + cur) * st_stride_ + gso, gdn_state_ + ((size_t)sl * 2 + nxt) * st_stride_ + gso, praw_, s_);
            gdn_out_sig(praw_, pz_, L.ssm_norm, T, pao_, xb_, D::rms_eps, s_); pt.mark(2);
            GemvSeg so = {L.ssm_out.fmt, L.ssm_out.w, nullptr, L.ssm_out.N, L.ssm_out.K}; lin_gemm(&so, 1, T, xb_); gemm_seg(D::n_embd, 0, D::n_embd, py_, T); pt.mark(1);
        } else {
            const int ai = attn_index(il);
            const size_t off = (size_t)ai * max_ctx_ * D::n_head_kv * D::head_dim, offv = (size_t)ai * (max_ctx_ + 128) * D::n_head_kv * D::head_dim;
            GemvSeg segs[3] = {{L.wq.fmt, L.wq.w, nullptr, L.wq.N, L.wq.K}, {L.wk.fmt, L.wk.w, nullptr, L.wk.N, L.wk.K}, {L.wv.fmt, L.wv.w, nullptr, L.wv.N, L.wv.K}};
            lin_gemm(segs, 3, T, xb_);
            const int Nt = 12288 + 1024;
            gemm_seg(Nt, 0, 12288, pq_, T); gemm_seg(Nt, 12288, 512, pk_, T); gemm_seg(Nt, 12800, 512, pv_, T); pt.mark(1);
            attn_stage1_24_2(pq_, pk_, pv_, T, L.q_norm, L.k_norm, D::rope_base, pos0, kc_ + off, vc_ + offv, max_ctx_, D::rms_eps, s_);
            attn_prefill_24_2(pq_, kc_ + off, vc_ + offv, T, pos0, max_ctx_, xb_, s_);   // xb_ bf16 [T][6144]
            pt.mark(3);
            GemvSeg so = {L.wo.fmt, L.wo.w, nullptr, L.wo.N, L.wo.K}; lin_gemm(&so, 1, T, xb_); gemm_seg(D::n_embd, 0, D::n_embd, py_, T); pt.mark(1);
        }
        pend = Pending{}; pend.y = py_; pend.inject = pinj_;
        hc_read_pf(L, true, T, pend, pinj2_); pt.mark(0);
        // MoE over T rows: exact f32 router GEMM, top-10, sorted slots -> grouped WMMA GEMMs
        blas_.gemm_f32(pmixed_, L.router, prl_, T, D::n_exp + 1, D::n_embd, s_);
        fn::moe_route(prl_, peid_, prw_, T, s_);
        {   // slots sorted by expert on the host -> tiles of <= 16 columns; grouped WMMA GEMMs read each expert once per tile
            const int NS = T * (D::n_used + 1); constexpr int NE = D::n_used + 1;
            std::vector<int> eid(NS); CK(hipMemcpy(eid.data(), peid_, (size_t)NS * 4, hipMemcpyDeviceToHost));
            std::vector<int> order(NS), kpos(NS), rowtok(NS); for (int i = 0; i < NS; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return eid[a] != eid[b] ? eid[a] < eid[b] : a < b; });
            // column tiles of up to 16*ct sorted positions per expert: the expert's weights are read once per tile
            static const int ct_env = getenv("HIPSTER_MOE_CT") ? atoi(getenv("HIPSTER_MOE_CT")) : 1;   // bench_moe: 16-column tiles are the fastest (docs/prefill-flash-next.md)
            const int ct = std::max(1, std::min(3, ct_env)), maxc = 16 * ct;
            std::vector<MoeTile> tiles, stiles; tiles.reserve(NS / maxc + D::n_exp + 2);
            for (int k = 0; k < NS; ++k) {
                const int te = order[k]; kpos[te] = k; rowtok[k] = te / NE;
                auto& lst = eid[te] < 0 ? stiles : tiles;
                if (lst.empty() || (k > 0 && eid[te] != eid[order[k - 1]]) || lst.back().ncols == maxc) lst.push_back({eid[te], k, 1}); else ++lst.back().ncols;
            }
            const int nt = (int)tiles.size(), nst = (int)stiles.size();
            CK(hipMemcpyAsync(d_kpos_, kpos.data(), (size_t)NS * 4, hipMemcpyHostToDevice, s_)); CK(hipMemcpyAsync(d_rowtok_, rowtok.data(), (size_t)NS * 4, hipMemcpyHostToDevice, s_));
            CK(hipMemcpyAsync(d_tiles_, tiles.data(), (size_t)nt * sizeof(MoeTile), hipMemcpyHostToDevice, s_));
            CK(hipMemcpyAsync(d_tiles_ + nt, stiles.data(), (size_t)nst * sizeof(MoeTile), hipMemcpyHostToDevice, s_)); pt.mark(4);
            const MoeTile* dst = d_tiles_ + nt;
            moe_gemm(L.gate_fmt, L.sh_gate.fmt, L.gate_exps, L.sh_gate.w, L.gate_eb, d_tiles_, nt, dst, nst, ct, d_rowtok_, pxq_, peg_, D::exp_ff, D::n_embd, s_);
            moe_gemm(L.up_fmt, L.sh_up.fmt, L.up_exps, L.sh_up.w, L.up_eb, d_tiles_, nt, dst, nst, ct, d_rowtok_, pxq_, peu_, D::exp_ff, D::n_embd, s_);
            fn::moe_silu_quant(peg_, peu_, pxq_, NS, D::exp_ff, s_);   // rows in sorted order (k)
            moe_gemm(L.down_fmt, L.sh_down.fmt, L.down_exps, L.sh_down.w, L.down_eb, d_tiles_, nt, dst, nst, ct, nullptr, pxq_, pdown_, D::n_embd, D::exp_ff, s_);
            fn::moe_combine_sorted(pdown_, prw_, d_kpos_, pymoe_, T, NE, s_); pt.mark(5);
        }
        pend = Pending{}; pend.y = pymoe_; pend.inject = pinj2_;
        if (dbg_layers_.count(il)) dbg_capture(il, py_ + (size_t)(T - 1) * D::n_embd, pymoe_ + (size_t)(T - 1) * D::n_embd, true);
    }
    fn::hc_combine(pR_, pymoe_, pinj2_, T, s_);                                                          // final residual for all rows
    CK(hipMemcpyAsync(R_, pR_ + (size_t)(T - 1) * D::wide, (size_t)D::wide * 4, hipMemcpyDeviceToDevice, s_));   // h_nextn = last row
    head(1, Pending{}, R_, logits_);                                                                     // logits for the last row
    pt.mark(7);
    auto& hist = hist_[sl]; for (int i = 0; i < T; ++i) hist.push_back(tokens[i]); if (hist.size() > 2) hist.erase(hist.begin(), hist.end() - 2);
    cur_[sl] = nxt; lT_[sl] = 0;
    CK(hipEventRecord(ev1_, s_)); CK(hipEventSynchronize(ev1_));
    float ms; CK(hipEventElapsedTime(&ms, ev0_, ev1_));
    return ms;
}

void Qwen4Exp::topk(const float* logits, int T, int k, int* ids, float* vals) {
    argmax_topk(logits, T, D::n_vocab, k, d_ids_, d_vals_, s_);
    CK(hipStreamSynchronize(s_));
    CK(hipMemcpy(ids, d_ids_, T * k * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(vals, d_vals_, T * k * 4, hipMemcpyDeviceToHost));
}

}  // namespace hip
