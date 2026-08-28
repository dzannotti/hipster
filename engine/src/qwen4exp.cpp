#include "qwen4exp.h"
#include "quant.h"
#include "../kernels/ops.h"
#include "../kernels/fn.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) throw std::runtime_error(std::string("HIP: ") + hipGetErrorString(e) + " @" + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace hip {
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

Qwen4Exp::Qwen4Exp(const std::string& path, int max_ctx) : max_ctx_(max_ctx) {
    gguf_ = new GGUF(path);
    GGUF& g = *gguf_;
    if (g.arch() != "qwen4exp") throw std::runtime_error("not a qwen4exp GGUF: " + g.arch());
    CK(hipStreamCreate(&s_)); CK(hipEventCreate(&ev0_)); CK(hipEventCreate(&ev1_));
    // PLE constants
    { auto* m = g.kv("qwen4exp.ple.layer_multipliers"); auto* o = g.kv("qwen4exp.ple.head_offsets"); auto* v = g.kv("qwen4exp.ple.head_vocab_sizes");
      for (auto& x : m->arr) ple_mult_.push_back((uint64_t)x.i ? (uint64_t)x.i : x.u); for (auto& x : o->arr) ple_off_.push_back(x.i ? x.i : (int64_t)x.u); for (auto& x : v->arr) ple_vocab_.push_back(x.i ? x.i : (int64_t)x.u);
      eos_ = (int)g.kv_u32("qwen4exp.ple.eos_token_id");
      ple_table_ = &g.get("per_layer_token_embd.weight"); }
    layers_.resize(D::n_layer);
    for (int il = 0; il < D::n_layer; ++il) {
        auto& L = layers_[il]; const std::string p = "blk." + std::to_string(il) + ".";
        L.hc_attn_norm = upload_f32(g, p + "hc_attn_norm.weight", D::wide); L.hc_attn_inject = upload_f32(g, p + "hc_attn_inject.weight", (size_t)D::wide * 4, true);
        L.hc_ffn_norm = upload_f32(g, p + "hc_ffn_norm.weight", D::wide); L.hc_ffn_inject = upload_f32(g, p + "hc_ffn_inject.weight", (size_t)D::wide * 4, true);
        L.hc_attn_down = upload_lin(g, p + "hc_attn_down.weight"); L.hc_attn_up = upload_lin(g, p + "hc_attn_up.weight");
        L.hc_ffn_down = upload_lin(g, p + "hc_ffn_down.weight"); L.hc_ffn_up = upload_lin(g, p + "hc_ffn_up.weight");
        if (D::is_attn(il)) {
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
        if (il % 8 == 7) fprintf(stderr, "  loaded %d layers, %.1f GiB\n", il + 1, weight_bytes_ / 1073741824.0);
    }
    { const GTensor& t = g.get("token_embd.weight"); if (t.type != GType::Q8_0) throw std::runtime_error("token_embd must be Q8_0");
      std::vector<uint8_t> stage(t.nbytes); g.read_tensor(t, stage.data());
      CK(hipMalloc(&tok_embd_, t.nbytes)); CK(hipMemcpy(tok_embd_, stage.data(), t.nbytes, hipMemcpyHostToDevice)); weight_bytes_ += t.nbytes; }
    head_norm_ = upload_f32(g, "output_hc_norm.weight", D::wide); head_down_ = upload_lin(g, "output_hc_down.weight"); head_up_ = upload_lin(g, "output_hc_up.weight");
    output_ = upload_lin(g, "output.weight");
    // state
    const size_t st = (size_t)D::n_gdn * D::gdn_v_heads * D::gdn_dim * D::gdn_dim, cs = (size_t)D::n_gdn * 3 * D::gdn_qkv;
    for (int i = 0; i < 2; ++i) { CK(hipMalloc(&gdn_state_[i], st * 4)); CK(hipMalloc(&conv_state_[i], cs * 4)); }
    CK(hipMalloc(&ple_state_, (size_t)D::ple_hist * D::wide * 4));
    const size_t kv = (size_t)D::n_attn * max_ctx_ * D::n_head_kv * D::head_dim, kvt = (size_t)D::n_attn * (max_ctx_ + 128) * D::n_head_kv * D::head_dim;
    CK(hipMalloc(&kc_, kv * 2)); CK(hipMalloc(&vc_, kvt * 2)); CK(hipMemset(vc_, 0, kvt * 2));
    const int T = D::max_T;
    CK(hipMalloc(&R_, (size_t)T * D::wide * 4)); CK(hipMalloc(&xn_, (size_t)T * D::wide * 4)); CK(hipMalloc(&lo_, (size_t)T * D::hc_lr * 4)); CK(hipMalloc(&up_, (size_t)T * D::wide * 4));
    CK(hipMalloc(&mixed_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&inject_, (size_t)T * 20 * 4 * 4)); CK(hipMalloc(&inject2_, (size_t)T * 20 * 4 * 4)); CK(hipMalloc(&ymoe_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&y_, (size_t)T * D::n_embd * 4));
    CK(hipMalloc(&big0_, (size_t)T * D::gdn_qkv * 4)); CK(hipMalloc(&big1_, (size_t)T * D::gdn_qkv * 4)); CK(hipMalloc(&big2_, (size_t)T * D::gdn_qkv * 4));
    CK(hipMalloc(&logits_, (size_t)T * D::n_vocab * 4)); CK(hipMalloc(&emb_, (size_t)T * D::n_embd * 4));
    CK(hipMalloc(&rlogits_, (size_t)T * (D::n_exp + 1) * 4)); CK(hipMalloc(&rw_, (size_t)T * (D::n_used + 1) * 4)); CK(hipMalloc(&eid_, (size_t)T * (D::n_used + 1) * 4));
    CK(hipMalloc(&eg_, (size_t)T * (D::n_used + 1) * D::exp_ff * 4)); CK(hipMalloc(&eu_, (size_t)T * (D::n_used + 1) * D::exp_ff * 4));
    CK(hipMalloc(&shx_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&sg_, (size_t)T * 4));
    const size_t xk = (size_t)16 * 17408;
    CK(hipMalloc(&xq_.q, xk)); CK(hipMalloc(&xq_.d, xk / 32 * 4)); CK(hipMalloc(&xq_.s, xk / 32 * 4));
    CK(hipMemset(xq_.q, 0, xk)); CK(hipMemset(xq_.d, 0, xk / 32 * 4)); CK(hipMemset(xq_.s, 0, xk / 32 * 4));
    CK(hipMalloc(&d_tok_, T * 4)); CK(hipMalloc(&d_ids_, T * 16 * 4)); CK(hipMalloc(&d_vals_, T * 16 * 4));
    reset();
}

void Qwen4Exp::reset() {
    for (int i = 0; i < 2; ++i) {
        CK(hipMemset(gdn_state_[i], 0, (size_t)D::n_gdn * D::gdn_v_heads * D::gdn_dim * D::gdn_dim * 4));
        CK(hipMemset(conv_state_[i], 0, (size_t)D::n_gdn * 3 * D::gdn_qkv * 4));
    }
    CK(hipMemset(ple_state_, 0, (size_t)D::ple_hist * D::wide * 4));
    hist_.clear(); cur_ = 0; last_T_ = 0;
}

// lanes per row for the T == 1 GEMV, from bench_gemv on this checkpoint's tensors (docs/decode-flash-next.md):
// Q8_0 K=6144 (ssm_out, wo): 2 lanes 199 GB/s vs 32 lanes 142; Q8_0 K=320 (hc up): 8 lanes 150 vs 4 lanes 104;
// Q8_0 K=2560 (qkv, gate, attn_q): 32 lanes 190-197, everything else worse.
static int fn_tpr(const FnLin& l) {
    if (l.K == 320) return 8;
    if (l.K == 6144) return 2;
    return 32;
}
void Qwen4Exp::gemv(const FnLin& l, float* y, int ncol) {
    if (ncol == 1) gemv_q8(l.fmt, l.w, xq_, y, l.N, l.K, 1, fn_tpr(l), 1, s_);
    else gemv_q8(l.fmt, l.w, xq_, y, l.N, l.K, ncol, 4, 3, s_);
}

// gated-residual read: (pending write of the previous block folded into the norm) R -> mixed (f32 + xq) and inject [T][4]
void Qwen4Exp::hc_block(const FnLayer& L, bool ffn, int T, const Pending& p, float* mixed, float* inject_out) {
    fn::HcNormArgs na; na.R = R_; na.y = p.y; na.inject = p.inject;
    na.w_norm = ffn ? L.hc_ffn_norm : L.hc_attn_norm; na.xn = xn_; na.zero_lo = lo_; na.zero_y = ffn ? ymoe_ : nullptr;   // the ffn read zeroes the MoE accumulator
    (void)inject_out;
    fn::hc_norm(na, xq_, T, D::rms_eps, s_);
    const FnLin& dn = ffn ? L.hc_ffn_down : L.hc_attn_down; const FnLin& up = ffn ? L.hc_ffn_up : L.hc_attn_up;
    if (T == 1) gemv_splitk(dn.fmt, dn.w, xq_, lo_, dn.N, dn.K, 8, false, s_);   // 320 rows x 10240: fill the GPU with 8 K-splits
    else gemv(dn, lo_, T);
    fn::hc_silu_quant(lo_, xq_, T, s_);
    gemv(up, up_, T);
    fn::hc_mix(xn_, up_, ffn ? L.hc_ffn_inject : L.hc_attn_inject, mixed, inject_out, xq_, T, s_);
}

// n-gram rows for T tokens: host hash + IQ4_NL dequant from the mmap'd table, then key/value GEMVs and the gate/conv kernel
void Qwen4Exp::ple(const int* tokens, int T) {
    const auto& L = layers_[D::ple_layer];
    std::vector<float> emb((size_t)T * D::n_embd);
    std::vector<int> seq = hist_; for (int i = 0; i < T; ++i) seq.push_back(tokens[i]);
    const int base = (int)hist_.size();
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
                dequant_row((int)ple_table_->type, ple_table_->data + (size_t)row * gtype_row_bytes(ple_table_->type, D::ple_dim), emb.data() + (size_t)i * D::n_embd + h * D::ple_dim, D::ple_dim);
            }
        }
    }
    CK(hipMemcpyAsync(emb_, emb.data(), emb.size() * 4, hipMemcpyHostToDevice, s_));
    quantize_x_q8(emb_, xq_, T, D::n_embd, s_);
    gemv(L.ple_key, up_, T);       // key [T][10240]
    gemv(L.ple_value, y_, T);      // value [T][2560]
    fn::ple_apply(R_, up_, y_, L.ple_nk, L.ple_nq, L.ple_nc, L.ple_conv, ple_state_, T, D::rms_eps, s_);
}

float Qwen4Exp::forward(const int* tokens, int T, int pos0) {
    if (T < 1 || T > D::max_T) throw std::runtime_error("bad T");
    if (pos0 + T > max_ctx_) throw std::runtime_error("pos >= max_ctx");
    if (pos0 + T > D::qsa_dense_limit) fprintf(stderr, "warning: %d cached tokens > QSA budget; dense attention is no longer exact\n", pos0 + T);
    CK(hipEventRecord(ev0_, s_));
    CK(hipMemcpyAsync(d_tok_, tokens, T * 4, hipMemcpyHostToDevice, s_));
    fn::embed_q8_0(tok_embd_, d_tok_, T, emb_, D::n_embd, s_);
    fn::init_wide(emb_, R_, T, s_);
    const int nxt = cur_ ^ 1;
    Pending pend;
    for (int il = 0; il < D::n_layer; ++il) {
        auto& L = layers_[il];
        if (il == D::ple_layer) {   // PLE edits R itself: flush the pending MoE write of the previous layer first
            fn::hc_combine(R_, ymoe_, inject2_, T, s_);
            pend = Pending{};
            ple(tokens, T);
        }
        hc_block(L, false, T, pend, mixed_, inject_);
        if (!D::is_attn(il)) {
            const int gi = gdn_index(il);
            float* qkv_raw = big0_; float* z = big1_; float* b = big2_; float* a = big2_ + (size_t)T * 64; float* qkv = big2_ + (size_t)T * 128; float* ao = big1_ + (size_t)T * D::gdn_z;
            GemvSeg segs[4] = {{L.qkv.fmt, L.qkv.w, qkv_raw, L.qkv.N, L.qkv.K}, {L.zgate.fmt, L.zgate.w, z, L.zgate.N, L.zgate.K},
                               {L.beta.fmt, L.beta.w, b, L.beta.N, L.beta.K}, {L.alpha.fmt, L.alpha.w, a, L.alpha.N, L.alpha.K}};
            if (T == 1) gemv_multi(segs, 4, xq_, s_); else for (auto& sg : segs) gemv_q8(sg.fmt, sg.w, xq_, sg.y, sg.N, sg.K, T, 4, 3, s_);
            const size_t cso = (size_t)gi * 3 * D::gdn_qkv, gso = (size_t)gi * D::gdn_v_heads * D::gdn_dim * D::gdn_dim;
            gdn_conv(qkv_raw, T, conv_state_[cur_] + cso, conv_state_[nxt] + cso, L.conv_w, qkv, D::gdn_qkv, s_);
            gdn_step_sig(qkv, z, b, a, T, L.dt_bias, L.a_neg, gdn_state_[cur_] + gso, gdn_state_[nxt] + gso, L.ssm_norm, ao, xq_, D::rms_eps, s_);
            gemv(L.ssm_out, y_, T);
        } else {
            const int ai = attn_index(il);
            float* qf = big0_; float* k = big1_; float* v = big1_ + (size_t)T * 512; float* ao = big2_;
            GemvSeg segs[3] = {{L.wq.fmt, L.wq.w, qf, L.wq.N, L.wq.K}, {L.wk.fmt, L.wk.w, k, L.wk.N, L.wk.K}, {L.wv.fmt, L.wv.w, v, L.wv.N, L.wv.K}};
            if (T == 1) gemv_multi(segs, 3, xq_, s_); else for (auto& sg : segs) gemv_q8(sg.fmt, sg.w, xq_, sg.y, sg.N, sg.K, T, 4, 3, s_);
            const size_t off = (size_t)ai * max_ctx_ * D::n_head_kv * D::head_dim, offv = (size_t)ai * (max_ctx_ + 128) * D::n_head_kv * D::head_dim;
            attn_decode_24_2(qf, k, v, T, L.q_norm, L.k_norm, D::rope_base, pos0, kc_ + off, vc_ + offv, max_ctx_, ao, xq_, D::rms_eps, s_);
            gemv(L.wo, y_, T);
        }
        pend = Pending{}; pend.y = y_; pend.inject = inject_;
        hc_block(L, true, T, pend, mixed_, inject2_);
        // MoE: router (exact f32) -> top-10 -> expert GEMVs -> combine with the shared expert
        fn::f32_gemv(L.router, mixed_, rlogits_, D::n_exp + 1, D::n_embd, T, s_);   // row 512 = the shared-expert gate
        fn::moe_route(rlogits_, eid_, rw_, T, s_);                                   // writes 11 slots per token: 10 routed + shared (-1)
        // gate|up for the 10 routed experts and the shared expert in ONE launch (same shapes; shared uses ids == -1)
        MoeSegs gu = {{L.gate_exps, L.up_exps}, {L.sh_gate.w, L.sh_up.w}, {eg_, eu_}, L.gate_eb};
        if (L.sh_gate.fmt != L.sh_up.fmt || L.up_fmt != L.gate_fmt) throw std::runtime_error("gate/up format mismatch");
        fn::moe_gemv(L.gate_fmt, L.sh_gate.fmt, gu, 2, eid_, D::n_used + 1, xq_, D::exp_ff, D::n_embd, T, s_);
        fn::moe_silu_quant(eg_, eu_, xq_, T * (D::n_used + 1), D::exp_ff, s_);
        MoeSegs dn = {{L.down_exps, nullptr}, {L.sh_down.w, nullptr}, {ymoe_, nullptr}, L.down_eb, rw_};   // down + weighted combine into ymoe_
        fn::moe_gemv(L.down_fmt, L.sh_down.fmt, dn, 1, eid_, D::n_used + 1, xq_, D::n_embd, D::exp_ff, T, s_);
        pend = Pending{}; pend.y = ymoe_; pend.inject = inject2_;   // the write half is folded into the next read's norm
    }
    // head: final gated-residual read (no inject), then the LM head
    { fn::HcNormArgs na; na.R = R_; na.y = pend.y; na.inject = pend.inject; na.w_norm = head_norm_; na.xn = xn_; na.zero_lo = lo_;
      fn::hc_norm(na, xq_, T, D::rms_eps, s_); }
    if (T == 1) gemv_splitk(head_down_.fmt, head_down_.w, xq_, lo_, head_down_.N, head_down_.K, 8, false, s_); else gemv(head_down_, lo_, T);
    fn::hc_silu_quant(lo_, xq_, T, s_); gemv(head_up_, up_, T);
    fn::hc_mix(xn_, up_, nullptr, mixed_, nullptr, xq_, T, s_);
    gemv(output_, logits_, T);
    for (int i = 0; i < T; ++i) hist_.push_back(tokens[i]);
    if (hist_.size() > 2) hist_.erase(hist_.begin(), hist_.end() - 2);
    last_T_ = T;
    CK(hipEventRecord(ev1_, s_)); CK(hipEventSynchronize(ev1_));
    float ms; CK(hipEventElapsedTime(&ms, ev0_, ev1_));
    return ms;
}

void Qwen4Exp::accept(int m) {
    if (m != last_T_) throw std::runtime_error("Flash-Next: partial accept not implemented yet");
    cur_ ^= 1; last_T_ = 0;
}

void Qwen4Exp::topk(const float* logits, int T, int k, int* ids, float* vals) {
    argmax_topk(logits, T, D::n_vocab, k, d_ids_, d_vals_, s_);
    CK(hipStreamSynchronize(s_));
    CK(hipMemcpy(ids, d_ids_, T * k * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(vals, d_vals_, T * k * 4, hipMemcpyDeviceToHost));
}

}  // namespace hip
