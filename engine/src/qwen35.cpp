#include "qwen35.h"
#include "quant.h"
#include "../kernels/ops.h"
#include <cstdio>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cstdlib>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) throw std::runtime_error(std::string("HIP: ") + hipGetErrorString(e) + " @" + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace hip {
using D = Qwen35Dims;

static int gdn_index(int il) { return il - (il + 1) / 4; }   // 0..47 for GDN layers
static int attn_index(int il) { return (il + 1) / 4 - 1; }   // 0..15 for attention layers

// One arena for all weights (2 MiB-aligned bump allocator).
static uint8_t* g_arena = nullptr; static size_t g_arena_off = 0, g_arena_cap = 0;
static void* arena_alloc(size_t bytes) {
    if (!g_arena) { g_arena_cap = (size_t)18 << 30; if (hipMalloc(&g_arena, g_arena_cap) != hipSuccess) throw std::runtime_error("arena alloc failed"); }
    const size_t align = bytes >= (1u << 20) ? (2u << 20) : 256;
    g_arena_off = (g_arena_off + align - 1) / align * align;
    if (g_arena_off + bytes > g_arena_cap) {   // overflow (e.g. the DFlash2 draft after the 27B): a second 4 GiB arena
        g_arena_cap = (size_t)4 << 30; if (hipMalloc(&g_arena, g_arena_cap) != hipSuccess) throw std::runtime_error("arena alloc failed"); g_arena_off = 0;
        if (bytes > g_arena_cap) throw std::runtime_error("arena exhausted");
    }
    void* p = g_arena + g_arena_off; g_arena_off += bytes; return p;
}

Lin Qwen35::upload_lin(const GGUF& g, const std::string& name) {
    const GTensor& t = g.get(name);
    Lin l; l.K = (int)t.ne[0]; l.N = (int)t.ne[1];
    const size_t rb_src = gtype_row_bytes(t.type, l.K);
    std::vector<uint8_t> tmp, stage(t.nbytes); g.read_tensor(t, stage.data());   // pread + drop cache
    const uint8_t* src = stage.data();
    switch (t.type) {
        case GType::Q4_K: l.fmt = WFmt::Q4_K; l.bytes = t.nbytes; break;
        case GType::Q5_K: l.fmt = WFmt::Q5_K; l.bytes = t.nbytes; break;
        case GType::IQ4_XS: l.fmt = WFmt::IQ4_XS; l.bytes = t.nbytes; break;
        case GType::Q6_K: l.fmt = WFmt::Q6_K_SOA; l.bytes = t.nbytes; tmp.resize(t.nbytes);
            for (int r = 0; r < l.N; ++r) repack_row_q6_K(stage.data() + (size_t)r * rb_src, tmp.data() + (size_t)r * rb_src, l.K);
            src = tmp.data(); break;
        case GType::Q8_0: l.fmt = WFmt::Q8_0_SOA; l.bytes = t.nbytes; tmp.resize(t.nbytes);
            for (int r = 0; r < l.N; ++r) repack_row_q8_0(stage.data() + (size_t)r * rb_src, tmp.data() + (size_t)r * rb_src, l.K);
            src = tmp.data(); break;
        default: {   // Q3_K, IQ3_S, IQ4_NL, ...: dequantise and requantise to Q8_0 SoA
            l.fmt = WFmt::Q8_0_SOA; const size_t rb = wfmt_row_bytes(WFmt::Q8_0_SOA, l.K); l.bytes = rb * l.N; tmp.resize(l.bytes);
            std::vector<float> row(l.K);
            for (int r = 0; r < l.N; ++r) { dequant_row((int)t.type, stage.data() + (size_t)r * rb_src, row.data(), l.K); quantize_row_q8_0_soa(row.data(), tmp.data() + (size_t)r * rb, l.K); }
            src = tmp.data();
            report_ += "  " + name + ": " + gtype_name(t.type) + " -> Q8_0 (" + std::to_string(l.bytes >> 20) + " MiB)\n";
        }
    }
    l.w = (uint8_t*)arena_alloc(l.bytes); CK(hipMemcpy(l.w, src, l.bytes, hipMemcpyHostToDevice));
    weight_bytes_ += l.bytes;
    return l;
}

float* Qwen35::upload_f32(const GGUF& g, const std::string& name, size_t expect) {
    const GTensor& t = g.get(name);
    if (t.type != GType::F32 || t.nbytes != expect * 4) throw std::runtime_error("unexpected f32 tensor " + name);
    std::vector<uint8_t> stage(t.nbytes); g.read_tensor(t, stage.data());
    float* d = (float*)arena_alloc(t.nbytes); CK(hipMemcpy(d, stage.data(), t.nbytes, hipMemcpyHostToDevice));
    weight_bytes_ += t.nbytes;
    return d;
}

AttnBlock Qwen35::upload_attn(const GGUF& g, const std::string& p) {
    AttnBlock a;
    a.wq = upload_lin(g, p + "attn_q.weight"); a.wk = upload_lin(g, p + "attn_k.weight"); a.wv = upload_lin(g, p + "attn_v.weight"); a.wo = upload_lin(g, p + "attn_output.weight");
    a.q_norm = upload_f32(g, p + "attn_q_norm.weight", D::head_dim); a.k_norm = upload_f32(g, p + "attn_k_norm.weight", D::head_dim);
    return a;
}

Qwen35::Qwen35(const std::string& path, int max_ctx, int n_slots) : max_ctx_(max_ctx), n_slots_(n_slots) {
    timing_ = getenv("HIPSTER_TIMING") != nullptr; prefetch_on_ = getenv("HIPSTER_PREFETCH") != nullptr; for (float& t : tacc_) t = 0;
    GGUF g(path);
    if (g.arch() != "qwen35") throw std::runtime_error("not a qwen35 GGUF: " + g.arch());
    CK(hipStreamCreate(&s_)); CK(hipEventCreate(&ev0_)); CK(hipEventCreate(&ev1_)); CK(hipEventCreate(&tev_[0])); CK(hipEventCreate(&tev_[1]));
    layers_.resize(D::n_layer);
    for (int il = 0; il < D::n_layer; ++il) {
        auto& L = layers_[il]; const std::string p = "blk." + std::to_string(il) + ".";
        L.attn_norm = upload_f32(g, p + "attn_norm.weight", D::n_embd);
        L.post_norm = upload_f32(g, p + "post_attention_norm.weight", D::n_embd);
        L.ffn_gate = upload_lin(g, p + "ffn_gate.weight"); L.ffn_up = upload_lin(g, p + "ffn_up.weight"); L.ffn_down = upload_lin(g, p + "ffn_down.weight");
        if (D::is_attn(il)) {
            L.attn = upload_attn(g, p);
        } else {
            L.qkv = upload_lin(g, p + "attn_qkv.weight"); L.zgate = upload_lin(g, p + "attn_gate.weight");
            L.beta = upload_lin(g, p + "ssm_beta.weight"); L.alpha = upload_lin(g, p + "ssm_alpha.weight"); L.ssm_out = upload_lin(g, p + "ssm_out.weight");
            L.conv_w = upload_f32(g, p + "ssm_conv1d.weight", (size_t)D::conv_k * D::gdn_qkv);
            L.dt_bias = upload_f32(g, p + "ssm_dt.bias", D::gdn_v_heads); L.a_neg = upload_f32(g, p + "ssm_a", D::gdn_v_heads);
            L.ssm_norm = upload_f32(g, p + "ssm_norm.weight", D::gdn_dim);
        }
    }
    { const std::string p = "blk." + std::to_string(D::n_layer) + ".";
      mtp_.eh_proj = upload_lin(g, p + "nextn.eh_proj.weight");
      mtp_.enorm = upload_f32(g, p + "nextn.enorm.weight", D::n_embd); mtp_.hnorm = upload_f32(g, p + "nextn.hnorm.weight", D::n_embd);
      mtp_.attn_norm = upload_f32(g, p + "attn_norm.weight", D::n_embd); mtp_.post_norm = upload_f32(g, p + "post_attention_norm.weight", D::n_embd);
      mtp_.head_norm = g.find(p + "nextn.shared_head_norm.weight") ? upload_f32(g, p + "nextn.shared_head_norm.weight", D::n_embd) : nullptr;
      mtp_.attn = upload_attn(g, p);
      mtp_.ffn_gate = upload_lin(g, p + "ffn_gate.weight"); mtp_.ffn_up = upload_lin(g, p + "ffn_up.weight"); mtp_.ffn_down = upload_lin(g, p + "ffn_down.weight"); }
    { const GTensor& t = g.get("token_embd.weight"); if (t.type != GType::Q4_K) throw std::runtime_error("token_embd must be Q4_K");
      std::vector<uint8_t> stage(t.nbytes); g.read_tensor(t, stage.data());
      tok_embd_ = (uint8_t*)arena_alloc(t.nbytes); CK(hipMemcpy(tok_embd_, stage.data(), t.nbytes, hipMemcpyHostToDevice)); weight_bytes_ += t.nbytes; }
    output_norm_ = upload_f32(g, "output_norm.weight", D::n_embd);
    if (!mtp_.head_norm) mtp_.head_norm = output_norm_;
    output_ = upload_lin(g, "output.weight");
    // state + work (per slot, double-buffered)
    st_stride_ = (size_t)D::n_gdn * D::gdn_v_heads * D::gdn_dim * D::gdn_dim; cs_stride_ = (size_t)D::n_gdn * 3 * D::gdn_qkv;
    CK(hipMalloc(&gdn_state_, st_stride_ * 2 * n_slots_ * 4)); CK(hipMalloc(&conv_state_, cs_stride_ * 2 * n_slots_ * 4));
    cur_.assign(n_slots_, 0); lr0_.assign(n_slots_, 0); lT_.assign(n_slots_, 0);
    const size_t kv = (size_t)D::n_attn * max_ctx_ * D::n_head_kv * D::head_dim;
    const size_t kvt = (size_t)D::n_attn * (max_ctx_ + 128) * D::n_head_kv * D::head_dim;   // V^T rows padded
    kv_slot_ = kv; kvt_slot_ = kvt;
    kv8_ = getenv("HIPSTER_KV8") != nullptr;
    if (kv8_ && n_slots_ > 1) throw std::runtime_error("HIPSTER_KV8 is single-slot");
    if (!kv8_) {
        CK(hipMalloc(&kc_, kv * 2 * n_slots_)); CK(hipMalloc(&vc_, kvt * 2 * n_slots_)); CK(hipMemset(vc_, 0, kvt * 2 * n_slots_));
        CK(hipMalloc(&mkc_, kv / D::n_attn * 2)); CK(hipMalloc(&mvc_, kvt / D::n_attn * 2)); CK(hipMemset(mvc_, 0, kvt / D::n_attn * 2));
    } else {   // int8: K + 8 scales per row, V^T + 1 scale per position (half the bytes of f16)
        auto alloc8 = [&](KV8& c, int layers) {
            const size_t nk = (size_t)layers * max_ctx_ * D::n_head_kv, nv = (size_t)layers * (max_ctx_ + 128) * D::n_head_kv;
            CK(hipMalloc(&c.k, nk * D::head_dim)); CK(hipMalloc(&c.ks, nk * 8 * 2)); CK(hipMalloc(&c.v, nv * D::head_dim)); CK(hipMemset(c.v, 0, nv * D::head_dim)); CK(hipMalloc(&c.vs, nk * 2)); CK(hipMemset(c.vs, 0, nk * 2));
        };
        alloc8(kv8c_, D::n_attn); alloc8(mkv8c_, 1);
    }
    CK(hipMalloc(&sv_qkv_, (size_t)D::n_gdn * MAXR * D::gdn_qkv * 4));
    CK(hipMalloc(&sv_beta_, (size_t)D::n_gdn * MAXR * 64 * 4)); CK(hipMalloc(&sv_alpha_, (size_t)D::n_gdn * MAXR * 64 * 4));
    const int T = D::max_prefill;
    CK(hipMalloc(&pf_qkv_, (size_t)T * D::gdn_qkv * 4)); CK(hipMalloc(&pf_qn_, (size_t)T * (4096 + 96) * 4));
    CK(hipMalloc(&pf_raw_, (size_t)T * D::gdn_z * 4)); CK(hipMalloc(&pf_ao_, (size_t)T * D::gdn_z * 4));
    { const size_t nsp = (max_ctx_ + 127) / 128 + 1; CK(hipMalloc(&pO_, (size_t)4 * nsp * 48 * 256 * 4)); CK(hipMalloc(&pm_, (size_t)4 * nsp * 48 * 4)); CK(hipMalloc(&pl_, (size_t)4 * nsp * 48 * 4)); }
    CK(hipMalloc(&xb_, (size_t)T * 2 * D::n_embd * 2 > (size_t)T * D::n_ff * 2 ? (size_t)T * 2 * D::n_embd * 2 : (size_t)T * D::n_ff * 2));
    CK(hipMalloc(&wscratch_, (size_t)2 * D::n_ff * D::n_embd * 2)); CK(hipMalloc(&gout_, (size_t)T * 2 * D::n_ff * 2)); CK(hipMalloc(&gout2_, (size_t)T * D::n_embd * 2));   // merged gate+up is the largest segment set
    for (int i = 0; i < 2; ++i) { CK(hipMalloc(&wscratch2_[i], (size_t)2 * D::n_ff * D::n_embd * 2)); CK(hipEventCreateWithFlags(&scratch_free_[i], hipEventDisableTiming)); CK(hipEventRecord(scratch_free_[i], s_)); }
    { int lo, hi; CK(hipDeviceGetStreamPriorityRange(&lo, &hi)); CK(hipStreamCreateWithPriority(&s2_, hipStreamNonBlocking, hi)); }   // high priority: interleave with the GEMM
    CK(hipEventCreateWithFlags(&pf_ready_, hipEventDisableTiming));
    CK(hipMalloc(&x_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&xn_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&y_, (size_t)T * D::n_embd * 4));
    CK(hipMalloc(&h_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&mh_, (size_t)T * D::n_embd * 4)); CK(hipMalloc(&cat_, (size_t)T * 2 * D::n_embd * 4));
    CK(hipMalloc(&big0_, (size_t)T * D::n_ff * 4)); CK(hipMalloc(&big1_, (size_t)T * D::n_ff * 4)); CK(hipMalloc(&big2_, (size_t)T * D::n_ff * 4));
    CK(hipMalloc(&logits_, (size_t)MAXR * D::n_vocab * 4)); CK(hipMalloc(&mlogits_, (size_t)D::max_T * D::n_vocab * 4));
    const size_t xk = (size_t)T * D::n_ff;   // sized for prefill (in-kernel quantisers write T rows)
    CK(hipMalloc(&xq_.q, xk)); CK(hipMalloc(&xq_.d, xk / 32 * 4)); CK(hipMalloc(&xq_.s, xk / 32 * 4));
    CK(hipMemset(xq_.q, 0, xk)); CK(hipMemset(xq_.d, 0, xk / 32 * 4)); CK(hipMemset(xq_.s, 0, xk / 32 * 4));
    CK(hipMalloc(&d_tok_, T * 4)); CK(hipMalloc(&d_ids_, MAXR * 16 * 4)); CK(hipMalloc(&d_vals_, MAXR * 16 * 4));
    attn_mq_ = !(getenv("HIPSTER_ATTN") && std::string(getenv("HIPSTER_ATTN")) == "old");
    { apart_.spl = std::max(512, (max_ctx_ / 32 + 511) / 512 * 512); apart_.nsplit_max = (max_ctx_ + apart_.spl - 1) / apart_.spl;
      const size_t np = (size_t)4 * n_slots_ * apart_.nsplit_max * 64;
      CK(hipMalloc(&apart_.O, np * 256 * 4)); CK(hipMalloc(&apart_.m, np * 4)); CK(hipMalloc(&apart_.l, np * 4)); }
    reset();
}

Qwen35::~Qwen35() {}

void Qwen35::clear_kv(bool k, bool v) {
    if (k) CK(hipMemset(kc_, 0, kv_slot_ * 2 * n_slots_));
    if (v) CK(hipMemset(vc_, 0, kvt_slot_ * 2 * n_slots_));
}
void Qwen35::reset_slot(int s) {
    CK(hipMemset(gdn_state_ + (size_t)s * 2 * st_stride_, 0, st_stride_ * 2 * 4)); CK(hipMemset(conv_state_ + (size_t)s * 2 * cs_stride_, 0, cs_stride_ * 2 * 4));
    cur_[s] = 0; lT_[s] = 0;
}
void Qwen35::reset() {
    CK(hipMemset(gdn_state_, 0, st_stride_ * 2 * n_slots_ * 4)); CK(hipMemset(conv_state_, 0, cs_stride_ * 2 * n_slots_ * 4));
    for (int s = 0; s < n_slots_; ++s) { cur_[s] = 0; lT_[s] = 0; }
    last_T_ = 0;
}

void Qwen35::gemv(const Lin& l, float* y, int ncol) {
    if (!gm_) { lin(l, y, ncol); return; }   // same T-invariant kernel selection as every other GEMV (LM head incl.)
    // measured policy (docs/decode-gemv.md): 1 column -> direct tpr=32; 2..4 -> LDS-x tpr=4; >4 -> WMMA (Q4_K only so far)
    static const int pol = getenv("HIPSTER_GEMV_POLICY") ? atoi(getenv("HIPSTER_GEMV_POLICY")) : 0;   // debug: 1 = direct for all, 2 = LDS for all
    if (ncol == 1) gemv_q8(l.fmt, l.w, xq_, y, l.N, l.K, 1, 32, 1, s_);
    else if (pol == 1) gemv_q8(l.fmt, l.w, xq_, y, l.N, l.K, ncol, 4, 1, s_);
    else if (pol == 2 || ncol <= 4 || l.fmt != WFmt::Q4_K) gemv_q8(l.fmt, l.w, xq_, y, l.N, l.K, ncol, 4, 3, s_);
    else gemv_q8(l.fmt, l.w, xq_, y, l.N, l.K, ncol, 1, 4, s_);
}

void Qwen35::tmark(int cat) {   // accumulate GPU time since the previous mark into category cat (timing mode only)
    if (!timing_) return;
    CK(hipEventRecord(tev_[1], s_)); CK(hipEventSynchronize(tev_[1]));
    float ms; CK(hipEventElapsedTime(&ms, tev_[0], tev_[1])); tacc_[cat] += ms;
    CK(hipEventRecord(tev_[0], s_));
}
// Weight dequantisation runs on a second stream one GEMM ahead into the other scratch buffer:
// prefetch(segs) queues it (after the GEMM that last used that buffer), lin_multi waits for it.
void Qwen35::prefetch(const GemvSeg* segs, int n) {
    if (!prefetch_on_ || pf_pending_) return;   // one outstanding prefetch; measured a net loss (docs/prefill-27b.md), off by default
    const int b = pf_buf_ ^ 1;
    CK(hipStreamWaitEvent(s2_, scratch_free_[b], 0));   // buffer b: its previous GEMM must be done
    int Nt = 0; const int K = segs[0].K;
    for (int i = 0; i < n; ++i) { dequant_bf16(segs[i].fmt, segs[i].w, wscratch2_[b] + (size_t)Nt * K, segs[i].N, K, s2_); Nt += segs[i].N; }
    CK(hipEventRecord(pf_ready_, s2_));
    pf_pending_ = true; pf_w_ = segs[0].w; pf_n_ = n;
}
void Qwen35::lin(const Lin& l, float* y, int M) {
    GemvSeg s = {l.fmt, l.w, y, l.N, l.K}; lin_multi(&s, 1, M);
}

// Several weights sharing the input: GEMV path -> gemv_multi; GEMM path -> dequant all segments into
// one [sum N][K] bf16 matrix, one GEMM, then split the bf16 output rows into the f32 targets.
void Qwen35::lin_multi(const GemvSeg* segs, int n, int M) {
    if (!gm_) {
        // one kernel shape for every T: each column is reduced in the same order as the T=1 pass (T-invariant numerics,
        // the verify pass is bit-identical to T single passes). HIPSTER_GEMV_FAST=1 restores the old ncol-tuned kernel.
        // HIPSTER_GEMV=wmma (default): int8-WMMA kernel for every T (T-invariant, ~the T=1 cost at T=8); multi: the
        // 32-lane multi-column kernel (T-invariant, slow at T=8); fast: the old ncol-tuned kernels (not T-invariant).
        static const std::string mode = getenv("HIPSTER_GEMV") ? getenv("HIPSTER_GEMV") : "wmma";
        static const bool fast = mode == "fast";
        if (mode == "wmma" && gemv_wmma_ok(segs, n)) { gemv_wmma(segs, n, xq_, M, 0, s_); tmark(T_GEMV); return; }
        if (M == 1 || !fast) { gemv_multi(segs, n, xq_, M, s_); tmark(T_GEMV); return; }
        for (int i = 0; i < n; ++i) gemv_q8(segs[i].fmt, segs[i].w, xq_, segs[i].y, segs[i].N, segs[i].K, M, 4, 3, s_);
        tmark(T_GEMV); return;
    }
    int Nt = 0; const int K = segs[0].K;
    uint16_t* ws;
    if (pf_pending_ && pf_w_ == segs[0].w && pf_n_ == n) {   // prefetched on s2_
        CK(hipStreamWaitEvent(s_, pf_ready_, 0)); pf_pending_ = false; pf_buf_ ^= 1; ws = wscratch2_[pf_buf_];
        for (int i = 0; i < n; ++i) Nt += segs[i].N;
    } else {
        ws = wscratch_;
        for (int i = 0; i < n; ++i) { dequant_bf16(segs[i].fmt, segs[i].w, ws + (size_t)Nt * K, segs[i].N, K, s_); Nt += segs[i].N; }
    }
    tmark(T_DEQ);
    const bool splitk = K > 8192 && n == 1;   // hipBLASLt: 37 TFLOPS at K=17408, 52 at K=8704 (docs/prefill-27b.md)
    if (!splitk) blas_.gemm(xb_, ws, gout_, M, Nt, K, s_);
    else {   // D = A[:, :K/2] W[:, :K/2]^T + A[:, K/2:] W[:, K/2:]^T ; operands are [rows][K] so a column half is a strided view
        blas_.gemm_ld(xb_, K, ws, K, gout_, M, Nt, K / 2, s_);
        blas_.gemm_ld(xb_ + K / 2, K, ws + K / 2, K, gout2_, M, Nt, K / 2, s_);
    }
    if (ws != wscratch_) CK(hipEventRecord(scratch_free_[pf_buf_], s_));   // this buffer may be refilled after the GEMM
    tmark(T_GEMM);
    if (next_pfn_ > 0) { prefetch(next_pf_, next_pfn_); next_pfn_ = 0; }
    int off = 0;
    for (int i = 0; i < n; ++i) { if (segs[i].y) bf16_to_f32_seg2(gout_, splitk ? gout2_ : nullptr, Nt, off, segs[i].N, segs[i].y, M, s_); off += segs[i].N; }
    tmark(T_SPLIT);
}

KV8 Qwen35::layer_kv8(int ai) const {
    const size_t nk = (size_t)ai * max_ctx_ * D::n_head_kv, nv = (size_t)ai * (max_ctx_ + 128) * D::n_head_kv;
    return KV8{kv8c_.k + nk * D::head_dim, kv8c_.ks + nk * 8, kv8c_.v + nv * D::head_dim, kv8c_.vs + nk};
}

// xq_ (GEMV) / xb_ (GEMM) hold the normalised input [T]; writes the projected output into y [T][n_embd]
void Qwen35::attn_block(const AttnBlock& a, int T, int pos0, uint16_t* kc, uint16_t* vc, float* y, const KV8* c8, const RowBatch* rb) {
    float* qf = big0_; float* k = big1_; float* v = big1_ + (size_t)T * 1024; float* ao = big2_;
    GemvSeg segs[3] = {{a.wq.fmt, a.wq.w, qf, a.wq.N, a.wq.K}, {a.wk.fmt, a.wk.w, k, a.wk.N, a.wk.K}, {a.wv.fmt, a.wv.w, v, a.wv.N, a.wv.K}};
    if (gm_) { GemvSeg keep[2]; int kn = next_pfn_; for (int i = 0; i < kn; ++i) keep[i] = next_pf_[i];   // gate+up set by caller; o-proj must come first
                        next_pf_[0] = {a.wo.fmt, a.wo.w, nullptr, a.wo.N, a.wo.K}; next_pfn_ = 1; next2_pfn_ = kn; for (int i = 0; i < kn; ++i) next2_pf_[i] = keep[i]; }
    lin_multi(segs, 3, T);
    if (gm_ && next2_pfn_ > 0) { for (int i = 0; i < next2_pfn_; ++i) next_pf_[i] = next2_pf_[i]; next_pfn_ = next2_pfn_; next2_pfn_ = 0; }
    if (kv8_) {   // int8 KV path (kc/vc are unused; c8 selects the layer's caches)
        if (!gm_) attn_decode_i8(qf, k, v, T, a.q_norm, a.k_norm, D::rope_base, pos0, *c8, max_ctx_, ao, xq_, D::rms_eps, s_);
        else { attn_stage1_i8(qf, k, v, T, a.q_norm, a.k_norm, D::rope_base, pos0, *c8, max_ctx_, D::rms_eps, s_); attn_prefill_i8(qf, *c8, T, pos0, max_ctx_, xb_, s_); }
    } else if (!gm_ && getenv("HIPSTER_GQA_ATTN")) {   // experimental: slower at 32K and not yet exact (docs/decode-27b.md)
        attn_decode_gqa(qf, k, v, T, a.q_norm, a.k_norm, D::rope_base, pos0, kc, vc, max_ctx_, ao, xq_, pO_, pm_, pl_, D::rms_eps, s_);
    } else if (!gm_ && attn_mq_) {   // multi-query: K/V read once per kv head for all heads x rows; rows grouped by slot
        const RowBatch* rbp = rb;
        if (!rbp) { arb_.n = T; for (int t = 0; t < T; ++t) { arb_.pos[t] = pos0 + t; arb_.kv[t] = 0; } rbp = &arb_; }
        agroups_.n = 0;   // rows grouped by slot, at most 8 rows (one wave each) per group
        for (int t = 0; t < T; ++t) { if (t == 0 || rbp->kv[t] != rbp->kv[t - 1] || agroups_.T[agroups_.n - 1] == 8) { agroups_.r0[agroups_.n] = t; agroups_.T[agroups_.n] = 0; ++agroups_.n; } ++agroups_.T[agroups_.n - 1]; }
        attn_decode_mq(qf, k, v, T, a.q_norm, a.k_norm, D::rope_base, *rbp, agroups_, kc, vc, kv_slot_, kvt_slot_, max_ctx_, ao, xq_, apart_, D::rms_eps, s_);
    } else if (rb) {   // batched rows (several slots): per-row positions and KV slots
        attn_decode_b(qf, k, v, T, a.q_norm, a.k_norm, D::rope_base, *rb, kc, vc, kv_slot_, kvt_slot_, max_ctx_, ao, xq_, D::rms_eps, s_);
    } else if (!gm_ || getenv("HIPSTER_SCALAR_ATTN")) {
        attn_decode(qf, k, v, T, a.q_norm, a.k_norm, D::rope_base, pos0, kc, vc, max_ctx_, ao, xq_, D::rms_eps, s_);
        if (gm_) f32_to_bf16(ao, xb_, (size_t)T * D::n_head * D::head_dim, s_);
    } else {
        attn_stage1(qf, k, v, T, a.q_norm, a.k_norm, D::rope_base, pos0, kc, vc, max_ctx_, D::rms_eps, s_);
        attn_prefill(qf, kc, vc, T, pos0, max_ctx_, xb_, s_);
    }
    tmark(T_ATTN);
    lin(a.wo, y, T);
}

void Qwen35::ffn_block(const Lin& g, const Lin& u, const Lin& d, int T, float* y) {
    GemvSeg fs[2] = {{g.fmt, g.w, big0_, g.N, g.K}, {u.fmt, u.w, big1_, u.N, u.K}};
    if (gm_) { fs[0].y = nullptr; fs[1].y = nullptr; }   // GEMM path: silu reads the bf16 GEMM output directly, no split
    lin_multi(fs, 2, T);   // consumes next_pf_ (= down)
    if (next2_pfn_ > 0) { for (int i = 0; i < next2_pfn_; ++i) next_pf_[i] = next2_pf_[i]; next_pfn_ = next2_pfn_; next2_pfn_ = 0; }
    if (!gm_) silu_mul_quant(big0_, big1_, xq_, T * D::n_ff, s_);
    else silu_mul_bf16_seg(gout_, 2 * D::n_ff, 0, D::n_ff, xb_, T, s_);
    tmark(T_OTHER);
    lin(d, y, T);   // consumes next_pf_ (= next layer's first group)
}

float Qwen35::forward(const SlotReq* reqs, int S) {
    if (S < 1 || S > n_slots_) throw std::runtime_error("bad S");
    int T = 0; for (int i = 0; i < S; ++i) T += reqs[i].T;
    const bool gm = S == 1 && T > MAXR;   // GEMM (prefill) path: one slot; up to MAXR rows go through the GEMV path (short prompts)
    if (T < 1 || T > D::max_prefill || (!gm && T > MAXR)) throw std::runtime_error("bad T");
    gm_ = gm;
    const int slot0 = reqs[0].slot, pos0 = reqs[0].pos;   // used by the single-sequence paths
    SeqBatch sb; RowBatch rb; std::vector<int> toks(T);
    { int r0 = 0; for (int i = 0; i < S; ++i) { const SlotReq& r = reqs[i];
        if (r.slot < 0 || r.slot >= n_slots_ || r.T < 1 || r.pos + r.T > max_ctx_) throw std::runtime_error("bad request");
        sb.s[i] = {r0, r.T, r.slot * 2 + cur_[r.slot], r.slot * 2 + (cur_[r.slot] ^ 1)};
        for (int t = 0; t < r.T; ++t) { toks[r0 + t] = r.tokens[t]; if (!gm) { rb.pos[r0 + t] = r.pos + t; rb.kv[r0 + t] = r.slot; } }
        lr0_[r.slot] = r0; lT_[r.slot] = r.T; r0 += r.T; }
      sb.n = S; rb.n = gm ? 0 : T; }
    const bool batched = !gm && (S > 1 || slot0 != 0);
    const RowBatch* rbp = batched ? &rb : nullptr;
    const float* cs_in = conv_state_ + (size_t)(slot0 * 2 + cur_[slot0]) * cs_stride_; float* cs_out = conv_state_ + (size_t)(slot0 * 2 + (cur_[slot0] ^ 1)) * cs_stride_;
    const float* gs_in = gdn_state_ + (size_t)(slot0 * 2 + cur_[slot0]) * st_stride_; float* gs_out = gdn_state_ + (size_t)(slot0 * 2 + (cur_[slot0] ^ 1)) * st_stride_;
    CK(hipEventRecord(ev0_, s_));
    CK(hipMemcpyAsync(d_tok_, toks.data(), T * 4, hipMemcpyHostToDevice, s_));
    embed_q4_K(tok_embd_, d_tok_, T, x_, D::n_embd, s_);
    const float* pending = nullptr;
    auto norm_in = [&](const float* y, const float* w) {   // residual add + norm -> GEMV/GEMM input
        if (gm) add_rmsnorm_bf16(x_, y, w, xb_, T, D::n_embd, D::rms_eps, s_);
        else add_rmsnorm_quant(x_, y, w, xn_, xq_, T, D::n_embd, D::rms_eps, s_);
        tmark(T_OTHER);
    };
    if (timing_) { CK(hipEventRecord(tev_[0], s_)); }
    // segment lists per layer, in execution order, for one-ahead weight prefetch
    auto seg_first = [&](int il, GemvSeg* s) -> int {   // qkv-group or q/k/v
        auto& L = layers_[il];
        if (!D::is_attn(il)) { s[0] = {L.qkv.fmt, L.qkv.w, nullptr, L.qkv.N, L.qkv.K}; s[1] = {L.zgate.fmt, L.zgate.w, nullptr, L.zgate.N, L.zgate.K}; s[2] = {L.beta.fmt, L.beta.w, nullptr, L.beta.N, L.beta.K}; s[3] = {L.alpha.fmt, L.alpha.w, nullptr, L.alpha.N, L.alpha.K}; return 4; }
        s[0] = {L.attn.wq.fmt, L.attn.wq.w, nullptr, L.attn.wq.N, L.attn.wq.K}; s[1] = {L.attn.wk.fmt, L.attn.wk.w, nullptr, L.attn.wk.N, L.attn.wk.K}; s[2] = {L.attn.wv.fmt, L.attn.wv.w, nullptr, L.attn.wv.N, L.attn.wv.K}; return 3;
    };
    for (int il = 0; il < D::n_layer; ++il) {
        auto& L = layers_[il];
        const Lin& proj_out = D::is_attn(il) ? L.attn.wo : L.ssm_out;
        GemvSeg pfs[4]; int pfn;
        if (gm && il == 0) { pfn = seg_first(0, pfs); prefetch(pfs, pfn); }
        norm_in(pending, L.attn_norm);
        if (dfl_) for (int k = 0; k < 5; ++k) if (dfl_->target_layers[k] == il)   // DFlash2: residual entering this layer -> feature slot k
            CK(hipMemcpy2DAsync(dfl_feat_ + (size_t)k * D::n_embd, (size_t)5 * D::n_embd * 4, x_, (size_t)D::n_embd * 4, (size_t)D::n_embd * 4, T, hipMemcpyDeviceToDevice, s_));
        if (gm) { next_pf_[0] = {proj_out.fmt, proj_out.w, nullptr, proj_out.N, proj_out.K}; next_pfn_ = 1; }
        if (!D::is_attn(il)) {
            const int gi = gdn_index(il);
            float* qkv_raw = gm ? pf_qkv_ : sv_qkv_ + (size_t)gi * MAXR * D::gdn_qkv;
            float* b = gm ? big2_ : sv_beta_ + (size_t)gi * MAXR * 64; float* a = gm ? big2_ + (size_t)T * 64 : sv_alpha_ + (size_t)gi * MAXR * 64;
            float* z = big1_; float* qkv = big0_; float* ao = gm ? pf_ao_ : big2_;
            GemvSeg segs[4] = {{L.qkv.fmt, L.qkv.w, qkv_raw, L.qkv.N, L.qkv.K}, {L.zgate.fmt, L.zgate.w, z, L.zgate.N, L.zgate.K},
                               {L.beta.fmt, L.beta.w, b, L.beta.N, L.beta.K}, {L.alpha.fmt, L.alpha.w, a, L.alpha.N, L.alpha.K}};
            lin_multi(segs, 4, T);
            const size_t cso = (size_t)gi * 3 * D::gdn_qkv, gso = (size_t)gi * D::gdn_v_heads * D::gdn_dim * D::gdn_dim;
            if (batched) {
                gdn_conv_b(qkv_raw, sb, cs_stride_, conv_state_ + cso, conv_state_ + cso, L.conv_w, qkv, D::gdn_qkv, s_);
                gdn_step_b(qkv, z, b, a, sb, st_stride_, L.dt_bias, L.a_neg, gdn_state_ + gso, gdn_state_ + gso, L.ssm_norm, ao, xq_, D::rms_eps, s_);
            } else {
                gdn_conv(qkv_raw, T, cs_in + cso, cs_out + cso, L.conv_w, qkv, D::gdn_qkv, s_);
                if (!gm) gdn_step(qkv, z, b, a, T, L.dt_bias, L.a_neg, gs_in + gso, gs_out + gso, L.ssm_norm, ao, xq_, D::rms_eps, s_);
                else {   // prefill: prep -> tiled scan -> out (writes bf16 into xb_ directly)
                    float* qn = pf_qn_; float* kn = pf_qn_ + (size_t)T * 2048; float* bt = pf_qn_ + (size_t)T * 4096; float* dc = bt + (size_t)T * 48; float* raw = pf_raw_;
                    gdn_prep(qkv, b, a, L.dt_bias, L.a_neg, T, qn, kn, bt, dc, D::rms_eps, s_);
                    gdn_scan(qn, kn, qkv, bt, dc, T, gs_in + gso, gs_out + gso, raw, s_);
                    gdn_out(raw, z, L.ssm_norm, T, ao, xb_, D::rms_eps, s_);
                }
            }
            tmark(T_GDN);
            if (gm) { next_pf_[0] = {L.ffn_gate.fmt, L.ffn_gate.w, nullptr, L.ffn_gate.N, L.ffn_gate.K}; next_pf_[1] = {L.ffn_up.fmt, L.ffn_up.w, nullptr, L.ffn_up.N, L.ffn_up.K}; next_pfn_ = 2; }
            lin(L.ssm_out, y_, T);
        } else {
            const size_t off = (size_t)attn_index(il) * max_ctx_ * D::n_head_kv * D::head_dim, offv = (size_t)attn_index(il) * (max_ctx_ + 128) * D::n_head_kv * D::head_dim;
            if (gm) { next_pf_[0] = {L.ffn_gate.fmt, L.ffn_gate.w, nullptr, L.ffn_gate.N, L.ffn_gate.K}; next_pf_[1] = {L.ffn_up.fmt, L.ffn_up.w, nullptr, L.ffn_up.N, L.ffn_up.K}; next_pfn_ = 2; }
            KV8 c8 = kv8_ ? layer_kv8(attn_index(il)) : KV8{};
            attn_block(L.attn, T, pos0, kc_ + (batched ? 0 : (size_t)slot0 * kv_slot_) + off, vc_ + (batched ? 0 : (size_t)slot0 * kvt_slot_) + offv, y_, &c8, rbp);
        }
        norm_in(y_, L.post_norm);
        if (gm) { next_pf_[0] = {L.ffn_down.fmt, L.ffn_down.w, nullptr, L.ffn_down.N, L.ffn_down.K}; next_pfn_ = 1;
                  // and after the down-proj: the next layer's first group
                  if (il + 1 < D::n_layer) next2_pfn_ = seg_first(il + 1, next2_pf_); else next2_pfn_ = 0; }
        ffn_block(L.ffn_gate, L.ffn_up, L.ffn_down, T, y_);
        pending = y_;
    }
    add_rmsnorm_quant(x_, pending, output_norm_, h_, xq_, T, D::n_embd, D::rms_eps, s_);   // h for all rows (+ q8 of all rows)
    if (gm) {   // logits for the last row only: quantised row T-1 -> row 0 of xq_
        CK(hipMemcpyAsync(xq_.q, xq_.q + (size_t)(T - 1) * D::n_embd, D::n_embd, hipMemcpyDeviceToDevice, s_));
        CK(hipMemcpyAsync(xq_.d, xq_.d + (size_t)(T - 1) * (D::n_embd / 32), D::n_embd / 32 * 4, hipMemcpyDeviceToDevice, s_));
        CK(hipMemcpyAsync(xq_.s, xq_.s + (size_t)(T - 1) * (D::n_embd / 32), D::n_embd / 32 * 4, hipMemcpyDeviceToDevice, s_));
        gemv(output_, logits_, 1);
    } else gemv(output_, logits_, T);
    last_T_ = T; last_gm_ = gm; gm_ = false;
    CK(hipEventRecord(ev1_, s_)); CK(hipEventSynchronize(ev1_));
    CK(hipEventElapsedTime(&last_ms_, ev0_, ev1_));
    return last_ms_;
}

void Qwen35::accept(int slot, int m) {
    if (slot < 0 || slot >= n_slots_ || lT_[slot] == 0) throw std::runtime_error("accept: slot had no rows in the last pass");
    const int T = lT_[slot], r0 = lr0_[slot];
    if (m < 1 || m > T) throw std::runtime_error("bad accept");
    if (last_gm_ && m != T) throw std::runtime_error("prefill must accept all tokens");
    const int cur = slot * 2 + cur_[slot], nxt = slot * 2 + (cur_[slot] ^ 1);
    if (m < T) {   // replay: recompute the state after m tokens from the saved inputs (cur is intact)
        float* qkv = big0_;
        for (int il = 0; il < D::n_layer; ++il) {
            if (D::is_attn(il)) continue;
            const auto& L = layers_[il]; const int gi = gdn_index(il);
            const size_t cso = (size_t)gi * 3 * D::gdn_qkv, gso = (size_t)gi * D::gdn_v_heads * D::gdn_dim * D::gdn_dim;
            const float* qkv_raw = sv_qkv_ + ((size_t)gi * MAXR + r0) * D::gdn_qkv;
            gdn_conv(qkv_raw, m, conv_state_ + cur * cs_stride_ + cso, conv_state_ + nxt * cs_stride_ + cso, L.conv_w, qkv, D::gdn_qkv, s_);
            gdn_step(qkv, nullptr, sv_beta_ + (size_t)gi * MAXR * 64 + (size_t)r0 * D::gdn_v_heads, sv_alpha_ + (size_t)gi * MAXR * 64 + (size_t)r0 * D::gdn_v_heads, m, L.dt_bias, L.a_neg,   // rows are [T][48]
                     gdn_state_ + cur * st_stride_ + gso, gdn_state_ + nxt * st_stride_ + gso, L.ssm_norm, nullptr, xq_, D::rms_eps, s_);
        }
    }
    cur_[slot] ^= 1; lT_[slot] = 0;
    bool any = false; for (int s = 0; s < n_slots_; ++s) any |= lT_[s] != 0; if (!any) last_T_ = 0;
}

void Qwen35::mtp_forward(const int* tokens, const float* h, int T, int pos0) {
    if (T < 1 || T > D::max_prefill) throw std::runtime_error("bad T");
    const bool gm = T > D::max_T; gm_ = gm;
    CK(hipMemcpyAsync(d_tok_, tokens, T * 4, hipMemcpyHostToDevice, s_));
    embed_q4_K(tok_embd_, d_tok_, T, xn_, D::n_embd, s_);                          // e [T][5120]
    if (gm) norm2_concat_bf16(xn_, mtp_.enorm, D::n_embd, h, mtp_.hnorm, D::n_embd, xb_, T, D::rms_eps, s_);
    else norm2_concat_quant(xn_, mtp_.enorm, D::n_embd, h, mtp_.hnorm, D::n_embd, cat_, xq_, T, D::rms_eps, s_);   // [e_norm | h_norm]
    lin(mtp_.eh_proj, x_, T);                                                         // inpSA
    auto norm_in = [&](const float* y, const float* w, float* xn_out) {
        if (gm) add_rmsnorm_bf16(x_, y, w, xb_, T, D::n_embd, D::rms_eps, s_);
        else add_rmsnorm_quant(x_, y, w, xn_out, xq_, T, D::n_embd, D::rms_eps, s_);
    };
    norm_in(nullptr, mtp_.attn_norm, xn_);
    attn_block(mtp_.attn, T, pos0, mkc_, mvc_, y_, &mkv8c_);
    norm_in(y_, mtp_.post_norm, xn_);
    ffn_block(mtp_.ffn_gate, mtp_.ffn_up, mtp_.ffn_down, T, y_);
    add_rmsnorm_quant(x_, y_, mtp_.head_norm, mh_, xq_, T, D::n_embd, D::rms_eps, s_);
    if (!gm) gemv(output_, mlogits_, T);   // prefill catch-up needs no draft logits
    gm_ = false;
}

void Qwen35::topk(const float* logits, int T, int k, int* ids, float* vals) {
    argmax_topk(logits, T, D::n_vocab, k, d_ids_, d_vals_, s_);
    CK(hipStreamSynchronize(s_));
    CK(hipMemcpy(ids, d_ids_, T * k * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(vals, d_vals_, T * k * 4, hipMemcpyDeviceToHost));
}

}  // namespace hip
