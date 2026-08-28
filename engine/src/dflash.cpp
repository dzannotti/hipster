// DFlash2 draft for the 27B: 5-layer block draft (incoai/Qwen3.8-27B-DFlash2) driven by residuals captured at target
// layers 6/20/34/48/62. Encoder: g = rmsnorm(fc . [5 x 5120 features]) -> per draft layer K (norm + rope) / V into the draft
// KV. Draft block: [id_last, MASK x nd] at positions n.., non-causal windowed attention over the block + the encoded cache,
// 2-tap dynamic grouped conv around attention and FFN, target lm head, top-16 selector lattice, greedy walk on the host.
#include "qwen35.h"
#include <stdexcept>
#include <algorithm>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) throw std::runtime_error(std::string("HIP: ") + hipGetErrorString(e) + " @" + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace hip {
using namespace hip::dfl;

void Qwen35::load_dflash(const std::string& path) {
    dfl_gguf_ = new GGUF(path); const GGUF& g = *dfl_gguf_;
    if (g.arch() != "dflash") throw std::runtime_error("not a dflash gguf");
    dfl_ = new DflashDraft;
    auto kv_i = [&](const char* k, int def) { const GValue* v = g.kv(k); return v ? (int)(v->kind == GValue::U32 ? v->u : v->i) : def; };
    dfl_->mask_id = kv_i("tokenizer.ggml.mask_token_id", 248070);
    { const GValue* v = g.kv("dflash.target_layers"); if (!v || v->arr.size() != 5) throw std::runtime_error("dflash.target_layers");
      for (int i = 0; i < 5; ++i) dfl_->target_layers[i] = (int)(v->arr[i].kind == GValue::U32 ? v->arr[i].u : v->arr[i].i); }
    const int nl = kv_i("dflash.block_count", 5);
    dfl_->fc = upload_lin(g, "fc.weight"); dfl_->selector_hidden = upload_lin(g, "selector_hidden.weight");
    dfl_->enc_norm = upload_f32(g, "enc.output_norm.weight", D::n_embd); dfl_->out_norm = upload_f32(g, "output_norm.weight", D::n_embd);
    for (const char* nm : {"selector_predecessor.weight", "selector_successor.weight"}) {
        const GTensor& t = g.get(nm); if (t.type != GType::Q4_K || t.ne[0] != RANK) throw std::runtime_error(std::string("selector must be Q4_K [V][256]: ") + nm);
        std::vector<uint8_t> stage(t.nbytes); g.read_tensor(t, stage.data());
        void* d; CK(hipMalloc(&d, t.nbytes)); CK(hipMemcpy(d, stage.data(), t.nbytes, hipMemcpyHostToDevice)); weight_bytes_ += t.nbytes;
        (nm[9] == 'p' ? dfl_->sel_pred : dfl_->sel_succ) = d;
    }
    for (int il = 0; il < nl; ++il) {
        const std::string p = "blk." + std::to_string(il) + "."; DflashLayer L;
        L.attn_norm = upload_f32(g, p + "attn_norm.weight", D::n_embd); L.ffn_norm = upload_f32(g, p + "ffn_norm.weight", D::n_embd);
        L.q_norm = upload_f32(g, p + "attn_q_norm.weight", HD); L.k_norm = upload_f32(g, p + "attn_k_norm.weight", HD);
        L.attn_conv_base = upload_f32(g, p + "attn_conv_base", (size_t)E * 4); L.ffn_conv_base = upload_f32(g, p + "ffn_conv_base", (size_t)E * 4);
        L.wq = upload_lin(g, p + "attn_q.weight"); L.wk = upload_lin(g, p + "attn_k.weight"); L.wv = upload_lin(g, p + "attn_v.weight"); L.wo = upload_lin(g, p + "attn_output.weight");
        L.attn_conv_proj = upload_lin(g, p + "attn_conv_proj.weight"); L.ffn_conv_proj = upload_lin(g, p + "ffn_conv_proj.weight");
        L.ffn_gate = upload_lin(g, p + "ffn_gate.weight"); L.ffn_up = upload_lin(g, p + "ffn_up.weight"); L.ffn_down = upload_lin(g, p + "ffn_down.weight");
        dfl_->layers.push_back(L);
    }
    const int T = MAXR;
    // every pass captures all its rows into dfl_feat_ (prefill included: 419 MB)
    CK(hipMalloc(&dfl_feat_, (size_t)D::max_prefill * 5 * E * 4)); CK(hipMalloc(&dfl_g_, (size_t)T * E * 4)); CK(hipMalloc(&dfl_x_, (size_t)T * E * 4)); CK(hipMalloc(&dfl_y_, (size_t)T * E * 4));
    CK(hipMalloc(&dfl_blk_, (size_t)MAXR * sizeof(dfl::Blk)));
    CK(hipMalloc(&dfl_q_, (size_t)T * NQ * HD * 4)); CK(hipMalloc(&dfl_k_, (size_t)T * NKV * HD * 4)); CK(hipMalloc(&dfl_v_, (size_t)T * NKV * HD * 4)); CK(hipMalloc(&dfl_o_, (size_t)T * NQ * HD * 4));
    CK(hipMalloc(&dfl_delta_, (size_t)T * 4 * N_GROUPS * 4)); CK(hipMalloc(&dfl_logits_, (size_t)T * D::n_vocab * 4)); CK(hipMalloc(&dfl_gate_, (size_t)T * RANK * 4));
    CK(hipMalloc(&dfl_lattice_, (size_t)T * LATTICE * 4)); CK(hipMalloc(&dfl_pos_, T * 4)); CK(hipMalloc(&dfl_tok_, T * 4));
    dfl_layer_stride_ = (size_t)NKV * max_ctx_ * HD; dfl_slot_stride_ = dfl_layer_stride_ * nl;
    CK(hipMalloc(&dfl_kc_, dfl_slot_stride_ * n_slots_ * 2)); CK(hipMalloc(&dfl_vc_, dfl_slot_stride_ * n_slots_ * 2));
    report_ += "  dflash draft: " + std::to_string(nl) + " layers, targets " + std::to_string(dfl_->target_layers[0]) + ".." + std::to_string(dfl_->target_layers[4]) + ", mask " + std::to_string(dfl_->mask_id) + "\n";
}

void Qwen35::dflash_encode(int slot, int T, int pos0) {
    uint16_t* kcs = dfl_kc_ + (size_t)slot * dfl_slot_stride_; uint16_t* vcs = dfl_vc_ + (size_t)slot * dfl_slot_stride_;
    for (int c0 = 0; c0 < T; c0 += MAXR) {   // rows in chunks of MAXR through the GEMV path (a prefill's rows included)
        const int Tc = std::min(MAXR, T - c0);
        std::vector<int> pos(Tc); for (int i = 0; i < Tc; ++i) pos[i] = pos0 + c0 + i;
        CK(hipMemcpyAsync(dfl_pos_, pos.data(), Tc * 4, hipMemcpyHostToDevice, s_));
        static const bool dbg = getenv("DFL_DBG") != nullptr;
        auto chk = [&](const char* what) { if (!dbg) return; hipError_t e = hipStreamSynchronize(s_); fprintf(stderr, "  encode %s: %s (gm_=%d Tc=%d lr0=%d c0=%d xq=%p feat=%p g=%p)\n", what, hipGetErrorString(e), (int)gm_, Tc, lr0_[slot], c0, (void*)xq_.q, (void*)dfl_feat_, (void*)dfl_g_); };
        quantize_x_q8(dfl_feat_ + (size_t)(lr0_[slot] + c0) * 5 * E, xq_, Tc, 5 * E, s_); chk("quant");
        lin(dfl_->fc, dfl_g_, Tc); chk("fc");
        add_rmsnorm_quant(dfl_g_, nullptr, dfl_->enc_norm, xn_, xq_, Tc, E, D::rms_eps, s_); chk("norm");
        for (size_t l = 0; l < dfl_->layers.size(); ++l) {
            const DflashLayer& L = dfl_->layers[l];
            GemvSeg segs[2] = {{L.wk.fmt, L.wk.w, dfl_k_, L.wk.N, L.wk.K}, {L.wv.fmt, L.wv.w, dfl_v_, L.wv.N, L.wv.K}};
            lin_multi(segs, 2, Tc);
            qk_rope_kv(nullptr, dfl_k_, dfl_v_, nullptr, L.k_norm, D::rope_base, dfl_pos_, Tc, Tc, nullptr, 0, kcs + l * dfl_layer_stride_, vcs + l * dfl_layer_stride_, max_ctx_, D::rms_eps, s_);
        }
    }
}

int Qwen35::dflash_draft_b(const int* slots, const int* id_last, const int* n, int S, int nd, int* out) {
    const int B = 1 + nd, T = S * B;
    if (B > D::max_T || T > MAXR || S < 1) throw std::runtime_error("dflash_draft: bad block/rows");
    std::vector<int> tok(T, dfl_->mask_id), pos(T); std::vector<dfl::Blk> blk(S);
    for (int b = 0; b < S; ++b) { tok[b * B] = id_last[b]; for (int i = 0; i < B; ++i) pos[b * B + i] = n[b] + i; blk[b] = {slots[b], n[b] + B, id_last[b]}; }
    CK(hipMemcpyAsync(dfl_tok_, tok.data(), T * 4, hipMemcpyHostToDevice, s_)); CK(hipMemcpyAsync(dfl_pos_, pos.data(), T * 4, hipMemcpyHostToDevice, s_));
    CK(hipMemcpyAsync(dfl_blk_, blk.data(), S * sizeof(dfl::Blk), hipMemcpyHostToDevice, s_));
    embed_q4_K(tok_embd_, dfl_tok_, T, dfl_x_, E, s_);
    const XQ8 noq{nullptr, nullptr, nullptr}; const float* pend = nullptr;
    for (size_t l = 0; l < dfl_->layers.size(); ++l) {
        const DflashLayer& L = dfl_->layers[l];
        uint16_t* kcl = dfl_kc_ + l * dfl_layer_stride_; uint16_t* vcl = dfl_vc_ + l * dfl_layer_stride_;   // + slot * dfl_slot_stride_ inside the kernels
        add_rmsnorm_quant(dfl_x_, pend, L.attn_norm, xn_, xq_, T, E, D::rms_eps, s_);
        lin(L.attn_conv_proj, dfl_delta_, T);
        conv(xn_, dfl_delta_, L.attn_conv_base, 0, T, B, nullptr, xq_, s_);
        GemvSeg qkv[3] = {{L.wq.fmt, L.wq.w, dfl_q_, L.wq.N, L.wq.K}, {L.wk.fmt, L.wk.w, dfl_k_, L.wk.N, L.wk.K}, {L.wv.fmt, L.wv.w, dfl_v_, L.wv.N, L.wv.K}};
        lin_multi(qkv, 3, T);
        qk_rope_kv(dfl_q_, dfl_k_, dfl_v_, L.q_norm, L.k_norm, D::rope_base, dfl_pos_, T, B, dfl_blk_, dfl_slot_stride_, kcl, vcl, max_ctx_, D::rms_eps, s_);
        attend(dfl_q_, kcl, vcl, dfl_pos_, T, B, dfl_blk_, dfl_slot_stride_, max_ctx_, dfl_o_, xq_, s_);
        lin(L.wo, dfl_y_, T);
        conv(dfl_y_, dfl_delta_, L.attn_conv_base, 1, T, B, big2_, noq, s_);
        add_rmsnorm_quant(dfl_x_, big2_, L.ffn_norm, xn_, xq_, T, E, D::rms_eps, s_);
        lin(L.ffn_conv_proj, dfl_delta_, T);
        conv(xn_, dfl_delta_, L.ffn_conv_base, 0, T, B, nullptr, xq_, s_);
        GemvSeg gu[2] = {{L.ffn_gate.fmt, L.ffn_gate.w, big0_, L.ffn_gate.N, L.ffn_gate.K}, {L.ffn_up.fmt, L.ffn_up.w, big1_, L.ffn_up.N, L.ffn_up.K}};
        lin_multi(gu, 2, T);
        silu_mul_quant(big0_, big1_, xq_, T * D::n_ff, s_);
        lin(L.ffn_down, dfl_y_, T);
        conv(dfl_y_, dfl_delta_, L.ffn_conv_base, 1, T, B, big2_, noq, s_);
        pend = big2_;
    }
    add_rmsnorm_quant(dfl_x_, pend, dfl_->out_norm, xn_, xq_, T, E, D::rms_eps, s_);
    lin(output_, dfl_logits_, T);
    argmax_topk(dfl_logits_, T, D::n_vocab, TOPK, d_ids_, d_vals_, s_);
    lin(dfl_->selector_hidden, dfl_gate_, T);
    selector(d_ids_, d_vals_, dfl_gate_, dfl_->sel_pred, dfl_->sel_succ, dfl_blk_, S, B, dfl_lattice_, s_);
    std::vector<float> lat((size_t)T * LATTICE);
    CK(hipMemcpyAsync(lat.data(), dfl_lattice_, lat.size() * 4, hipMemcpyDeviceToHost, s_)); CK(hipStreamSynchronize(s_));
    for (int b = 0; b < S; ++b) {
        int pred = 0;
        for (int i = 1; i < B; ++i) {
            const float* row = lat.data() + (size_t)(b * B + i) * LATTICE; const float* sc = row + TOPK + pred * TOPK;
            pred = (int)(std::max_element(sc, sc + TOPK) - sc); out[b * nd + i - 1] = (int)row[pred];
        }
    }
    return nd;
}

}  // namespace hip
