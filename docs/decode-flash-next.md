# Qwen3.8-Flash-Next decode — measurements log

Engine: `engine/src/qwen4exp.cpp` (+ `engine/kernels/fn.hip`). Driver: `./build.sh './build/runfn <shard1> docs/ref/fn-<prompt>.json'`.
Reference: llama.cpp ROCm (franken branch) with the n-gram table on disk (`tools/ref-server-fn.sh`).
Streamed bytes per token: 6.43 GB (docs/roofline.md) → floor 37 t/s.

## 2026-08-28 · first pass (T ≤ 8 decode, dense attention, no QSA / MTP / prefill GEMM)
- Load: 76.8 GiB to GTT via pread (page cache dropped per tensor); ~1.3 GB/s; the n-gram table is
  never uploaded (16 IQ4_NL rows of 90 B gathered per token from the mmap on the host).
- Correctness (code prompt, 29 tokens): top-1 identical, greedy **12/12** identical to llama.cpp,
  max |Δlogprob| over top-10 = 0.62 (tail tokens at logprob −9).
- **57.0 ms/token = 17.5 t/s** (llama.cpp on this box: 18.0 ROCm / 26.7 Vulkan bare). Effective
  streamed bandwidth 113 GB/s = 47% of roof → profile next.

Profile (rocprofv3, per token): kernels 40.6 ms of 57 wall → **~16 ms of launch gaps (~1150 launches)**.
Q8_0 GEMVs (hc down/up, ssm_out, wo, shared expert, PLE) 389 launches 14.5 ms (3.3 MB ones at ~90 GB/s
— ramp-bound); qkv+z+β+α multi 9.0 ms (at roof); MoE gate/up (Q4_K) 4.4 ms (~190 GB/s); MoE down
(Q5_1) 5.3 ms (~97 GB/s); router f32 1.8; hc_mix 1.4; gdn_step 1.1; the rest < 1 ms each.
Plan: fuse the hyper-connection read (norm+down GEMV, up GEMV+mix), merge expert gate/up (+ shared
expert) into one launch, fold combine into the norm, router+shexp-gate in one launch; Q5_1 decoder.

## 2026-08-28 · fusion pass: 57.0 → 35.4 ms/token (17.5 → **28.2 t/s**), still 12/12 greedy-exact (Δlogprob ≤ 0.42)
Kernel time 33.7 ms + 1.7 ms launch gaps (was 40.6 + 16). Weight bytes per token re-counted from the tensor
list: **6.1 GB → floor 25.4 ms = 39 t/s**; we are at 72% of it. Steps, each measured (`tools/prof-kernels.sh`):
1. gate|up for the 10 routed experts **and the shared expert in one launch** (shared = slot with id −1, its own
   Q8_0 decoder as a second template parameter); router and shared-expert gate concatenated into one 513-row
   f32 GEMV; combine folded away. 57 → 42 ms.
2. **Lanes per row is a per-shape decision**, from `bench_gemv` on this checkpoint (GB/s at ncol=1):

   | tensor | shape | 32 lanes | 16 | 8 | 4 | 2 | 1 | chosen |
   |---|---|---|---|---|---|---|---|---|
   | ffn_down_exps Q5_1 (flat) | 1.3M × 640 | 107 | 194 | **224** | 207 | 140 | 71 | 16 (gather: 152; 8: 144; 4: 97; 32: 95) |
   | ffn_gate_exps Q4_K (flat) | 327K × 2560 | 233 | 236 | 231 | 221 | 224 | 204 | 32 (gather 201–206 for all) |
   | ssm_out / wo Q8_0 | 2560 × 6144 | 142 | 147 | 128 | 187 | **199** | 183 | 2 |
   | attn_qkv Q8_0 | 10240 × 2560 | **192** | 159 | 154 | 141 | 137 | 138 | 32 |
   | attn_q Q8_0 | 12288 × 2560 | **189** | 150 | 146 | 155 | 146 | 135 | 32 |
   | hc_*_up Q8_0 | 10240 × 320 | 88 | 135 | **150** | 104 | 100 | 102 | 8 |
   | hc_*_down Q8_0 | 320 × 10240 | 120 (40 blocks) | | | | | | split-K ×8 + atomics: 170 |

   The short-K rows (640, 320) are not bandwidth-bound: on cache-hot data the flat kernel takes 54 µs at 8 lanes vs
   134 at 4 and 190 at 32 for the same 28160 rows, i.e. the wave shape drives issue efficiency, not lane idleness.
   Lesson recorded: the bench's env knob was cached in a `static`, which made the gather look lane-independent
   for an hour — knobs are now plain globals set by the bench.
3. Launch count 1150 → ~1000 and the small kernels: `hc_mix` 14 → 3.2 µs (4 contended atomic addresses → per-block
   partials summed by the consumer), `moe_route` 11.5 → 7.6 µs (wave-level top-10 with shuffles instead of 10
   block-wide argmax rounds), MoE combine folded into the down-projection launch (weighted atomicAdd into a
   buffer the norm zeroes), the write half of every hyper-connection folded into the next read's norm, all
   memsets gone. 42 → 35.4 ms.

Profile now (ms/token): expert GEMVs 9.4 (1.81 GB, 193 GB/s), qkv/z/β/α + q/k/v multi 9.0 (1.96 GB, 218),
Q8_0 GEMVs 8.7 (lm head 2.8 at 241; ssm_out/wo/hc_up at ~180), split-K hc_down 1.9, router f32 1.45,
gdn_step 1.1, hc_norm 1.0, moe_route 0.4, hc_mix 0.3, rest < 0.2. What is left is ~8 ms of per-launch ramp/tail
on ~1 GB-class launches plus ~2.5 ms of small kernels — no single item above 2 ms. The multiplier from here is
the MTP block (blk.48 in `/srv/models/qwen3.8-flash-next/mtp/*.gguf`: attention + indexer + MoE + hc +
`nextn.eh_proj/enorm/hnorm`), then requantising the 1.96 GB/token of Q8_0 dense projections (32% of the bytes).
Todo noted: `k_topk` over 248K logits takes 270 µs — needs a proper two-stage argmax before serving.
