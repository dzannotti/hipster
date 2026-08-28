# Qwen3.8-27B single-token decode — measurements log

Engine: `engine/src/qwen35.cpp` (fixed shapes, f32 activations, int8 GEMV inputs, f16 KV).
Driver: `./build.sh './build/run27b <gguf> docs/ref/<prompt>.json'` — compares the next-token
top-10 logprobs and the greedy continuation with llama.cpp (CPU reference, `tools/ref-server.sh`).
Streamed bytes per token: 16.95 GB (17.67 GB loaded incl. the 0.72 GB embedding gather) → floor
70.6 ms = 14.2 t/s at 240 GB/s.

## 2026-08-28 · first full pass

- Load: 16.46 GiB to GTT; Q3_K/IQ3_S/IQ4_NL tensors (10 of 866) converted to Q8_0-SoA at load
  (+0.55 GB streamed; 3% — a per-format decoder would recover it, low priority).
- Correctness vs llama.cpp, prompt "The capital of France is": top-10 next-token set identical
  (9/10 ids), max |Δlogprob| 0.19 (top-1: −0.488 vs −0.449); greedy continuation identical for 12
  tokens ("Paris.\nThe capital of Germany is Berlin.\nThe").
- **86.0 ms/token = 11.6 t/s** (197 GB/s streamed, 82% of roof). rocprofv3 breakdown per token:

| kernel | calls/token | ms/token | note |
|---|---:|---:|---|
| GEMV Q5_K (7.9 GB) | 191 | 36.4 | 218 GB/s |
| GEMV IQ4_XS (3.1 GB) | 70 | 14.3 | 218 GB/s |
| GEMV Q6_K-SoA (2.9 GB incl. LM head) | 50 | 12.8 | 224 GB/s |
| GEMV Q4_K (2.3 GB) | 68 | 10.7 | 217 GB/s |
| GEMV Q8_0-SoA (1.1 GB) | 118 | 5.7 | 187 GB/s |
| rmsnorm | 129 | 1.1 | 8.8 µs each — 3× too slow for 20 KB |
| gdn_step | 48 | 1.0 | 6 MB state r/w per layer = at its own roof |
| quantize / add / conv / silu / attn / rope | ~440 | 0.7 | |
| launch gaps (wall − kernels) | ~640 launches | 3.0 | |

Gap to floor = 15.4 ms: ~9 ms is GEMV ramp/tail (450 launches × ~15–20 µs), ~3 ms launch gaps,
~2 ms small kernels. Plan, by size: fuse same-input GEMVs into one launch (qkv+z+β+α, q+k+v,
gate+up: 450 → ~250 launches); fuse add+rmsnorm+quantize and silu·mul+quantize; then Q8_0.
- Longer prompts: "code" (29 tokens) and "story" (19-token chat template) both greedy-identical for
  12 generated tokens; top-1 |Δlogprob| 0.002–0.04, worst tail token 0.32 at logprob −8.6.

## 2026-08-28 · fusions, and what did not help

- Fused: same-input GEMVs into one launch (`gemv_multi`: qkv+z+β+α, q+k+v, gate+up), residual
  add + rmsnorm + q8-quantise, silu·mul + quantise, in-kernel quantise of the GDN and attention
  outputs. Launches/token ≈ 640 → ≈ 350. **86.0 → 81.6 ms/token (12.25 t/s, 208 GB/s streamed,
  87% of floor).** Greedy output unchanged (12/12 on all three prompts).
- Did not help (measured, reverted to default): persistent GEMV grids capped at 160–1280 blocks
  (81.5–83.6 ms, noise); one 18 GiB weight arena instead of 866 allocations (81.6 ms) — kept for
  simplicity, but page-table locality is not the loss.
- Remaining ~11 ms: the GEMV bench already shows it — 34–45 MB tensors rotated cold reach 87–89%
  where a 682 MB tensor reaches 98%. Per-launch cold start inside the memory system, not
  scheduling. Ideas not yet tried: prefetch the next layer's weights into MALL (32 MB) during the
  GDN/attention kernels; larger effective launches by fusing the down-projection with the next
  layer's qkv (different inputs — needs a dependency-free formulation). Parked: the 3–4× lever is
  speculation, which is next.
- Non-temporal weight loads (`__builtin_nontemporal_load` on the 16-byte weight loads, per the
  Nathan-fork suggestion): **80.1 → 132.0 ms/token**. On gfx11 the nt bits bypass L2 and break
  the coalescer; reverted (compile switch `HIPSTER_NT_WEIGHTS` kept at 0).
- KV cache moved to head-major `[layer][kv_head][max_ctx][256]` (per-head stride was 2048 B =
  2 of 16 memory channels on this part; see docs/forks/nathan-strix-halo-llamacpp.md). No effect
  at short context, exact output; the gain shows at depth.

## 2026-08-28 (cont.) — decode at depth (prefill driver, `HIPSTER_DEPTH=1`)

| depth | old per-q-head kernel (K/V read 6×) | GQA-packed split-KV WMMA kernel (512-position splits) |
|---:|---:|---:|
| 2048 | 81.4 ms/token | 82.5 |
| 8192 | 83.1 | |
| 32768 | 95.1 (10.5 t/s) | 97.3 |

KV at 32K is 2 GB per token → 8.7 ms at the roof; both kernels lose ~15 ms, the new one for lack
of parallelism (256 single-wave blocks). 128-position splits are being measured. At 262K the f16
KV alone is 17 GB per token (≥ 70 ms) — int8 KV is mandatory for the 262K × 4-slot target.
