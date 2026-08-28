# Quantisation quality references (collected 2026-08-28)

## Qwen3.8-Flash-Next — unsloth's KLD table vs BF16 (355 GB) [V: unsloth.ai/docs/models/qwen3.8-next]
| variant | file | mean KLD | top-1 agreement |
|---|---:|---:|---:|
| UD-Q4_K_XL (ours) | 111.3 GB | 0.0447 | 93.5% |
| UD-IQ4_XS | 93.7 GB | 0.0792 | 91.1% |
| UD-Q3_K_XL | 90 GB | 0.0997 | 90.4% |
| UD-IQ3_XXS | 82 GB | 0.1565 | 87.6% |
| UD-Q2_K_XL | 78.9 GB | 0.2133 | 85.2% |
No Q4_K_XS / Q4_K_S / Q4_K_M / MXFP4 variants exist in the unsloth repo. Third-party (AtomicChat) ladder: 4.27 bpw
Q4_K_M-mix 92.9 GB KLD 0.084 (experts IQ2_S/IQ3_S/IQ4_NL, everything else Q8_0), BF16 PPL 4.04 ± 0.02 (4096 ctx).
Per-token bytes are what decode speed follows, not file size: the 28.8 GB n-gram table streams from SSD and 10/512
experts are read, so IQ4_XS experts would cut ~6% of the 6.1 GB/token; the 1.96 GB of Q8_0 dense projections are the
lever (docs/ideas.md "Quantisation choice"). Any candidate must pass the KL-vs-reference + needle gate.
