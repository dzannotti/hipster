# Serving — `hipster-serve` (OpenAI-compatible, Flash-Next)

`tools/serve.sh [--port 8090] [--ctx 16384] [--chunk 2048] [--mtp 2]` runs `build/hipster-serve` inside the ROCm image
(model: the UD-Q4_K_XL-MTP shards; `MODEL=...` overrides). One slot, requests served in order. Startup ≈ 60 s (78.6 GiB
of weights via pread + GEMM warm-up). Stop with `docker stop hipster-serve`.

A built-in chat page is served at `/` (streaming, multi-turn, thinking toggle + effort, temperature, max tokens; shows time to first token and t/s).

llama.cpp's web UI is served at `/` (static files from `--ui-dir`, default the fn-tree build's `tools/ui/dist` mounted read-only
inside the container; `/props` and per-chunk `timings` are provided in the shape it reads, so its prompt/generation t/s readout works).

Endpoints: `GET /health`, `GET /props`, `GET /v1/models`, `POST /v1/chat/completions` (`messages`, `tools`, `stream`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `chat_template_kwargs: {enable_thinking, reasoning_effort}` or top-level
`enable_thinking`/`reasoning_effort`), `POST /v1/completions` (raw `prompt` string, no template).
- Thinking is on by default (the model's template): the reply carries `reasoning_content` (streamed as
  `delta.reasoning_content`) and `content`; `"chat_template_kwargs": {"enable_thinking": false}` turns it off.
- Tools: rendered with the model's `<tools>` block; a `<tool_call><function=…>` reply is parsed into OpenAI
  `tool_calls` (`arguments` as a JSON string) with `finish_reason: "tool_calls"`; `role: tool` messages are rendered
  as `<tool_response>` turns.
- temperature 0 (default): greedy with MTP drafts (exact). temperature > 0: plain decode with top-k/top-p sampling.
- The tokenizer is ours (`front/tokenizer.cpp`, byte-level BPE with the qwen35 pre-tokenizer), 45/45 identical to
  llama.cpp's `/tokenize` on `docs/ref/tok-tests.json` (`build/tok_test`).

```
curl -s localhost:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "Write one sentence about the sea."}],
  "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": false}}'
```
First measurements (2026-08-28): generation 41 t/s with MTP n=2 (acceptance ~80% on these prompts); prefill 318 t/s on a
265-token prompt (short prompts are launch-bound; 2K chunks run at ~1000 t/s, 16K prompts at ~800 t/s).
Not there yet: prompt/prefix caching across turns, concurrent requests (the engine has multi-slot decode; the server
does not use it yet), `stop` strings, `logprobs`, `n > 1`, vision, the 27B behind the same endpoints.

## 2026-08-28 · the 27B behind the same endpoints (DFlash2 drafts)

`hipster-serve` picks the engine from the GGUF architecture (`qwen35` → 27B, `qwen4exp` → Flash-Next) behind one
token-level `Backend` interface (prefill chunk, draft, verify, accept, step). 27B: `--draft <DFlash2 gguf>` enables the
block draft (`--ndraft 7`), otherwise plain decode; prefill chunks are capped at 4096 (its GEMM path).

    MODEL=/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf DRAFT=models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf tools/serve.sh --ctx 8192

First measurement (chat completion, 37-token prompt, 200 tokens of Python, greedy): `prompt_ms 1135`,
`predicted_per_second 45.9` (DFlash2 n=7), UI/streaming/thinking/tools unchanged. Still one slot per server; the
engine's 2-slot lockstep mode (79 t/s aggregate in `dflash27b`) is not scheduled from the HTTP layer yet.

### 2-slot serving (27B)

`--slots 2`: `http::serve` now runs one thread per connection; requests are queued to a scheduler thread that admits jobs
into free slots (prefill, other slots pause), then runs lockstep rounds over the active slots — one batched DFlash2 draft
pass, one verify pass over all rows, per-slot accept/encode/stream. Greedy output is identical alone or concurrent
(`A alone == A concurrent: True`).

| request mix (200–300 tokens each, greedy, 40-token prompts) | per request | aggregate (wall, incl. prefills) |
|---|---|---|
| 1 code | 45.8 t/s | 45.8 |
| 2 concurrent code | 32.7 / 38.5 | **61.4** |
| code + prose | 30.6 / 19.1 | 32.9 (prose accepts ~30% of the drafts) |

The 40-token prompts cost 1.1 s each through the GEMM path (every weight dequantised to bf16 per chunk). Prompts of
≤ 96 tokens now go through 16-row GEMV passes (a single-slot pass of up to 16 rows takes the GEMV path; attention groups
are split at 8 rows): `prompt_ms 372`, and with the prefill share smaller the two concurrent code requests reach
**68.4 t/s aggregate** (600 tokens in 8.8 s wall, per request 35.7 / 38.5, text identical to the solo run).

### Flash-Next with slots (plain batched decode)

`--slots N` (≤ 8) on the Flash-Next GGUF: the same scheduler, one row per active slot per pass (its MTP drafts are used
only with one slot for now). `prefill()` took a slot argument; it had been writing K/V through the single-slot rope/KV
kernel without the slot offset (reads used the slot), so any request prefilled into slot ≠ 0 decoded garbage — fixed
(`qsa_attention`, rb == null path). Checks: a request prefilled into slot 1 while slot 0 decodes matches its solo output;
each of 4 concurrent requests matches its solo output.

| Flash-Next | per request | aggregate (wall, incl. prefills) |
|---|---|---|
| 1 request, MTP n=2 | 43.5 t/s | 43.5 |
| 1 request on a 4-slot server (plain) | 27.4 | 27.4 |
| 4 concurrent (200-token replies, 40-token prompts) | 14.4–16.3 | **50.2** (63 while all four are active; one reply ended at 85 tokens) |

Flash-Next drafts over slots: batched MTP rounds (`FnBackend::draft`, one MTP pass per draft step over the active slots)
with the measured policy n=2 / n=1 / 0 for 1 / 2 / ≥3 active slots (docs/decode-flash-next.md). A 4-slot server:
1 request 43.7 t/s, 2 concurrent 48.1 aggregate (24.8 / 25.5), 4 concurrent 50.8 aggregate (13–17 each), every output
identical to the plain-batched and solo runs. The fix behind it: Flash-Next decode attention used a different key-split
count above 8 rows, which broke exactness for any multi-slot pass with more than 8 rows.

Client disconnects: an `emit` that fails (socket gone) marks the job and the scheduler finishes it at the end of the
round, freeing the slot (a 3000-token streaming request killed after 2 s: the next request was served 0.7 s later).
