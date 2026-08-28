# Serving — `hipster-serve` (OpenAI-compatible, Flash-Next)

`tools/serve.sh [--port 8090] [--ctx 16384] [--chunk 2048] [--mtp 2]` runs `build/hipster-serve` inside the ROCm image
(model: the UD-Q4_K_XL-MTP shards; `MODEL=...` overrides). One slot, requests served in order. Startup ≈ 60 s (78.6 GiB
of weights via pread + GEMM warm-up). Stop with `docker stop hipster-serve`.

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (`messages`, `tools`, `stream`, `max_tokens`,
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
