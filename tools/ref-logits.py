#!/usr/bin/env python3
"""Reference from llama.cpp (tools/ref-server.sh): token ids, greedy continuation, top-10 logprobs
of the first generated token. Writes docs/ref/<name>.json."""
import json, sys, urllib.request, os
import os
URL = os.environ.get("REF_URL", "http://127.0.0.1:18081")
PREFIX = os.environ.get("REF_PREFIX", "")
def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
prompts = {
    "france": "The capital of France is",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return",
    "story": "<|im_start|>user\nWrite one sentence about the sea.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
}
os.makedirs("docs/ref", exist_ok=True)
for name, text in prompts.items():
    ids = post("/tokenize", {"content": text, "add_special": False})["tokens"]
    r = post("/completion", {"prompt": ids, "n_predict": 12, "temperature": 0, "n_probs": 10, "top_k": 1, "return_tokens": True, "cache_prompt": False})
    gen = r["tokens"] if "tokens" in r else [t["id"] for t in r["completion_probabilities"]]
    out = {"text": text, "prompt_ids": ids, "gen_ids": gen, "content": r["content"],
           "first_top10": [(p["id"], p["token"], p["logprob"]) for p in r["completion_probabilities"][0]["top_logprobs"]] if r.get("completion_probabilities") else None}
    json.dump(out, open(f"docs/ref/{PREFIX}{name}.json", "w"), indent=1)
    print(name, len(ids), "prompt tokens ->", repr(r["content"][:60]), "| top:", out["first_top10"][:3] if out["first_top10"] else None)
