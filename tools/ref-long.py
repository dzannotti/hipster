#!/usr/bin/env python3
"""Long-context reference from llama.cpp on the GPU (f16 KV, no speculation): builds a haystack
prompt with a planted needle at ~40% depth and a question, records token ids, the greedy answer
and the top-10 logprobs at the last prompt position. usage: ref-long.py <target_tokens> <name>"""
import json, sys, urllib.request, os, random
URL = os.environ.get("REF_URL", "http://127.0.0.1:18082")
def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=36000).read())
target, name = int(sys.argv[1]), sys.argv[2]
corpus = open("/srv/models/.work/corpus/doc.txt").read().split("\n\n")
random.seed(7)
needle = "\n\nNote for the reader: the secret passphrase for the vault is 'amber-falcon-73'. Keep it safe.\n\n"
head = "<|im_start|>user\nHere is a long document. Read it carefully; a question follows at the end.\n\n"
tail = "\n\nQuestion: what is the secret passphrase for the vault? Answer with only the passphrase.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# grow the haystack until the token count reaches the target
paras, ids = [], []
while True:
    paras.append(random.choice(corpus).strip())
    text = head + "\n\n".join(paras[: max(1, int(len(paras) * 0.4))]) + needle + "\n\n".join(paras[max(1, int(len(paras) * 0.4)):]) + tail
    ids = post("/tokenize", {"content": text, "add_special": False})["tokens"]
    if len(ids) >= target: break
print(name, len(ids), "tokens; querying reference ...", flush=True)
r = post("/completion", {"prompt": ids, "n_predict": 24, "temperature": 0, "n_probs": 10, "top_k": 1, "return_tokens": True, "cache_prompt": False})
out = {"text_head": text[:200], "prompt_ids": ids, "gen_ids": r["tokens"], "content": r["content"],
       "first_top10": [(p["id"], p["token"], p["logprob"]) for p in r["completion_probabilities"][0]["top_logprobs"]],
       "timings": r.get("timings")}
os.makedirs("docs/ref", exist_ok=True)
json.dump(out, open(f"docs/ref/{name}.json", "w"))
print(name, "->", repr(r["content"]), "| prefill", r.get("timings", {}).get("prompt_ms"), "ms")
