#!/usr/bin/env bash
# llama.cpp reference for Qwen3.8-Flash-Next on the GPU (ROCm build of the franken branch, n-gram table on disk). Port 18083.
set -euo pipefail
case "${1:-start}" in
  start)
    docker run -d --rm --name hipster-ref-fn -p 127.0.0.1:18083:8080 --security-opt label=disable \
      --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 --security-opt seccomp=unconfined --ipc host \
      -e LLAMA_ATTN_ROT_DISABLE=1 -v /srv/models/qwen3.8-flash-next:/models/fn:ro mimiron/llamacpp:b6fdd0ac8 \
      /opt/llamacpp/bin/llama-server --host 0.0.0.0 --port 8080 -ngl 999 -fit off -fa on -c ${2:-8192} -b 2048 -ub 2048 -np 1 \
      --model /models/fn/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --ngram-on-disk --ngram-io-threads 64 --no-warmup >/dev/null
    echo started;;
  stop) docker stop hipster-ref-fn >/dev/null 2>&1 || true; echo stopped;;
esac
