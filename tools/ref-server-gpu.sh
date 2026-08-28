#!/usr/bin/env bash
# llama.cpp reference on the GPU (ROCm build, f16 KV, no speculation) for long-context checks. Port 18082.
set -euo pipefail
case "${1:-start}" in
  start)
    docker run -d --rm --name hipster-ref-gpu -p 127.0.0.1:18082:8080 --security-opt label=disable \
      --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 --security-opt seccomp=unconfined --ipc host \
      -v /srv/models/qwen3.8-27b:/models/qwen38:ro mimiron/llamacpp:b6fdd0ac8 \
      /opt/llamacpp/bin/llama-server --host 0.0.0.0 --port 8080 -ngl 999 -fit off -fa on -c ${2:-40960} -b 2048 -ub 2048 \
      --model /models/qwen38/Qwen3.8-27B-UD-Q4_K_XL.gguf --no-warmup -np 1 >/dev/null
    echo started;;
  stop) docker stop hipster-ref-gpu >/dev/null 2>&1 || true; echo stopped;;
esac
