#!/usr/bin/env bash
# llama.cpp reference for correctness: CPU-only server on the 27B (no GPU contention), port 18081.
# usage: tools/ref-server.sh start|stop
set -euo pipefail
case "${1:-start}" in
  start)
    docker run -d --rm --name hipster-ref -p 127.0.0.1:18081:8080 --security-opt label=disable \
      -v /srv/models/qwen3.8-27b:/models/qwen38:ro mimiron/llamacpp:b6fdd0ac8 \
      /opt/llamacpp/bin/llama-server --host 0.0.0.0 --port 8080 --device none -ngl 0 -t 24 -c 4096 \
      --model /models/qwen38/Qwen3.8-27B-UD-Q4_K_XL.gguf --no-warmup >/dev/null
    echo started;;
  stop) docker stop hipster-ref >/dev/null 2>&1 || true; echo stopped;;
esac
