#!/usr/bin/env bash
# Run hipster-serve inside the ROCm image (the host has no ROCm). usage: tools/serve.sh [--port 8090] [--ctx 16384] [--mtp 2] ...
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-/models/qwen3.8-flash-next/UD-Q4_K_XL-MTP/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00005.gguf}
PORT=8090; for ((i=1; i<=$#; i++)); do [ "${!i}" = "--port" ] && j=$((i+1)) && PORT=${!j}; done
exec docker run --rm --name hipster-serve -p 127.0.0.1:$PORT:$PORT --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 \
  --security-opt seccomp=unconfined --security-opt label=disable --ipc host -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 \
  -v "$ROOT":/src -v /srv/models:/models:ro -w /src "${IMG:-mimiron/rocm:10.0.0}" ./build/hipster-serve --model "$MODEL" --host 0.0.0.0 --port $PORT "$@"
