#!/usr/bin/env bash
# Run hipster-serve inside the ROCm image (the host has no ROCm). usage: tools/serve.sh [--port 8090] [--ctx 16384] [--mtp 2] ...
# Published on all interfaces (LAN access); BIND=127.0.0.1 restricts it to this box.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-/models/qwen3.8-flash-next/UD-Q4_K_XL-MTP/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00005.gguf}
# 27B: MODEL=/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf DRAFT=models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf tools/serve.sh
DRAFT=${DRAFT:-}
PORT=8090; for ((i=1; i<=$#; i++)); do [ "${!i}" = "--port" ] && j=$((i+1)) && PORT=${!j}; done
exec docker run --rm --name hipster-serve -p ${BIND:-0.0.0.0}:$PORT:$PORT --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 \
  --security-opt seccomp=unconfined --security-opt label=disable --ipc host -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 \
  -v "$ROOT":/src -v /srv/models:/models:ro -w /src "${IMG:-mimiron/rocm:10.0.0}" ./build/hipster-serve --model "$MODEL" ${DRAFT:+--draft "$DRAFT"} --host 0.0.0.0 --port $PORT "$@"
