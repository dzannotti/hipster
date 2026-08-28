#!/usr/bin/env bash
# Build (and optionally run) inside the ROCm dev image. usage: ./build.sh [cmd...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
IMG=${IMG:-mimiron/rocm:10.0.0}
docker run --rm --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 \
  --security-opt seccomp=unconfined --security-opt label=disable --ipc host \
  -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 \
  -v "$ROOT":/src -v /srv/models:/models:ro -w /src "$IMG" bash -c '
    set -e
    cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ > /dev/null
    if ! cmake --build build > /tmp/build.log 2>&1; then grep -E "error|FAILED" /tmp/build.log | head -20; echo BUILD_FAILED; exit 1; fi
    echo BUILD_OK
    '"$*"
