#!/usr/bin/env bash
# Hardware roofline for gfx1151 (Strix Halo). Runs inside the ROCm dev image.
# Results are recorded in docs/roofline.md -- re-run when the driver/ROCm changes.
set -euo pipefail
IMG=${IMG:-mimiron/rocm:10.0.0}
HERE=$(cd "$(dirname "$0")" && pwd)
exec docker run --rm --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 \
  --security-opt seccomp=unconfined --security-opt label=disable --ipc host \
  -v "$HERE":/w -w /w "$IMG" bash -c '
    set -e; mkdir -p /tmp/b
    hipcc -O3 --offload-arch=gfx1151 -o /tmp/b/bw     bandwidth.hip      2>/dev/null
    hipcc -O3 --offload-arch=gfx1151 -o /tmp/b/launch launch.hip         2>/dev/null
    hipcc -O3 --offload-arch=gfx1151 -o /tmp/b/gemm   gemm_hipblaslt.hip -lhipblaslt 2>/dev/null
    echo "## bandwidth";  /tmp/b/bw 8
    echo "## launch";     /tmp/b/launch
    echo "## gemm (hipBLASLt, f16 in / f16 out, f32 accumulate)"
    for s in "2048 5120 17408" "512 5120 17408" "64 5120 17408" "8 5120 17408" "2048 17408 5120" "2048 2560 640"; do /tmp/b/gemm $s; done
    /tmp/b/gemm 2048 5120 17408 b'
