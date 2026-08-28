#!/usr/bin/env bash
# Build and run the HRX launch benchmark inside the ROCm image against a
# prebuilt hrx-system release. usage: HRX_DIR=/path/with/pub+deps [RUNS=2] ./bench/hrx/run.sh
#   HRX_DIR must contain pub/ (hrx-public-linux-x86_64-v0.3.0.tar.zst extracted)
#   and deps/ (hrx-public-deps-linux-x86_64-v0.3.0.tar.zst extracted; only used
#   when HRX_ROCR=bundled, default is the container's /opt/rocm ROCR).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMG=${IMG:-mimiron/rocm:10.0.0}
HRX_DIR=${HRX_DIR:?set HRX_DIR to the directory holding pub/ and deps/}
HRX_ROCR=${HRX_ROCR:-rocm}
RUNS=${RUNS:-2}
docker run --rm --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 \
  --security-opt seccomp=unconfined --security-opt label=disable --ipc host \
  -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 -e HRX_GPU_DEBUG=1 \
  -v "$ROOT":/src -v "$HRX_DIR":/hrx -w /src "$IMG" bash -c '
    set -e
    H=/hrx/pub; LLVM=/opt/rocm/lib/llvm/bin
    mkdir -p build/hrx
    $LLVM/clang -x c -std=c23 --target=amdgcn-amd-amdhsa -mcpu=gfx1151 -nogpulib -O3 -fvisibility=hidden \
      bench/hrx/kernels.c -o build/hrx/kernels.hsaco
    DEFS=""; grep -q hrx_graph_create $H/include/hrx/hrx_runtime.h && DEFS="$DEFS -DHRX_HAS_GRAPHS"
    grep -q target_family $H/include/hrx/hrx_runtime.h || DEFS="$DEFS -DHRX_OLD_LOAD_API"
    echo "libhrx header API defines:$DEFS"
    $LLVM/clang -O2 -I$H/include $DEFS bench/hrx/launch_hrx.c -o build/hrx/launch_hrx \
      -L$H/lib -lhrx -Wl,-rpath,$H/lib
    echo "=================== HIP baseline (bench/roofline/launch.hip), same session ==================="
    $LLVM/clang++ -x hip --offload-arch=gfx1151 -O2 -w bench/roofline/launch.hip -o build/hrx/launch_hip && ./build/hrx/launch_hip
    if [ "'"$HRX_ROCR"'" = bundled ]; then LP=$H/lib:/hrx/deps/lib:/hrx/deps/lib/rocm_sysdeps/lib; else LP=$H/lib:/opt/rocm/lib; fi
    for r in $(seq '"$RUNS"'); do
      echo "=================== HRX run $r (ROCR: '"$HRX_ROCR"') ==================="
      LD_LIBRARY_PATH=$LP timeout 60 ./build/hrx/launch_hrx build/hrx/kernels.hsaco || echo "hrx run exited rc=$?"
    done
    echo "=================== HIP baseline again (after HRX) ==================="
    ./build/hrx/launch_hip
  '
