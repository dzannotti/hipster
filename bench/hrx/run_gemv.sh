#!/usr/bin/env bash
# Dispatch the engine's k_gemv_q8<Q4_K,4,1> through libhrx and through HIP,
# back to back. usage: HRX_DIR=/path/with/pub [GEMV_N=93184] ./bench/hrx/run_gemv.sh
# (N=93184 rows x 2880 B = 256 MB so the matrix streams from DRAM; 32 MB fits the MALL)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMG=${IMG:-mimiron/rocm:10.0.0}
HRX_DIR=${HRX_DIR:?set HRX_DIR to the directory holding pub/}
docker run --rm --device /dev/kfd --device /dev/dri --group-add 105 --group-add 39 \
  --security-opt seccomp=unconfined --security-opt label=disable --ipc host \
  -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 -e HRX_GPU_DEBUG=1 \
  -v "$ROOT":/src -v "$HRX_DIR":/hrx -w /src "$IMG" bash -c '
    set -e
    H=/hrx/pub; LLVM=/opt/rocm/lib/llvm/bin
    mkdir -p build/hrx
    # device-only HIP compile yields a clang offload bundle; unbundle the gfx1151 code object.
    # (rebuilt only when missing or REBUILD=1, so an in-progress engine edit does not block the bench)
    if [ ! -f build/hrx/gemv_dev.hsaco ] || [ "'"${REBUILD:-0}"'" = 1 ]; then
      $LLVM/clang++ -x hip --offload-arch=gfx1151 --offload-device-only -O3 -w engine/kernels/gemv.hip -o build/hrx/gemv_bundle
      $LLVM/clang-offload-bundler --unbundle --input=build/hrx/gemv_bundle --output=build/hrx/gemv_dev.hsaco --type=o --targets=hip-amdgcn-amd-amdhsa--gfx1151
    fi
    $LLVM/clang++ -x hip --offload-arch=gfx1151 -O2 -w bench/hrx/gemv_hip.hip -o build/hrx/gemv_hip
    DEFS=""; grep -q target_family $H/include/hrx/hrx_runtime.h || DEFS="-DHRX_OLD_LOAD_API"
    $LLVM/clang -O2 -I$H/include $DEFS bench/hrx/gemv_hrx.c -o build/hrx/gemv_hrx -L$H/lib -lhrx -Wl,-rpath,$H/lib
    echo "=== HIP ==="; ./build/hrx/gemv_hip build/hrx/gemv_dev.hsaco '"${GEMV_N:-93184}"'
    echo "=== HRX ==="; LD_LIBRARY_PATH=$H/lib:/opt/rocm/lib timeout 120 ./build/hrx/gemv_hrx build/hrx/gemv_dev.hsaco '"${GEMV_N:-93184}"' || echo "gemv_hrx rc=$?"
    echo "=== HIP again ==="; ./build/hrx/gemv_hip build/hrx/gemv_dev.hsaco '"${GEMV_N:-93184}"'
  '
