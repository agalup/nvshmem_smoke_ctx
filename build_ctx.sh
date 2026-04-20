#!/usr/bin/env bash
set -euo pipefail
NVSHMEM_HOME="${NVSHMEM_HOME:-/usr/local/nvshmem-3.3.24-cuda13}"
SRC="$(dirname "$0")/smoke_ctx.cu"
OUT="$(dirname "$0")/smoke_ctx"
nvcc -std=c++20 -rdc=true -arch=sm_80 -O2 \
    -I"$NVSHMEM_HOME/include" \
    -o "$OUT" \
    "$SRC" \
    "$NVSHMEM_HOME/lib/libnvshmem_device.a" \
    -L"$NVSHMEM_HOME/lib" -lnvshmem_host \
    -Xlinker -rpath -Xlinker "$NVSHMEM_HOME/lib" \
    -lcuda
echo "built $OUT"
