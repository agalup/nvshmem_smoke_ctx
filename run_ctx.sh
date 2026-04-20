#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RDMAV_DRIVERS="${RDMAV_DRIVERS:-mlx5}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"
DIR="$(dirname "$0")"
BIN="$DIR/smoke_ctx"
if [ ! -x "$BIN" ]; then
    "$DIR/build_ctx.sh"
fi
exec nvshmrun -np 2 "$BIN"
