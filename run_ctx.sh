#!/usr/bin/env bash
set -u
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RDMAV_DRIVERS="${RDMAV_DRIVERS:-mlx5}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"
DIR="$(dirname "$0")"
BIN="$DIR/smoke_ctx"
if [ ! -x "$BIN" ]; then
    "$DIR/build_ctx.sh"
fi
exec nvshmrun -np 2 "$BIN"
