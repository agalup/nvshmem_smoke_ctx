# NVSHMEM 3.3.24 — fails in a non-primary CUDA context

Minimal reproducer: NVSHMEM 3.3.24 aborts during `nvshmem_init()` when the
current CUDA context is anything other than the device's primary context.

Just `cuCtxCreate` + `nvshmem_init` — no extra runtime or library required.

## Build and run

```bash
./build_ctx.sh                 # nvcc -> ./smoke_ctx
./run_ctx.sh                   # nvshmrun -np 2 ./smoke_ctx
```

Env (set by `run_ctx.sh`, override before calling if needed):

- `NVSHMEM_HOME` — install root (default `/usr/local/nvshmem-3.3.24-cuda13`)
- `CUDA_VISIBLE_DEVICES=0,1` — both GPUs visible
- `NVSHMEM_REMOTE_TRANSPORT=none`, `RDMAV_DRIVERS=mlx5` — quiet, intra-node only

## Expected output

```
PE 0: my_ctx=0x... primary=0x... (distinct? yes)
PE 0: before nvshmem_init
[putget.cpp:58] cuda failed with invalid resource handle
[nvshmemi_prepare_and_post_rma:296 aborting]
```

`PE ?: after nvshmem_init` **never prints** — NVSHMEM aborts inside its own init.

## Why it fails

`cudaErrorInvalidResourceHandle` (33) means a CUDA handle was used in a
context other than the one it was created in. The test itself never changes
contexts after `cuCtxCreate`, so the mismatch happens inside `nvshmem_init`:
NVSHMEM internally calls `cuDevicePrimaryCtxRetain`, creates streams/events
in one of the two contexts (primary or the caller's), then issues an
operation in the other one. Crash before `nvshmem_init` returns.

**Not documented.** Searched NVSHMEM 3.6.5 docs
(`docs.nvidia.com/nvshmem/api/cuda-interactions.html`, `.../using.html`) —
no mention of a primary-context requirement. Behavior is nonetheless
deterministic on 3.3.24.

## Compare with working case

The companion folder `../nvshmem_smoke_basic/` does the same exchange but
uses `cudaSetDevice` (primary context) instead of `cuCtxCreate`. Runs clean:

```
PE 0: slots=[0,101] expect=[0,101]
PE 1: slots=[100,1] expect=[100,1]
```

Only material difference between the two: which CUDA context is current at
`nvshmem_init`. Primary works; non-primary fails.

## Implication

Any stack that manages its own CUDA contexts (anything using `cuCtxCreate`
instead of the default primary) cannot call device-side NVSHMEM APIs from
those contexts. NVSHMEM calls have to stay in the primary context.

To file upstream: the sources here are a complete repro — attach `smoke_ctx.cu`,
`build_ctx.sh`, `run_ctx.sh` to an issue at https://github.com/NVIDIA/nvshmem/issues.
