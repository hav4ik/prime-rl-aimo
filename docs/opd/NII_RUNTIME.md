# NII runtime — launching the container on the NII cluster

How to run `aimo-opd-sft` under Singularity/Apptainer on NII. The image is fully baked; runtime writes
go only to `/tmp` + `$HOME`. Nguyen's proven runtime setup is the reference (see [[CONTAINER_COMPARISON]]).

## InfiniBand (multi-node NCCL)
NII sets these **host** env vars before running and adds the IB device bind — Nguyen's proven setup,
which our container supports natively:
```bash
export NCCL_IB_HCA=mlx5_ibn1,mlx5_ibn2,mlx5_ibn3,mlx5_ibn4,mlx5_ibn5,mlx5_ibn6,mlx5_ibn7,mlx5_ibn8
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CROSS_NIC=1
apptainer run --nv --bind /dev/infiniband:/dev/infiniband  aimo-opd-sft.sif   # (+ your other binds)
```
**Container-side support is already present** (inherited from the prime-rl base — it's a distributed
trainer): `libibverbs.so.1`, `libmlx5.so.1` + the `libmlx5-rdmav34.so` provider (drives the `mlx5_ibn*`
HCAs), `librdmacm.so.1`, `ibverbs-providers` (v50.0); NCCL (bundled with torch, `libnccl.so.2`) has IB
transport built in. So:
- The 3 `NCCL_IB_*` vars are read by NCCL **at runtime** — nothing to bake.
- `--bind /dev/infiniband` exposes the host IB devices; the container has the verbs libs to use them.
- We deliberately do **NOT** bake these vars — the HCA names are NII-host-specific; they stay host-side,
  exactly as Nguyen does. (Nguyen installs the same `rdma-core`/`ibverbs-providers`; his `-dev` variants
  are build-time headers, not needed at runtime.)

## GPU
`--nv` injects the host driver + `libcuda.so.1`. Our cu128 stack runs on the NII cu13 host driver via
CUDA backward-compat (see [[CUDA_VERSIONING]]).

## Writable paths (immutable FS)
Only `/tmp` (NII team storage) + `$HOME` are writable; everything else is read-only. All caches sit under
`/tmp/imochallenge/cache/*` (baked env vars); daemon logs under `/tmp/imochallenge/logs`; the git-pulled
OPD env under `/tmp/imochallenge/opd-env` (via `opd-env-sync`).

## Remote-shell daemon (default entrypoint)
The entrypoint runs the crash-resilient relay daemon (`/opt/venv-daemon`). Provide at runtime:
`HF_TOKEN`, `RELAY_SPACE`, and optional `CLIENT_ID`. Training runs **through** the relay shell sessions
(both prime-rl OPD and olmo-core SFT share `/app/.venv`).

## Secrets
Pass `HF_TOKEN` / `WANDB_API_KEY` / `GITHUB_TOKEN` via the environment at runtime; never baked into the
image.
