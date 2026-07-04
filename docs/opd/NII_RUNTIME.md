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

## Running the OPD operator command on our image
Reference: `aimo-proof-pilot/operator_commands/prime_rl_opd_4xh200_muon_imo_...sh`. Our image doesn't
bake Nguyen's `train.py` launcher, so:
1. Pull the operator repo: `eval "$(opd-env-sync)"` → `/tmp/imochallenge/opd-env/aimo-proof-pilot`
   (+ PYTHONPATH so `proof_opd_env` resolves).
2. Stage model + data locally; point the operator env vars at them (`PRIME_OPD_MODEL_PATH`,
   `PRIME_OPD_DATASET_PATH`, `PRIME_OPD_VERIFIABLE_DATASET_PATH`).
3. Run the git-pulled `train.py` with OUR venv + fetch DISABLED (so it uses our baked, worker-bug-free
   prime-rl instead of re-pulling nguyen599/main):
   - `/app/.venv/bin/python $REPO/src/train.py` (not `/usr/bin/python /app/train.py`)
   - replace `--fetch-update` with `--no-fetch-update --no-ensure-runtime-training-deps`
4. Launch detached: `opd-run opd1 bash <edited-operator>.sh`; monitor via `opd-status` /
   `tail -f /tmp/imochallenge/logs/opd1.log`.
All other flags (muon, opd, teacher fp8, olmo3_sink_fa3, deepseek_v4, ulysses CP, fp8) are baked.

## Checkpointing & disk
prime-rl `CheckpointConfig` defaults (verified): **`interval=None`, `keep_last=None`, `keep_interval=None`**.
- **The reference operator command sets no ckpt config → NO full checkpoints are written** (the loop
  saves only when `config.ckpt.interval` is set AND `step % interval == 0`, `rl/train.py:303-305`).
  Policy weights go to the rollout/teacher engine each step, but via NCCL/in-memory by default — not disk.
- **If you enable checkpointing, retention is UNBOUNDED** (`keep_last=None` → keep all). Set `keep_last`
  (e.g. 2–3) or the disk fills.
- **Sizing (32B):** image ~22.5 GB; model bf16 ~64 GB (fp8 ~32 GB; +64 GB if pulled via HF cache instead
  of a local path); caches ~10–20 GB; **each full checkpoint ~200–450 GB** (bf16 weights + fp32 optimizer
  state). Budget ~150 GB for a smoke run, ~1 TB for real training with checkpoints. Stage the model as a
  LOCAL path and checkpoint to a MOUNTED volume (not container `/tmp`).
