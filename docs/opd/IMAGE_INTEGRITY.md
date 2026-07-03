# AIMO OPD image — integrity checks

Every verification performed while building `chankhavu/aimo-opd:v1` (the prime-rl OPD image),
so its correctness is auditable and re-runnable. Each item lists **what** is guaranteed, **how**
it was checked, and the **result**. Commands run against the built image via
`docker run --rm [--gpus all] --entrypoint … chankhavu/aimo-opd:v1`.

The image is built by overlaying this fork's `src/prime_rl` onto `primeintellect/prime-rl:main`
(`Dockerfile.opd`) — no from-scratch build. See `CUDA_VERSIONING.md` for the version rationale.

---

## 1. Overlay integrity — the fork's code is the image's code, on the exact same venv

- **Fork deps ≡ upstream deps.** `pyproject.toml` AND `uv.lock` are **byte-identical** between the
  fork and upstream `PrimeIntellect-ai/prime-rl@main` (`diff` → 0 lines both files). So the
  prebuilt upstream image already contains this fork's exact venv; only Python source differs.
- **`prime_rl` is an editable namespace package.** Verified `prime_rl.__path__ == ['/app/src/prime_rl',
  '/app/packages/prime-rl-configs/src/prime_rl']` and `prime_rl.trainer.models.__path__ ==
  ['/app/src/prime_rl/trainer/models']` — so overlaying those paths adds new files (olmo3_sink) and
  overrides changed ones with no reinstall.
- **Overlay took effect.** Base image: `olmo3_sink present: False`. Built image: `olmo3_sink present:
  True`, `Olmo3SinkForCausalLM import: OK`, `configs.trainer` resolves to the overlaid file.

## 2. Attention-sink backends — registered, selectable, and numerically correct

- **Both backends register** (`register_olmo3_sink()` → `ALL_ATTENTION_FUNCTIONS`): `olmo3_sink_fa3`
  and `olmo3_sink_fa2` = `True`; both selectable in the `AttnImplementation` config alias.
- **FA3 fallback path** verified: `_fa3_fwd_accepts_sink()` = `False` on stock FA3 3.0.0, so the
  exact `o·α` post-processing runs (forward is the closed-form sink; backward reuses FA3's native
  backward fed the sink-inclusive `lse`).
- **FA2 sink kernel — gradient parity on GPU** (RTX 3090, sm_80 cubins). Vs an fp32 reference:
  - forward matches to bf16 (`9.85e-3`), MHA and GQA;
  - `sinks.grad` matches to ~`1e-3` relative;
  - **sink-off ≡ plain FA2 bit-for-bit** (`fwd/dq/dk/dv` diff = `0.00e+00`) → the adapter adds zero
    error beyond FA2's own bf16 backward.
- **Test suite** (`tests/unit/train/models/`): `test_olmo3_sink_kernels.py` (FA2/FA3 forward + dsink
  parity, FA3 skips off-Hopper) and `test_olmo3_sink_guards.py` (whitelist raise + vLLM convert).
  In-container result: **kernels 4 passed / 4 skipped (FA2 runs, FA3 needs sm_90); guards 7 passed.**

## 3. FP8 (deep-gemm)

- Installed (`deep-gemm==2.5.0+891d57b`) and imports cleanly even without a GPU (JIT is lazy).
- On the OLMo3-sink dense path it is inert unless FP8 is explicitly enabled (`config.fp8=true`,
  `use_deep_gemm=true`) — imports are soft (`deep_gemm=None` on ImportError).

## 4. CUDA versioning integrity (details in `CUDA_VERSIONING.md`)

- **cu128 core:** torch `2.11.0+cu128`, vLLM `0.23.0+cu129` (abi3, runs on the cu12.8 runtime),
  flash-attn `2.8.3+cu128`, all `nvidia-*-cu12` libs at `12.8.x`.
- **cu13 sliver = deep-gemm only, and minimal.** `ldd` on the standalone `deep_gemm/_C.*.so` shows
  it needs exactly `libcudart.so.13` + `libnvrtc.so.13` (both baked in `/opt/cuda13`, `ldconfig`'d) —
  **not** cuBLAS13/cuDNN13, because its kernels JIT. (vLLM's bundled deep_gemm is a separate cu12
  build linking `.so.12`.)
- **Runs on CUDA-13 hosts** via driver backward-compat (cu128 floor ≈ driver 570; any cu13 host
  clears it). No cu13-only rebuild needed or possible (vLLM 0.23 has no cu13 wheel).

## 5. Compiler toolchain — self-contained, never resolves a host (noexec) tool

Rationale: on NII/Singularity a host `nvcc`/tool resolved via PATH (or a host CUDA bound onto
`/usr/local/cuda/bin`) sits on a noexec mount → `torch.compile`/JIT die with an uncaught
`PermissionError`. So every compiler must resolve to an in-image, executable binary.

- **`nvcc` is bind-safe and first on PATH.** `command -v nvcc` → `/opt/opd/bin/nvcc`, a wrapper
  that `exec`s the venv's **self-contained** cu13 nvcc (bundles its own ptxas/cicc/nvvm/headers,
  under `/app/.venv/…/nvidia/cu13/bin`, a path Singularity won't bind).
- **Build-time compile gate.** The Dockerfile RUN compiles a CUDA source with the wrapper nvcc +
  in-image gcc (`in-image nvcc + gcc compile OK`); the build **fails** if the toolchain can't compile.
- **gcc/g++ in-image + pinned.** `/usr/bin/gcc`, `/usr/bin/g++` (image rootfs, never bind-shadowed);
  `CC=/usr/bin/gcc CXX=/usr/bin/g++ NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc"` so they're used
  regardless of PATH. `cmake` and `ninja` are in the venv.
- **PATH ordering:** in-image dirs first (`/opt/opd/bin`, …), host-fillable dirs (`/usr/local/nvidia/bin`)
  LAST — they can never shadow a compiler. This mirrors the proven SFT v2.2.1 (`/opt/fields/bin`) design.
- **`CUDA_HOME` is bind-safe (covers torch + FlashInfer).** These two resolve nvcc/headers/libs via
  `CUDA_HOME`, **not** PATH, so the wrapper alone misses them. A full cu12.8 toolkit is copied to
  `/opt/opd/cuda` (Singularity won't bind `/opt`) and `CUDA_HOME`/`CUDA_PATH` point there — cu12.8,
  matching cu128 torch, no cu13 mix. VERIFIED on GPU: with the host `/usr/local/cuda/bin/nvcc`
  `chmod 000`'d (noexec-mount mimic), `torch.utils.cpp_extension.load_inline` still compiles + runs
  off `/opt/opd/cuda`; flashinfer's JIT reads the same `CUDA_HOME`.
- **deep-gemm JIT** pointed at the bind-safe cu13 wrapper (`DG_JIT_NVCC_COMPILER=/opt/opd/bin/nvcc`),
  matching its cu13 wheel.

## 6. Immutable-FS / runtime writes — only `/tmp` and `$HOME`, tidily namespaced

NII mounts the team's data storage at `/tmp` (so `/tmp` is **not** empty). Nothing is assumed about
`/tmp`'s contents; every cache this container writes is namespaced under one umbrella:

- **`/tmp/imochallenge/cache/<tool>`** for all 15 cache/JIT/scratch dirs: `torchinductor`, `triton`,
  `vllm`, `tilelang`, `torch_extensions`, `nv_compute` (nvcc), `deep_gemm`, `xdg`, `wandb`,
  `wandb_cache`, `hf`, `torch`, `flashinfer` (via `FLASHINFER_WORKSPACE_BASE`), and `cutlass`
  (`CUTE_DSL_CACHE_DIR` — nvidia-cutlass-dsl, imported by vLLM/FlashInfer on every run).
- **FlashInfer gotcha (caught + fixed):** FlashInfer ignores a generic `FLASHINFER_CACHE_DIR`; it
  derives its JIT dir from `FLASHINFER_WORKSPACE_BASE` (defaulting to `$HOME`). Verified its JIT dir
  now resolves under the umbrella (`…/flashinfer/.cache/flashinfer/…/cached_ops`). It also ships AOT
  cubins, so most ops don't JIT at all.
- **No runtime package installation** — all Python deps (incl. deep-gemm) are baked at build time.
  Only *kernel JIT* (deep-gemm/TileLang/FlashInfer/inductor) happens at runtime, compiling in place
  with the baked nvcc into the writable umbrella — allowed by the deployment.

## 7. Runtime user

- Runs as **root** (`id -un` → `root`, uid 0) throughout — no dependency on a non-root uid. On
  Singularity the `USER` directive is ignored anyway (runs as the invoking host user); all baked
  files are world-readable/executable (e.g. the nvcc wrapper is `0755`), so either uid works.

---

## 8. Adversarial audit

Two rounds of parallel adversarial sub-agents audited the image + `dev` branch against the real
image/repo (GPU tests, real container builds). **Full findings, root causes, and the action plan
are in `AUDIT.md`.** Headlines:
- **CRITICAL (pending): `worker/__init__.py` ImportErrors the rollout worker** — a bad-merge on
  `main` kept two LoRA-monkeypatch calls that upstream deleted for vLLM 0.24, breaking olmo3_sink
  registration in every vLLM worker, on any version. See AUDIT §A.
- **vLLM 0.24 upgrade (pending):** the base image lags at 0.23; the validated path is
  `uv sync --locked --no-dev` (copy full workspace incl `deps/`, sync before deep-gemm; this
  replaces the round-2 overlay-guard). See AUDIT §B.
- **Fixed round-2:** the `CUDA_HOME` host-nvcc bypass (§5), the cutlass cache leak (§6), the
  build-file version-control gap, and a round-1 `ptxas` PATH hole.

Verified **clean**: sink gradients (fp64 brute-force, 9 regimes — a missing-α bug would show
40–130% error; the kernel sits at bf16 noise), rebase/dev-branch integrity (`range-diff` identical),
fork ≡ upstream `pyproject.toml`+`uv.lock` (identical blob hashes). The `olmo3_sink` vLLM integration
is **model registration on vLLM's native `Attention(sinks=)`, not a sink patch**.

---

## Re-running the checks

The in-container probes above are quick (`docker run --rm [--gpus all] …`). The GPU gradient-parity
tests are the pytest suite:
```
docker run --rm --gpus all -v $PWD/tests/...:/tmp/... --entrypoint /app/.venv/bin/python \
  chankhavu/aimo-opd:v1 -m pytest test_olmo3_sink_kernels.py test_olmo3_sink_guards.py
```
FA3 cases require a Hopper (sm_90) host; FA2/guards run on any CUDA GPU (sm_80+).
