# OPD image — adversarial audit & action plan

Record of the two-round adversarial audit of `chankhavu/aimo-opd:v1` (the prime-rl OPD image)
and the `dev` branch: what was checked, what broke, what's fixed, and what's still pending.
Companion to `IMAGE_INTEGRITY.md` (verification log) and `CUDA_VERSIONING.md` (CUDA rationale).

## How it was audited
Two rounds of **parallel adversarial sub-agents**, each mandated to *break* a specific set of
claims against the real image + repo (GPU tests, real container builds over the network) — not
to reason abstractly.
- **Round 1** (5 agents): overlay/deps + vLLM bump, compiler toolchain, cache/immutable-FS, sink
  numerics, git/branch state.
- **Round 2** (5 agents, after the round-1 fixes + the upstream vLLM 0.23→0.24 bump): Dockerfile
  round-2 edits, `uv sync` 0.24 upgrade, vLLM sink/registration reality, regression, and the
  worker-bug follow-up.

## Status at a glance
| # | Item | Severity | Status |
|---|---|---|---|
| A | `worker/__init__.py` ImportError (rollout worker dead) | **CRITICAL** | **PENDING** |
| B | vLLM 0.24 upgrade via `uv sync` | upgrade | **PENDING** (recipe validated) |
| C | Dead FlashInfer assertion in `Dockerfile.opd` verify RUN | MED | **PENDING** |
| D | Doc/header vLLM-version incoherence (0.23 vs 0.24) | MED | **PENDING** |
| E | `LD_LIBRARY_PATH` lists host-bindable `/usr/local/cuda/lib64` | LOW | PENDING (optional) |
| F | `flash_attn/cute` uses `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` (outside umbrella) | LOW | PENDING (optional) |
| 1 | Overlay base/lock skew guard | HIGH | FIXED r2 (obsoleted by B) |
| 2 | `CUDA_HOME` host-nvcc bypass (torch + FlashInfer) | HIGH | FIXED r2 |
| 3 | `nvidia-cutlass-dsl` cache leak | MED | FIXED r2 (`CUTE_DSL_CACHE_DIR`) |
| 4 | Image-build files untracked / disk-only | HIGH | FIXED r2 (committed to `dev`) |
| 5 | Submodule `research-environments` rewind | MED | FIXED r2 |
| 6 | round-1 `ptxas` resolving to host `/usr/local/cuda/bin` | — | FIXED r2 (PATH reorder) |

Verified **clean** (no action): sink gradients, native-sink-matches-trainer, git/dev integrity,
fork≡upstream `pyproject`+`uv.lock`, overlay namespace, FP8 install.

---

## A. CRITICAL — `worker/__init__.py` ImportErrors the rollout worker
**Symptom.** `src/prime_rl/inference/vllm/worker/__init__.py` imports
`monkey_patch_skip_lora_module_warnings` (not defined in `patches.py`) and calls
`monkey_patch_LRUCacheWorkerLoRAManager()` (defined nowhere). vLLM loads this package in every
rollout **worker** process (via `worker_extension_cls` → `NCCLWeightUpdateWorker`), so the import
raises `ImportError`, the worker dies, and `Olmo3SinkForCausalLM` never registers there — **rollout
model-load fails on any vLLM version.** (The API-server-side `register_olmo3_sink_model()` in
`server.py` does not help worker load.)

**Root cause — a bad merge on `main`, not the feat branch.**
- Upstream `6836d325c` ("refactor(inference): drop 2 LoRA reload monkeypatches for vLLM 0.24
  native load_inplace") **deleted** both functions and their `worker/__init__.py` calls.
- The fork's `3055c0610` ("Add OLMo3Sink model support") added `register_olmo3_sink_model()` to
  `worker/__init__.py` while those LoRA calls still existed (pre-0.24).
- The `main` sync-merge `6ee9a5dc` mis-resolved the conflict: it took upstream's `patches.py`
  (functions gone) but kept the fork's `worker/__init__.py` (calls retained). Upstream's own
  `worker/__init__.py` is self-consistent; ours is not.

**Fix** — finish upstream's refactor: drop the two stale LoRA calls + the dangling import, keep
`register_olmo3_sink_model()` and the still-valid patches (`minimax_m2_for_lora`, `no_moe_lora`,
`fp32_lm_head`). The corrected file equals upstream's 0.24 `worker/__init__.py` plus the one
olmo3_sink registration line. This is version-independent and should land regardless of B.

---

## B. vLLM 0.24 upgrade — `uv sync --locked` is correct, with two mandatory corrections
**Why an upgrade is needed.** Upstream bumped the lock to vLLM **0.24.0** (PR #2921) but its
published image `primeintellect/prime-rl:main` still ships **0.23.0** (a publish lag). Overlaying
our 0.24-targeted code onto the 0.23 base leaves `prime_rl`'s dist-info registering the old plugin
entry point (`transformers_v5_compat`, since renamed to `apply_shared_vllm_patches`) and the plugin
hard-imports 0.24-only `vllm.parser.*`. So we upgrade the venv ourselves rather than wait.

**The naive plan fails / is destructive — two corrections are mandatory** (proven by real builds):
1. **Copy the full `deps/` tree.** uv validates all 70 `[tool.uv.sources]` path-deps during
   workspace discovery — including two *new* envs (`arxivmath_v1`, `s1_deepresearch_v1`) absent
   from the base. Omit `deps/` and sync hard-fails: `Distribution not found at: .../arxivmath_v1`.
2. **Run `uv sync` BEFORE the deep-gemm install.** Exact-mode sync uninstalls `deep-gemm` (it lives
   in the unselected `disagg`/`all` extra). Order sync first, or pass `--inexact`.
- Note: exact sync strips the 64 RL-env editables (69→5 editables). **Harmless for OPD inference**
  (even shrinks the image). Add `--extra envs` only if an env is needed.
- Cost: **~14 min + network** (a real layer, dominated by the ~450 MB `0.24.0+cu129` wheel), not a
  2 MB overlay. It **replaces** the round-2 overlay-guard (sync *performs* the upgrade the guard
  merely asserted — drop the guard).

**Verified after the corrected sync:** `vllm 0.24.0+cu129` (the locked GitHub wheel, *not* PyPI
generic), `torch 2.11.0+cu128` unchanged, `transformers 5.6.2`, `flashinfer-python 0.6.12`,
dist-info entry point refreshed to `apply_shared_vllm_patches` and it loads, `apply_shared_vllm_patches()`
runs with no ImportError, `Olmo3SinkForCausalLM in ModelRegistry`.

**Rejected alternative.** `uv pip install vllm==0.24.0` pulls the **PyPI-generic** 0.24 (not
`+cu129`) and drags CUDA-13 runtime libs onto the cu128 torch base — a lock divergence with real
runtime-skew risk. Do not use it.

**`Dockerfile.opd` change** — replace the `COPY src/prime_rl … + overlay-guard` block with, placed
**before** the deep-gemm/cmake blocks:
```dockerfile
COPY --chown=root:root pyproject.toml uv.lock /app/
COPY --chown=root:root src/prime_rl /app/src/prime_rl
COPY --chown=root:root packages /app/packages
COPY --chown=root:root deps /app/deps
RUN cd /app && uv sync --locked --no-dev
```

---

## C. Dead FlashInfer assertion (`Dockerfile.opd` verify RUN)
The verify RUN's FlashInfer half does `fe.cuda_home if hasattr(fe,'cuda_home') else os.environ['CUDA_HOME']`
— `flashinfer.jit.env` has no `cuda_home` attribute, so it always re-reads the env var we just set:
a tautology that checks nothing. **Runtime is fine** (FlashInfer's real resolver
`flashinfer.jit.cpp_ext.get_cuda_path()` returns `/opt/opd/cuda`); only the check is dead. Fix:
assert on `get_cuda_path()`.

## D. Doc/header vLLM-version incoherence
`CUDA_VERSIONING.md` says 0.24 throughout; `IMAGE_INTEGRITY.md §4` still says 0.23; `Dockerfile.opd`
header still claims "base already contains the exact venv (…vLLM 0.23…)" directly above the guard
that aborts *because* base 0.23 ≠ lock 0.24. Reconcile once B lands (image becomes genuinely 0.24).

## E/F. Optional hardening
- **E** — `LD_LIBRARY_PATH` still lists host-bindable `/usr/local/cuda/lib64` (the CUDA_HOME
  *compiler* fix didn't touch the runtime-lib path). Mitigated: it's last and torch's venv-RPATH
  `nvidia-*-cu12` libs win. Optionally prepend `/opt/opd/cuda/lib64`.
- **F** — `flash_attn/cute` + `vllm_flash_attn/cute` read `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`
  (default outside the umbrella, still writable on NII). Optionally add it under `IMO_CACHE_ROOT`.

---

## olmo3_sink × vLLM — it is model registration, NOT a sink patch (clarification)
Correcting earlier loose framing: **vLLM natively supports attention sinks** —
`Attention(..., sinks=...)` (gpt-oss `s_aux`, predates 0.24). The olmo3_sink "vLLM code" does **not**
patch vLLM's attention; `vllm_adapter.py` builds vLLM's native `Attention(sinks=self.sinks)`, and
`register_olmo3_sink_model()` only does `ModelRegistry.register_model("Olmo3SinkForCausalLM", …)` —
it registers the *model class* vLLM doesn't ship, wiring the checkpoint's `sinks` weights into the
native kernel.

- **Rollout matches training numerically:** vLLM's `s_aux` = the trainer's
  `out = out_base · exp(lse_base − logaddexp(lse_base, sink))` (a softmax column of value 0). Same
  formula, fp32 softmax both sides.
- **Constraint:** vLLM sinks require Hopper/FA3 (compute ≥ 9.0) — it *asserts* on non-Hopper rather
  than silently diverging.
- The `apply_shared_vllm_patches` "plugin" is prime-rl's **unrelated** bundle of monkey-patches
  (qwen3, minimax, KV-xfer, transformers-compat…); olmo3_sink registration is merely tacked onto
  its end and *also* called directly in `server.py`/`worker/__init__.py`. That direct path is what
  bug **A** breaks.

---

## Findings verified CLEAN (no action)
- **Sink gradients** (round 1): fp64 brute-force across 9 regimes (MHA/GQA, causal/non-causal,
  sliding-window, packed varlen); a controlled missing-α bug shows 40–130% error, the kernel sits
  at bf16 noise. All four grads (dq/dk/dv/dsink) exact.
- **CUDA_HOME fix (2)**: GPU-proven — host `nvcc` `chmod 000`'d, `load_inline` still compiles off
  `/opt/opd/cuda`. Toolkit copy is complete + bind-safe; deep-gemm/torch/FlashInfer resolve the
  right compilers; PATH host-safe.
- **Overlay guard (1)**: version-parse robust despite the dual (x86_64/aarch64) lock entries.
- **Git/`dev`**: rebase content-preserving (`range-diff` identical), `dev` = main+feat, local ==
  origin, commit `358479bc4` has exactly the intended files, submodule clean.
- **fork ≡ upstream** `pyproject.toml` + `uv.lock` (identical blob hashes) — the only skew is
  upstream's own published-image lag.
