# Container comparison — ours vs Nguyen's (Manh's)

Single source of truth for how our combined container (`chankhavu/aimo-opd-sft`, built from
`prime-rl-aimo` + `aimo-olmo3-sft`) differs from Nguyen Manh's (`aimo-proof-pilot`). Companion to
`AUDIT.md` and `CUDA_VERSIONING.md`.

## TL;DR — two philosophies
- **Ours:** everything **baked** (immutable, reproducible), **prime-rl stays bit-for-bit official**
  (cu128, stable vLLM 0.24), single-image. Built for a locked-down NII deploy.
- **Nguyen's:** **runtime-fetch** (clones + pip-installs prime-rl/verl/vLLM at container start),
  **multi-engine**, and **prime-rl deviated** onto cu130 + a vLLM dev nightly. Built as a
  competition-submission image (update code without rebuilding).

## Side-by-side

| Dimension | Ours (`aimo-opd-sft`) | Nguyen's (`aimo-proof-pilot`) |
|---|---|---|
| Fork lineage | `hav4ik/prime-rl-aimo` (fork of `nguyen599/prime-rl`) | `nguyen599/prime-rl` (fork of PrimeIntellect) |
| prime-rl (OPD) | **baked**, official stack | **runtime-fetched** at container start |
| olmo-core (SFT) | **baked**, same venv | **baked** (system Python) |
| open-instruct / verl | not baked (fetchable backups) | baked deps + runtime-fetched |
| Megatron | no | yes |
| **CUDA line** | **cu128** (both engines) | **cu130** |
| **vLLM** | **0.24.0+cu129** (stable, official) | **0.23.1rc1.dev699+cu130** (dev nightly, for cu130 compat) |
| transformers | **5.6.2** (prime-rl's official pin) | 5.8.1 (system choice) |
| torch | 2.11.0+cu128 | 2.11.0+cu130 |
| Env model | one `/app/.venv` (prime-rl + olmo-core) + tiny daemon venv | one system Python (everything together) |
| **Runtime installs** | **none** (all baked) | **yes** (git clone + pip at launch) |
| Base OS | Ubuntu 24.04 (prime-rl base) | Ubuntu 22.04 (pytorch base) |
| Entrypoint | **remote-shell daemon** (crash-resilient PID 1) | `train.py` (multi-engine launcher) |
| SFT recipe | native OLMo-core (AI2 base) | open-instruct-wrapped OLMo-core |
| olmo3_sink | nguyen's/Yi-Chia's + **our FA2 backend + audit fixes** | nguyen's/Yi-Chia's |
| Caches | `/tmp/imochallenge/cache/*` (tidy umbrella) | `/tmp/olmo_train_runtime_cache/*` (scattered) + `/cache` (chmod 777) |
| NII compiler safety | bind-safe `/opt/opd/cuda` nvcc, host dirs last | plain `/usr/local/cuda` (not bind-safe) |
| Worker bug (below) | **fixed** (or being fixed) | **present** (his `main` == `6ee9a5dc`) |

## Why we diverge (the load-bearing decisions)
- **prime-rl is the primary vehicle → keep it exactly official.** Nguyen runs it on cu130 + a vLLM
  dev nightly (deviating from the stack PrimeIntellect tests). We keep cu128 / vLLM 0.24 / torch
  2.11 / transformers 5.6.2 unchanged. (Nguyen's simplicity comes *from* that deviation — see AUDIT.)
- **Bake, don't fetch.** Runtime clone+pip is where unforeseen errors live on a locked-down cluster;
  we bake everything so runtime writes hit only `/tmp` + `$HOME`.
- **One env, not two.** olmo-core is unpinned on transformers and works on prime-rl's 5.6.2 (it has
  `Olmo3Config`); both are cu128; olmo-core reuses prime-rl's flash-attn. The *only* real dep clash
  is `rich` (prime-rl needs ≥14 via textual; olmo-core's cached-path caps <14) — resolved by keeping
  prime-rl's `rich 15` and forcing cached-path onto it (verified safe; rich 14/15 are additive).

## vLLM version — the state to remember
- **Base image `primeintellect/prime-rl:main`: still 0.23.0** (publish lag — upstream bumped the
  lock but hasn't republished the image).
- **New prime-rl code (`main @ 6ee9a5dc`, synced 2026-07-03): pins 0.24.0+cu129** (upstream PR
  #2921, `b2b4f3f7`, 2026-07-02).
- **Ours:** upgrade the baked venv to 0.24 via `uv sync` (AUDIT item B) — stay on the official wheel.
- **Nguyen's:** overrides with the cu130 vLLM **dev nightly** because the stable 0.24 wheel is cu129
  (won't run on his cu130 torch). So he *runs* 0.23.1rc1.dev699, not 0.24.

## The worker bug (affects BOTH forks — brand new)
`src/prime_rl/inference/vllm/worker/__init__.py` calls `monkey_patch_LRUCacheWorkerLoRAManager()` and
imports `monkey_patch_skip_lora_module_warnings` — both **removed from `patches.py`** by upstream
`6836d325c` (dropped for vLLM 0.24's native inplace-load). Importing the worker package → hard
`ImportError` → the vLLM rollout worker dies → `Olmo3SinkForCausalLM` never registers → **OPD rollout
cannot load the model, on any vLLM version.**

- **Origin:** the 0.24-sync merge `6ee9a5dc`, dated **2026-07-03** (the sync you just pulled). Before
  it, `patches.py` still had the functions, so the worker imported fine.
- **Why Nguyen's OPD "runs great":** his W&B runs **predate this merge** (pre-0.24 code). His `main`
  now carries the same bug (`main == 6ee9a5dc`); his next OPD run on current main will hit it.
- **Fix:** finish upstream's refactor — drop the 2 stale calls + the dangling import from
  `worker/__init__.py`, keep `register_olmo3_sink_model()`. ~5-line deletion. Worth PR-ing to nguyen599.

## Bottom line
Same roots (shared `prime-rl` main, same olmo3_sink), opposite build philosophies. Ours trades
Nguyen's multi-engine/runtime-fetch flexibility for **reproducibility + an untouched official
prime-rl**, in a single baked image with a crash-resilient remote-shell entrypoint.

## Why Nguyen's 0.24 branch still runs — the nightly-vLLM speculation
Nguyen's prime-rl `main` (`6ee9a5dc`) is on the **0.24 code** — `patches.py` hard-imports
`vllm.parser.{qwen3,minimax_m2,engine}` (0.24-era modules) *and* the worker `ImportError` bug is
present. Yet his multi-engine plugin loads. The likely reason: **he runs a vLLM *nightly*
(`0.23.1rc1.dev699+gf5a8d7337`, built from vLLM `main` at `f5a8d733`).** Despite the `0.23.1rc1`
version *string*, a nightly off vLLM `main` is typically **ahead of the 0.24 release branch point**,
so it **already contains `vllm.parser.*`** — which makes the 0.24-era imports resolve. So the
nightly sidesteps the *plugin-load* issue even though his branch is on 0.24 code. **(Speculation —
not verified against the wheel's contents; the version string is misleading.)**

Two caveats:
- The nightly does **not** fix the worker `ImportError` — that's pure prime-rl code. His past W&B
  runs avoided it by **predating the `6ee9a5dc` merge**, not via the nightly. His next OPD run on
  current `main` will still hit it.
- Running a nightly is exactly the **deviation from an official *released* vLLM** we chose to avoid.

## Our resolution: `dev-vllm023` (stable 0.23, no nightly, no bugs)
We branch from the fork's pre-0.24-merge state (`6ee9a5dc^1`) — **prime_rl 0.6.0 @ vLLM 0.23**,
worker-consistent, **zero `vllm.parser` imports**, entry point (`transformers_v5_compat`) matching
the base image's dist-info. On the stable 0.23 base it just works — no nightly, no `uv sync`, no
worker fix. **Verified in `aimo-opd-sft:v2`:** plugin loads, worker package imports, olmo3_sink
registers, olmo-core coexists in `/app/.venv` (rich 15, transformers 5.6.2, torch 2.11+cu128).
