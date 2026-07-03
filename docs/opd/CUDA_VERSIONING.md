# CUDA versioning in the AIMO OPD image

How CUDA toolkit/runtime versions are chosen and mixed in `chankhavu/aimo-opd` (the
prime-rl OPD image), and why. TL;DR: the image is **CUDA 12.8 (cu128) at its core**,
carries a **tiny CUDA-13 sliver** for one dependency, and uses **cu12 TransformerEngine** —
and it **runs unchanged on CUDA-13 hosts**. "cu13-only" is neither possible (cleanly) nor
beneficial.

---

## 1. The version matrix (what's actually baked)

| Component | Version | CUDA build | Source |
|---|---|---|---|
| **torch / vision / audio** | 2.11.0 | **cu128** | `pytorch-cu128` index (prime-rl lock) |
| **vLLM** | 0.24.0 | **cu129** (abi3) | prebuilt wheel; runs on the cu12.8 runtime |
| **flash-attn (FA2)** | 2.8.3 | **cu128** / torch2.11 | prebuilt wheel; cubins `sm_80/90/100/120` |
| **flash_attn_3 (FA3)** | 3.0.0 | cu128 | `pytorch-cu128-test` index; Hopper (sm_90) only |
| **flash-attn-4 (FA4/cute)** | 4.0.0b11 | JIT | git build |
| **flashinfer-python** | 0.6.12 | cu12 | prebuilt |
| **deep-gemm (standalone)** | 2.5.0+891d57b | **cu13** | PrimeIntellect wheel — see §4 |
| deep-gemm (vLLM-bundled) | in `vllm/third_party` | cu12 | ships with vLLM |
| **TransformerEngine** | 2.x | **cu12** | see §5 |
| tilelang / quack / liger / ring-flash-attn | — | cu12 / JIT | prime-rl deps |
| `nvidia-*-cu12` runtime libs | 12.8.x | cu128 | torch's bundled wheels |
| transformers | 5.6.2 | — | pinned by vLLM 0.24 (not olmo-core's 5.8.x) |

**Base images:** builder `nvidia/cuda:12.8.1-cudnn-devel`, runtime `nvidia/cuda:12.8.1-base`.
The runtime carries a **trimmed CUDA-12.8 toolkit** (nvcc/nvvm/ptxas + headers + libcudart/
libnvrtc, for TileLang/FlashInfer/deep-gemm runtime JIT).

---

## 2. Why cu128 is the core (and not cu13)

The **entire prime-rl prebuilt wheel set is cu12-based**, and there is no clean cu13
equivalent:

- **vLLM 0.24.0 is published only as `+cu129`** (abi3). There is **no `+cu13` vLLM 0.24
  wheel** — going cu13 would mean building vLLM from source (huge).
- **torch is pinned to 2.11.0+cu128**, and FA2/FA3 are the cu128 prebuilt wheels.

The image is built by **overlaying this fork's `src/prime_rl` onto `primeintellect/prime-rl:main`**
(the fork's `pyproject.toml` + `uv.lock` are byte-identical to upstream — see the overlay in
`Dockerfile.opd`). Re-resolving the whole tree against cu13 indices would:

1. **Abandon the overlay-on-upstream strategy** → build from scratch (slow, and every kernel
   becomes a source-build risk).
2. **Hit missing cu13 wheels** — vLLM 0.24 being the blocker.

And critically, **there is no runtime benefit** (§3). `nguyen599/aimo-proof-pilot` *is* a
cu130 image, but it only works because it **runtime-clones + pip-installs prime-rl into
`/tmp`** — which violates the immutable requirement (all deps baked, nothing installed at
runtime). For a self-contained OPD image, cu128 is what prime-rl's wheels dictate.

---

## 3. cu128 runs on CUDA-13 hosts (the key fact)

A GPU container brings its **own CUDA userspace** (torch's bundled `nvidia-*-cu12` libs); only
the **driver** comes from the host. NVIDIA drivers are **backward compatible** — a newer
driver runs applications built against older toolkits. So:

- **cu128 image on a CUDA-13 host (driver ≥580)** → runs. ✅ (backward compat — the direction
  we need for NII/VastAI cu13 clusters)
- The direction that would break is the opposite (a cu13-built image on an old cu12 driver).

**The only host requirement is the driver:** ≥ ~570 (the cu128 floor). Any CUDA-13 host clears
it. The host's CUDA *toolkit* version is irrelevant; nothing in the container reads it.

vLLM's `+cu129` tag is likewise just its *build* toolkit — the wheel links `libcudart.so.12`
and runs on the bundled cu12.8 runtime via CUDA 12.x intra-major ABI stability (the lock
resolves the actual runtime libs to 12.8.x, no cu129 toolkit is shipped).

---

## 4. The cu13 sliver — for deep-gemm, and only deep-gemm

`Dockerfile.cuda` copies exactly **three files** from `nvidia/cuda:13.0.1-runtime` into
`/opt/cuda13/lib64` and registers them with `ldconfig`:

```
libcudart.so.13   libnvrtc.so.13   libnvrtc-builtins.so.13
```

Why: the **standalone `deep_gemm`** wheel (PrimeIntellect's build, our FP8-blockwise path) is
compiled against the CUDA-13 toolkit, so its `_C.*.so` has `DT_NEEDED` on `libcudart.so.13` +
`libnvrtc.so.13` (verified with `ldd`). Because cu12 (`.so.12`) and cu13 (`.so.13`) have
distinct SONAME majors, both load in the same process with no collision — torch loads `.so.12`,
deep-gemm loads `.so.13`.

**deep-gemm needs *only* those two libs — not the full CUDA-13 runtime** — because its GEMM
kernels are **JIT-compiled at runtime**, not calls into cuBLAS/cuDNN. That is the whole reason
a 3-file sliver suffices. (Note: vLLM's *bundled* `deep_gemm` under `vllm/third_party` is a
separate, **cu12** build linking `.so.12`; the cu13 one is the prime-rl `Float8BlockwiseLinear`
path.)

deep-gemm is opt-in (`INSTALL_FP8`, and gated at runtime by `fp8=true` / `use_deep_gemm=true`).
Its JIT cache is `DG_JIT_CACHE_DIR=/tmp/imochallenge/cache/deep_gemm` (writable on the read-only rootfs).

---

## 5. TransformerEngine: cu12, not cu13

TE 2.16.1 publishes both `transformer-engine-cu12` and `transformer-engine-cu13`. We use the
**cu12** build, because:

- TE uses **cuBLAS and cuDNN** for its GEMMs/layernorms/attention. The **cu13** TE therefore
  needs the **full CUDA-13 runtime** (cuBLAS13, cuDNN-for-13, …) — which the sliver does **not**
  provide, and which we don't want to bake (GBs, no benefit).
- The **cu12** TE links the same cu12.8 libs torch already bundles → consistent with the cu128
  core.

This is the general rule: a cu13 dependency is fine **only if** it needs nothing beyond the
`libcudart.so.13`/`libnvrtc.so.13` sliver (like deep-gemm's JIT kernels). A cu13 dependency that
pulls cuBLAS13/cuDNN13 (like TE) must instead use its cu12 build.

TE is optional for OPD (the default layout uses the Muon optimizer; TE is only needed for the
fork's TE FusedAdamW optimizer, and imports lazily).

---

## 6. Immutable-FS note (nothing installed at runtime)

All Python dependencies are **baked at build time**; the container never pip-installs at
runtime. The one runtime-side activity is **JIT kernel compilation** by deep-gemm / TileLang /
FlashInfer on first use — that is *compilation in place*, not installation, and it needs only:

- **nvcc/nvrtc** — baked. A full cu12.8 toolkit is copied to a **bind-safe** `/opt/opd/cuda`
  (`CUDA_HOME`/`CUDA_PATH` point there) so torch + FlashInfer never resolve a host `nvcc`; deep-gemm
  uses the cu13 nvcc wrapper + cu13 nvrtc sliver. And
- a **writable cache** — every cache/JIT dir is namespaced under `/tmp/imochallenge/cache/<tool>`
  (the deployment guarantees `/tmp` and `$HOME` writable, and `/tmp` is the team's mounted storage,
  so we never assume it's empty; see the cache-dir `ENV` block in `Dockerfile.opd`).

If *zero* runtime compilation is ever required (not just zero installation), the JIT caches
would need pre-warming for the exact GPU arch — a separate step, not currently done.

---

## 7. Summary

- **Core = cu128**, dictated by prime-rl's pinned wheels (torch 2.11+cu128, vLLM 0.24+cu129,
  FA2/FA3 cu128). Not a free choice.
- **cu13 sliver** (3 libs) = deep-gemm's minimal need; it JITs its kernels, so no cuBLAS13/
  cuDNN13.
- **TE = cu12**, because cu13 TE would need the full CUDA-13 runtime.
- **Runs on CUDA-13 hosts** via driver backward-compat → no reason to build cu13.
- **"cu13-only" is blocked** by the missing cu13 vLLM 0.24 wheel and would force runtime
  installs (as `nguyen599/aimo-proof-pilot` does), violating the immutable requirement.
