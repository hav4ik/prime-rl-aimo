#!/usr/bin/env bash
# Build a standalone uv venv for OLMo-core SFT (phase-1), isolated from prime-rl's
# /app/.venv so the two transformers pins never collide (prime-rl needs ==5.6.2;
# OLMo-core tracks latest). Mirrors the validated recipe in
# sft-images/aimo-olmo3-sft (docker/base/Dockerfile.olmo-core-official,
# hav4ik/OLMo-core@olmo3-sft), retargeted to this image's torch 2.11+cu128 base
# and Hopper (sm_90) so the flash-attn kernels come from prebuilt wheels rather
# than the SFT recipe's sm_120 source builds.
#
# Runs in a cudnn-devel stage: transformer-engine still compiles from source.
set -euxo pipefail

OLMO_VENV="${OLMO_VENV:-/opt/olmo/.venv}"
OLMO_CORE_REPO="${OLMO_CORE_REPO:-https://github.com/hav4ik/OLMo-core.git}"
OLMO_CORE_REF="${OLMO_CORE_REF:-olmo3-sft}"
OLMO_CORE_DIR="${OLMO_CORE_DIR:-/opt/OLMo-core}"

TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_CU_INDEX="${TORCH_CU_INDEX:-https://download.pytorch.org/whl/cu128}"
FA3_INDEX="${FA3_INDEX:-https://download.pytorch.org/whl/test/cu128}"
# Same prebuilt FA2 wheel prime-rl uses (cu128 / torch2.11 / cp312, covers sm_90).
FA2_WHEEL="${FA2_WHEEL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl}"
TE_VERSION="${TE_VERSION:-2.9}"
LIGER_VERSION="${LIGER_VERSION:-0.6.4}"
RING_FLASH_ATTN_VERSION="${RING_FLASH_ATTN_VERSION:-0.1.8}"
# Target arch for the ONE kernel we compile from source (TransformerEngine).
# Default Hopper (H200); this is independent of the build box's own GPU, so a
# non-Hopper builder still produces sm_90 cubins. Add ";100" to also cover B200.
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# 1. Isolated venv against system python3.12 (matches prime-rl's interpreter so the
#    venv relocates cleanly when copied into the runtime image).
uv venv --python 3.12 --seed "${OLMO_VENV}"
PY="${OLMO_VENV}/bin/python"
pip() { uv pip install --python "${PY}" --no-cache-dir "$@"; }

pip --upgrade pip setuptools wheel packaging ninja

# 2. Torch core (cu128), pinned equal to prime-rl's lock so both venvs agree on the
#    CUDA userspace ABI baked into the wheels.
pip --index-url "${TORCH_CU_INDEX}" "torch==${TORCH_VERSION}" torchvision torchaudio

# 3. Flash-attn 2 — prebuilt wheel; Hopper (sm_90) is covered, so no source build.
pip "${FA2_WHEEL}"

# 4. Flash-attn 3 — stock, from the pytorch test index (same source prime-rl uses).
#    TE imports it as `flash_attn_3.flash_attn_interface`; the wheel exposes the
#    module as top-level `flash_attn_interface`, so shim a package alias if needed.
pip --index-url "${FA3_INDEX}" flash_attn_3
"${PY}" - <<'PY'
import importlib.util, os, pathlib, site
if importlib.util.find_spec("flash_attn_3") is None and importlib.util.find_spec("flash_attn_interface") is not None:
    sp = pathlib.Path(site.getsitepackages()[0])
    pkg = sp / "flash_attn_3"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "flash_attn_interface.py").write_text("from flash_attn_interface import *  # noqa: F401,F403\n")
    print("shimmed flash_attn_3.flash_attn_interface ->", pkg)
PY

# 5. TransformerEngine (fp8 path). Compiles from source -> needs nvcc (devel stage).
pip --no-build-isolation "transformer-engine[pytorch]==${TE_VERSION}"

# 6. ring-flash-attn + the transformers-5.x shim: its hf_adapter imports
#    is_flash_attn_greater_or_equal_2_10, which transformers 5.x removed. Without
#    the shim `import ring_flash_attn` raises and CP silently falls back.
pip --no-build-isolation "ring-flash-attn==${RING_FLASH_ATTN_VERSION}"
F="$("${PY}" -c 'import transformers.modeling_flash_attention_utils as m; print(m.__file__)')"
grep -q 'is_flash_attn_greater_or_equal_2_10' "${F}" || \
    echo "from transformers.utils import is_flash_attn_greater_or_equal_2_10  # shim: ring_flash_attn on transformers 5.x" >> "${F}"

# 7. liger-kernel.
pip --no-build-isolation "liger-kernel==${LIGER_VERSION}"

# 8. OLMo-core source (editable) + its SFT extras. grouped_gemm is intentionally
#    omitted: it is a MoE kernel and OLMo3 is dense.
git clone --depth 1 --branch "${OLMO_CORE_REF}" "${OLMO_CORE_REPO}" "${OLMO_CORE_DIR}"
pip -e "${OLMO_CORE_DIR}[beaker,wandb,transformers,fla,torchao,dion]"

# 9. Sanity: the venv imports the core stack. Pure-Python imports (torch, olmo_core)
#    must succeed and fail the build if not. The CUDA-extension imports are
#    best-effort: `docker build` has no GPU, and importing a Hopper-built TE on a
#    GPU-less builder can abort — so we report but do not gate the build on them.
"${PY}" - <<'PY'
import torch, olmo_core
print("olmo-core venv OK | torch", torch.__version__, "| olmo_core", olmo_core.__version__)
for mod in ("flash_attn", "flash_attn_interface", "flash_attn_3.flash_attn_interface", "transformer_engine"):
    try:
        __import__(mod)
        print(f"  import {mod}: ok")
    except Exception as exc:  # noqa: BLE001 -- GPU-less builder / lazy CUDA init
        print(f"  import {mod}: deferred ({type(exc).__name__}: {exc})")
PY
