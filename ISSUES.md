# Olmo3Sink × prime-rl: verified bugs and required fixes

**Scope.** Fork: `nguyen599/prime-rl` (Olmo3Sink port on top of `PrimeIntellect-ai/prime-rl` main).
Model: OLMo3 + per-head learnable attention sinks (`self_attn.sinks`, `[num_heads]`), trained, sinks carry real
attention mass — they are **load-bearing** and cannot be ignored or ablated.
Pinned deps: `transformers==5.6.2` (from upstream `pyproject.toml`). Trainer model is a HF `Olmo3Attention`
subclass dispatching through `ALL_ATTENTION_FUNCTIONS`; rollout is a custom vLLM adapter
(`src/prime_rl/trainer/models/olmo3_sink/vllm_adapter.py`) that **does** apply sinks via vLLM's
`Attention(..., sinks=...)`.

All findings below were verified against the actual fork diff, the upstream repo, the unpacked
transformers 5.6.2 wheel, and the FA3 C++ patch (`fa3_attention_sink.patch`). Nothing here is speculative.

---

## BUG 1 (critical): Ulysses CP never wraps this model's attention — cp ≥ 2 silently trains the wrong model

**Documented config** (`docs/olmo3-sink.md`): `impl = "custom"`, `attn = "flash_attention_3"`, `cp = 2`,
`cp_style = "ulysses"`.

**Verified mechanism:**

1. `substitute_ulysses_attn` (upstream `src/prime_rl/trainer/models/layers/ulysses_attn.py`) patches
   `_compute_attention` on exactly three classes: prime-rl's internal `FlashAttention`,
   `AfmoeFlashAttention`, `Qwen3_5MoeGatedFlashAttention`. `Olmo3SinkAttention` is none of these
   (it is a HF subclass dispatching via `ALL_ATTENTION_FUNCTIONS`), so this patch never touches it.
2. `substitute_hf_ulysses_attn` (same file) does two things:
   - Monkey-patches the module attribute `transformers.modeling_flash_attention_utils._flash_attention_forward`.
     **Dead code in transformers 5.6.2**: `transformers/integrations/flash_attention.py` binds
     `_flash_attention_forward` at module top, and `modeling_utils.py` (line ~69) imports
     `flash_attention_forward` from it at transformers import time — before CP setup ever runs. This kills the
     module patch for **both** the `"flash_attention_2"` and `"flash_attention_3"` interface keys (both map to
     the same bound `flash_attention_forward` in 5.6.2's `_global_mapping`).
   - Re-registers **only** `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]`. Neither `"flash_attention_3"`
     (the documented config) nor the fork's own `"olmo3_sink_fa3"` key ever gets a Ulysses wrapper.
3. `setup_cp_params` (upstream `src/prime_rl/utils/cp.py`) still shards `input_ids`/`position_ids` across CP
   ranks and publishes full `cu_seqlens` into `ULYSSES_PARAMS` — which nothing on this model's path reads.
   `Olmo3SinkModel.forward` then derives varlen metadata from its **local sharded** `position_ids`.

**Failure signature:** no all-to-all ever happens; each CP rank runs attention over its local sequence shard
as if it were a complete sequence. RoPE positions look right (position_ids are sharded, not renumbered), loss
computes, memory fits, "one trainer step completed" — but **no token attends across shard boundaries**.
The existing smoke test cannot detect this. All cp ≥ 2 runs and their loss curves are invalid.

**Fix (~100 lines):** register a Ulysses-aware attention entry for `"olmo3_sink_fa3"`:

- all-to-all seq→head on q/k/v (reuse upstream `_all_to_all_seq_to_head` / `_all_to_all_head_to_seq`);
- slice the sink to the local head chunk: `sinks[rank*h_local : (rank+1)*h_local]` (head chunks map
  contiguously by rank in the existing a2a layout);
- call `fa3_varlen_attn_with_sink_kernel` with the **full** `cu_seqlens`/`max_seqlen` from `ULYSSES_PARAMS`
  (NOT the local metadata threaded by `Olmo3SinkModel.forward`);
- all-to-all head→seq on the output;
- backward: scatter the local `[H/cp]` `dsink` into a full-size zero tensor so cross-rank gradient reduction
  reconstructs the complete sink gradient. **Verify reduction semantics:** if CP ranks are folded into the
  data-parallel gradient reduction as *averaging* replicas, disjoint head slices need a `×cp` correction
  (or a sum-reduce). Make the parity test (below) a **gradient**-parity test to catch this.

---

## BUG 2 (critical): sinks are silently NOT applied in training under the documented `attn = "flash_attention_3"`

**Verified mechanism:**

- transformers 5.6.2 builds a `supports_mapping` from `inspect.signature(flash_attn_varlen_func)`
  (`modeling_flash_attention_utils.py`, `_lazy_define_process_function`, line ~217) and forwards `s_aux` to the
  kernel **only if** the wrapper signature declares `s_aux` or `learnable_sink` (line ~642). Otherwise `s_aux`
  is dropped **silently — no warning**.
- The FA3 patch (`fa3_attention_sink.patch`) modifies **only C++/CUDA** (`flash.h`, `flash_api.cpp`,
  `flash_fwd_launch_template.h`, `mainloop_fwd_sm90_tma_gmma_ws.hpp`, `softmax.h`), adding an optional trailing
  `sink` arg (default `None`) to `torch.ops.flash_attn_3.fwd`. The Python `flash_attn_varlen_func` wrapper
  signature is **not** extended.
- Therefore under `attn="flash_attention_3"`, even with the patched FA3 build installed, transformers drops
  `s_aux` and the patched op runs with `sink=None`: **training is sink-free**, `sinks.grad` is `None`.
- Compounding: the vLLM rollout adapter **does** load and apply sinks → systematic per-token
  train/rollout logprob mismatch (breaks the on-policy property of OPD).
- FA2 corollary (verified): stock FA2's wrapper has no sink parameter either, and the Ulysses FA2 wrapper
  swallows `s_aux` in `**kw`. So `attn="flash_attention_2"` is also silently sink-free, and as the code stands
  **no CP-enabled path applies sinks at all**.

**Fix:**

1. Pin config to `attn = "olmo3_sink_fa3"` everywhere (docs, TOMLs, launch scripts).
2. Add a hard guard in `Olmo3SinkAttention.forward` mirroring the existing `sdpa` raise: **whitelist** the
   sink-correct differentiable backends — `"olmo3_sink_fa3"`, the `fa3_sink.py` post-processing fallback,
   and `"eager"` — and raise on everything else (including `flash_attention_2` and `flash_attention_3`).
   Do not rely on transformers to warn; it won't.

---

## BUG 3 (minor, upstream): SWA window off-by-one in `substitute_hf_ulysses_attn`

Upstream `substitute_hf_ulysses_attn` uses `window_size = (sliding_window, sliding_window)`; the convention
everywhere else (upstream `substitute_ulysses_attn`, the fork's adapter, and HF's own
`_flash_attention_forward`, which uses `(sliding_window − 1, ...)`) is window-includes-self. One extra left
token of context if that path fires. Not currently exercised by this model. File as an **upstream** issue.

---

## BUG 4 (latent, loud): TP > 1 breaks sink weight transfer to vLLM

The vLLM adapter's checkpoint `load_weights` correctly TP-shards sinks (narrows to the local head slice), but
the NCCL quantized weight-transfer path applies tensors via bare `param.copy_(tensor)`
(upstream `src/prime_rl/inference/vllm/worker/weight_transfer.py`), and
`convert_layer_to_vllm_kernel` emits the **full** `[num_heads]` sinks tensor. At vLLM `tp ≥ 2` this is a shape
mismatch → hard crash (loud, not silent). Current single-GPU-per-server layout (tp=1) is fine.
Fix or assert `tp == 1` before scaling the rollout servers. (FP8 transfer is fine: the `ndim == 2` filter
correctly keeps sinks unquantized.)

---

## Required tests (gate all multi-GPU-trainer runs on these)

1. **cp=2 vs cp=1 parity** on a packed batch: compare per-token logprobs **and gradients** (including
   `sinks.grad`) against cp=1 and against the sink-aware eager reference
   (`eager_attention_forward_with_sink`, already in the fork — it is the exact fp32 reference).
2. **Post-step-1 assertion:** `sinks.grad is not None` and nonzero. This single check catches both Bug 1
   (no Ulysses → still passes, so it is necessary but not sufficient) and Bug 2 (sink dropped → fails).
3. Re-verify any completed single-GPU runs actually used `attn="olmo3_sink_fa3"` and not the documented
   `"flash_attention_3"` string — the latter trained sink-free with no error message.

## What is confirmed correct (do not touch)

- FA3 in-kernel sink math (forward epilogue + native-backward reuse) and the closed-form
  `dsink = −Σ_t exp(sink − lse')·⟨o, do⟩` (independently rederived — exact).
- `custom_op` + `register_fake` wrapping (torch.compile-safe, no graph breaks).
- Sink-aware eager reference path; per-layer-type RoPE (YaRN on full-attention layers only);
  meta-load rotary buffer reinit + unit test.
- Weight-sync content: `self_attn.sinks` is included in `convert_layer_to_vllm_kernel` passthrough and
  correctly excluded from FP8 quantization. vLLM adapter checkpoint loading TP-shards sinks correctly.
- Upstream OPD exists (`src/prime_rl/orchestrator/algo/opd.py` + `configs/debug/algorithms/opd*.toml`), so the
  frozen-teacher-vLLM layout rides on maintained code.
