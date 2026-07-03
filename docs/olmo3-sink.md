# Olmo3Sink

Olmo3Sink is the OLMo3 training/inference path used for proof-reasoning experiments. It starts from the Hugging Face OLMo3 architecture and adds trainable attention sinks plus packed-sequence metadata reuse for FlashAttention backends.

## Code Layout

| Path | Purpose |
|---|---|
| `src/prime_rl/trainer/models/olmo3_sink/configuration_olmo3_sink.py` | `Olmo3SinkConfig`, registered as `model_type = "olmo3_sink"`. |
| `src/prime_rl/trainer/models/olmo3_sink/modeling_olmo3_sink.py` | Trainer-side model implementation with attention sinks and OLMo3 per-layer RoPE handling. |
| `src/prime_rl/trainer/models/olmo3_sink/vllm_adapter.py` | vLLM adapter with packed `qkv_proj`, packed `gate_up_proj`, and per-head sink loading. |
| `src/prime_rl/trainer/models/olmo3_sink/converting_olmo3_sink.py` | Layer conversion for vLLM kernel-format weight transfer, including optional FP8 quantized transfer. |

`Olmo3SinkConfig` and `Olmo3SinkForCausalLM` are registered in Prime-RL's custom model mapping, so `[trainer.model] impl = "custom"` loads the trainer-side implementation directly.

## Training Configuration

Use the custom trainer model implementation for Olmo3Sink:

```toml
[trainer.model]
impl = "custom"
attn = "olmo3_sink_fa3"
cp = 2
cp_style = "ulysses"
fp8 = false
```

`attn` must be `"olmo3_sink_fa3"` (or `"eager"` for a CPU/debug reference). `Olmo3SinkAttention.forward` whitelists only those two backends and raises on the generic ones (`flash_attention_2`, `flash_attention_3`, `sdpa`), because they silently drop the `s_aux` sink logit — a sink-free run with no error. The `olmo3_sink_fa3` interface applies the sink in-kernel (patched FA3) or via a post-processing fallback on stock FA3.

Context parallelism (`cp`) is useful for long contexts on dense 32B models. The current 4xH200 smoke layout uses `cp = 2` over two trainer GPUs.

## Attention Sink Implementation

The per-head learnable sink (`self_attn.sinks`, gpt-oss style) is applied in **two independent places** — the trainer and the vLLM rollout — using separate kernels. Both apply the sink; neither is a fallback for the other.

### Trainer

`attn = "olmo3_sink_fa3"` routes attention through `src/prime_rl/trainer/models/olmo3_sink/fa3_sink_kernel.py`, which selects one of two paths at runtime based on the installed FlashAttention-3:

- **In-kernel** — when a patched FA3 whose `flash_attn_3::fwd` op accepts a `sink` argument is installed (`_fa3_fwd_accepts_sink()` returns `True`). The sink is applied in the kernel epilogue; FA3's native backward plus a `dsink` reduction supply the gradients.
- **Post-processing** — on stock FA3, where the op has no `sink` argument. Stock FA3 returns the sink-free output `o` and `lse_base = log Σ_j exp(s_j)`; the sink is then applied in PyTorch as `o · exp(lse_base − logaddexp(lse_base, sink))`. This forward is exact for any packing; the backward reuses the same FA3 native backward + `dsink` as the in-kernel path.

The path is selected automatically — installing a patched FA3 switches the trainer to the in-kernel path with no config or code change. The two paths are numerically identical up to **one extra bf16 rounding** on the attention output: the post-processing path quantizes `o` to bf16 before the scale, whereas the in-kernel path scales in fp32 and rounds once. The difference is unbiased and ~2⁻⁸.

Training backends are whitelisted in `Olmo3SinkAttention.forward`: `olmo3_sink_fa3` (the sink-aware FA3 kernel above), `olmo3_sink_fa2` (the same post-processing sink on FlashAttention-2, for non-Hopper GPUs — see below), and `eager` (an fp32 sink-aware reference, `eager_attention_forward_with_sink`). Generic backends (`flash_attention_2`, `flash_attention_3`, `sdpa`) raise, because they silently drop the sink logit. `torch`'s `flex_attention` can also apply the sink via `score_mod` (autograd-native, any arch) but is not currently registered.

#### `olmo3_sink_fa2` (non-Hopper debug)

FA3 runs only on Hopper (sm_90). `fa2_sink_kernel.py` provides the **same exact post-processing sink** on FlashAttention-2, which ships cubins for `sm_80/86/90/100/120` — so the sink can run on Ampere or Blackwell (e.g. sm_120) for debugging. Select it with `attn = "olmo3_sink_fa2"`. Its forward is identical to the FA3 fallback (`o · exp(lse_base − logaddexp(lse_base, sink))`), and the backward reuses FA2's native varlen backward fed the sink-inclusive `lse` plus the same `dsink`. It is **single-GPU only** — there is no FA2 Ulysses wrapper, so `cp > 1` with `olmo3_sink_fa2` raises. Forward + `dsink` parity against an fp32 reference (and that a strongly-negative sink reduces to plain FA2) is covered by `tests/unit/train/models/test_olmo3_sink_kernels.py`.

### Rollout (vLLM)

`vllm_adapter.py` applies the sink through vLLM's native attention layer, `Attention(..., sinks=self.sinks)` — vLLM's own gpt-oss sink support. It does **not** use the trainer's `fa3_sink_kernel.py`, so the rollout sink is independent of the trainer's in-kernel/post-processing selection (and independent of whether a patched FA3 is installed).

### OPD consistency

OPD requires the trainer and rollout to produce consistent per-token logprobs. Both sides apply the sink, so the on-policy property holds. They use different attention kernels (trainer FA3 vs vLLM's backend), so a small numerical difference at the sink exists on top of the usual trainer-vs-vLLM kernel difference; it is within OPD's tolerance. A patched (in-kernel, fp32) trainer path nudges the trainer's numerics slightly closer to vLLM's in-kernel sink but is not required. The step-1 `assert_sink_grad_nonzero` canary guards against a trainer configuration that silently dropped the sink.

## FP8 Support

Trainer FP8 is enabled with:

```toml
[trainer.model]
fp8 = true
```

This uses Prime-RL's existing DeepGEMM blockwise FP8 path by replacing eligible `nn.Linear` modules with `Float8BlockwiseLinear`. It applies to dense OLMo3Sink projections such as attention and MLP linears. Layer norms, embeddings, LM head, and `self_attn.sinks` stay in their normal dtype.

For vLLM inference, use vLLM quantization:

```toml
[inference.vllm_extra]
quantization = "fp8"
```

If NCCL quantized weight transfer is enabled, Olmo3Sink emits vLLM adapter names directly:

- `self_attn.qkv_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_up_proj.weight`
- `mlp.down_proj.weight`
- matching `*.weight_scale_inv` tensors for FP8 weights

## Current 4xH200 OPD Layout

The current Modal test layout is:

| GPU | Role |
|---|---|
| 0 | policy vLLM rollout server |
| 1-2 | trainer with `cp = 2` |
| 3 | frozen OPD teacher vLLM server |

Known successful smoke:

- Dataset: `submissions-instructions/test.csv`
- Context: 2,048
- Trainer: 2 GPUs, `cp = 2`, `cp_style = "ulysses"`
- Policy vLLM: 1 GPU, FP8
- Teacher vLLM: 1 GPU, FP8
- Result: one trainer step completed, peak trainer memory about 99.7 GiB.

Current longer-context test command:

```bash
bash /workspace/submissions-instructions/operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

Default settings in that command:

- Dataset: `/workspace/submissions-instructions/imo_data_1959_2024.csv`
- Columns: `question` and `solution`
- Context length: 16,384
- Rollout max completion tokens: 12,288
- Batch size: 2
- Group size: 2
- Optimizer: Muon
- Trainer FP8: enabled
- Policy/teacher vLLM quantization: FP8

Override these from the shell when needed:

```bash
PRIME_OPD_CTX_LEN=16384 \
PRIME_OPD_COMPLETION_TOKENS=8192 \
MAX_TRAIN_STEPS=1 \
bash /workspace/submissions-instructions/operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

## Practical Notes

- Keep `--fetch-update` enabled in submission-side commands so Modal and server runs pick up the latest `submissions-instructions` and `prime-rl` commits.
- For first-pass debugging, keep `wandb_mode = "disabled"` and run one step.
- Increase rollout context before increasing batch size. For proof data, long completions are usually the first pressure point.
- If vLLM memory is tight, lower `max_num_seqs`, `max_num_batched_tokens`, or `rollout_max_completion_tokens` before changing the trainer layout.
