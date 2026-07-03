"""CPU-safe guards for the verified Olmo3Sink issues (see ISSUES.md).

Covers:
- BUG 2: `Olmo3SinkAttention.forward` must hard-raise on generic attention
  backends that would silently drop the `s_aux` sink logit.
- BUG 4: the vLLM kernel-format conversion emits the *full* `[num_heads]` sink
  (never TP-sharded, never FP8-quantized). This is exactly the tensor the NCCL
  weight-transfer path copies verbatim, so it is only shape-compatible with a
  vLLM worker at `tp == 1`. The test pins that contract so any future TP-sharding
  fix is a deliberate, reviewed change rather than a silent shape drift.
"""

from __future__ import annotations

import pytest
import torch

from prime_rl.trainer.models.olmo3_sink.configuration_olmo3_sink import Olmo3SinkConfig
from prime_rl.trainer.models.olmo3_sink.converting_olmo3_sink import convert_layer_to_vllm_kernel
from prime_rl.trainer.models.olmo3_sink.modeling_olmo3_sink import Olmo3SinkAttention


def _tiny_config(attn_impl: str) -> Olmo3SinkConfig:
    config = Olmo3SinkConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        intermediate_size=16,
        vocab_size=32,
        max_position_embeddings=16,
        layer_types=["full_attention"],
        attention_dropout=0.0,
    )
    config._attn_implementation = attn_impl
    return config


def _attention_inputs(config: Olmo3SinkConfig):
    batch, seq = 1, 4
    hidden = torch.randn(batch, seq, config.hidden_size)
    # cos=1, sin=0 -> identity RoPE, so we exercise only the backend dispatch/guard.
    cos = torch.ones(batch, seq, config.head_dim)
    sin = torch.zeros(batch, seq, config.head_dim)
    return hidden, (cos, sin)


@pytest.mark.parametrize("attn_impl", ["sdpa", "flash_attention_2", "flash_attention_3", "fa4"])
def test_generic_backend_is_rejected(attn_impl):
    """BUG 2: generic backends may drop `s_aux`, so the forward must refuse them."""
    config = _tiny_config(attn_impl)
    attn = Olmo3SinkAttention(config, layer_idx=0)
    attn.sinks.data.fill_(0.0)
    hidden, position_embeddings = _attention_inputs(config)

    with pytest.raises(ValueError, match="attention sinks"):
        attn(hidden, position_embeddings=position_embeddings, attention_mask=None)


def test_eager_backend_is_allowed():
    """BUG 2: the sink-aware eager backend is whitelisted and must run."""
    config = _tiny_config("eager")
    attn = Olmo3SinkAttention(config, layer_idx=0)
    attn.sinks.data.fill_(0.0)
    hidden, position_embeddings = _attention_inputs(config)

    out, _ = attn(hidden, position_embeddings=position_embeddings, attention_mask=None)
    assert out.shape == hidden.shape


def test_vllm_kernel_conversion_emits_full_unsharded_sinks():
    """BUG 4: sinks are emitted at full `[num_heads]`, unsharded and unquantized.

    The NCCL kernel-transfer path copies this tensor verbatim into the vLLM param,
    whose shape is `[num_heads / tp]`. Emitting the full tensor is therefore only
    valid at `tp == 1`; pin the shape so a TP-sharding fix is an explicit change.
    """
    num_heads = 8
    sinks = torch.arange(num_heads, dtype=torch.float32)
    state_dict = {
        "model.layers.0.self_attn.sinks": sinks,
        "model.layers.0.self_attn.q_norm.weight": torch.ones(num_heads),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(num_heads),
    }

    out = convert_layer_to_vllm_kernel(state_dict, layer_idx=0)

    emitted = out["model.layers.0.self_attn.sinks"]
    assert emitted.shape == (num_heads,)
    assert torch.equal(emitted, sinks)


def test_vllm_kernel_conversion_keeps_sinks_unquantized_under_fp8():
    """BUG 4 corollary: the `ndim == 2` FP8 filter keeps the 1-D sink in full precision."""
    num_heads = 8
    sinks = torch.arange(num_heads, dtype=torch.float32)
    # 1-D-only state dict: no 2-D weight is present, so no FP8 kernel is invoked.
    state_dict = {"model.layers.0.self_attn.sinks": sinks}

    out = convert_layer_to_vllm_kernel(state_dict, layer_idx=0, quantize_fp8=True)

    emitted = out["model.layers.0.self_attn.sinks"]
    assert emitted.dtype == torch.float32
    assert torch.equal(emitted, sinks)
    assert "model.layers.0.self_attn.sinks_scale_inv" not in out
