# Copyright 2026 proof-pilot. Apache-2.0.
"""transformers attention-interface adapter for the FA2 sink (post-processing).

Registers `attn_implementation="olmo3_sink_fa2"`. Identical [B,H,S,D] -> varlen
[total,H,D] + cu_seqlens conversion as the FA3 adapter (`attention.py`), but backs
onto FlashAttention-2 so the sink runs on non-Hopper GPUs (Ampere sm_80/86, Blackwell
sm_100/sm_120) for debugging. See `fa2_sink_kernel.py`. Deliberately NOT registered in
the mask interface, so `create_causal_mask` returns None and we rely on cu_seqlens.
"""
from __future__ import annotations

import torch
from transformers import AttentionInterface
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence,
    prepare_fa_kwargs_from_position_ids,
)

from .fa2_sink_kernel import fa2_varlen_attn_with_sink_kernel

ATTN_NAME = "olmo3_sink_fa2"


def fa2_sink_attention_forward(
    module,
    query: torch.Tensor,   # [B, Hq, S, D]
    key: torch.Tensor,     # [B, Hkv, S, D]
    value: torch.Tensor,   # [B, Hkv, S, D]
    attention_mask=None,   # ignored: varlen uses cu_seqlens
    scaling: float | None = None,
    dropout: float = 0.0,
    sliding_window: int | None = None,
    s_aux: torch.Tensor | None = None,
    **kwargs,
):
    B, Hq, S, D = query.shape
    Hkv = key.shape[1]
    sink = s_aux if s_aux is not None else module.sinks

    # [B, H, S, D] -> varlen [B*S, H, D]
    q = query.transpose(1, 2).reshape(B * S, Hq, D)
    k = key.transpose(1, 2).reshape(B * S, Hkv, D)
    v = value.transpose(1, 2).reshape(B * S, Hkv, D)

    cu_q = kwargs.get("cu_seq_lens_q")
    cu_k = kwargs.get("cu_seq_lens_k")
    max_q = kwargs.get("max_length_q")
    max_k = kwargs.get("max_length_k")
    if cu_q is None:
        position_ids = kwargs.get("position_ids")
        if position_ids is not None and _is_packed_sequence(position_ids, B):
            (cu_q, cu_k), (mq, mk) = prepare_fa_kwargs_from_position_ids(position_ids)
            max_q, max_k = int(mq), int(mk)
        else:
            cu_q = torch.arange(0, (B + 1) * S, S, device=q.device, dtype=torch.int32)
            cu_k = cu_q
            max_q = max_k = S

    window = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    out = fa2_varlen_attn_with_sink_kernel(
        q, k, v, sink, cu_q, cu_k, max_q, max_k,
        softmax_scale=scaling, causal=True, window_size=window,
    )  # [B*S, Hq, D]
    return out.reshape(B, S, Hq, D), None


def register_fa2_sink_attention() -> None:
    AttentionInterface.register(ATTN_NAME, fa2_sink_attention_forward)
