# Copyright 2026 proof-pilot. Apache-2.0.
"""FlashAttention-2 attention sink via post-processing (arch-agnostic).

Mirrors `fa3_sink_kernel.py` but backs onto FA2's varlen forward/backward instead of
the patched FA3 op. Its purpose is to run the sink on **non-Hopper** GPUs (Ampere
sm_80/86, Blackwell sm_100/sm_120) for debugging, since FA3 is Hopper-only.

Stock FA2 has no sink argument, so the sink is applied as the exact post-processing
correction (identical math to the FA3 stock fallback):

    o_sink = o_nosink · exp(lse_base − logaddexp(lse_base, sink))
    lse    = logaddexp(lse_base, sink)

The forward is exact for any packing. The backward reuses FA2's native varlen
backward, fed the sink-corrected `out` and the **sink-inclusive** `lse`: FA's backward
recomputes the softmax weights as exp(scores − lse), so a sink-inclusive `lse` makes it
reconstruct the sink attention weights — giving exact `dq/dk/dv`. The `dsink` term reuses
the same closed-form Triton reduction as the FA3 path.

Forward and backward are each a `torch.library.custom_op` with a registered fake, so
Dynamo treats them as opaque nodes (torch.compile-safe), matching `fa3_sink_kernel.py`.
"""
from __future__ import annotations

import torch
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)

from .fa3_sink_kernel import _dsink  # reuse the arch-agnostic Triton dsink reduction


# --- forward as an opaque, fake-backed custom op -----------------------------------------
@torch.library.custom_op("olmo3_sink::fa2_sink_fwd", mutates_args=())
def _fa2_sink_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink: torch.Tensor,
    cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
    scale: float, causal: bool, wl: int, wr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # FA2 varlen forward: out [total_q, H, D], softmax_lse [H, total_q].
    out, lse_base, _S, _rng = _flash_attn_varlen_forward(
        q, k, v, cu_q, cu_k, max_q, max_k,
        0.0, scale, causal, wl, wr, 0.0, None, False, None,
    )
    sink = sink.to(torch.float32).view(-1, 1)                                          # [H, 1]
    lse = torch.logaddexp(lse_base.to(torch.float32), sink)                            # [H, T]
    alpha = torch.exp(lse_base.to(torch.float32) - lse).transpose(0, 1).unsqueeze(-1)  # [T, H, 1]
    return (out * alpha.to(out.dtype)).contiguous(), lse


@_fa2_sink_fwd.register_fake
def _(q, k, v, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr):
    T, H = q.shape[0], q.shape[1]
    return torch.empty_like(q), q.new_empty((H, T), dtype=torch.float32)


# --- backward as an opaque, fake-backed custom op ----------------------------------------
@torch.library.custom_op("olmo3_sink::fa2_sink_bwd", mutates_args=())
def _fa2_sink_bwd(
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor,
    cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
    scale: float, causal: bool, wl: int, wr: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    # Feed the sink-corrected `out` + sink-inclusive `lse`: FA2's backward recomputes
    # P = exp(scores - lse), so this reconstructs the sink attention weights -> exact dq/dk/dv.
    _flash_attn_varlen_backward(
        do, q, k, v, out, lse, dq, dk, dv, cu_q, cu_k, max_q, max_k,
        0.0, scale, causal, wl, wr, 0.0, None, False, None,
    )
    dsink = _dsink(out, do, sink, lse)
    return dq, dk, dv, dsink


@_fa2_sink_bwd.register_fake
def _(do, q, k, v, out, lse, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr):
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(sink))


def _setup_context(ctx, inputs, output):
    q, k, v, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, sink, cu_q, cu_k)
    ctx.max_q, ctx.max_k = max_q, max_k
    ctx.scale, ctx.causal, ctx.wl, ctx.wr = scale, causal, wl, wr


def _backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse, sink, cu_q, cu_k = ctx.saved_tensors
    dq, dk, dv, dsink = torch.ops.olmo3_sink.fa2_sink_bwd(
        grad_out.contiguous(), q, k, v, out, lse, sink, cu_q, cu_k,
        ctx.max_q, ctx.max_k, ctx.scale, ctx.causal, ctx.wl, ctx.wr)
    # grads for (q, k, v, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr)
    return dq, dk, dv, dsink, None, None, None, None, None, None, None, None


_fa2_sink_fwd.register_autograd(_backward, setup_context=_setup_context)


def fa2_varlen_attn_with_sink_kernel(
    q, k, v, sink, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=True, window_size=(-1, -1),
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    wl, wr = window_size
    out, _lse = _fa2_sink_fwd(
        q, k, v, sink.to(torch.float32), cu_seqlens_q, cu_seqlens_k,
        int(max_seqlen_q), int(max_seqlen_k), float(softmax_scale), bool(causal), int(wl), int(wr))
    return out
