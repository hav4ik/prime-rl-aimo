"""GPU parity tests for the Olmo3Sink FA2/FA3 attention-sink kernels.

Both kernels apply the gpt-oss per-head sink. FA3 needs Hopper (sm_90) — the stock
FA3 op runs only there — so those cases skip on other GPUs; FA2 runs on sm_80+.
Each kernel is checked against an fp32 closed-form reference: the forward, the sink
gradient (`dsink`), and that a strongly-negative sink reduces to plain (sink-free)
attention. The exact `dq/dk/dv` values track FlashAttention's own bf16 backward, so
they are only smoke-checked (finite + nonzero), not asserted against fp32.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


def _ref(q, k, v, sink, scale, with_sink=True):
    """fp32 exact gpt-oss sink attention. q,k,v: [B,Hq,S,D]/[B,Hkv,S,D]; sink: [Hq]."""
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    k = k.repeat_interleave(Hq // Hkv, dim=1)
    v = v.repeat_interleave(Hq // Hkv, dim=1)
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    causal = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(causal, float("-inf"))
    if with_sink:
        sinks = sink.float().view(1, Hq, 1, 1).expand(B, Hq, S, 1)
        comb = torch.cat([scores, sinks], dim=-1)
    else:
        comb = scores
    comb = comb - comb.amax(-1, keepdim=True)
    probs = comb.softmax(-1)
    if with_sink:
        probs = probs[..., :-1]  # drop the sink column
    return probs @ v.float()  # [B,Hq,S,D]


@pytest.fixture(scope="module", autouse=True)
def _register():
    from prime_rl.trainer.models.olmo3_sink.register import register_olmo3_sink

    register_olmo3_sink()


def _kernel(name):
    if name == "fa3":
        cap = torch.cuda.get_device_capability()
        if cap[0] != 9:
            pytest.skip(f"FA3 needs Hopper sm_90, got sm_{cap[0]}{cap[1]}")
        from prime_rl.trainer.models.olmo3_sink.fa3_sink_kernel import (
            fa3_varlen_attn_with_sink_kernel as fn,
        )
        return fn
    from prime_rl.trainer.models.olmo3_sink.fa2_sink_kernel import (
        fa2_varlen_attn_with_sink_kernel as fn,
    )
    return fn


def _inputs(B, S, Hq, Hkv, D):
    torch.manual_seed(0)

    def mk(h):
        return torch.randn(B * S, h, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
    sink = torch.randn(Hq, device="cuda", dtype=torch.float32, requires_grad=True)
    cu = torch.arange(0, (B + 1) * S, S, device="cuda", dtype=torch.int32)
    return q, k, v, sink, cu


def _to_bhsd(x, B, S, H, D):
    return x.reshape(B, S, H, D).transpose(1, 2)


@pytest.mark.parametrize("kernel", ["fa2", "fa3"])
@pytest.mark.parametrize("Hkv", [4, 2], ids=["mha", "gqa"])
def test_sink_forward_matches_fp32_reference(kernel, Hkv):
    fn = _kernel(kernel)
    B, S, Hq, D = 1, 128, 4, 64
    scale = D**-0.5
    q, k, v, sink, cu = _inputs(B, S, Hq, Hkv, D)
    out = fn(q, k, v, sink, cu, cu, S, S, softmax_scale=scale, causal=True)  # [T,Hq,D]
    out_ref = _ref(_to_bhsd(q, B, S, Hq, D), _to_bhsd(k, B, S, Hkv, D), _to_bhsd(v, B, S, Hkv, D), sink, scale)
    assert (_to_bhsd(out, B, S, Hq, D).float() - out_ref).abs().max() < 3e-2  # bf16 tolerance


@pytest.mark.parametrize("kernel", ["fa2", "fa3"])
def test_sink_gradient_matches_reference(kernel):
    fn = _kernel(kernel)
    B, S, Hq, Hkv, D = 1, 128, 4, 4, 64
    scale = D**-0.5
    q, k, v, sink, cu = _inputs(B, S, Hq, Hkv, D)
    qr, kr, vr, sr = (t.detach().clone().requires_grad_() for t in (q, k, v, sink))
    w = torch.arange(1, D + 1, device="cuda").float()

    out = fn(q, k, v, sink, cu, cu, S, S, softmax_scale=scale, causal=True)
    (_to_bhsd(out, B, S, Hq, D) * w).sum().backward()
    out_ref = _ref(_to_bhsd(qr, B, S, Hq, D), _to_bhsd(kr, B, S, Hkv, D), _to_bhsd(vr, B, S, Hkv, D), sr, scale)
    (out_ref * w).sum().backward()

    assert sink.grad is not None and sink.grad.abs().sum() > 0, "sink gradient did not flow"
    torch.testing.assert_close(sink.grad, sr.grad, rtol=5e-2, atol=1e-1)  # dsink vs fp32 ref
    for g in (q.grad, k.grad, v.grad):
        assert torch.isfinite(g).all() and g.abs().sum() > 0  # bf16 exactness is FA's, not tested here


@pytest.mark.parametrize("kernel", ["fa2", "fa3"])
def test_strongly_negative_sink_reduces_to_plain_attention(kernel):
    fn = _kernel(kernel)
    B, S, Hq, Hkv, D = 1, 128, 4, 4, 64
    scale = D**-0.5
    q, k, v, _sink, cu = _inputs(B, S, Hq, Hkv, D)
    off = torch.full((Hq,), -30.0, device="cuda", dtype=torch.float32)  # sink effectively disabled
    out = fn(q, k, v, off, cu, cu, S, S, softmax_scale=scale, causal=True)
    out_ref = _ref(_to_bhsd(q, B, S, Hq, D), _to_bhsd(k, B, S, Hkv, D), _to_bhsd(v, B, S, Hkv, D), off, scale, with_sink=False)
    assert (_to_bhsd(out, B, S, Hq, D).float() - out_ref).abs().max() < 3e-2
