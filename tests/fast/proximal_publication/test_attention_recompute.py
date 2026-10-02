"""Exercise cache lifetime with real reentrant autograd checkpoints on CPU.

The small differentiable kernel substitutes for GPU-only FA4; the manual GPU
check compares the real FA4 outputs and Q/K/V gradients.
"""

import weakref
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import CheckpointFunction, checkpoint

from miles_plugins.proximal import attention_recompute as cache


def test_pending_checkpoints_keep_their_outputs_and_gradients(monkeypatch):
    calls = []
    monkeypatch.setattr(
        CheckpointFunction,
        "forward",
        staticmethod(partial(cache._checkpoint_forward, CheckpointFunction.forward)),
    )

    def kernel(q, k, v, *, causal, return_lse=False):
        calls.append(1)
        assert causal and return_lse
        return q.square() + k * v, q.float().sum(-1)

    class Kernel(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            result, _ = cache._cached_forward(kernel, q, k, v, causal=True)
            ctx.save_for_backward(q, k, v)
            return result

        @staticmethod
        def backward(ctx, *grad_outputs):
            (grad,) = grad_outputs
            q, k, v = ctx.saved_tensors
            return 2 * q * grad, v * grad, k * grad

    def layer(x):
        return Kernel.apply(x * 1, x * 2, x * 3)

    # Different inputs with identical shapes catch cross-microbatch cache reuse.
    inputs = [torch.full((3, 1, 4), n, dtype=torch.bfloat16, requires_grad=True) for n in (1, 2)]
    outputs = [checkpoint(layer, x, use_reentrant=True) for x in inputs]
    assert len(calls) == 2
    for x, output in zip(reversed(inputs), reversed(outputs), strict=True):
        torch.testing.assert_close(output, 7 * x.square(), rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(x.grad, 14 * x, rtol=0, atol=0)
    assert len(calls) == 2  # Both recomputes used the saved kernel results.
    assert cache._ACTIVE.get() is None


def test_failed_forward_releases_cache_and_restores_context(monkeypatch):
    monkeypatch.setattr(
        CheckpointFunction,
        "forward",
        staticmethod(partial(cache._checkpoint_forward, CheckpointFunction.forward)),
    )

    outputs = []

    def kernel(x, *args, **kwargs):
        out = x.clone()
        outputs.append(weakref.ref(out))
        return out, None

    def layer(x):
        cache._cached_forward(kernel, x, x, x, causal=True)
        raise RuntimeError("layer failed")

    x = torch.ones(3, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(RuntimeError, match="layer failed"):
        checkpoint(layer, x, use_reentrant=True)
    assert outputs[0]() is None
    assert cache._ACTIVE.get() is None
