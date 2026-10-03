"""Check FA4 outputs and gradients in the pinned B300 training image.

    PYTHONPATH=. uv run --no-sync python tests/manual/attention_recompute.py

Use trainer_replay for training time and memory measurements.
"""

import torch

from miles_plugins.proximal import attention_recompute as cache


def check(lengths):
    from flash_attn.cute.interface import flash_attn_varlen_func
    from megatron.core.tensor_parallel.random import checkpoint

    torch.manual_seed(42)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    inputs = [
        torch.randn(sum(lengths), h, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True) for h in (6, 1, 1)
    ]
    dout = torch.randn_like(inputs[0])

    def attention(q, k, v):
        # Mirrors QKV projection under Megatron's no-grad first forward.
        out, _ = flash_attn_varlen_func(
            q * 1,
            k * 1,
            v * 1,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            causal=True,
        )
        return out

    def run():
        for x in inputs:
            x.grad = None
        out = checkpoint(attention, False, *inputs)
        out.backward(dout)
        return out

    baseline = run().detach().clone()
    grads = []
    for x in inputs:
        assert x.grad is not None
        grads.append(x.grad.clone())
    originals = cache.enable()
    try:
        torch.testing.assert_close(run(), baseline, rtol=0, atol=0)
        errors = []
        for x, ref in zip(inputs, grads, strict=True):
            assert x.grad is not None
            relative_error = ((x.grad.float() - ref.float()).norm() / ref.float().norm()).item()
            assert relative_error < 0.005, relative_error
            errors.append(relative_error)
        print(f"PASS lengths={lengths}, Q/K/V gradient relative L2 errors={errors}")
    finally:
        cache.disable(originals)


if __name__ == "__main__":
    for lengths in ([1024, 384], [131072], [150000, 110000], [247936, 384]):
        check(lengths)
