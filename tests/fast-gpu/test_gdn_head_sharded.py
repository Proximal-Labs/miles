"""Head-sharded GDN (miles_plugins.models.layers.delta_rule_attention) against the replicated
HF-style reference it replaces, at TP = world size.

Run with:
    torchrun --nproc_per_node=1 tests/fast-gpu/test_gdn_head_sharded.py
    torchrun --nproc_per_node=2 tests/fast-gpu/test_gdn_head_sharded.py
    torchrun --nproc_per_node=4 tests/fast-gpu/test_gdn_head_sharded.py

Every rank builds the same replicated reference (full heads, plain nn.Linear) and the sharded module,
copies the reference weights into its shard, and checks the forward output and the gradients of the
input and of every parameter (gathered back to the full layout).
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.models.layers.delta_rule_attention import GatedDeltaRule
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads, qkv_flat_to_group_major
from miles_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet
from miles_plugins.models.qwen3_next import Qwen3NextGatedDeltaNet

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

HIDDEN = 256
HEADS = DeltaRuleHeads(num_k_heads=4, num_v_heads=8, head_k_dim=64, head_v_dim=64)
CONV = 4
EPS = 1e-6


class ReplicatedGDN(nn.Module):
    """The pre-sharding implementation: every rank holds all heads. ``family`` picks the HF
    projection layout (qwen3_5: qkv | z | b | a; qwen3_next: qkvz | ba, group-major)."""

    def __init__(self, family: str, dtype):
        super().__init__()
        self.family = family
        h = HEADS
        if family == "qwen3_5":
            self.in_proj_qkv = nn.Linear(HIDDEN, h.qkv_dim, bias=False)
            self.in_proj_z = nn.Linear(HIDDEN, h.value_dim, bias=False)
            self.in_proj_b = nn.Linear(HIDDEN, h.num_v_heads, bias=False)
            self.in_proj_a = nn.Linear(HIDDEN, h.num_v_heads, bias=False)
        else:
            self.in_proj_qkvz = nn.Linear(HIDDEN, h.qkv_dim + h.value_dim, bias=False)
            self.in_proj_ba = nn.Linear(HIDDEN, 2 * h.num_v_heads, bias=False)
        self.conv1d = ShortConvolution(hidden_size=h.qkv_dim, kernel_size=CONV, bias=False)
        self.dt_bias = nn.Parameter(torch.rand(h.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(h.num_v_heads).uniform_(1, 16)))
        self.norm = FusedRMSNormGated(h.head_v_dim, eps=EPS, activation="silu")
        self.out_proj = nn.Linear(h.value_dim, HIDDEN, bias=False)
        self.to(dtype=dtype)
        self.A_log.data = self.A_log.data.float()

    def _split(self, x):
        h = HEADS
        if self.family == "qwen3_5":
            q, k, v = self.in_proj_qkv(x).split([h.key_dim, h.key_dim, h.value_dim], dim=-1)
            return q, k, v, self.in_proj_z(x), self.in_proj_b(x), self.in_proj_a(x)
        r, hv = h.v_per_k, h.head_v_dim
        grouped = self.in_proj_qkvz(x).view(*x.shape[:-1], h.num_k_heads, 2 * h.head_k_dim + 2 * r * hv)
        q, k, v, z = grouped.split([h.head_k_dim, h.head_k_dim, r * hv, r * hv], dim=-1)
        b, a = self.in_proj_ba(x).view(*x.shape[:-1], h.num_k_heads, 2 * r).split([r, r], dim=-1)
        return q.flatten(-2), k.flatten(-2), v.flatten(-2), z.flatten(-2), b.flatten(-2), a.flatten(-2)

    def forward(self, x, cu_seqlens):
        h = HEADS
        bsz, seq_len, _ = x.shape
        q, k, v, z, b, a = self._split(x)
        mixed, _ = self.conv1d(x=torch.cat([q, k, v], dim=-1), cu_seqlens=cu_seqlens)
        q, k, v = mixed.split([h.key_dim, h.key_dim, h.value_dim], dim=-1)
        q = q.reshape(bsz, seq_len, -1, h.head_k_dim).repeat_interleave(h.v_per_k, dim=2)
        k = k.reshape(bsz, seq_len, -1, h.head_k_dim).repeat_interleave(h.v_per_k, dim=2)
        v = v.reshape(bsz, seq_len, -1, h.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        out, _ = chunk_gated_delta_rule(
            q, k, v, g=g, beta=b.sigmoid(), use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens
        )
        out = self.norm(out.reshape(-1, h.head_v_dim), z.reshape(-1, h.head_v_dim))
        return self.out_proj(out.reshape(bsz, seq_len, -1))


def shard(full: torch.Tensor, dim: int, tp: int, rank: int) -> torch.Tensor:
    return full.chunk(tp, dim=dim)[rank].contiguous()


def load_sharded_from_reference(sharded, ref: ReplicatedGDN, tp: int, rank: int):
    """Copy the reference (flat, full) weights into this rank's shard of the sharded module."""
    with torch.no_grad():
        if ref.family == "qwen3_5":
            grouped = qkv_flat_to_group_major(ref.in_proj_qkv.weight, HEADS)
            sharded.in_proj_qkv.weight.copy_(shard(grouped, 0, tp, rank))
            for name in ("in_proj_z", "in_proj_b", "in_proj_a"):
                getattr(sharded, name).weight.copy_(shard(getattr(ref, name).weight, 0, tp, rank))
        else:
            sharded.in_proj_qkvz.weight.copy_(shard(ref.in_proj_qkvz.weight, 0, tp, rank))
            sharded.in_proj_ba.weight.copy_(shard(ref.in_proj_ba.weight, 0, tp, rank))
        sharded.conv1d.weight.copy_(shard(qkv_flat_to_group_major(ref.conv1d.weight, HEADS), 0, tp, rank))
        sharded.A_log.copy_(shard(ref.A_log, 0, tp, rank))
        sharded.dt_bias.copy_(shard(ref.dt_bias, 0, tp, rank))
        sharded.norm.weight.copy_(ref.norm.weight)
        sharded.out_proj.weight.copy_(shard(ref.out_proj.weight, 1, tp, rank))


def gather(local: torch.Tensor, dim: int, group) -> torch.Tensor:
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def check_family(family: str, tp: int, rank: int, tp_group, config, log):
    dtype = torch.bfloat16
    torch.manual_seed(7)
    ref = ReplicatedGDN(family, dtype).cuda()
    cls = Qwen3_5GatedDeltaNet if family == "qwen3_5" else Qwen3NextGatedDeltaNet
    sharded = cls(config, HEADS, GatedDeltaRule("fla", "silu"), CONV, EPS, tp_group)
    load_sharded_from_reference(sharded, ref, tp, rank)

    torch.manual_seed(11)
    x = torch.randn(1, 512, HIDDEN, device="cuda", dtype=dtype)
    cu_seqlens = torch.tensor([0, 200, 512], device="cuda", dtype=torch.int32)
    x_ref = x.clone().requires_grad_()
    x_new = x.clone().requires_grad_()

    out_ref = ref(x_ref, cu_seqlens)
    out_new = sharded(x_new, cu_seqlens, None)
    torch.manual_seed(13)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_new.backward(grad_out)

    errors = {
        "out": rel_err(out_ref, out_new),
        "dx": rel_err(x_ref.grad, x_new.grad),
        "norm.weight": rel_err(ref.norm.weight.grad, sharded.norm.weight.grad),
        "A_log": rel_err(ref.A_log.grad, gather(sharded.A_log.grad, 0, tp_group)),
        "dt_bias": rel_err(ref.dt_bias.grad, gather(sharded.dt_bias.grad, 0, tp_group)),
        "out_proj": rel_err(
            ref.out_proj.weight.grad,
            gather(
                (
                    sharded.out_proj.weight.main_grad
                    if hasattr(sharded.out_proj.weight, "main_grad")
                    else sharded.out_proj.weight.grad
                ),
                1,
                tp_group,
            ),
        ),
    }
    conv_ref = qkv_flat_to_group_major(ref.conv1d.weight.grad, HEADS)
    errors["conv1d"] = rel_err(conv_ref, gather(sharded.conv1d.weight.grad, 0, tp_group))
    if family == "qwen3_5":
        qkv_ref = qkv_flat_to_group_major(ref.in_proj_qkv.weight.grad, HEADS)
        errors["in_proj_qkv"] = rel_err(qkv_ref, gather(sharded.in_proj_qkv.weight.grad, 0, tp_group))
        for name in ("in_proj_z", "in_proj_b", "in_proj_a"):
            errors[name] = rel_err(
                getattr(ref, name).weight.grad, gather(getattr(sharded, name).weight.grad, 0, tp_group)
            )
    else:
        for name in ("in_proj_qkvz", "in_proj_ba"):
            errors[name] = rel_err(
                getattr(ref, name).weight.grad, gather(getattr(sharded, name).weight.grad, 0, tp_group)
            )

    log(f"[{family} TP={tp}] " + "  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
    # bf16 GEMMs in TE vs torch differ in accumulation order; a relative error of a few 1e-3 is the
    # usual floor, anything near 1 means a head or a layout is wrong.
    bad = {k: v for k, v in errors.items() if v > 2e-2}
    return bad


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=world,
        sequence_parallel=False,
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()
    log = print if rank == 0 else (lambda *_: None)
    failures = {}
    for family in ("qwen3_5", "qwen3_next"):
        bad = check_family(family, world, rank, tp_group, config, log)
        if bad:
            failures[family] = bad
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    log(f"TP={world} head-sharded GDN matches the replicated reference")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
