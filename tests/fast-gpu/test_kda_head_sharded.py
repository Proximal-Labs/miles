"""Head-sharded KDA (KimiDeltaAttention + KimiDeltaRule) against a replicated HF-style reference
(Kimi-K3 / GLM-5.3-flash layout), at TP = world size.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_kda_head_sharded.py
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.models.layers.delta_rule_attention import KimiDeltaAttention, KimiDeltaRule
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads, qkv_flat_to_group_major

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

HIDDEN = 256
HEADS = DeltaRuleHeads(num_k_heads=8, num_v_heads=8, head_k_dim=64, head_v_dim=64)
CONV = 4
EPS = 1e-6
LOWER_BOUND = -5.0


class ReplicatedKDA(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        size, d = HEADS.value_dim, HEADS.head_v_dim
        self.q_proj = nn.Linear(HIDDEN, size, bias=False)
        self.k_proj = nn.Linear(HIDDEN, size, bias=False)
        self.v_proj = nn.Linear(HIDDEN, size, bias=False)
        self.q_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.f_a_proj = nn.Linear(HIDDEN, d, bias=False)
        self.f_b_proj = nn.Linear(d, size, bias=False)
        self.b_proj = nn.Linear(HIDDEN, HEADS.num_v_heads, bias=False)
        self.g_proj = nn.Linear(HIDDEN, size, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(HEADS.num_v_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.rand(size))
        self.o_norm = FusedRMSNormGated(d, eps=EPS, activation="sigmoid")
        self.o_proj = nn.Linear(size, HIDDEN, bias=False)
        self.to(dtype=dtype)
        self.A_log.data = self.A_log.data.float()
        self.dt_bias.data = self.dt_bias.data.float()

    def forward(self, x, cu_seqlens):
        bsz, seq_len, _ = x.shape
        h, d = HEADS.num_v_heads, HEADS.head_v_dim
        q, _ = self.q_conv1d(self.q_proj(x), cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(self.k_proj(x), cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(self.v_proj(x), cu_seqlens=cu_seqlens)
        forget = self.f_b_proj(self.f_a_proj(x))
        out, _ = chunk_kda(
            q=q.view(bsz, seq_len, h, d),
            k=k.view(bsz, seq_len, h, d),
            v=v.view(bsz, seq_len, h, d),
            g=forget.view(bsz, seq_len, h, d),
            beta=self.b_proj(x).float().sigmoid(),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        out = self.o_norm(out.reshape(-1, d), self.g_proj(x).reshape(-1, d))
        return self.o_proj(out.reshape(bsz, seq_len, -1))


def shard(full, dim, tp, rank):
    return full.chunk(tp, dim=dim)[rank].contiguous()


def load_from_reference(sharded, ref, tp, rank):
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "b_proj", "g_proj", "f_b_proj"):
            getattr(sharded, name).weight.copy_(shard(getattr(ref, name).weight, 0, tp, rank))
        sharded.f_a_proj.weight.copy_(ref.f_a_proj.weight)
        conv = torch.cat([ref.q_conv1d.weight, ref.k_conv1d.weight, ref.v_conv1d.weight], dim=0)
        sharded.conv1d.weight.copy_(shard(qkv_flat_to_group_major(conv, HEADS), 0, tp, rank))
        sharded.A_log.copy_(shard(ref.A_log, 0, tp, rank))
        sharded.dt_bias.copy_(shard(ref.dt_bias, 0, tp, rank))
        sharded.norm.weight.copy_(ref.o_norm.weight)
        sharded.out_proj.weight.copy_(shard(ref.o_proj.weight, 1, tp, rank))


def gather(local, dim, group):
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=world,
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()
    dtype = torch.bfloat16
    torch.manual_seed(7)
    ref = ReplicatedKDA(dtype).cuda()
    sharded = KimiDeltaAttention(config, HEADS, KimiDeltaRule(LOWER_BOUND), CONV, EPS, tp_group)
    load_from_reference(sharded, ref, world, rank)

    torch.manual_seed(11)
    x = torch.randn(1, 512, HIDDEN, device="cuda", dtype=dtype)
    cu_seqlens = torch.tensor([0, 200, 512], device="cuda", dtype=torch.int32)
    x_ref, x_new = x.clone().requires_grad_(), x.clone().requires_grad_()
    out_ref = ref(x_ref, cu_seqlens)
    out_new = sharded(x_new, cu_seqlens, None)
    torch.manual_seed(13)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_new.backward(grad_out)

    errors = {
        "out": rel_err(out_ref, out_new),
        "dx": rel_err(x_ref.grad, x_new.grad),
        "norm.weight": rel_err(ref.o_norm.weight.grad, sharded.norm.weight.grad),
        "f_a_proj": rel_err(ref.f_a_proj.weight.grad, sharded.f_a_proj.weight.grad),
        "A_log": rel_err(ref.A_log.grad, gather(sharded.A_log.grad, 0, tp_group)),
        "dt_bias": rel_err(ref.dt_bias.grad, gather(sharded.dt_bias.grad, 0, tp_group)),
        "out_proj": rel_err(ref.o_proj.weight.grad, gather(sharded.out_proj.weight.grad, 1, tp_group)),
    }
    conv_ref = qkv_flat_to_group_major(
        torch.cat([ref.q_conv1d.weight.grad, ref.k_conv1d.weight.grad, ref.v_conv1d.weight.grad], dim=0), HEADS
    )
    errors["conv1d"] = rel_err(conv_ref, gather(sharded.conv1d.weight.grad, 0, tp_group))
    for name in ("q_proj", "k_proj", "v_proj", "b_proj", "g_proj", "f_b_proj"):
        errors[name] = rel_err(getattr(ref, name).weight.grad, gather(getattr(sharded, name).weight.grad, 0, tp_group))
    if rank == 0:
        print(f"[kda TP={world}] " + "  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
    bad = {k: v for k, v in errors.items() if v > 2e-2}
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    if bad:
        print(f"FAILED: {bad}")
        sys.exit(1)
    if rank == 0:
        print(f"TP={world} head-sharded KDA matches the replicated reference")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
