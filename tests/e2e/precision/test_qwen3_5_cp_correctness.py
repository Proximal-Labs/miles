"""Correctness test for the head-sharded Qwen3.5 GDN core with native fla context parallelism.

Run with:
    torchrun --nproc_per_node=2 tests/e2e/precision/test_qwen3_5_cp_correctness.py   # CP=2
    torchrun --nproc_per_node=4 tests/e2e/precision/test_qwen3_5_cp_correctness.py   # CP=4

Every rank runs the full sequence without CP as the reference, then its contiguous shard with a
``cp_context``; outputs and input gradients gathered over CP must match the reference.
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.backends.training_utils.cp_utils import build_fla_cp_context
from miles_plugins.models.layers.delta_rule_attention import GatedDeltaRule
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads
from miles_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=120, suite="nightly-stage-c-4-gpu-mi350", labels=["precision"])

HIDDEN = 256
HEADS = DeltaRuleHeads(num_k_heads=2, num_v_heads=4, head_k_dim=64, head_v_dim=64)


def build_gdn(config, tp_group):
    torch.manual_seed(42)
    module = Qwen3_5GatedDeltaNet(config, HEADS, GatedDeltaRule("fla", "silu"), 4, 1e-6, tp_group)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * 0.05)
        module.A_log.copy_(torch.log(torch.empty_like(module.A_log).uniform_(1, 16)))
    return module


def test_cp_forward_backward(rank, world_size, config):
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    model = build_gdn(config, tp_group)

    total_seq_len = 128 * world_size
    torch.manual_seed(123)
    full_hidden = torch.randn(1, total_seq_len, HIDDEN, device=device, dtype=dtype)
    full_cu = torch.tensor([0, total_seq_len], dtype=torch.int32, device=device)

    # Reference: the whole sequence, no CP.
    x_ref = full_hidden.clone().requires_grad_()
    ref_out = model(x_ref, full_cu, None)
    ref_out.sum().backward()
    ref_grad = x_ref.grad.clone()
    model.zero_grad(set_to_none=True)

    # CP: this rank's contiguous shard with the fla context built from the global boundaries.
    local_seq_len = total_seq_len // world_size
    local_hidden = full_hidden[:, rank * local_seq_len : (rank + 1) * local_seq_len].clone().requires_grad_()
    cp_context = build_fla_cp_context(full_cu, cp_group, model.conv_kernel_size, device)
    cp_out = model(local_hidden, cp_context.cu_seqlens, cp_context)
    cp_loss = cp_out.sum()
    dist.all_reduce(cp_loss, op=dist.ReduceOp.SUM)
    cp_loss.backward()

    def gather(local):
        parts = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(parts, local.contiguous())
        return torch.cat(parts, dim=1)

    full_cp_out = gather(cp_out.detach())
    full_cp_grad = gather(local_hidden.grad)

    if rank == 0:
        out_max_diff = (ref_out.detach().float() - full_cp_out.float()).abs().max().item()
        grad_max_diff = (ref_grad.float() - full_cp_grad.float()).abs().max().item()
        print(f"\n=== CP={world_size} Correctness Test ===")
        print(f"Forward  max abs diff: {out_max_diff:.6e}")
        print(f"Backward max abs diff: {grad_max_diff:.6e}")
        # bf16 tolerance: 1e-2 is generous for bf16 accumulated ops
        if not (out_max_diff < 1e-2 and grad_max_diff < 1e-2):
            print("FAILED!")
            sys.exit(1)
        print(f"CP={world_size} test PASSED!")


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=world_size)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        context_parallel_size=world_size,
    )
    try:
        test_cp_forward_backward(rank, world_size, config)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    # Self-bootstrap under torchrun when invoked as `python3 file.py` (the
    # CUDA CI runner's mode). Already inside torchrun => RANK is set.
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=4", __file__])
    main()
