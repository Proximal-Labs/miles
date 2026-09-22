"""Forward+backward time and peak memory of one GDN layer: replicated (every TP rank runs all heads,
the pre-sharding implementation) vs head-sharded. Qwen3.5-35B-A3B dims by default.

    torchrun --nproc_per_node=2 tests/manual/perf_gdn_head_sharded.py [seq_len]
"""

import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.layers.delta_rule_attention import GatedDeltaRule
from miles_plugins.models.layers.delta_rule_layout import DeltaRuleHeads
from miles_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fast-gpu"))
import test_gdn_head_sharded as ab  # noqa: E402

HIDDEN = 2048
HEADS = DeltaRuleHeads(num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128)


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3, torch.cuda.max_memory_allocated() / 2**30


def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    ab.HIDDEN, ab.HEADS = HIDDEN, HEADS
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=16,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=world,
        sequence_parallel=False,
    )
    tp_group = parallel_state.get_tensor_model_parallel_group()
    dtype = torch.bfloat16
    x = torch.randn(1, seq_len, HIDDEN, device="cuda", dtype=dtype, requires_grad=True)
    cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

    ref = ab.ReplicatedGDN("qwen3_5", dtype).cuda()
    t_ref, m_ref = bench(lambda: ref(x, cu).float().sum().backward())
    ref = None
    torch.cuda.empty_cache()
    new = Qwen3_5GatedDeltaNet(config, HEADS, GatedDeltaRule("fla", "silu"), 4, 1e-6, tp_group)
    t_new, m_new = bench(lambda: new(x, cu, None).float().sum().backward())
    if rank == 0:
        print(f"GDN layer fwd+bwd, seq_len={seq_len}, TP={world}, Qwen3.5-35B-A3B dims")
        print(f"  replicated : {t_ref:8.2f} ms   peak {m_ref:6.2f} GiB")
        print(
            f"  head-shard : {t_new:8.2f} ms   peak {m_new:6.2f} GiB   speedup x{t_ref / t_new:.2f}  mem x{m_ref / m_new:.2f}"
        )
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__, *sys.argv[1:]])
    main()
