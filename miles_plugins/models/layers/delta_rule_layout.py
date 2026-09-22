"""Row layouts of the fused delta-rule projections.

HF stores the fused q/k/v projection and the short-conv weight as flat blocks ``[Q_all, K_all, V_all]``.
Megatron shards the output dimension in contiguous chunks, so the mcore copy is *group-major*: one
block per key head ``g`` holding ``[q_g, k_g, v_{g*R} .. v_{g*R+R-1}]`` where ``R = num_v_heads //
num_k_heads``. A TP chunk of the group-major tensor is then exactly the heads that rank owns, for any
TP size. Qwen3-Next already ships its fused qkvz weight group-major; Qwen3.5 ships flat.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeltaRuleHeads:
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int

    @property
    def v_per_k(self) -> int:
        assert self.num_v_heads % self.num_k_heads == 0
        return self.num_v_heads // self.num_k_heads

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def qkv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def group_qkv_dim(self) -> int:
        return 2 * self.head_k_dim + self.v_per_k * self.head_v_dim

    def local(self, tp_size: int) -> "DeltaRuleHeads":
        assert self.num_k_heads % tp_size == 0, f"{self.num_k_heads} key heads do not split across TP={tp_size}"
        return DeltaRuleHeads(
            self.num_k_heads // tp_size, self.num_v_heads // tp_size, self.head_k_dim, self.head_v_dim
        )


def qkv_flat_to_group_major(weight: torch.Tensor, heads: DeltaRuleHeads) -> torch.Tensor:
    """Permute dim 0 of a ``[Q_all, K_all, V_all]`` tensor into group-major order."""
    assert weight.shape[0] == heads.qkv_dim, (weight.shape, heads)
    q, k, v = weight.split([heads.key_dim, heads.key_dim, heads.value_dim], dim=0)
    rest = weight.shape[1:]
    q = q.reshape(heads.num_k_heads, heads.head_k_dim, *rest)
    k = k.reshape(heads.num_k_heads, heads.head_k_dim, *rest)
    v = v.reshape(heads.num_k_heads, heads.v_per_k * heads.head_v_dim, *rest)
    return torch.cat([q, k, v], dim=1).reshape(heads.qkv_dim, *rest).contiguous()


def qkv_group_major_to_flat(weight: torch.Tensor, heads: DeltaRuleHeads) -> torch.Tensor:
    """Inverse of :func:`qkv_flat_to_group_major`."""
    assert weight.shape[0] == heads.qkv_dim, (weight.shape, heads)
    rest = weight.shape[1:]
    grouped = weight.reshape(heads.num_k_heads, heads.group_qkv_dim, *rest)
    q, k, v = grouped.split([heads.head_k_dim, heads.head_k_dim, heads.v_per_k * heads.head_v_dim], dim=1)
    return torch.cat([q.reshape(-1, *rest), k.reshape(-1, *rest), v.reshape(-1, *rest)], dim=0).contiguous()


def split_group_major_qkv(
    mixed: torch.Tensor, heads: DeltaRuleHeads
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[..., G * group_qkv_dim]`` -> q ``[..., G, hk]``, k ``[..., G, hk]``, v ``[..., G * R, hv]``."""
    lead = mixed.shape[:-1]
    grouped = mixed.reshape(*lead, heads.num_k_heads, heads.group_qkv_dim)
    q, k, v = grouped.split([heads.head_k_dim, heads.head_k_dim, heads.v_per_k * heads.head_v_dim], dim=-1)
    return q, k, v.reshape(*lead, heads.num_v_heads, heads.head_v_dim)


def cat_group_major_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: DeltaRuleHeads) -> torch.Tensor:
    """Inverse of :func:`split_group_major_qkv` for flat ``[..., dim]`` inputs."""
    lead = q.shape[:-1]
    q = q.reshape(*lead, heads.num_k_heads, heads.head_k_dim)
    k = k.reshape(*lead, heads.num_k_heads, heads.head_k_dim)
    v = v.reshape(*lead, heads.num_k_heads, heads.v_per_k * heads.head_v_dim)
    return torch.cat([q, k, v], dim=-1).reshape(*lead, heads.num_k_heads * heads.group_qkv_dim)
