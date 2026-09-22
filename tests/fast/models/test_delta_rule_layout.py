import torch

from miles_plugins.models.layers.delta_rule_layout import (
    DeltaRuleHeads,
    cat_group_major_qkv,
    qkv_flat_to_group_major,
    qkv_group_major_to_flat,
    split_group_major_qkv,
)

HEADS = DeltaRuleHeads(num_k_heads=4, num_v_heads=8, head_k_dim=16, head_v_dim=32)


def test_flat_and_group_major_are_inverse():
    weight = torch.randn(HEADS.qkv_dim, 24)
    grouped = qkv_flat_to_group_major(weight, HEADS)
    assert grouped.shape == weight.shape
    assert torch.equal(qkv_group_major_to_flat(grouped, HEADS), weight)
    conv = torch.randn(HEADS.qkv_dim, 1, 4)
    assert torch.equal(qkv_group_major_to_flat(qkv_flat_to_group_major(conv, HEADS), HEADS), conv)


def test_tp_chunk_of_group_major_holds_that_ranks_heads():
    """Chunk r of the group-major tensor is exactly [q, k, v] of key heads r*G/tp .. (r+1)*G/tp."""
    tp = 2
    weight = torch.randn(HEADS.qkv_dim, 8)
    q, k, v = weight.split([HEADS.key_dim, HEADS.key_dim, HEADS.value_dim], dim=0)
    local = HEADS.local(tp)
    for rank, chunk in enumerate(qkv_flat_to_group_major(weight, HEADS).chunk(tp, dim=0)):
        k_heads = slice(rank * local.num_k_heads, (rank + 1) * local.num_k_heads)
        v_heads = slice(rank * local.num_v_heads, (rank + 1) * local.num_v_heads)
        expected = qkv_flat_to_group_major(
            torch.cat(
                [
                    q.view(HEADS.num_k_heads, HEADS.head_k_dim, -1)[k_heads].reshape(-1, 8),
                    k.view(HEADS.num_k_heads, HEADS.head_k_dim, -1)[k_heads].reshape(-1, 8),
                    v.view(HEADS.num_v_heads, HEADS.head_v_dim, -1)[v_heads].reshape(-1, 8),
                ]
            ),
            local,
        )
        assert torch.equal(chunk, expected)


def test_activation_split_matches_weight_layout():
    """A group-major projection output splits into the same q/k/v a flat projection would give."""
    hidden = 24
    weight = torch.randn(HEADS.qkv_dim, hidden)
    x = torch.randn(2, 5, hidden)
    flat_out = x @ weight.T
    q_ref, k_ref, v_ref = flat_out.split([HEADS.key_dim, HEADS.key_dim, HEADS.value_dim], dim=-1)
    grouped_out = x @ qkv_flat_to_group_major(weight, HEADS).T
    q, k, v = split_group_major_qkv(grouped_out, HEADS)
    assert torch.allclose(q.flatten(-2), q_ref, atol=1e-6)
    assert torch.allclose(k.flatten(-2), k_ref, atol=1e-6)
    assert torch.allclose(v.flatten(-2), v_ref, atol=1e-6)
    assert torch.allclose(cat_group_major_qkv(q.flatten(-2), k.flatten(-2), v.flatten(-2), HEADS), grouped_out)
