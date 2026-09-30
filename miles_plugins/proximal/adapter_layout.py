"""The tensors a published LoRA adapter may hold, checked before it can serve.

SGLang files each adapter tensor under the first ``layers.N.`` in its name and keeps the
last tensor written per module. A tensor outside the text decoder therefore either
vanishes (a vision-tower adapter) or overwrites a decoder layer's (an MTP layer's
``mtp.layers.0.*`` replaces decoder layer 0's), and a partial stacked group serves zeros
for its missing members. Each case serves a policy other than the one trained, with no
error, so a publication that could cause one is refused.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

_TENSOR = re.compile(
    r"^(?:base_model\.model\.)?model\.(?:language_model\.)?layers\.(?P<layer>\d+)\."
    r"(?P<parent>(?:[a-z_]+\.)*)(?P<leaf>[a-z_]+)\.lora_(?P<side>[AB])\.weight$"
)

# Exported tensors SGLang stacks into one module, keyed by that module's name. Megatron
# exports a fused module's members together, and SGLang serves a member the adapter
# lacks as zeros.
_STACKED: dict[str, frozenset[str]] = {
    "qkv_proj": frozenset({"q_proj", "k_proj", "v_proj"}),
    "gate_up_proj": frozenset({"gate_proj", "up_proj"}),
    "in_proj_qkvz": frozenset({"in_proj_qkv", "in_proj_z"}),
    "in_proj_ba": frozenset({"in_proj_b", "in_proj_a"}),
}

_SAMPLE = 5


def exported_leaves(serving_targets: Iterable[str]) -> frozenset[str]:
    """The adapter tensor leaf names that serve ``serving_targets`` (SGLang module names).

    A stacked module stands for all its members, and naming one member stands for the
    whole stack: SGLang serves the stacked module either way.
    """
    leaves: set[str] = set()
    for target in serving_targets:
        members = _STACKED.get(target) or next((g for g in _STACKED.values() if target in g), frozenset({target}))
        leaves.update(members)
    return frozenset(leaves)


def adapter_layout_problem(
    shapes: Mapping[str, Sequence[int]], *, serving_targets: Sequence[str], rank: int
) -> str | None:
    """Why an adapter with these tensor shapes would not serve as trained, or None if it would."""
    expected = exported_leaves(serving_targets)
    sides: dict[tuple[int, str], set[str]] = defaultdict(set)
    layer_leaves: dict[int, set[str]] = defaultdict(set)
    outside, untargeted, wrong_rank = [], [], []
    for name, shape in shapes.items():
        match = _TENSOR.match(name)
        if match is None:
            outside.append(name)
            continue
        if match["leaf"] not in expected:
            untargeted.append(name)
            continue
        if len(shape) != 2 or shape[0 if match["side"] == "A" else 1] != rank:
            wrong_rank.append(f"{name} {tuple(shape)}")
        layer = int(match["layer"])
        sides[(layer, match["parent"] + match["leaf"])].add(match["side"])
        layer_leaves[layer].add(match["leaf"])
    if outside:
        return _problem(
            "outside the text decoder layers, so SGLang would drop them or file them under another layer", outside
        )
    if untargeted:
        return _problem(f"for modules outside the serving targets {sorted(serving_targets)}", untargeted)
    if wrong_rank:
        return _problem(f"not rank {rank}", wrong_rank)
    unpaired = [f"layer {layer} {module}" for (layer, module), found in sorted(sides.items()) if found != {"A", "B"}]
    if unpaired:
        return _problem("missing their lora_A or lora_B half", unpaired)
    partial = [
        f"layer {layer} {sorted(leaves & group)} without {sorted(group - leaves)}"
        for layer, leaves in sorted(layer_leaves.items())
        for group in _STACKED.values()
        if leaves & group and not group <= leaves
    ]
    if partial:
        return _problem("in partial stacked groups, whose missing members SGLang would serve as zeros", partial)
    unused = sorted(expected - set().union(*layer_leaves.values()))
    if unused:
        return f"No adapter tensors for targeted modules {unused}: their training target matched no layer"
    return None


def _problem(what: str, items: list[str]) -> str:
    return f"{len(items)} adapter tensor(s) {what}: {sorted(items)[:_SAMPLE]}"
