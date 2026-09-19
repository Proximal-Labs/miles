import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter, WeightRenaming

from miles.utils.hf_lora_targets import matches_hf_lora_target


@dataclass(frozen=True)
class HfWeightMapping:
    parameter_shapes: dict[str, tuple[int, ...]]
    conversions: tuple = ()

    @classmethod
    def from_config(cls, config):
        for auto_model in (AutoModelForCausalLM, AutoModelForImageTextToText):
            if type(config) in auto_model._model_mapping:
                # Only structure is needed; never allocate or load base weights.
                with torch.random.fork_rng(devices=[]), torch.device("meta"):
                    model = auto_model.from_config(config, attn_implementation="eager")
                shapes = {
                    name: tuple(param.shape)
                    for name, param in model.named_parameters(remove_duplicate=False)
                    if param.ndim in (2, 3)
                }
                return cls(shapes, tuple(get_model_conversion_mapping(model, add_legacy=False)))
        # Custom HF implementations without native conversion rules retain their checkpoint namespace.
        return cls({})

    def _resolve(self, name):
        if name.removesuffix(".weight") in self.parameter_shapes:
            name = name.removesuffix(".weight")
        if name in self.parameter_shapes:
            return name, None, None, name
        renamed = name
        for conversion in self.conversions:
            if isinstance(conversion, WeightRenaming):
                renamed, _ = conversion.rename_source_key(renamed)
        for conversion in self.conversions:
            if isinstance(conversion, WeightConverter):
                target, source_pattern = conversion.rename_source_key(renamed)
                if source_pattern is not None:
                    assert (
                        len(conversion.target_patterns) == 1
                    ), f"HF target binding does not support one-to-many conversion of {name!r}"
                    operation_types = tuple(type(op) for op in conversion.operations)
                    assert operation_types in (
                        (MergeModulelist,),
                        (Concatenate,),
                        (MergeModulelist, Concatenate),
                    ), f"HF target binding does not support conversion of {name!r}: {conversion.operations}"
                    if operation_types == (MergeModulelist, Concatenate):
                        assert (
                            conversion.operations[0].dim != conversion.operations[1].dim
                        ), f"HF stacking and concatenation must use distinct axes for {name!r}"
                    return target, conversion, source_pattern, renamed
        return renamed, None, None, renamed

    def model_parameter(self, checkpoint_name):
        return self._resolve(checkpoint_name)[0]

    def validate_coverage(self, checkpoint_names, targets):
        resolved = defaultdict(list)
        for name in checkpoint_names:
            target, conversion, source_pattern, renamed = self._resolve(name)
            resolved[target].append((conversion, source_pattern, renamed))
        unexpected = {
            name
            for name in resolved
            if (self.parameter_shapes and name not in self.parameter_shapes)
            or not any(matches_hf_lora_target(name.removesuffix(".weight"), target) for target in targets)
        }
        assert not unexpected, f"Weights include unselected HF parameters: {sorted(unexpected)}"
        if self.parameter_shapes:
            expected = {
                name
                for name in self.parameter_shapes
                if any(matches_hf_lora_target(name.removesuffix(".weight"), target) for target in targets)
            }
            missing = expected - resolved.keys()
            assert not missing, f"Weights are missing HF parameters: {sorted(missing)}"
        for target in targets:
            assert any(
                matches_hf_lora_target(name.removesuffix(".weight"), target) for name in resolved
            ), f"Weights are missing HF target {target!r}"
        for name, sources in resolved.items():
            self._validate_sources(name, sources)

    def _validate_sources(self, name, sources):
        if any(conversion is None for conversion, _, _ in sources):
            assert all(
                conversion is None for conversion, _, _ in sources
            ), f"Mixed packed and unpacked sources for HF parameter {name!r}"
            return
        conversion = sources[0][0]
        assert all(item[0] is conversion for item in sources), f"Ambiguous HF conversion for {name!r}"
        patterns = {pattern for _, pattern, _ in sources}
        assert patterns == set(conversion.source_patterns), f"Incomplete HF projections for {name!r}: {patterns}"
        for op in conversion.operations:
            if isinstance(op, MergeModulelist):
                assert name in self.parameter_shapes, f"HF shape is required to validate stacked parameter {name!r}"
                expected = set(range(self.parameter_shapes[name][op.dim]))
                for pattern in conversion.source_patterns:
                    assert pattern.count("*") == 1, f"Unsupported HF stacking pattern {pattern!r}"
                    regex = re.escape(pattern).replace(r"\*", r"(\d+)") + "$"
                    indices = set()
                    for _, source_pattern, source in sources:
                        if source_pattern == pattern:
                            match = re.search(regex, source)
                            assert match is not None, f"Cannot identify stacked index in {source!r}"
                            indices.add(int(match[1]))
                    assert indices == expected, (
                        f"Incomplete HF stack for {name!r}, {pattern!r}: "
                        f"missing {sorted(expected - indices)}, unexpected {sorted(indices - expected)}"
                    )
