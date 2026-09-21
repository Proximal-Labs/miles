from argparse import Namespace
from dataclasses import dataclass


def build_offline_conversion_config(megatron_args: Namespace) -> Namespace:
    values = vars(megatron_args)
    return Namespace(
        backend=megatron_args,
        hf_checkpoint=values.get("hf_checkpoint"),
        extra_high_precision_layers_megatron=values.get("extra_high_precision_layers_megatron"),
        sglang=values.get("sglang")
        or _OfflineSglangConfig(
            values={
                "moe_runner_backend": values.get("sglang_moe_runner_backend", "auto"),
                "moe_a2a_backend": values.get("sglang_moe_a2a_backend", "none"),
            }
        ),
    )


@dataclass(frozen=True)
class _OfflineSglangConfig:
    values: dict[str, str]

    def common_value(self, name: str) -> str:
        return self.values[name]
