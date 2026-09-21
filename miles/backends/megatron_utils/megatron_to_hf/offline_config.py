from argparse import Namespace
from dataclasses import dataclass


def build_offline_conversion_config(megatron_args: Namespace, origin_hf_dir: str | None = None) -> Namespace:
    values = vars(megatron_args)
    export_metadata = values.get("export_metadata") or values
    return Namespace(
        backend=megatron_args,
        hf_checkpoint=origin_hf_dir if origin_hf_dir is not None else export_metadata.get("hf_checkpoint"),
        extra_high_precision_layers_megatron=export_metadata.get("extra_high_precision_layers_megatron"),
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
