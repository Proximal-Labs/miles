from argparse import Namespace
from typing import Any

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import (
    MegatronArgsNamespace,
    MegatronTrainerConfig,
    compute_trainer_args,
)
from miles.utils.args.runtime import AllConfig, TrainerConfig


_IMMUTABLE_DUPLICATED_TRAINER_FIELDS = frozenset(
    {
        "async_save",
        "distributed_backend",
        "distributed_timeout_minutes",
        "eval_interval",
        "global_batch_size",
        "micro_batch_size",
        "num_layers",
        "save",
        "save_interval",
        "seed",
        "wandb_project",
    }
)
# Ref/teacher checkpoint loading temporarily overrides these backend fields and restores them afterward.
_MUTABLE_DUPLICATED_TRAINER_FIELDS = frozenset({"ckpt_step", "load"})


# TODO: After zhichen's training backend refactor, make per-trainer argument computation consume structured configs without flattening Miles and backend fields.
def compute_trainer_config(all_config: AllConfig, *, trainer: MegatronTrainerConfig) -> TrainerConfig:
    base_backend_values = (
        all_config.raw_megatron.base_args
        if all_config.train_backend == "megatron"
        else vars(all_config.raw_fsdp)
    )
    base_args = Namespace(**(dict(all_config) | base_backend_values))
    values = vars(compute_trainer_args(args=base_args, trainer=trainer))

    backend_cls = MegatronArgsNamespace if all_config.train_backend == "megatron" else FsdpArgsNamespace
    values["backend"] = backend_cls(**{name: values[name] for name in base_backend_values})

    return TrainerConfig.model_validate(
        {name: values[name] for name in TrainerConfig.model_fields if name in values}
    )


def validate_shared_trainer_fields(args: TrainerConfig) -> None:
    _validate_duplicated_fields(
        values_a=dict(args),
        values_b=vars(args.backend),
        expected_immutable_fields=_IMMUTABLE_DUPLICATED_TRAINER_FIELDS,
        expected_mutable_fields=_MUTABLE_DUPLICATED_TRAINER_FIELDS,
        actual_mutable_fields=type(args)._mutable_fields | type(args.backend)._mutable_fields,
    )


def _validate_duplicated_fields(
    values_a: dict[str, Any],
    values_b: dict[str, Any],
    expected_immutable_fields: set[str] | frozenset[str],
    expected_mutable_fields: set[str] | frozenset[str],
    actual_mutable_fields: set[str] | frozenset[str],
) -> None:
    duplicated_fields = values_a.keys() & values_b.keys()
    if unclassified := duplicated_fields - (expected_immutable_fields | expected_mutable_fields):
        raise ValueError(f"Unclassified duplicated fields: {sorted(unclassified)}")
    if mutable := duplicated_fields & expected_immutable_fields & actual_mutable_fields:
        raise ValueError(f"Duplicated fields must be immutable: {sorted(mutable)}")

    for name in duplicated_fields:
        if values_a[name] != values_b[name]:
            raise ValueError(f"Duplicated field {name!r} differs: {values_a[name]!r} != {values_b[name]!r}")
