from typing import Any, ClassVar, Literal

from miles.utils.args.enhanced_argparse_namespace import EnhancedArgparseNamespace


class FsdpArgsNamespace(EnhancedArgparseNamespace):
    backend_name: Literal["fsdp"]
    _mutable_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "ckpt_step",
            "finetune",
            "load",
            "lr_decay_iters",
            "no_load_optim",
            "no_load_rng",
            "rank",
            "start_rollout_id",
            "train_iters",
            "world_size",
        }
    )

    def __init__(self, *, backend_name: Literal["fsdp"] = "fsdp", **values: Any) -> None:
        assert backend_name == "fsdp", f"Invalid FSDP backend name: {backend_name!r}"
        super().__init__(backend_name=backend_name, **values)
