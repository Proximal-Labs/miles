from typing import ClassVar

from miles.utils.args.enhanced_argparse_namespace import EnhancedArgparseNamespace


class FsdpArgsNamespace(EnhancedArgparseNamespace):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset(
        {"rank", "world_size", "start_rollout_id", "train_iters", "lr_decay_iters", "load", "ckpt_step"}
    )
