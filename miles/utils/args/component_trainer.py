from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.schema import BaseConfig


class TrainerOnlyConfig(BaseConfig):
    megatron: MegatronArgsNamespace
    trainer_role: str
