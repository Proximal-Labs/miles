from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.schema import BaseConfig


class TrainerOnlyConfig(BaseConfig):
    trainer_backend: MegatronArgsNamespace | FsdpArgsNamespace
    trainer_role: str
