from typing import Annotated, Literal

from pydantic import Field

from miles.utils.args.runtime import InferenceControllerConfig, MultiLoraConfig, RolloutConfig, TrainerConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.connection_config import StaticConnConfig


class RolloutWorkerConfig(FrozenStrictBaseModel):
    kind: Literal["rollout"]
    args: RolloutConfig


class MultiLoraWorkerConfig(FrozenStrictBaseModel):
    kind: Literal["multi_lora"]
    args: MultiLoraConfig


class InferenceWorkerConfig(FrozenStrictBaseModel):
    kind: Literal["inference_controller", "inference_registration_reporter"]
    args: InferenceControllerConfig


class TrainerWorkerConfig(FrozenStrictBaseModel):
    kind: Literal["trainer", "trainer_controller"]
    args: TrainerConfig


class RuntimeConfig(FrozenStrictBaseModel):
    worker: Annotated[
        RolloutWorkerConfig | MultiLoraWorkerConfig | InferenceWorkerConfig | TrainerWorkerConfig,
        Field(discriminator="kind"),
    ]
    static_connections: StaticConnConfig
