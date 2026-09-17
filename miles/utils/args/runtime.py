from argparse import Namespace
from dataclasses import fields
from typing import Any, Self

from pydantic import ConfigDict, model_validator

from miles.backends.fsdp_utils.arguments import FSDPArgs
from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import (
    MegatronArgsNamespace,
    MegatronConfig,
    MegatronTrainerConfig,
    compute_trainer_args,
)
from miles.utils.args.component_multi_lora import MultiLoraOnlyConfig
from miles.utils.args.component_orchestrator import OrchestratorOnlyConfig
from miles.utils.args.component_rollout import InferenceControllerOnlyConfig, RolloutOnlyConfig
from miles.utils.args.component_shared import SglangFieldsConfig
from miles.utils.args.component_trainer import TrainerOnlyConfig
from miles.utils.args.configs.algo import AlgoConfig
from miles.utils.args.configs.ci import CiConfig
from miles.utils.args.configs.cluster import ClusterConfig
from miles.utils.args.configs.custom_megatron_plugins import CustomMegatronPluginsConfig
from miles.utils.args.configs.dashboard import DashboardConfig
from miles.utils.args.configs.data import DataConfig
from miles.utils.args.configs.debug import DebugConfig
from miles.utils.args.configs.eval import EvalConfig
from miles.utils.args.configs.fault_tolerance import FaultToleranceConfig
from miles.utils.args.configs.lora import LoraConfig
from miles.utils.args.configs.mlflow import MlflowConfig
from miles.utils.args.configs.mtp_training import MtpTrainingConfig
from miles.utils.args.configs.network import NetworkConfig
from miles.utils.args.configs.on_policy_distillation import OnPolicyDistillationConfig
from miles.utils.args.configs.prefill_decode_disaggregation import PrefillDecodeDisaggregationConfig
from miles.utils.args.configs.prometheus import PrometheusConfig
from miles.utils.args.configs.reward_model import RewardModelConfig
from miles.utils.args.configs.rollout import RolloutRelatedConfig
from miles.utils.args.configs.rollout_buffer import RolloutBufferConfig
from miles.utils.args.configs.router import RouterConfig
from miles.utils.args.configs.run_uuid import RunUuidConfig
from miles.utils.args.configs.session import SessionConfig
from miles.utils.args.configs.tensorboard import TensorboardConfig
from miles.utils.args.configs.train import TrainConfig
from miles.utils.args.configs.wandb import WandbConfig
from miles.utils.args.runtime_base import BaseLeafConfig


class OrchestratorConfig(
    BaseLeafConfig,
    OrchestratorOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    pass


class TrainerConfig(
    BaseLeafConfig,
    TrainerOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    @classmethod
    def from_all_config(cls, args: "AllConfig", *, trainer: MegatronTrainerConfig) -> Self:
        backend_values = (
            args.raw_megatron.base_args if args.train_backend == "megatron" else vars(args.fsdp)
        )
        base = Namespace(**(dict(args) | backend_values))
        trainer_args = compute_trainer_args(args=base, trainer=trainer)
        values = vars(trainer_args)
        if args.train_backend == "megatron":
            trainer_backend = MegatronArgsNamespace.from_args(
                trainer_args,
                names=set(args.raw_megatron.base_args),
            )
        else:
            fsdp_names = {field.name for field in fields(FSDPArgs)} | {
                "calculate_per_token_loss",
                "clip_grad",
                "no_save_optim",
            }
            trainer_backend = FsdpArgsNamespace.from_args(trainer_args, names=fsdp_names)
            if trainer_backend.fsdp_cpu_offload:
                values["offload_train"] = False
        if trainer.role == "critic":
            values["loss_type"] = "value_loss"
        values.update(
            trainer_backend=trainer_backend,
            trainer_id=trainer.trainer_id,
            trainer_model_id=trainer.model_id,
            trainer_role=trainer.role,
        )
        return cls.from_config(values)

    @model_validator(mode="before")
    @classmethod
    def _validate_trainer_backend(cls, values: Any) -> Any:
        if not isinstance(values, dict) or "trainer_backend" not in values:
            return values
        backend_class: type[MegatronArgsNamespace | FsdpArgsNamespace]
        match values.get("train_backend", cls.model_fields["train_backend"].default):
            case "megatron":
                backend_class = MegatronArgsNamespace
            case "fsdp":
                backend_class = FsdpArgsNamespace
            case backend:
                raise ValueError(f"Unsupported training backend: {backend!r}")
        return values | {"trainer_backend": backend_class._validate(values["trainer_backend"])}


class InferenceControllerConfig(
    BaseLeafConfig,
    InferenceControllerOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    pass


class RolloutConfig(
    BaseLeafConfig,
    RolloutOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    pass


class MultiLoraConfig(
    BaseLeafConfig,
    MultiLoraOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    pass


class AllConfig(
    BaseLeafConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
    OrchestratorOnlyConfig,
    RolloutOnlyConfig,
    InferenceControllerOnlyConfig,
    MultiLoraOnlyConfig,
):
    # TODO: Remove extra="allow" after backend, custom, and derived fields have explicit config owners.
    model_config = ConfigDict(extra="allow")

    raw_megatron: MegatronConfig
    fsdp: FsdpArgsNamespace
