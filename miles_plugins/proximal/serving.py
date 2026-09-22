"""Miles-owned serving pool: one typed deployment, SGLang arguments derived by Miles.

The pool is the set of Modal GPU replicas that serve immutable LoRA versions to
platform rollouts. Miles owns its setup so inference and training cannot drift:

- the replica image is the Miles image, so SGLang is Miles's pinned build;
- base model, tokenizer, LoRA rank/alpha/targets come from the same run config
  that produces the trainer's arguments (see runtime.training_argv);
- SGLang server arguments are rendered and re-parsed by Miles's own helpers.

The platform only records the deployment's URL in its endpoint registry.
This module is pure configuration; serving_app.py is the Modal deployment.
"""

from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import Field, model_validator

from miles_plugins.proximal.contracts import Contract, Positive, RunConfig
from miles_plugins.proximal.gateway import GatewayConfig
from miles_plugins.proximal.modal_volume import VolumeDestination
from miles_plugins.proximal.replica import ReplicaConfig
from miles_plugins.proximal.snapshot import Nonempty

ENGINE_PORT = 30000
GATEWAY_PORT = 8000


class ServingDeployment(Contract):
    """Operational shape of the pool. Research semantics stay in the run config."""

    app_name: Nonempty
    image: Annotated[str, Field(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")]  # Miles image, pinned by digest.
    gpu: Nonempty
    tensor_parallel: Positive
    routing_region: Nonempty
    min_replicas: Annotated[int, Field(ge=0)]
    max_replicas: Positive
    target_concurrency: Positive
    scaledown_window_seconds: Positive
    startup_timeout_seconds: Positive
    # Base weights staged on an existing Volume at the tokenizer path's basename.
    base_volume: VolumeDestination
    base_mount: PurePosixPath
    adapter_mount: PurePosixPath
    local_cache: PurePosixPath
    max_loaded_adapters: Positive
    reasoning_parser: Nonempty
    tool_call_parser: Nonempty
    # Modal secret holding the gateway credential under ``gateway_key_env``.
    gateway_secret: Nonempty
    gateway_key_env: Nonempty
    # Additional SGLang flags, validated by SGLang's own parser at render time.
    extra_engine_args: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> "ServingDeployment":
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas exceeds max_replicas")
        for path in (self.base_mount, self.adapter_mount, self.local_cache):
            if not path.is_absolute():
                raise ValueError("Container paths must be absolute")
        if self.adapter_mount == self.base_mount:
            raise ValueError("Adapters and base weights use separate Volumes")
        forbidden = {"--enable-lora", "--lora-paths", "--max-lora-rank", "--lora-target-modules", "--model-path"}
        if any(arg.split("=", 1)[0] in forbidden for arg in self.extra_engine_args):
            raise ValueError("LoRA and model flags are derived from the run config, not passed as extras")
        return self


def engine_model_path(run: RunConfig, deployment: ServingDeployment) -> str:
    return str(deployment.base_mount / Path(run.tokenizer_path).name)


def engine_server_args(run: RunConfig, deployment: ServingDeployment) -> dict[str, object]:
    """SGLang ServerArgs fields for one replica, derived from the run's LoRA contract.

    Mirrors Miles's multi-LoRA engine settings: named adapters are loaded at
    runtime by the gateway, so no startup ``lora_paths``.
    """
    # Local import: pulls torch; only the renderer and the replica need it.
    from miles.backends.megatron_utils.lora.utils import convert_target_modules_to_hf

    return {
        "model_path": engine_model_path(run, deployment),
        "served_model_name": run.base_model.name,
        "trust_remote_code": False,
        "device": "cuda",  # Explicit: replicas are GPU hosts; rendering must not probe this machine.
        "host": "127.0.0.1",
        "port": ENGINE_PORT,
        "tp_size": deployment.tensor_parallel,
        "enable_lora": True,
        "max_lora_rank": run.research.lora.rank,
        "lora_target_modules": convert_target_modules_to_hf(list(run.research.lora.target_modules)),
        "max_loras_per_batch": deployment.max_loaded_adapters,
        "max_loaded_loras": deployment.max_loaded_adapters,
        "reasoning_parser": deployment.reasoning_parser,
        "tool_call_parser": deployment.tool_call_parser,
        "skip_server_warmup": True,
        "enable_metrics": True,
    }


def engine_argv(run: RunConfig, deployment: ServingDeployment) -> list[str]:
    """Render through Miles's helper, append validated extras, and re-parse with SGLang."""
    from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv, server_args_to_argv

    argv = [*server_args_to_argv(engine_server_args(run, deployment)), *deployment.extra_engine_args]
    parse_server_args_argv(argv)  # SGLang rejects unknown or conflicting flags before any GPU is used.
    return argv


def gateway_config(run: RunConfig, deployment: ServingDeployment) -> GatewayConfig:
    return GatewayConfig(
        replica=ReplicaConfig(
            base_model=run.base_model,
            served_model_name=run.base_model.name,
            backend_url=f"http://127.0.0.1:{ENGINE_PORT}",
        ),
        api_key_env=deployment.gateway_key_env,
        engine_model_path=engine_model_path(run, deployment),
        max_loaded_adapters=deployment.max_loaded_adapters,
    )
