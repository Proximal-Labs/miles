"""Miles-owned serving pool: one typed deployment, SGLang arguments derived by Miles.

The pool is the set of Modal GPU replicas that serve immutable LoRA versions to
platform rollouts. Miles owns its setup so inference and training cannot drift:

- the replica image is the Miles image, so SGLang is Miles's pinned build;
- base model, tokenizer, LoRA rank/alpha/targets come from the same run config
  that produces the trainer's arguments (see runtime.training_argv);
- SGLang server arguments are rendered by Miles's helper, parsed by SGLang, and
  the *resolved* settings are checked against the derived ones, so no alias or
  extra flag can change the model, tokenizer, LoRA shape, parsers or address.

Capture runs in every replica (see serve_replica): the platform's endpoint registry
records the deployment's URL as a rollout_capture endpoint, and sticky routing keeps
each rollout on the replica that holds its session (contracts.AFFINITY_HEADER).
This module is pure configuration; serving_app.py is the Modal deployment.
"""

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, model_validator

from miles_plugins.proximal.contracts import Contract, Positive, RunConfig
from miles_plugins.proximal.gateway import GatewayConfig
from miles_plugins.proximal.modal_volume import VolumeDestination
from miles_plugins.proximal.replica import ReplicaConfig
from miles_plugins.proximal.snapshot import Nonempty

ENGINE_PORT = 30000
GATEWAY_PORT = 8000
# GPUs whose SGLang build has the trtllm_mha attention kernels (SM100/SM103).
BLACKWELL_GPUS = frozenset({"B200", "B300"})


class Attention(Contract):
    """SGLang's full-attention kernel for the replica.

    Left to SGLang, Qwen3.5-family hybrids (Qwen3.8) on Blackwell fall back to Triton with
    one-token KV pages; trtllm_mha measured 2x faster decode at 100k-token contexts.
    """

    backend: Literal["triton", "flashinfer", "fa3", "trtllm_mha"]
    page_size: Positive


class Speculation(Contract):
    """Speculative decoding with the checkpoint's own MTP head (SGLang's NEXTN), one chain.

    Verification is SGLang's default (target sampling), not a setting. Measured 2026-09-25
    on the pinned Miles image with Qwen3.8-27B and a LoRA adapter (600 five-token samples
    per prompt, compared against serving without speculation): default verification kept
    the sampled distribution (no near-impossible tokens, sampled-token logprobs within
    noise), while --speculative-use-rejection-sampling emitted tokens the model gives
    logprob -24 to -34 in ~2% of speculated positions. Returned logprobs are the served
    (LoRA) model's either way. The draft head runs without the adapter, which lowers
    acceptance but not correctness.
    """

    algorithm: Literal["NEXTN"]
    num_steps: Positive
    num_draft_tokens: Positive

    @model_validator(mode="after")
    def _chain(self) -> "Speculation":
        if self.num_draft_tokens != self.num_steps + 1:
            raise ValueError(
                "A single draft chain verifies num_steps + 1 tokens: set num_draft_tokens = num_steps + 1"
            )
        return self


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
    # Modal secret holding the gateway credential under ``gateway_key_env``.
    gateway_secret: Nonempty
    gateway_key_env: Nonempty
    # Modal secret holding the capture credentials the run config names: the trainer's
    # (capture.api_key_env) and the platform's (capture.platform_key_env). Capture runs
    # in every replica, next to the SGLang that serves its sessions.
    capture_secret: Nonempty
    # CPU cores for each replica: SGLang's tokenizer, scheduler and detokenizer processes
    # plus the front process (gateway and capture). Modal otherwise grants about one.
    cpu: Positive
    # Memory reserved for each replica: SGLang's host memory plus capture's live sessions,
    # which hold every in-flight rollout's token history.
    memory_mib: Positive
    # Whether Modal's proxy authenticates callers before they reach a replica. The
    # platform's agents call capture directly and hold only the capture credential, so a
    # pool serving platform rollouts sets this false; every route checks its own key.
    modal_proxy_auth: bool
    # SGLang's full-attention kernel; null leaves the choice to SGLang.
    attention: Attention | None
    # KV cache precision: "auto" keeps the checkpoint's dtype (BF16 for Qwen3.8); fp8_e4m3
    # halves KV per token, so twice the context fits, and changes attention numerics against
    # the BF16 trainer (rollout_logprobs behavior correction absorbs the gap).
    kv_cache_dtype: Literal["auto", "fp8_e4m3"]
    # Speculative decoding; null serves without it.
    speculative: Speculation | None
    # Operational SGLang flags only; see OPERATIONAL_ENGINE_SETTINGS.
    extra_engine_args: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> "ServingDeployment":
        if self.min_replicas != self.max_replicas:
            # Capture's sessions live in the replicas: scaling one down would drop the
            # rollouts it holds. A fixed size until scale-down drains sessions first.
            raise ValueError("Capture sessions live in the replicas; set min_replicas equal to max_replicas")
        for path in (self.base_mount, self.adapter_mount, self.local_cache):
            if not path.is_absolute():
                raise ValueError("Container paths must be absolute")
        if self.adapter_mount == self.base_mount:
            raise ValueError("Adapters and base weights use separate Volumes")
        if self.attention is not None and self.attention.backend == "trtllm_mha":
            if self.gpu.split(":")[0] not in BLACKWELL_GPUS:
                raise ValueError(
                    f"trtllm_mha attention needs a Blackwell GPU ({', '.join(sorted(BLACKWELL_GPUS))}), not {self.gpu}"
                )
            if self.attention.page_size == 1:
                raise ValueError("trtllm_mha attention needs KV pages larger than one token")
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
        "reasoning_parser": run.model_protocol.reasoning_parser,
        "tool_call_parser": run.model_protocol.tool_call_parser,
        "skip_server_warmup": True,
        "enable_metrics": True,
        "kv_cache_dtype": deployment.kv_cache_dtype,
        **_attention_args(deployment.attention),
        **_speculative_args(deployment.speculative),
    }


def _attention_args(attention: Attention | None) -> dict[str, object]:
    if attention is None:
        return {}
    return {"attention_backend": attention.backend, "page_size": attention.page_size}


def _speculative_args(speculative: Speculation | None) -> dict[str, object]:
    if speculative is None:
        return {}
    return {
        "speculative_algorithm": speculative.algorithm,
        "speculative_num_steps": speculative.num_steps,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": speculative.num_draft_tokens,
    }


# Extras may change only these resolved settings: memory, scheduling, CUDA graphs and
# logging. Anything else (load format, attention/sampling backends, quantization,
# model/tokenizer identity, LoRA shape, parsers, address) could change the served
# model or its numerics, so it is derived from the run config or not supported.
OPERATIONAL_ENGINE_SETTINGS = frozenset(
    {
        "mem_fraction_static",
        "max_running_requests",
        "max_total_tokens",
        "max_prefill_tokens",
        "chunked_prefill_size",
        "schedule_policy",
        "schedule_conservativeness",
        "cuda_graph_max_bs",
        "disable_cuda_graph",
        "log_level",
        "log_requests",
        "enable_cache_report",
        "watchdog_timeout",
    }
)


def _server_args_fields(server_args: object) -> tuple[str, ...]:
    # ServerArgs is a msgspec Struct, a dataclass, or a plain class depending on the SGLang build.
    import dataclasses

    if (struct_fields := getattr(type(server_args), "__struct_fields__", None)) is not None:
        return tuple(struct_fields)
    if dataclasses.is_dataclass(server_args):
        return tuple(field.name for field in dataclasses.fields(server_args))
    names = tuple(vars(server_args))
    if not names:
        raise TypeError("Cannot enumerate SGLang ServerArgs fields; refusing to validate extras blindly")
    return names


def engine_argv(run: RunConfig, deployment: ServingDeployment) -> list[str]:
    """Render through Miles's helper, parse with SGLang, and allow only operational extras.

    The resolved settings with and without the extras must differ only in
    OPERATIONAL_ENGINE_SETTINGS, whatever flag spelling or alias an extra used.
    """
    from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv, server_args_to_argv

    base_argv = server_args_to_argv(engine_server_args(run, deployment))
    baseline = parse_server_args_argv(base_argv)
    argv = [*base_argv, *deployment.extra_engine_args]
    resolved = parse_server_args_argv(argv)
    names = _server_args_fields(resolved)
    changed = sorted(name for name in names if getattr(resolved, name) != getattr(baseline, name))
    if not_operational := [name for name in changed if name not in OPERATIONAL_ENGINE_SETTINGS]:
        raise ValueError(f"Extra engine flags change non-operational settings: {not_operational}")
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
