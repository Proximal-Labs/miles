"""Training domain contracts; provider wire dictionaries stop at the adapters."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, ConfigDict, Field, FiniteFloat, SecretStr, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.modal_volume import VolumeDestination
from miles_plugins.proximal.snapshot import BaseModelIdentity, Digest, Nonempty, SnapshotReference

Positive = Annotated[int, Field(gt=0)]
SafeId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,100}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


def _endpoint(value: str) -> str:
    url = urlsplit(value)
    if url.username or url.password or url.query or url.fragment or not url.hostname:
        raise ValueError("Endpoint must have a host and no embedded credentials/query/fragment")
    if url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("Use HTTPS, or HTTP on loopback for CPU tests")
    return value.rstrip("/")


Endpoint = Annotated[str, AfterValidator(_endpoint)]


class Contract(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Harness(Contract):
    agent_type: Nonempty
    revision: Revision
    max_turns: Positive
    timeout_seconds: Positive
    p2p_enforce: bool


class Task(Contract):
    environment_id: Positive
    image_id: Positive
    source_commit_sha: Revision


class TaskDataset(Contract):
    schema_version: Literal[1] = 1
    project_id: Positive
    tasks: tuple[Task, ...]

    @model_validator(mode="after")
    def _nonempty_unique(self) -> "TaskDataset":
        if not self.tasks or len({task.environment_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Dataset needs nonempty, unique environment membership")
        return self


class Sampling(Contract):
    temperature: Annotated[FiniteFloat, Field(gt=0)]
    top_p: Annotated[FiniteFloat, Field(gt=0, le=1)]
    top_k: int
    max_tokens: Positive
    max_sequence_tokens: Positive
    # This first pass trains on untransformed model logprobs. Restrict the
    # sampling distribution so a serving flag cannot silently change their meaning.
    logprob_semantics: Literal["untransformed"]
    budget_policy: Literal["cap_to_remaining_context"]

    @model_validator(mode="after")
    def _supported_distribution(self) -> "Sampling":
        if self.temperature != 1 or self.top_p != 1 or self.top_k != -1:
            raise ValueError("First pass requires temperature=1, top_p=1, top_k=-1 for exact behavior logprobs")
        if self.max_tokens > self.max_sequence_tokens:
            raise ValueError("Per-turn token budget exceeds sequence budget")
        return self


class LoRA(Contract):
    """The trained adapter's shape. Single source for trainer args and serving engines."""

    rank: Positive
    alpha: Positive
    # Megatron module names, as Miles's --target-modules takes them.
    target_modules: Annotated[tuple[Nonempty, ...], Field(min_length=1)]


class Research(Contract):
    behavior_correction: Literal["rollout_logprobs"]
    lora: LoRA
    sampling: Sampling
    group_size: Annotated[int, Field(ge=2)]
    max_policy_lag: Annotated[int, Field(ge=0)]
    unused_groups: Literal["retry", "drop"]
    max_consecutive_failed_groups: Positive


class ModelProtocol(Contract):
    """How served text becomes reasoning and tool calls. Changes what the agent sees,
    so it is part of the training contract, shared by serving and capture."""

    reasoning_parser: Nonempty
    tool_call_parser: Nonempty


class Service(Contract):
    url: Endpoint
    api_key_env: Nonempty


class SharedDiskArtifacts(Contract):
    """One filesystem that every store reader and writer mounts with read-after-write
    visibility (e.g. a single host's persistent disk). No sync step."""

    kind: Literal["shared_disk"]


class ModalVolumeArtifacts(Contract):
    """``artifact_directory`` is where this Volume is mounted in every container.
    Writers commit before indexing a group; readers reload on a miss."""

    kind: Literal["modal_volume"]
    volume: VolumeDestination


ArtifactStorage = Annotated[SharedDiskArtifacts | ModalVolumeArtifacts, Field(discriminator="kind")]


class RunConfig(Contract):
    run_id: SafeId
    base_model: BaseModelIdentity
    dataset: TaskDataset
    harness: Harness
    research: Research
    platform: Service
    capture: Service
    inference_url: Endpoint
    # Header name -> environment variable name, never credential values.
    inference_header_env: dict[str, Nonempty]
    volume: VolumeDestination
    # Mount point for stored group payloads, sealed captures and publication staging.
    artifact_directory: Path
    artifact_storage: ArtifactStorage
    # Environment variable holding the Postgres DSN for the rollout store index.
    store_dsn_env: Nonempty
    tokenizer_path: Path
    tito_model: Literal["qwen3"]
    enable_thinking: bool
    model_protocol: ModelProtocol
    max_in_flight_samples: Positive
    completed_group_capacity: Positive
    request_timeout_seconds: Positive = 1800
    poll_interval_seconds: Annotated[FiniteFloat, Field(gt=0)] = 2.0

    @model_validator(mode="after")
    def _capacity(self) -> "RunConfig":
        if self.max_in_flight_samples < self.research.group_size:
            raise ValueError("In-flight capacity must accommodate one complete group")
        forbidden = {"host", "content-length", "transfer-encoding", "x-proximal-policy-sha256"}
        if any(name.lower() in forbidden for name in self.inference_header_env):
            raise ValueError("Invalid inference authentication header")
        return self


class TrainingContract(Contract):
    """Everything a stored group must share with the consuming trainer.

    Operational fields (URLs, timeouts, capacities) are excluded: changing them
    does not change what a group means as training data.
    """

    run_id: SafeId
    base_model: BaseModelIdentity
    dataset: TaskDataset
    harness: Harness
    behavior_correction: Literal["rollout_logprobs"]
    lora: LoRA
    sampling: Sampling
    group_size: int
    tokenizer: Nonempty
    tito_model: Literal["qwen3"]
    enable_thinking: bool
    model_protocol: ModelProtocol


def training_contract(config: RunConfig) -> TrainingContract:
    return TrainingContract(
        run_id=config.run_id,
        base_model=config.base_model,
        dataset=config.dataset,
        harness=config.harness,
        behavior_correction=config.research.behavior_correction,
        lora=config.research.lora,
        sampling=config.research.sampling,
        group_size=config.research.group_size,
        tokenizer=config.tokenizer_path.name,
        tito_model=config.tito_model,
        enable_thinking=config.enable_thinking,
        model_protocol=config.model_protocol,
    )


class Policy(Contract):
    run_id: SafeId
    version: Positive
    snapshot: SnapshotReference
    base_model: BaseModelIdentity


class Attempt(Contract):
    attempt_id: SafeId
    run_id: SafeId
    group_id: SafeId
    sample_index: Annotated[int, Field(ge=0)]
    dataset_sha256: Digest
    task: Task
    harness: Harness
    policy: Policy
    sampling: Sampling

    @model_validator(mode="after")
    def _run(self) -> "Attempt":
        if self.policy.run_id != self.run_id:
            raise ValueError("Policy belongs to another training run")
        return self


class SessionHandle(Contract):
    session_id: SafeId
    base_url: Endpoint
    api_key: SecretStr
    request_sha256: Digest


class CaptureReceipt(Contract):
    session_id: SafeId
    request_sha256: Digest
    policy: Policy
    payload_sha256: Digest
    num_calls: Positive
    num_tokens: Positive


class Grade(Contract):
    run_id: SafeId
    rollout_id: Nonempty
    request_sha256: Digest
    reward: FiniteFloat
    status: Literal["success", "completed"]
    artifacts_url: str | None


class AcceptedAttempt(Contract):
    attempt: Attempt
    capture: CaptureReceipt
    grade: Grade


class PolicyEvidence(Contract):
    snapshot: SnapshotReference
    base_model: BaseModelIdentity
    request_model: Nonempty


def canonical_bytes(value: Contract) -> bytes:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()


def digest(value: Contract) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_run_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate_json(Path(path).read_bytes())
