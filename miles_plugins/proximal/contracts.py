"""Training domain contracts; provider wire dictionaries stop at the adapters."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, ConfigDict, Field, FiniteFloat, model_validator

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


class TruncatedImportanceSampling(Contract):
    """The decoupled off-policy correction (Miles's ``--use-tis``). The trainer recomputes
    each token's log-prob before the step and centers the PPO ratio on it; every token's
    gradient is then weighted by the rollout-to-trainer importance ratio, truncated to
    ``[clip_low, clip]``. It holds up with groups several policy versions old, where
    ``rollout_logprobs`` clips around a stale policy."""

    kind: Literal["truncated_importance_sampling"]
    clip: Annotated[FiniteFloat, Field(gt=1)]
    clip_low: Annotated[FiniteFloat, Field(ge=0, lt=1)]


# ``rollout_logprobs``: the PPO ratio's denominator is the rollout engine's log-prob
# (Miles's ``--use-rollout-logprobs``), with no recomputation.
BehaviorCorrection = Literal["rollout_logprobs"] | TruncatedImportanceSampling


def behavior_correction_args(correction: BehaviorCorrection) -> dict[str, object]:
    """The Miles arguments that select a behavior correction; exactly one is enabled."""
    if correction == "rollout_logprobs":
        return {"use_rollout_logprobs": True, "use_tis": False}
    assert isinstance(correction, TruncatedImportanceSampling)
    return {
        "use_rollout_logprobs": False,
        "use_tis": True,
        "tis_clip": correction.clip,
        "tis_clip_low": correction.clip_low,
    }


def behavior_correction_argv(correction: BehaviorCorrection) -> list[str]:
    """``behavior_correction_args`` as Miles's argv: a true boolean is a bare flag, a false one is omitted."""
    argv: list[str] = []
    for name, value in behavior_correction_args(correction).items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not False:
            argv += [flag, str(value)]
    return argv


class Research(Contract):
    behavior_correction: BehaviorCorrection
    lora: LoRA
    sampling: Sampling
    group_size: Annotated[int, Field(ge=2)]
    max_policy_lag: Annotated[int, Field(ge=0)]
    unused_groups: Literal["retry", "drop"]
    max_consecutive_failed_groups: Positive


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
# Chat-template families capture can render with (Miles's --tito-model). Each binds the
# SGLang reasoning and tool-call parsers the replicas must use (check_tito_protocol).
# Inkling is not listed: its template takes a numeric reasoning_effort, which has no
# platform equivalent.
TitoModel = Literal["qwen3", "qwen35", "qwen36", "qwen38small", "qwennext"]
# The efforts a family's fixed template renders. Capture passes the run's effort to
# these templates; a family not listed renders none.
TEMPLATE_REASONING_EFFORTS: dict[str, frozenset[str]] = {
    "qwen38small": frozenset({"xhigh", "medium", "low"}),
}


class ModelProtocol(Contract):
    """How served text becomes reasoning and tool calls, and what effort the harness
    requests. Changes what the agent sees, so it is part of the training contract,
    shared by serving, capture and the platform run request."""

    reasoning_parser: Nonempty
    tool_call_parser: Nonempty
    # Sent to the platform for every run; capture rejects a model call asking otherwise
    # and renders it through the template (TEMPLATE_REASONING_EFFORTS).
    reasoning_effort: ReasoningEffort


class Service(Contract):
    url: Endpoint
    api_key_env: Nonempty


class CaptureService(Service):
    """``api_key_env``: Miles's admin credential. ``platform_key_env``: the credential the
    platform's endpoint registry holds to call rollout routes on behalf of agent-px."""

    platform_key_env: Nonempty


class PlatformRoute(Contract):
    """How the platform sends a run's model calls to capture: the registry entry
    (``endpoint_name``) under the platform model id (``model``). The registry derives
    ``<capture url>/rollouts/<platform rollout id>/v1`` as each rollout's base URL.

    ``endpoint_name`` unset routes by the model's default endpoint: a training node that
    registers a new capture endpoint on each start (``training.RealPlatform``). The
    platform pins the resolved endpoint for each run's lifetime."""

    model: Nonempty
    endpoint_name: Nonempty | None


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
    platform_route: PlatformRoute
    capture: CaptureService
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
    tito_model: TitoModel
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
        rendered = TEMPLATE_REASONING_EFFORTS.get(self.tito_model)
        if rendered is not None and self.model_protocol.reasoning_effort not in rendered:
            raise ValueError(
                f"The {self.tito_model} template renders reasoning effort {', '.join(sorted(rendered))}, "
                f"not {self.model_protocol.reasoning_effort!r}"
            )
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
    behavior_correction: BehaviorCorrection
    lora: LoRA
    sampling: Sampling
    group_size: int
    tokenizer: Nonempty
    tito_model: TitoModel
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


class ServingContract(Contract):
    """What a serving deployment binds for every run it serves: the base model, how its
    capture renders and parses turns, the adapter shape its engines load, and its
    sequence ceiling.

    Everything else about a run (run id, tasks, harness, token budgets within the
    ceiling) travels with each session's attempt, so a new run on the same deployment
    needs no redeploy. Changing any field here does: the replicas were started with it.
    """

    base_model: BaseModelIdentity
    tokenizer: Nonempty
    tito_model: TitoModel
    enable_thinking: bool
    model_protocol: ModelProtocol
    lora_rank: Positive
    lora_target_modules: tuple[Nonempty, ...]
    max_sequence_tokens: Positive


def serving_contract(config: RunConfig) -> ServingContract:
    return ServingContract(
        base_model=config.base_model,
        tokenizer=config.tokenizer_path.name,
        tito_model=config.tito_model,
        enable_thinking=config.enable_thinking,
        model_protocol=config.model_protocol,
        lora_rank=config.research.lora.rank,
        lora_target_modules=config.research.lora.target_modules,
        max_sequence_tokens=config.research.sampling.max_sequence_tokens,
    )


def serving_mismatches(config: RunConfig, deployed: ServingContract) -> list[str]:
    """Why a deployment cannot serve this run; empty when it can."""
    run = serving_contract(config)
    same = ("base_model", "tokenizer", "tito_model", "enable_thinking", "model_protocol", "lora_target_modules")
    reasons = [
        f"{name}: the run has {getattr(run, name)!r}, the deployment {getattr(deployed, name)!r}"
        for name in same
        if getattr(run, name) != getattr(deployed, name)
    ]
    if run.lora_rank > deployed.lora_rank:
        reasons.append(f"lora rank {run.lora_rank} exceeds the deployment's maximum {deployed.lora_rank}")
    if run.max_sequence_tokens > deployed.max_sequence_tokens:
        reasons.append(
            f"max_sequence_tokens {run.max_sequence_tokens} exceeds the deployment's {deployed.max_sequence_tokens}"
        )
    return reasons


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


# The platform names a run's rollouts ``<run id>-rollout-<index>``; Miles runs have one.
ROLLOUT_SUFFIX = "-rollout-0"

# Modal routes requests that carry the same ``Modal-Session-Id`` to the same container.
# Capture keeps each rollout's token history in the replica that serves it, so every
# caller of a rollout's routes sends this header: the trainer (create, seal, fetch,
# release) and the platform's agent (chat calls, from the registry's rollout_capture
# client). The value is the SHA-256 of the platform rollout ID, on both sides.
AFFINITY_HEADER = "Modal-Session-Id"


def platform_rollout_id(attempt_id: str) -> str:
    """The platform rollout ID of an attempt's single-instance run."""
    return f"{attempt_id}{ROLLOUT_SUFFIX}"


def affinity_headers(rollout_id: str) -> dict[str, str]:
    """Pin every call for one rollout to the replica that holds its session."""
    return {AFFINITY_HEADER: hashlib.sha256(rollout_id.encode()).hexdigest()}


class SessionHandle(Contract):
    """A registered attempt's capture session. ``base_url`` is the rollout route the
    platform derives for this run; no per-session credential leaves Miles. Calls about
    the session carry ``affinity_headers(rollout_id)``."""

    session_id: SafeId
    rollout_id: Nonempty
    base_url: Endpoint
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


class PinnedDataset:
    """A run's dataset hash and task set, computed once: per-attempt checks must not
    re-serialize the whole pinned dataset (7,473 tasks is ~1 MB, ~8 ms per digest)."""

    def __init__(self, dataset: TaskDataset):
        self.dataset = dataset
        self.sha256 = digest(dataset)
        self.tasks = frozenset(dataset.tasks)


_PINNED: dict[int, PinnedDataset] = {}


def pinned_dataset(dataset: TaskDataset) -> PinnedDataset:
    """The cached hash and task set of a dataset object (contracts are immutable)."""
    entry = _PINNED.get(id(dataset))
    if entry is None or entry.dataset is not dataset:  # The entry holds the object, so its id is never reused.
        entry = _PINNED[id(dataset)] = PinnedDataset(dataset)
    return entry


def read_run_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate_json(Path(path).read_bytes())
