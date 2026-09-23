"""Strict complete-group eligibility and the durable batch query at Miles's DataBuffer seam."""

import asyncio
import math
from typing import cast

from miles.rollout.fully_async_data_buffer import DataBuffer, DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample
from miles_plugins.proximal.contracts import (
    AcceptedAttempt,
    Policy,
    RunConfig,
    digest,
    pinned_dataset,
    read_run_config,
)
from miles_plugins.proximal.data_source import ConsumptionLedger
from miles_plugins.proximal.store import RolloutStore


def accepted(sample: Sample) -> AcceptedAttempt:
    # Miles metadata is a framework I/O boundary. Narrow it immediately.
    return AcceptedAttempt.model_validate_json(sample.metadata["proximal_accepted"])


def validate_sample(sample: Sample, evidence: AcceptedAttempt) -> None:
    if evidence.capture.policy != evidence.attempt.policy:
        raise ValueError("Capture policy does not match attempt")
    expected = digest(evidence.attempt)
    if evidence.capture.request_sha256 != expected or evidence.grade.request_sha256 != expected:
        raise ValueError("Grade and captured tokens do not certify the same attempt")
    if evidence.grade.run_id != evidence.attempt.attempt_id or sample.reward != evidence.grade.reward:
        raise ValueError("Sample reward lacks matching platform evidence")
    if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
        raise ValueError("Incomplete captured sample")
    sample.validate()  # type: ignore[no-untyped-call]  # Upstream Sample validation.
    if not sample.tokens or len(sample.tokens) > evidence.attempt.sampling.max_sequence_tokens:
        raise ValueError("Invalid captured sequence length")
    if sample.loss_mask is None or not any(sample.loss_mask) or any(x not in (0, 1) for x in sample.loss_mask):
        raise ValueError("Missing assistant-token loss mask")
    if sample.rollout_log_probs is None or any(not math.isfinite(x) or x > 1e-6 for x in sample.rollout_log_probs):
        raise ValueError("Missing/invalid behavior logprobs")
    spans = sample.all_weight_version_spans
    covered: set[int] = set()
    for span in spans:
        if span.version != str(evidence.attempt.policy.version):
            raise ValueError("Mixed or unbound policy versions within a rollout")
        covered.update(range(span.abs_start, span.abs_end))
    offset = len(sample.tokens) - sample.response_length
    if any(offset + i not in covered for i, mask in enumerate(sample.loss_mask) if mask):
        raise ValueError("Assistant tokens are missing verified policy provenance")


def validate_group(config: RunConfig, samples: list[Sample]) -> Policy:
    """A complete group under this run's contract, with full per-sample evidence.

    Runs on put (before storing) and again on get (after loading), so a stored
    payload is never trusted just because it has an index row.
    """
    if len(samples) != config.research.group_size:
        raise ValueError("Incomplete prompt group")
    proofs = [accepted(sample) for sample in samples]
    first = proofs[0].attempt
    indices = set()
    for sample, proof in zip(samples, proofs, strict=True):
        validate_sample(sample, proof)
        attempt = proof.attempt
        if (attempt.run_id, attempt.group_id, attempt.task, attempt.policy) != (
            first.run_id,
            first.group_id,
            first.task,
            first.policy,
        ):
            raise ValueError("Prompt group mixes tasks/policies")
        # Every member, not only the first: each carries its own harness/sampling/dataset.
        if (
            attempt.run_id != config.run_id
            or attempt.harness != config.harness
            or attempt.sampling != config.research.sampling
            or attempt.dataset_sha256 != pinned_dataset(config.dataset).sha256
            or attempt.task not in pinned_dataset(config.dataset).tasks
            or attempt.policy.base_model != config.base_model
        ):
            raise ValueError("Group member was produced under a different training contract")
        indices.add(attempt.sample_index)
    if len(indices) != config.research.group_size:
        raise ValueError("Repeated sample in prompt group")
    return first.policy


class PlatformDataBuffer(DataBuffer):
    """Miles's DataBuffer seam, backed by the durable rollout store.

    put: validate one finished group, persist it (it is a paid rollout), then
    pause the producer while more than ``completed_group_capacity`` fresh
    unconsumed groups are waiting, like Miles's default bounded buffer.
    get: the batch query. Select the oldest fresh, live-lineage group this
    training run has not consumed, record it in the checkpointed ledger, return it.
    Nothing lives only in memory: a restarted process sees the same store.
    """

    def __init__(self, input: DataBufferConstructorInput) -> None:
        self.config = read_run_config(input.args.proximal_config)
        self._unused = input.unused_handler_fn
        self._store: RolloutStore | None = None
        self._ledger: ConsumptionLedger | None = None
        self._stored = asyncio.Event()
        self._consumed_event = asyncio.Event()
        self._version: int | None = None
        self._failures = 0
        self._consumed = 0
        self._persisted = 0

    def attach(self, *, store: RolloutStore, ledger: ConsumptionLedger) -> None:
        """Composition root wiring: the store connection and the checkpointed ledger."""
        self._store, self._ledger = store, ledger

    def _parts(self) -> tuple[RolloutStore, ConsumptionLedger]:
        if self._store is None or self._ledger is None:
            raise RuntimeError("PlatformDataBuffer needs attach() from PlatformRolloutFn before use")
        return self._store, self._ledger

    async def put(self, input: DataBufferInput) -> None:
        store, ledger = self._parts()
        if any(not isinstance(sample, Sample) for sample in input.group):
            raise ValueError("Platform training requires one linear sample per attempt")
        samples = [cast(Sample, sample) for sample in input.group]
        if len(samples) != self.config.research.group_size:
            raise ValueError("Incomplete prompt group")
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            self._failures += 1
            if self._failures >= self.config.research.max_consecutive_failed_groups:
                raise RuntimeError("Platform rollout failure budget exhausted; stopping producer")
            self._unused(input.prompt_group)
            return
        policy = validate_group(self.config, samples)
        group_id = accepted(samples[0]).attempt.group_id
        self._failures = 0
        await store.add_group(group_id, policy, samples)
        self._persisted += 1
        self._stored.set()
        # Backpressure: stop the producer while enough fresh work is already waiting.
        while True:
            version = self._version
            if version is None:
                current = await store.current_policy()
                version = current.version if current is not None else policy.version
            waiting = await store.count(
                min_version=version - self.config.research.max_policy_lag,
                max_version=version,
                exclude=ledger.ids(),
            )
            if waiting <= self.config.completed_group_capacity:
                return
            self._consumed_event.clear()
            try:
                await asyncio.wait_for(self._consumed_event.wait(), self.config.poll_interval_seconds)
            except TimeoutError:
                pass

    async def get(self, current_version: int | None = None, **context: object) -> DataBufferInput:
        if type(current_version) is not int or current_version < 1:
            raise ValueError("Training must supply its committed policy version")
        store, ledger = self._parts()
        self._version = current_version
        min_version = current_version - self.config.research.max_policy_lag
        ledger.prune(below_version=min_version)
        while True:
            rows = await store.select(
                min_version=min_version, max_version=current_version, exclude=ledger.ids(), limit=1
            )
            if rows:
                row = rows[0]
                header, samples = await store.load(row)
                if validate_group(self.config, samples) != header.policy:
                    raise ValueError(f"Stored group {row.group_id} evidence names a different policy than its index")
                ledger.add(row.group_id, row.policy_version)
                self._consumed += 1
                self._consumed_event.set()
                return DataBufferInput(prompt_group=samples, group=list(samples))
            self._stored.clear()
            try:
                # A store written by another process is found by polling, not only by events.
                await asyncio.wait_for(self._stored.wait(), self.config.poll_interval_seconds)
            except TimeoutError:
                pass

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "rollout/platform/persisted_groups": float(self._persisted),
            "rollout/platform/consumed_groups": float(self._consumed),
            "rollout/platform/consecutive_failed_groups": float(self._failures),
        }
        self._persisted = self._consumed = 0
        return metrics
