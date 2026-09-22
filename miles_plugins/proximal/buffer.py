"""Strict complete-group eligibility at Miles's existing async DataBuffer seam."""

import asyncio
import math
from typing import cast

from miles.rollout.fully_async_data_buffer import DataBuffer, DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample
from miles_plugins.proximal.contracts import AcceptedAttempt, digest, read_run_config


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


class PlatformDataBuffer(DataBuffer):
    def __init__(self, input: DataBufferConstructorInput) -> None:
        self.config = read_run_config(input.args.proximal_config)
        self._unused = input.unused_handler_fn
        self._queue: asyncio.Queue[DataBufferInput] = asyncio.Queue(self.config.completed_group_capacity)
        self._failures = 0
        self._stale = 0
        self._consumed = 0

    async def put(self, input: DataBufferInput) -> None:
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
            indices.add(attempt.sample_index)
        if len(indices) != self.config.research.group_size:
            raise ValueError("Repeated sample in prompt group")
        self._failures = 0
        await self._queue.put(input)

    async def get(self, current_version: int | None = None, **context: object) -> DataBufferInput:
        if type(current_version) is not int or current_version < 1:
            raise ValueError("Training must supply its committed policy version")
        while True:
            item = await self._queue.get()
            version = accepted(cast(Sample, item.group[0])).attempt.policy.version
            lag = current_version - version
            if lag < 0:
                raise ValueError("Rollout policy is ahead of trainer; checkpoint/run identity conflict")
            if lag > self.config.research.max_policy_lag:
                self._stale += 1
                self._unused(item.prompt_group)
                continue
            self._consumed += 1
            return item

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "rollout/platform/queued_groups": float(self._queue.qsize()),
            "rollout/platform/stale_groups": float(self._stale),
            "rollout/platform/consumed_groups": float(self._consumed),
            "rollout/platform/consecutive_failed_groups": float(self._failures),
        }
        self._stale = self._consumed = 0
        return metrics
