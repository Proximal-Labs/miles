"""Platform execution through Miles's continuously running rollout producer."""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TypeVar, assert_never

import httpx

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnInput, RolloutFnOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, decode_samples_and_merge_input_sample
from miles.utils.types import Sample
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.buffer import PlatformDataBuffer, validate_sample
from miles_plugins.proximal.clients import CaptureClient, IneligibleAttempt, LaunchFailed, PlatformClient
from miles_plugins.proximal.contracts import (
    AcceptedAttempt,
    Attempt,
    CaptureReceipt,
    FailedAttempt,
    Grade,
    LaunchRetry,
    Policy,
    Research,
    SessionHandle,
    Task,
    canonical_bytes,
    pinned_dataset,
    read_run_config,
)
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.options import add_arguments
from miles_plugins.proximal.preflight import canary
from miles_plugins.proximal.storage import write_immutable
from miles_plugins.proximal.store import RolloutStore, open_store, sample_fields

logger = logging.getLogger(__name__)

# Releasing a capture session only frees the replica's memory: it runs in the
# background, so a rollout's sample reaches training without waiting for it.
RELEASE_TIMEOUT_SECONDS = 60
_releases: set["asyncio.Task[None]"] = set()
_HandoffResult = TypeVar("_HandoffResult")


async def _release(capture: CaptureClient, handle: SessionHandle, attempt_id: str) -> None:
    started = time.monotonic()
    try:
        await asyncio.wait_for(capture.release(handle), timeout=RELEASE_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.error("Capture release failed for attempt %s (%s)", attempt_id, type(exc).__name__)
        return
    if (elapsed := time.monotonic() - started) > 5:
        logger.warning("Capture release for attempt %s took %.1fs", attempt_id, elapsed)


async def wait_for_releases() -> None:
    """Drain this loop's releases without depending on queued done callbacks."""
    loop = asyncio.get_running_loop()
    while True:
        # Awaiting only finished tasks may never yield to their discard callbacks.
        # Drop them explicitly, including finished tasks left by an older loop.
        _releases.difference_update(task for task in tuple(_releases) if task.done())
        pending = [task for task in _releases if task.get_loop() is loop]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


async def _finish_handoff(task: asyncio.Task[_HandoffResult]) -> asyncio.CancelledError | None:
    """Finish a durable handoff before propagating any repeated shutdown signals."""
    cancelled = None
    while True:
        try:
            await asyncio.shield(task)
            return cancelled
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancelled = exc


async def _archive_failure(
    attempt: Attempt,
    error: Exception | asyncio.CancelledError,
    *,
    handle: SessionHandle | None,
    grade: Grade | None,
    capture: CaptureClient,
    platform: PlatformClient,
    directory: Path,
    store: RolloutStore | None,
) -> CaptureReceipt | None:
    # These are logical API lifetimes. Platform alone owns physical resources.
    try:
        await asyncio.wait_for(platform.cancel(attempt), timeout=30)
    except Exception as cancel_error:
        logger.error(
            "Logical cancellation failed for attempt %s (%s)", attempt.attempt_id, type(cancel_error).__name__
        )
    receipt = None
    payload_path = directory / "request.json"
    if handle is not None:
        try:
            receipt, partial = await asyncio.wait_for(capture.collect(handle, attempt), timeout=30)
        except Exception as collect_error:
            # A failed read does not prove the replica has no useful bytes.
            # Archive the unknown outcome, but do not authorize its deletion.
            logger.warning(
                "Capture retrieval failed for attempt %s (%s); no payload archived",
                attempt.attempt_id,
                type(collect_error).__name__,
            )
            receipt = None
        else:
            # Retrieval failure can leave no capture. A local storage failure
            # cannot: propagate it and retain the replica's recoverable bytes.
            payload_path = directory / "partial.safetensors"
            write_immutable(payload_path, partial)
    outcome = FailedAttempt(
        attempt=attempt,
        status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
        error_type=type(error).__name__,
        capture=receipt,
        grade=grade,
    )
    failed_path = directory / "failed.json"
    write_immutable(failed_path, canonical_bytes(outcome))
    if store is not None:
        await store.publish_artifact(payload_path, failed_path, record_id=f"failed-{attempt.attempt_id}")
    return receipt


async def execute_attempt(
    attempt: Attempt,
    sample: Sample,
    *,
    capture: CaptureClient,
    platform: PlatformClient,
    artifact_root: Path,
    store: RolloutStore | None = None,
) -> Sample:
    directory = artifact_root / attempt.attempt_id
    request_path = directory / "request.json"
    write_immutable(request_path, canonical_bytes(attempt))
    if store is not None:
        await store.publish_artifact(request_path, request_path, record_id=f"request-{attempt.attempt_id}")
    handle = None
    grade = None
    complete = False
    accepted_locally = False
    try:
        handle = await capture.create(attempt)
        grade = await platform.execute(attempt, handle)
        receipt, payload = await capture.collect(handle, attempt)
        # Capture seals with SessionCore's v1 allowlist, plus the support when the attempt replays it.
        decoded = decode_samples_and_merge_input_sample(
            payload, sample, fields=sample_fields(COMPUTED_FIELDS, attempt.sampling)
        )
        if len(decoded.samples) != 1:
            raise ValueError("A graded linear platform rollout must yield exactly one training sample")
        result = decoded.samples[0]
        result.reward = grade.reward
        evidence = AcceptedAttempt(attempt=attempt, capture=receipt, grade=grade)
        validate_sample(result, evidence)
        accepted_locally = True  # Validated paid work must survive a local write failure too.
        write_immutable(directory / "samples.safetensors", payload)
        write_immutable(directory / "accepted.json", canonical_bytes(evidence))
        if store is not None:
            publication = asyncio.create_task(
                store.publish_artifact(
                    directory / "samples.safetensors",
                    directory / "accepted.json",
                    record_id=f"capture-{attempt.attempt_id}",
                )
            )
            cancelled = await _finish_handoff(publication)
            if cancelled is not None:
                complete = True
                raise cancelled
        result.metadata["proximal_accepted"] = evidence.model_dump_json()
        complete = True
        return result
    except (Exception, asyncio.CancelledError) as exc:
        if not accepted_locally:
            archive = asyncio.create_task(
                _archive_failure(
                    attempt,
                    exc,
                    handle=handle,
                    grade=grade,
                    capture=capture,
                    platform=platform,
                    directory=directory,
                    store=store,
                )
            )
            cancelled = await _finish_handoff(archive)
            # A durable failure record alone does not preserve unread capture data.
            # LaunchFailed is the platform's explicit proof that no rollout ran.
            complete = archive.result() is not None or isinstance(exc, LaunchFailed)
            if cancelled is not None:
                raise cancelled from exc
        raise
    finally:
        # A staged accepted result whose handoff was interrupted must remain on the
        # replica. Its existing session expiry still bounds retention after a crash.
        if complete and handle is not None:
            task = asyncio.create_task(_release(capture, handle, attempt.attempt_id))
            _releases.add(task)
            task.add_done_callback(_releases.discard)


async def execute_with_launch_retry(
    attempt: Attempt,
    sample: Sample,
    *,
    capture: CaptureClient,
    platform: PlatformClient,
    artifact_root: Path,
    retry: LaunchRetry,
    rng: random.Random | None = None,
    store: RolloutStore | None = None,
) -> Sample:
    """``execute_attempt``, relaunching under a new attempt identity when the platform could
    not start the rollout (LaunchFailed). Each launch is its own platform run and capture
    session; the group, sample, task and policy stay the same."""
    rng = rng or random.Random()
    await asyncio.sleep(rng.uniform(0, retry.stagger_seconds))
    for launch in range(retry.attempts):
        try:
            return await execute_attempt(
                attempt, sample, capture=capture, platform=platform, artifact_root=artifact_root, store=store
            )
        except LaunchFailed as exc:
            if launch == retry.attempts - 1:
                raise
            delay = rng.uniform(0, min(retry.max_backoff_seconds, retry.backoff_seconds * 2**launch))
            logger.warning(
                "Launch %d of attempt %s failed (%s); relaunching in %.0fs", launch + 1, attempt.attempt_id, exc, delay
            )
            await asyncio.sleep(delay)
            attempt = relaunched(attempt)
    raise AssertionError("unreachable")


def relaunched(attempt: Attempt) -> Attempt:
    """The same group member under a new attempt identity: its own platform run and capture session."""
    return Attempt.model_validate({**attempt.model_dump(), "attempt_id": uuid.uuid4().hex})


def member_relaunches(research: Research) -> int:
    """How many failed members a group may relaunch before the group fails as a whole.

    Members are independent draws given the task and policy, so relaunching a failed one
    keeps the same distribution as retrying its whole group (each member is a draw that
    finished eligibly) without discarding its siblings' finished work. Under "retry" a
    group may spend ``group_size`` relaunches, never more extra attempts than one retry of
    the whole group, and then fails and its task is retried as before. Under "drop" a
    failed member still drops its group.
    """
    match research.unused_groups:
        case "retry":
            return research.group_size
        case "drop":
            return 0
        case _ as unreachable:
            assert_never(unreachable)


class _GroupRelaunches:
    """The relaunch budget a group's members share. Once one member gives up, the group cannot
    complete, so the others stop relaunching; attempts already running still finish."""

    def __init__(self, budget: int) -> None:
        self.remaining = budget
        self.given_up = False

    def take(self) -> bool:
        if self.given_up or self.remaining == 0:
            return False
        self.remaining -= 1
        return True


def _sample_index(sample: Sample) -> int:
    if sample.index is None:
        raise ValueError("Platform task source must assign sample identities")
    return sample.index


class PlatformRolloutFn(FullyAsyncRolloutFn):
    add_arguments = staticmethod(add_arguments)

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        self.config = read_run_config(input.args.proximal_config)
        self.authorization = authorize_run(
            self.config, yes_rollouts=input.args.proximal_yes_rollouts, yes_publish=input.args.proximal_yes_publish
        )
        self._client: httpx.AsyncClient | None = None
        self._capture: CaptureClient | None = None
        self._platform: PlatformClient | None = None
        self._store: RolloutStore | None = None
        self._member_relaunches = member_relaunches(self.config.research)

    # Async like FullyAsyncRolloutFn.__call__, which Miles's executor awaits; the
    # base class annotates the sync form.
    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:  # type: ignore[override]
        # Same lazy start as FullyAsyncRolloutFn, plus wiring the buffer to the
        # durable store and the checkpointed ledger owned by the task source.
        if not input.evaluation and self._worker is None:
            if not isinstance(self.data_source, PlatformTaskSource):
                raise ValueError("Platform rollouts require the platform task source")
            self._store = await open_store(self.config)
            await self._preflight()
            buffer = PlatformDataBuffer(
                DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
            )
            buffer.attach(store=self._store, ledger=self.data_source.consumed)
            self._output = buffer
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started platform rollout worker against the durable rollout store")
        return await super().__call__(input)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.request_timeout_seconds,
                limits=httpx.Limits(max_connections=self.config.max_in_flight_samples * 3),
            )
            self._capture = CaptureClient(self.authorization, self._client)
            self._platform = PlatformClient(self.authorization, self._client)
        return self._client

    async def _published_policy(self, store: RolloutStore) -> Policy:
        """The newest selectable policy, waiting for the trainer's first. An HTTP-synced
        policy is committed by the run-state writer after its archive, shortly after the
        sync returns and so possibly after the first rollout is asked for."""
        deadline = time.monotonic() + self.config.request_timeout_seconds
        while (policy := await store.current_policy()) is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("No published policy yet; the trainer publishes one before rollouts start")
            await asyncio.sleep(1)
        return policy

    async def _preflight(self) -> None:
        """Before any platform run: capture answers, seals, and ends an over-long turn cleanly."""
        assert self._store is not None
        policy = await self._published_policy(self._store)
        report = await canary(self.authorization, self._http(), policy)
        logger.info("Preflight canary passed: %s", report)

    def _max_in_flight_groups(self) -> int:
        # The finite collector uses Miles's sample scheduler: zero capacity stops
        # backfill while active groups finish. A failed group's buffer callback
        # reopens admission for its retry; an exhausted idle worker waits for close.
        if isinstance(self.data_source, PlatformTaskSource) and not self.data_source.has_samples:
            return 0
        return super()._max_in_flight_groups()

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        self._http()
        assert self._capture is not None and self._platform is not None and self._store is not None
        policy = await self._store.current_policy()
        if policy is None:
            raise RuntimeError("No published policy yet; the trainer publishes one before rollouts start")
        group_id = uuid.uuid4().hex
        attempts = [
            Attempt(
                attempt_id=uuid.uuid4().hex,
                run_id=self.config.run_id,
                group_id=group_id,
                sample_index=_sample_index(sample),
                dataset_sha256=pinned_dataset(self.config.dataset).sha256,
                task=Task.model_validate(sample.metadata["proximal_task"]),
                harness=self.config.harness,
                policy=policy,
                sampling=self.config.research.sampling,
            )
            for sample in prompt_group
        ]
        relaunches = _GroupRelaunches(self._member_relaunches)
        tasks = [
            asyncio.create_task(self._execute_member(attempt, sample, relaunches))
            for attempt, sample in zip(attempts, prompt_group, strict=True)
        ]
        if (sample_done := self._scheduler.sample_done_callback) is not None:
            # Miles's contract (generate_and_rm_group): each sample frees its submission
            # slot when it finishes, so the next group starts without waiting for this
            # group's slowest rollout.
            for task in tasks:
                task.add_done_callback(lambda _task: sample_done())
        try:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            # A failed rollout does not revoke the paid work of its siblings.
            # Let each finish and persist before rejecting/retrying the group.
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not isinstance(
                    outcome, (IneligibleAttempt, httpx.HTTPError)
                ):
                    raise outcome
            result = []
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
                result.append(outcome)
        except (IneligibleAttempt, httpx.HTTPError) as exc:
            # Messages carry only our own text or the request method, URL and status, never headers.
            logger.warning(
                "Platform group %s failed (%s: %s); no fabricated rewards", group_id, type(exc).__name__, exc
            )
            result = [replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group]
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return DataBufferInput(prompt_group=prompt_group, group=[sample for sample in result])

    async def _execute_member(self, attempt: Attempt, sample: Sample, relaunches: _GroupRelaunches) -> Sample:
        """One group member's sample. An attempt that fails ineligibly (an execution failure,
        never a graded zero) is relaunched as soon as it fails, under a new attempt identity with
        the same group, sample index, task and policy, while the group's budget lasts and its
        policy is still selectable. Its siblings keep running and are not re-run."""
        assert self._capture is not None and self._platform is not None
        try:
            while True:
                try:
                    return await execute_with_launch_retry(
                        attempt,
                        sample,
                        capture=self._capture,
                        platform=self._platform,
                        artifact_root=self.config.artifact_directory / self.config.run_id / "accepted",
                        retry=self.config.launch_retry,
                        store=self._store,
                    )
                except (IneligibleAttempt, httpx.HTTPError) as exc:
                    # Rechecked after the await: a sibling may have given up meanwhile.
                    if not relaunches.take() or not await self._selectable(attempt.policy) or relaunches.given_up:
                        raise
                    replacement = relaunched(attempt)
                    # Messages carry only our own text or the request method, URL and status, never headers.
                    logger.warning(
                        "Attempt %s (sample %d of group %s) failed (%s: %s); relaunching it as attempt %s, "
                        "%d relaunches left for the group",
                        attempt.attempt_id,
                        attempt.sample_index,
                        attempt.group_id,
                        type(exc).__name__,
                        exc,
                        replacement.attempt_id,
                        relaunches.remaining,
                    )
                    attempt = replacement
        except (Exception, asyncio.CancelledError):
            relaunches.given_up = True
            raise

    async def _selectable(self, policy: Policy) -> bool:
        """Whether a group sampled from ``policy`` can still be trained on: the batch query
        selects groups within ``max_policy_lag`` of the newest published policy. A stale
        group's failed member is not relaunched; the group fails and its task is retried
        (or dropped) under the current policy."""
        assert self._store is not None
        current = await self._store.current_policy()
        return current is not None and current.version - policy.version <= self.config.research.max_policy_lag

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        raise ValueError("Platform eval needs a separately pinned evaluation contract; training-only first pass")

    async def dispose(self) -> None:
        # Miles's executor disposes from its own loop; the worker, store and clients live on the worker's.
        if (worker := self._worker) is None:
            await self.close()
            return
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self.close(), worker.get_loop()))

    async def close(self) -> None:
        await super().close()
        await wait_for_releases()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._store is not None:
            await self._store.close()
            self._store = None
