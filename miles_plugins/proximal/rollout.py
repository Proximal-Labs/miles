"""Platform execution through Miles's continuously running rollout producer."""

import asyncio
import logging
import time
import uuid
from dataclasses import replace
from pathlib import Path

import httpx

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnInput, RolloutFnOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.utils.types import Sample
from miles_plugins.proximal.archive import publish_loop
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.buffer import PlatformDataBuffer, validate_sample
from miles_plugins.proximal.clients import CaptureClient, IneligibleAttempt, PlatformClient
from miles_plugins.proximal.contracts import (
    AcceptedAttempt,
    Attempt,
    SessionHandle,
    Task,
    canonical_bytes,
    pinned_dataset,
    read_run_config,
)
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.options import add_arguments
from miles_plugins.proximal.storage import write_immutable
from miles_plugins.proximal.store import CAPTURE_ARCHIVE_BATCH_SIZE, RolloutStore, open_store

logger = logging.getLogger(__name__)

# Releasing a capture session only frees the replica's memory: it runs in the
# background, so a rollout's sample reaches training without waiting for it.
RELEASE_TIMEOUT_SECONDS = 60
_releases: set["asyncio.Task[None]"] = set()


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
    """Finish the background session releases (at shutdown, and in tests)."""
    while _releases:
        await asyncio.gather(*list(_releases), return_exceptions=True)


async def execute_attempt(
    attempt: Attempt,
    sample: Sample,
    *,
    capture: CaptureClient,
    platform: PlatformClient,
    artifact_root: Path,
    store: RolloutStore,
) -> Sample:
    handle = await capture.create(attempt)
    platform_finished = False
    retain_session = False
    try:
        grade = await platform.execute(attempt, handle)
        platform_finished = True
        receipt, payload = await capture.collect(handle, attempt)
        retain_session = True
        decoded = decode_samples_and_merge_input_sample(payload, sample)
        if len(decoded.samples) != 1:
            raise ValueError("A graded linear platform rollout must yield exactly one training sample")
        result = decoded.samples[0]
        result.reward = grade.reward
        evidence = AcceptedAttempt(attempt=attempt, capture=receipt, grade=grade)
        validate_sample(result, evidence)
        directory = artifact_root / attempt.attempt_id
        write_immutable(directory / "samples.safetensors", payload)
        write_immutable(directory / "accepted.json", canonical_bytes(evidence))
        await store.enqueue_capture(evidence, directory / "samples.safetensors")
        retain_session = False
        result.metadata["proximal_accepted"] = evidence.model_dump_json()
        return result
    finally:
        # These are logical API lifetimes. Platform alone owns physical resources.
        # Cancellation is bounded and awaited before dropping the local session.
        if not platform_finished:
            try:
                await asyncio.wait_for(platform.cancel(attempt), timeout=30)
            except Exception as exc:
                logger.error("Logical cancellation failed for attempt %s (%s)", attempt.attempt_id, type(exc).__name__)
        if not retain_session:
            task = asyncio.create_task(_release(capture, handle, attempt.attempt_id))
            _releases.add(task)
            task.add_done_callback(_releases.discard)
        else:
            logger.error("Capture retained on replica: durable handoff failed for attempt %s", attempt.attempt_id)


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
        self._archive_worker: asyncio.Task[None] | None = None

    # Async like FullyAsyncRolloutFn.__call__, which Miles's executor awaits; the
    # base class annotates the sync form.
    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:  # type: ignore[override]
        # Same lazy start as FullyAsyncRolloutFn, plus wiring the buffer to the
        # durable store and the checkpointed ledger owned by the task source.
        if not input.evaluation and self._worker is None:
            if not isinstance(self.data_source, PlatformTaskSource):
                raise ValueError("Platform rollouts require the platform task source")
            self._store = await open_store(self.config)
            self._ensure_clients()
            assert self._platform is not None
            self._archive_worker = asyncio.create_task(publish_loop(self._store, self._platform))
            buffer = PlatformDataBuffer(
                DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
            )
            buffer.attach(store=self._store, ledger=self.data_source.consumed)
            self._output = buffer
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started platform rollout worker against the durable rollout store")
        return await super().__call__(input)

    def _ensure_clients(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.request_timeout_seconds,
                # Long archive PUTs must not consume the rollout HTTP capacity.
                limits=httpx.Limits(
                    max_connections=self.config.max_in_flight_samples * 3 + CAPTURE_ARCHIVE_BATCH_SIZE
                ),
            )
            self._capture = CaptureClient(self.authorization, self._client)
            self._platform = PlatformClient(self.authorization, self._client)

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        self._ensure_clients()
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
        tasks = [
            asyncio.create_task(
                execute_attempt(
                    attempt,
                    sample,
                    capture=self._capture,
                    platform=self._platform,
                    artifact_root=self.config.artifact_directory / self.config.run_id / "accepted",
                    store=self._store,
                )
            )
            for attempt, sample in zip(attempts, prompt_group, strict=True)
        ]
        try:
            result = await asyncio.gather(*tasks)
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
            if self._scheduler.sample_done_callback is not None:
                for _ in prompt_group:
                    self._scheduler.sample_done_callback()
        return DataBufferInput(prompt_group=prompt_group, group=[sample for sample in result])

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        raise ValueError("Platform eval needs a separately pinned evaluation contract; training-only first pass")

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await wait_for_releases()
            if self._archive_worker is not None:
                self._archive_worker.cancel()
                await asyncio.gather(self._archive_worker, return_exceptions=True)
                self._archive_worker = None
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            if self._store is not None:
                await self._store.close()
                self._store = None
