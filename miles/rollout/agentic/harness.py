"""Structured completion and request identity for instrumented agent harnesses."""

import asyncio
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx

from miles.rollout.session.v2.contexts import SessionContext


@dataclass(frozen=True)
class AgentResult:
    metadata: dict[str, Any] | None
    producer_finished: bool


@dataclass(frozen=True)
class GenerationRequest:
    """Reuse one instance for transport retries; construct another for resampling."""

    context: SessionContext
    previous_response_id: str | None = None
    retry_of: str | None = None
    supersedes: str | None = None
    idempotency_key: str = field(default_factory=lambda: uuid.uuid4().hex)

    def headers(self) -> dict[str, str]:
        return {
            **self.context.headers(),
            "X-Miles-Idempotency-Key": self.idempotency_key,
            **{
                header: value
                for header, value in (
                    ("X-Miles-Previous-Response-Id", self.previous_response_id),
                    ("X-Miles-Retry-Of", self.retry_of),
                    ("X-Miles-Supersedes", self.supersedes),
                )
                if value is not None
            },
        }


class AgentRun:
    """Join registered child/tool tasks before asserting producer completion."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self.client = client
        self._group = anyio.create_task_group()
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._producer_finished = False

    async def __aenter__(self):
        await self._group.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        joined = False
        try:
            await self._group.__aexit__(exc_type, exc, traceback)
            joined = exc_type is None
        finally:
            # cancellation can stop a joiner before it first awaits its asyncio task
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._closed = True
            self._producer_finished = joined and all(not task.cancelled() for task in self._tasks)

    def create_task(self, coroutine: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        try:
            self._group.start_soon(self._join_task, task)
        except RuntimeError:
            task.cancel()
            raise
        self._tasks.append(task)
        return task

    async def _join_task(self, task: asyncio.Task) -> None:
        await task

    async def register_context(self, context: SessionContext) -> SessionContext:
        if self._closed:
            raise RuntimeError("Agent run is closed; register contexts before joining.")
        response = await self.client.post(f"{self.base_url}/contexts", json=context.model_dump(exclude_none=True))
        response.raise_for_status()
        return context

    def result(self, metadata: dict[str, Any] | None = None) -> AgentResult:
        if not self._closed:
            raise RuntimeError("Join the agent run before returning its result.")
        return AgentResult(metadata=metadata, producer_finished=self._producer_finished)
