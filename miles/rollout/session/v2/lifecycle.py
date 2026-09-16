"""Admission and finalization for one session's in-flight generations."""

import asyncio
import uuid
from dataclasses import dataclass, field

from miles.rollout.session.errors import SessionConflictError


@dataclass(frozen=True)
class FinishedSession:
    snapshot_id: str
    complete: bool
    producer_finished: bool
    failed_requests: int
    unfinished_requests: int


@dataclass
class SessionLifecycle:
    """Own request admission and a shared, cancellation-safe finish operation.

    Call synchronous mutations under `lock`; inference and draining run outside it.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    finished: FinishedSession | None = None
    _pending: set[int] = field(default_factory=set)
    _next_ticket: int = 0
    _failed_requests: int = 0
    _idle: asyncio.Event = field(default_factory=asyncio.Event)
    _finish_task: asyncio.Task | None = None
    _finish_args: tuple[bool, float] | None = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def draining(self) -> bool:
        return self._finish_task is not None and self.finished is None

    def check_open(self) -> None:
        if self._finish_task is not None:
            raise SessionConflictError("Session is finishing; create a new session for additional generations.")

    def admit(self) -> int:
        self.check_open()
        ticket = self._next_ticket
        self._next_ticket += 1
        self._pending.add(ticket)
        self._idle.clear()
        return ticket

    def can_commit(self, ticket: int) -> bool:
        return ticket in self._pending and self.finished is None

    def resolve(self, ticket: int, *, failed: bool) -> None:
        if ticket not in self._pending:
            return
        self._pending.remove(ticket)
        self._failed_requests += int(failed)
        if not self._pending:
            self._idle.set()

    async def finish(self, *, producer_finished: bool, timeout: float) -> FinishedSession:
        async with self.lock:
            finish_args = (producer_finished, timeout)
            if self._finish_task is None:
                if not self._pending:
                    self._idle.set()
                self._finish_args = finish_args
                self._finish_task = asyncio.create_task(self._drain(producer_finished=producer_finished, timeout=timeout))
            elif finish_args != self._finish_args:
                raise SessionConflictError("Finish parameters changed; retry with the original parameters.")
            finish_task = self._finish_task
        # a disconnected caller must not cancel the session's finalization
        return await asyncio.shield(finish_task)

    async def _drain(self, *, producer_finished: bool, timeout: float) -> FinishedSession:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except TimeoutError:
            pass
        async with self.lock:
            self.finished = FinishedSession(
                snapshot_id=uuid.uuid4().hex,
                complete=producer_finished and not self._pending and not self._failed_requests,
                producer_finished=producer_finished,
                failed_requests=self._failed_requests,
                unfinished_requests=len(self._pending),
            )
            # clearing the tickets fences callbacks that arrive after the deadline
            self._pending.clear()
            return self.finished
