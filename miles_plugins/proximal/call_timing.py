"""Per-request timing for capture: one JSON line per chat call, to attribute latency.

An ASGI middleware opens a ``CallTiming`` for each HTTP request (first byte received to
last byte sent); the chat handler and the engine call mark points inside it. The record
carries the engine's response id, which the platform's agent journal also records as its
``providerRequestId``, so capture's view of a call joins the agent's view of it.

Marks are monotonic seconds since the request arrived. The gateway's own split (adapter
admission, upstream SGLang time) arrives in its ``Server-Timing`` header.
"""

import json
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_CURRENT: ContextVar["CallTiming | None"] = ContextVar("proximal_call_timing", default=None)


@dataclass
class CallTiming:
    path: str
    client: str
    new_connection: bool
    started: float = field(default_factory=time.perf_counter)
    wall_start: float = field(default_factory=time.time)
    marks: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        self.marks[name] = round(time.perf_counter() - self.started, 6)


def mark(name: str) -> None:
    """Mark a point in the current request, if it is being timed."""
    if (timing := _CURRENT.get()) is not None:
        timing.mark(name)


def note(**values: Any) -> None:
    """Attach facts (sizes, ids, engine timings) to the current request's record."""
    if (timing := _CURRENT.get()) is not None:
        timing.info.update(values)


class CallTimingMiddleware:
    """Times chat calls and writes one JSON line each to ``log_path``."""

    def __init__(self, app: ASGIApp, log_path: Path) -> None:
        self.app = app
        self.log_path = log_path
        self._connections: set[tuple[str, int]] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].endswith("/chat/completions"):
            await self.app(scope, receive, send)
            return
        host, port = scope.get("client") or ("", 0)
        new_connection = (host, port) not in self._connections
        self._connections.add((host, port))
        timing = CallTiming(path=scope["path"], client=f"{host}:{port}", new_connection=new_connection)
        token = _CURRENT.set(timing)
        status = 0
        sent = 0

        async def timed_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                timing.mark("body_received")
            return message

        async def timed_send(message: Message) -> None:
            nonlocal status, sent
            if message["type"] == "http.response.start":
                status = message["status"]
                timing.mark("response_start")
            elif message["type"] == "http.response.body":
                sent += len(message.get("body", b""))
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                timing.mark("response_sent")

        try:
            await self.app(scope, timed_receive, timed_send)
        finally:
            _CURRENT.reset(token)
            record = {
                # Which container served the call: sticky routing keeps one rollout on one.
                "container": os.environ.get("MODAL_TASK_ID", ""),
                "wall_start": timing.wall_start,
                "path": timing.path,
                "client": timing.client,
                "new_connection": timing.new_connection,
                "status": status,
                "response_bytes": sent,
                "marks": timing.marks,
                **timing.info,
            }
            with self.log_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
