"""Session-owned generation tasks and explicit retry intent."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.responses import Response

from miles.rollout.session.errors import MessageValidationError, SessionConflictError
from miles.rollout.session.v2.contexts import ContextId, SessionContext

_INTENT_HEADERS = {
    "x-miles-idempotency-key": "idempotency_key",
    "x-miles-retry-of": "retry_of",
    "x-miles-supersedes": "supersedes",
}


class GenerationIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: ContextId | None = None
    retry_of: ContextId | None = None
    supersedes: ContextId | None = None


def split_generation_headers(headers: dict) -> tuple[GenerationIntent, dict]:
    intent = {}
    forwarded = {}
    for header, value in headers.items():
        if header.lower() in _INTENT_HEADERS:
            intent[_INTENT_HEADERS[header.lower()]] = value
        else:
            forwarded[header] = value
    try:
        return GenerationIntent.model_validate(intent), forwarded
    except ValidationError as exc:
        raise MessageValidationError(f"Invalid generation headers: {exc}") from exc


def request_fingerprint(
    body: bytes,
    *,
    method: str,
    query: str,
    context: SessionContext | None,
    previous_response_id: str | None,
    intent: GenerationIntent,
) -> str:
    try:
        canonical = json.dumps(
            [
                json.loads(body),
                method,
                query,
                context.model_dump() if context else None,
                previous_response_id,
                intent.model_dump(),
            ],
            sort_keys=True,
            allow_nan=False,
        )
    except (ValueError, UnicodeDecodeError) as exc:
        raise MessageValidationError("Invalid generation request; use a finite JSON payload.") from exc
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class GenerationOperation:
    fingerprint: str
    task: asyncio.Task[Response]


@dataclass
class GenerationOperations:
    operations: dict[str, GenerationOperation] = field(default_factory=dict)

    def lookup(self, key: str, fingerprint: str) -> asyncio.Task[Response] | None:
        operation = self.operations.get(key)
        if operation is not None:
            if operation.fingerprint != fingerprint:
                raise SessionConflictError("Idempotency key was reused with different inputs; use a new key.")
            return operation.task
        if len(self.operations) >= 1024:
            raise SessionConflictError("Idempotency capacity reached; create a new session.")
        return None

    def remember(self, key: str, fingerprint: str, task: asyncio.Task[Response]) -> None:
        self.operations[key] = GenerationOperation(fingerprint, task)
        # disconnected callers may never retrieve a failed task's exception
        task.add_done_callback(_observe_completion)

    def cancel_pending(self) -> None:
        for operation in self.operations.values():
            if not operation.task.done():
                operation.task.cancel()


def _observe_completion(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()
