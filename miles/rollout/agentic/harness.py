"""Structured completion and request identity for instrumented agent harnesses."""

import uuid
from dataclasses import dataclass, field
from typing import Any


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
    idempotency_key: str = field(default_factory=lambda: uuid.uuid4().hex)

    def headers(self) -> dict[str, str]:
        headers = {**self.context.headers(), "X-Miles-Idempotency-Key": self.idempotency_key}
        if self.previous_response_id is not None:
            headers["X-Miles-Previous-Response-Id"] = self.previous_response_id
        return headers
