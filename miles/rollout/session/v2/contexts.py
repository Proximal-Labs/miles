"""Explicit execution identities, independent of token-prefix ancestry."""

import json
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from miles.rollout.session.errors import MessageValidationError, SessionConflictError

ContextId = Annotated[str, Field(strict=True, min_length=1, max_length=256, pattern=r"^[!-~]+$")]
_CONTEXT_HEADERS = {
    "x-miles-agent-run-id": "agent_run_id",
    "x-miles-context-id": "context_id",
    "x-miles-parent-agent-run-id": "parent_agent_run_id",
    "x-miles-parent-tool-call-id": "parent_tool_call_id",
    "x-miles-derived-from-context-id": "derived_from_context_id",
}
_PREVIOUS_RESPONSE_HEADER = "x-miles-previous-response-id"


class SessionContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_run_id: ContextId
    context_id: ContextId
    parent_agent_run_id: ContextId | None = None
    parent_tool_call_id: ContextId | None = None
    derived_from_context_id: ContextId | None = None

    def headers(self) -> dict[str, str]:
        return {
            header: value for header, name in _CONTEXT_HEADERS.items() if (value := getattr(self, name)) is not None
        }


def split_context_headers(headers: dict) -> tuple[SessionContext | None, str | None, dict]:
    identity = {}
    forwarded = {}
    previous_response_id = None
    for header, value in headers.items():
        name = header.lower()
        if name in _CONTEXT_HEADERS:
            identity[_CONTEXT_HEADERS[name]] = value
        elif name == _PREVIOUS_RESPONSE_HEADER:
            previous_response_id = value
        else:
            forwarded[header] = value
    try:
        context = SessionContext.model_validate(identity) if identity else None
    except ValidationError as exc:
        raise MessageValidationError(f"Invalid Miles context headers: {exc}") from exc
    if previous_response_id is not None and (not previous_response_id or context is None):
        raise MessageValidationError("A previous response requires explicit agent/context IDs and a nonempty ID.")
    return context, previous_response_id, forwarded


@dataclass
class SessionContexts:
    contexts: dict[str, SessionContext] = field(default_factory=dict)
    _rendering: dict[str, str] = field(default_factory=dict)

    def register(self, context: SessionContext, *, request: dict | None = None) -> None:
        existing = self.contexts.get(context.context_id)
        if existing is not None and existing != context:
            raise SessionConflictError("Context identity changed; use a new context_id.")
        if existing is None and len(self.contexts) >= 1024:
            raise SessionConflictError("Context capacity reached; create a new session.")
        contract = None
        if request is not None:
            contract = json.dumps(
                {key: request.get(key) for key in ("model", "lora_path", "tools", "chat_template_kwargs")},
                sort_keys=True,
                allow_nan=False,
            )
            rendering = self._rendering.get(context.context_id)
            if rendering is not None and rendering != contract:
                raise SessionConflictError(
                    "Context rendering changed; start a new context for model/template/tool changes."
                )
        self.contexts[context.context_id] = context
        if contract is not None:
            self._rendering[context.context_id] = contract
