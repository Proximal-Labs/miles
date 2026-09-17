"""Explicit execution identities, independent of token-prefix ancestry."""

import json
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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

    @model_validator(mode="after")
    def validate_parent(self):
        if self.parent_agent_run_id == self.agent_run_id:
            raise ValueError("an agent cannot be its own parent")
        if self.parent_tool_call_id is not None and self.parent_agent_run_id is None:
            raise ValueError("parent_tool_call_id requires parent_agent_run_id")
        return self

    def headers(self) -> dict[str, str]:
        return {
            header: value
            for header, name in _CONTEXT_HEADERS.items()
            if (value := getattr(self, name)) is not None
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
    _agent_parents: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    _rendering: dict[str, str] = field(default_factory=dict)

    def register(self, context: SessionContext) -> None:
        existing = self.contexts.get(context.context_id)
        if existing is not None:
            if existing != context:
                raise SessionConflictError("Context identity changed; use a new context_id.")
            return
        if len(self.contexts) >= 1024:
            raise SessionConflictError("Context capacity reached; create a new session.")
        parent = (context.parent_agent_run_id, context.parent_tool_call_id)
        if context.agent_run_id in self._agent_parents and self._agent_parents[context.agent_run_id] != parent:
            raise SessionConflictError("Agent parent changed; use a new agent_run_id.")
        if context.parent_agent_run_id is not None and context.parent_agent_run_id not in self._agent_parents:
            raise SessionConflictError("Unknown parent agent; register its context before the child.")
        if context.derived_from_context_id is not None:
            source = self.contexts.get(context.derived_from_context_id)
            if source is None or source.agent_run_id != context.agent_run_id:
                raise SessionConflictError("Compaction source must be a registered context of the same agent.")
        self.contexts[context.context_id] = context
        self._agent_parents[context.agent_run_id] = parent

    def bind_rendering(self, context_id: str, request: dict) -> None:
        contract = json.dumps(
            {key: request.get(key) for key in ("model", "lora_path", "tools", "chat_template_kwargs")},
            sort_keys=True,
            allow_nan=False,
        )
        existing = self._rendering.get(context_id)
        if existing is not None and existing != contract:
            raise SessionConflictError("Context rendering changed; start a new context for model/template/tool changes.")
        self._rendering[context_id] = contract
