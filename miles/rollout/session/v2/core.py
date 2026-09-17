import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass

from starlette.responses import Response

from miles.rollout.session.core import (
    JSON_MEDIA_TYPE,
    ProxyRequest,
    SessionCore,
    _chat_client_response,
    _render_json,
    _samples_response,
    extract_completion,
    prepare_chat_request,
    proxy_result_to_response,
)
from miles.rollout.session.errors import SessionConflictError, SessionNotFoundError, TokenizationError
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, encode_samples
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.rollout.session.v2.contexts import SessionContext, split_context_headers
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY, build_session_rollout_metrics
from miles.rollout.session.v2.operations import request_fingerprint, split_idempotency_header
from miles.rollout.session.v2.session_state import (
    SessionRegistryV2,
    SessionStateV2,
    commit_generation,
    position_for_request,
    prepare_pretokenized,
)
from miles.rollout.session.v2.tree_trajectory import MAX_NODES, TrajectoryNode
from miles.rollout.session.v2.utils import build_leaf_material, tree_metadata
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer
from miles.utils.function_registry import load_function

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedGeneration:
    ticket: int
    request: dict
    proxy_body: bytes
    parent: TrajectoryNode | None
    tokenizer: TITOTokenizer
    client_stream: bool
    request_timestamp: float
    context_id: str | None
    generation_id: str


class SessionCoreV2(SessionCore):
    """``SessionCore`` with tree serving: overrides the session-semantics
    methods (positioning/commit, metadata, samples op), inherits the
    transport shell (health, create/delete, raw proxy)."""

    def __init__(
        self, backend, registry: SessionRegistryV2, config, session_server_instance_id=None, *, use_addition_r3=False
    ):
        super().__init__(backend, registry, config, session_server_instance_id, use_addition_r3=use_addition_r3)
        # Import-path only in production: function_registry is process-local.
        self.sample_picker = load_function(config.session_sample_picker_path, sync_required=True)
        self.sample_postprocessor = load_function(config.session_sample_postprocessor_path, sync_required=True)

    def _session_metadata(self, session_id: str, session) -> dict:
        """Mirrors ``core.SessionCore._session_metadata``: token ids come from
        the active path, plus the ``tree`` block."""
        metadata: dict = {}
        try:
            mismatch = self.registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.active_token_ids()
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        metadata["tree"] = tree_metadata(session)
        if session.contexts.contexts:
            metadata["contexts"] = [
                context.model_dump(exclude_none=True) for context in session.contexts.contexts.values()
            ]
        if session.lifecycle.finished is not None:
            metadata["finalization"] = asdict(session.lifecycle.finished)
        return metadata

    async def get_session(self, session_id: str) -> Response:
        """Mirrors ``core.SessionCore.get_session``, serving ``active_records()``."""
        session = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, session)
        payload = GetSessionResponse(session_id=session_id, records=session.active_records(), metadata=metadata)
        return Response(
            content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
        )

    async def finish_session(self, session_id: str, *, producer_finished: bool, timeout: float) -> Response:
        session = self.registry.get_session(session_id)
        self.registry.retain_for_collection(session_id)
        finished = await session.lifecycle.finish(producer_finished=producer_finished, timeout=timeout)
        return Response(content=_render_json(asdict(finished)), media_type=JSON_MEDIA_TYPE)

    async def register_context(self, session_id: str, context: SessionContext) -> Response:
        session = self.registry.get_session(session_id)
        async with session.lock:
            session.lifecycle.check_open()
            session.contexts.register(context)
        return Response(content=_render_json(context.model_dump(exclude_none=True)), media_type=JSON_MEDIA_TYPE)

    async def collect_samples(
        self,
        session_id: str,
        *,
        max_seq_len: int | None,
        agent_metadata: dict | None = None,
        snapshot_id: str | None = None,
    ) -> Response:
        """Export a sealed session once; unsealed requests remain live previews."""
        session = self.registry.get_session(session_id)
        finished = session.lifecycle.finished
        if session.lifecycle.draining:
            raise SessionConflictError("Session is still draining; wait for finish before collecting samples.")
        if snapshot_id is not None and (finished is None or snapshot_id != finished.snapshot_id):
            raise SessionConflictError("Snapshot does not match this session; use the snapshot returned by finish.")
        export_args = json.dumps([max_seq_len, agent_metadata], sort_keys=True, allow_nan=False)
        if session.sample_export is not None:
            original_args, payload = session.sample_export
            if export_args != original_args:
                raise SessionConflictError("Export parameters changed; retry with the original parameters.")
            return _samples_response(payload)
        response = self._assemble_samples(session_id, session, max_seq_len=max_seq_len, agent_metadata=agent_metadata)
        if finished is not None and response.status_code == 200:
            session.sample_export = (export_args, response.body)
        return response

    def _assemble_samples(
        self, session_id: str, session: SessionStateV2, *, max_seq_len: int | None, agent_metadata: dict | None
    ) -> Response:
        metadata = self._session_metadata(session_id, session)
        if agent_metadata is not None:
            metadata["agent"] = agent_metadata
        if session.lifecycle.finished is not None and not session.lifecycle.finished.complete:
            return _samples_response(
                encode_samples([], metadata, empty_reason="incomplete", fields=COMPUTED_FIELDS_V2)
            )
        if not session.tree.nodes:
            return _samples_response(
                encode_samples([], metadata, empty_reason="no_records", fields=COMPUTED_FIELDS_V2)
            )

        try:
            material = build_leaf_material(
                self.config,
                session,
                self.registry,
                session_id=session_id,
                max_seq_len=max_seq_len,
                use_addition_r3=self.use_addition_r3,
            )
        except (AssertionError, ValueError) as exc:
            return Response(content=str(exc).encode(), status_code=422, media_type="text/plain")
        if not material:
            return _samples_response(
                encode_samples([], metadata, empty_reason="all_truncated", fields=COMPUTED_FIELDS_V2)
            )

        # Hook lane: a policy bug is a deterministic 422 carrying the hook's
        # identity, never a masked 500 (server death stays loud).
        try:
            picked = self.sample_picker(material, metadata)
            picked_ids = [id(sample) for sample in picked]
            allowed = {id(sample) for sample in material}
            if any(sample_id not in allowed for sample_id in picked_ids) or len(picked_ids) != len(set(picked_ids)):
                raise ValueError(
                    "pick hook must return a subset of its input samples without duplicates (pure selection)"
                )
            samples = self.sample_postprocessor(picked, metadata)
        except Exception as exc:
            body = (
                f"session sample hook failed (picker={self.config.session_sample_picker_path}, "
                f"postprocessor={self.config.session_sample_postprocessor_path}): {exc}"
            )
            return Response(content=body.encode(), status_code=422, media_type="text/plain")
        # Hooks may inspect or mutate session metadata, so publish the
        # authoritative server-owned value only at the wire boundary.
        if self.config.sglang_speculative_algorithm is not None:
            metadata[SESSION_ROLLOUT_METRICS_KEY] = build_session_rollout_metrics(session_id, session.tree.nodes)
        if session.lifecycle.finished is not None:
            metadata["finalization"] = asdict(session.lifecycle.finished)
        return _samples_response(encode_samples(samples, metadata, fields=COMPUTED_FIELDS_V2))

    async def chat_completions(
        self, session_id: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        """Record admitted generations while finish drains without holding the lock."""
        session = self.registry.get_session(session_id)
        context, previous_response_id, headers = split_context_headers(headers)
        key, headers = split_idempotency_header(headers)
        fingerprint = (
            request_fingerprint(
                body,
                method=method,
                query=query,
                context=context,
                previous_response_id=previous_response_id,
            )
            if key is not None
            else None
        )
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            operation = session.operations.lookup(key, fingerprint) if key is not None else None
            if operation is None:
                generation = self._prepare_generation(session, body, context, previous_response_id)
                if key is not None:
                    operation = asyncio.create_task(
                        self._execute_generation(
                            session_id,
                            session,
                            generation,
                            method=method,
                            query=query,
                            headers=headers,
                        )
                    )
                    session.operations.remember(key, fingerprint, operation)
        if operation is not None:
            response = await asyncio.shield(operation)
            return Response(content=response.body, status_code=response.status_code, headers=dict(response.headers))
        return await self._execute_generation(
            session_id, session, generation, method=method, query=query, headers=headers
        )

    async def _execute_generation(
        self,
        session_id: str,
        session: SessionStateV2,
        generation: _PreparedGeneration,
        *,
        method: str,
        query: str,
        headers: dict,
    ) -> Response:
        failed = True
        try:
            upstream = await self.backend.do_proxy(
                ProxyRequest(method=method, query=query),
                "v1/chat/completions",
                body=generation.proxy_body,
                headers={**headers, "X-SMG-Routing-Key": session_id},
            )
            if upstream["status_code"] != 200:
                # request rejections can be handled by the harness; server errors may lose sampled tokens
                failed = upstream["status_code"] >= 500 or upstream["status_code"] == 499
                return proxy_result_to_response(upstream)
            response, choice, assistant_message, completion_token_ids = extract_completion(upstream)
            assistant_message = generation.tokenizer.postprocess_completion(
                choice=choice, assistant_message=assistant_message, completion_token_ids=completion_token_ids
            )
            async with session.lock:
                if not session.closing and session.lifecycle.can_commit(generation.ticket):
                    record = SessionRecord(
                        timestamp=time.time(),
                        request_timestamp=generation.request_timestamp,
                        method=method,
                        path="/v1/chat/completions",
                        status_code=upstream["status_code"],
                        request=generation.request,
                        response=response,
                    )
                    commit_generation(
                        session,
                        parent=generation.parent,
                        request_messages=generation.request.get("messages", []),
                        assistant_message=assistant_message,
                        prompt_token_ids=generation.request["input_ids"],
                        completion_token_ids=completion_token_ids,
                        max_trim_tokens=generation.tokenizer.max_trim_tokens,
                        record=record,
                        response_id=response.get("id", ""),
                        finish_reason=choice.get("finish_reason") or "",
                        context_id=generation.context_id,
                        generation_id=generation.generation_id,
                    )
                    failed = False
                    session.lifecycle.resolve(generation.ticket, failed=False)
            client_response = _chat_client_response(upstream, response, generation.client_stream)
            client_response.headers["X-Miles-Generation-Id"] = generation.generation_id
            return client_response
        finally:
            async with session.lock:
                session.lifecycle.resolve(generation.ticket, failed=failed)

    def _prepare_generation(
        self,
        session: SessionStateV2,
        body: bytes,
        context: SessionContext | None,
        previous_response_id: str | None,
    ) -> _PreparedGeneration:
        session.lifecycle.check_open()
        if len(session.tree.nodes) + session.lifecycle.pending_count >= MAX_NODES:
            raise SessionConflictError(f"Session reached its {MAX_NODES}-generation capacity; create a new session.")
        request_body, client_stream, tokenizer = prepare_chat_request(body, self.config, self.registry.tito_tokenizer)
        context_id = context.context_id if context is not None else None
        request_messages = request_body.get("messages", [])
        position_for_request(
            session,
            request_messages,
            message_matcher=self.registry.message_matcher,
            context_id=context_id,
            previous_response_id=previous_response_id,
        )
        prompt_token_ids = prepare_pretokenized(
            session, request_messages, tools=request_body.get("tools"), tito_tokenizer=tokenizer
        )
        request_body["input_ids"] = prompt_token_ids
        self._maybe_request_addition_r3(request_body, session.active_token_ids(), prompt_token_ids)
        proxy_body = json.dumps(request_body).encode()
        if context is not None:
            session.contexts.register(context, request=request_body)
        return _PreparedGeneration(
            ticket=session.lifecycle.admit(),
            request=request_body,
            proxy_body=proxy_body,
            parent=session.active_leaf,
            tokenizer=tokenizer,
            client_stream=client_stream,
            request_timestamp=time.time(),
            context_id=context_id,
            generation_id=uuid.uuid4().hex,
        )
