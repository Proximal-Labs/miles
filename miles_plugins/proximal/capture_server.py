"""Authenticated, policy-bound HTTP adapter around Miles's real linear TITO core.

One CPU process owns live sessions. Sealed samples survive its restart on disk;
unfinished sessions are explicitly lost. This app exposes no unrecorded proxy.
Policy versions come from the rollout store, the single policy authority.
"""

import asyncio
import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.core import ProxyRequest, SessionCore
from miles.rollout.session.errors import SessionError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization, secret_env
from miles_plugins.proximal.contracts import (
    Attempt,
    CaptureReceipt,
    RunConfig,
    canonical_bytes,
    digest,
    pinned_dataset,
)
from miles_plugins.proximal.storage import write_immutable
from miles_plugins.proximal.store import RolloutStore


@dataclass
class LiveSession:
    attempt: Attempt
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sealed: bool = False


# agent-px rebuilds replayed assistant messages and re-serializes tool arguments
# compactly. This Miles matcher accepts JSON-equivalent arguments and nothing else;
# on a match the prefix tokens still come from the session's own checkpoint. With
# the strict matcher every such turn looks like a retry and is rolled back.
MESSAGE_MATCHER = "loose_tool_call"


def check_tito_protocol(config: RunConfig) -> None:
    """The replicas parse served text with ``model_protocol``'s parsers; they must be the
    ones the chat-template family binds, or the tool calls the agent sees and the tokens
    capture renders for them disagree."""
    from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType

    family = TITOTokenizerType.get_tokenizer_class(TITOTokenizerType(config.tito_model))
    for name in ("reasoning_parser", "tool_call_parser"):
        bound = getattr(family, name)
        configured = getattr(config.model_protocol, name)
        if bound is not None and configured != bound:
            raise ValueError(
                f"model_protocol.{name} is {configured!r}, but the {config.tito_model} template family binds {bound!r}"
            )


def capture_registry(config: RunConfig, tokenizer: Any) -> SessionRegistry:
    """The one way to build the capture session registry: TITO renderer + matcher."""
    from miles.utils.chat_template_utils import get_tito_tokenizer
    from miles.utils.chat_template_utils.message_matcher_hub import resolve_session_message_matcher

    check_tito_protocol(config)
    tito = get_tito_tokenizer(
        tokenizer, config.tito_model, chat_template_kwargs={"enable_thinking": config.enable_thinking}
    )
    return SessionRegistry(
        tokenizer, tito_tokenizer=tito, message_matcher=resolve_session_message_matcher(MESSAGE_MATCHER)
    )


def session_config(config: RunConfig) -> SessionServerConfig:
    return SessionServerConfig(
        host="127.0.0.1",
        port=0,
        instance_id=config.run_id,
        backend_url=config.inference_url,
        timeout=config.request_timeout_seconds,
        hf_checkpoint=str(config.tokenizer_path),
        chat_template_path=None,
        tito_model=config.tito_model,
        apply_chat_template_kwargs={"enable_thinking": config.enable_thinking},
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
        num_layers=None,
        moe_router_topk=None,
        save_debug_trajectory_data=None,
        # Named immutable adapters are selected by the bound transport, not lora_0.
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        use_session_server=True,
        # Informational: SessionCore reads the matcher from the registry (capture_registry).
        session_message_matcher=MESSAGE_MATCHER,
        pause_generation_mode=None,
        session_sample_picker_path=None,
        session_sample_postprocessor_path=None,
    )


ENGINE_ATTEMPTS = 3


class BoundTransport:
    def __init__(self, config: RunConfig, client: httpx.AsyncClient, sessions: dict[str, LiveSession]):
        self.config, self.client, self.sessions = config, client, sessions
        self.headers = {name: secret_env(env) for name, env in config.inference_header_env.items()}

    async def _post_engine(self, outbound: bytes, policy_sha256: str) -> httpx.Response:
        """One engine call, retried when the connection fails.

        Safe to retry: nothing is recorded until a reply is accepted, so a lost request or
        reply only costs a repeated generation. Retrying on transport failures does not
        depend on what was sampled, so it does not bias which completions are trained.
        """
        for attempt in range(ENGINE_ATTEMPTS):
            try:
                return await self.client.post(
                    f"{self.config.inference_url}/v1/chat/completions",
                    content=outbound,
                    headers={
                        **self.headers,
                        "Content-Type": "application/json",
                        "X-Proximal-Policy-Sha256": policy_sha256,
                    },
                    follow_redirects=False,
                )
            except httpx.TransportError:
                if attempt == ENGINE_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(0.2 * 2**attempt)
        raise AssertionError("unreachable")

    async def do_proxy(
        self, request: ProxyRequest, path: str, *, body: bytes, headers: dict[str, str]
    ) -> dict[str, object]:
        if request.session_id is None or path != "v1/chat/completions" or request.method != "POST" or request.query:
            raise ValueError("Only bound recorded chat requests may reach inference")
        attempt = self.sessions[request.session_id].attempt
        payload = orjson.loads(body)  # SDK boundary; SessionCore already rendered and validated input_ids.
        remaining = attempt.sampling.max_sequence_tokens - len(payload["input_ids"])
        if remaining <= 0:
            raise HTTPException(422, "Sequence token budget exhausted")
        payload["max_tokens"] = min(payload["max_tokens"], remaining)
        payload["model"] = f"{self.config.base_model.name}:miles-{attempt.policy.snapshot.sha256}"
        # The engine call is always complete, never streamed; the gateway requires it explicitly.
        payload["stream"] = False
        payload.pop("stream_options", None)
        outbound = orjson.dumps(payload)
        response = await self._post_engine(outbound, attempt.policy.snapshot.sha256)
        if response.status_code == 200:
            if response.headers.get("x-proximal-policy-sha256") != attempt.policy.snapshot.sha256:
                raise ValueError("Inference response lacks verified immutable adapter identity")
            if response.headers.get("x-proximal-base-revision") != attempt.policy.base_model.revision:
                raise ValueError("Inference response base revision differs")
            result = orjson.loads(response.content)
            if result.get("model") != payload["model"] or len(result.get("choices", [])) != 1:
                raise ValueError("Inference response model/choice count differs")
            meta = result["choices"][0]["meta_info"]
            # Convert verified immutable identity into Miles's publication ordinal.
            # Never trust the engine's global base-weight counter for a named LoRA.
            token_logprobs = meta.get("output_token_logprobs")
            if not isinstance(token_logprobs, list) or not token_logprobs:
                raise ValueError("Inference response lacks exact output token IDs/logprobs")
            for item in token_logprobs:
                if (
                    not isinstance(item, list)
                    or len(item) < 2
                    or type(item[1]) is not int
                    or item[1] < 0
                    or not isinstance(item[0], (float, int))
                    or not math.isfinite(item[0])
                    or item[0] > 1e-6
                ):
                    raise ValueError("Malformed output token/logprob evidence")
            if len(token_logprobs) > payload["max_tokens"]:
                raise ValueError("Inference exceeded the effective output token budget")
            meta.pop("weight_versions", None)
            meta["weight_version"] = str(attempt.policy.version)
            content = orjson.dumps(result)
        else:
            content = response.content
        return {
            "request_body": outbound,
            "response_body": content,
            "status_code": response.status_code,
            "headers": {"content-type": "application/json"},
        }


# The platform names a run's rollouts ``<run id>-rollout-<index>``; Miles runs have one.
ROLLOUT_SUFFIX = "-rollout-0"


def rollout_id(attempt_id: str) -> str:
    """The platform rollout ID of an attempt's single-instance run."""
    return f"{attempt_id}{ROLLOUT_SUFFIX}"


def normalize_agent_request(body: dict[str, Any], *, reasoning_effort: str) -> None:
    """Map agent-px's Chat Completions request onto the training sampling contract.

    - Cache hints do not affect sampling: dropped.
    - ``reasoning_effort`` must be the contract's value; the TITO renderer owns how
      thinking is rendered, so it is not forwarded.
    - ``max_completion_tokens`` is the per-turn budget, like ``max_tokens``.
    - ``strict`` tools make SGLang constrain decoding to the schema, so behavior
      logprobs would come from a different distribution than training computes.
      Dropped: sampling stays unconstrained and a malformed call is a real tool error.
    """
    body.pop("prompt_cache_key", None)
    body.pop("prompt_cache_retention", None)
    if (effort := body.pop("reasoning_effort", None)) is not None and effort != reasoning_effort:
        raise HTTPException(422, f"reasoning_effort {effort!r} differs from the training contract")
    if (cap := body.pop("max_completion_tokens", None)) is not None:
        if body.get("max_tokens", cap) != cap:
            raise HTTPException(422, "max_tokens and max_completion_tokens disagree")
        body["max_tokens"] = cap
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict):
                function.pop("strict", None)


class CaptureServer:
    def __init__(
        self,
        authorization: AuthorizedRun,
        *,
        tokenizer: Any,
        client: httpx.AsyncClient,
        store: RolloutStore,
    ):
        self.config = require_authorization(authorization)
        self.client = client  # Borrowed: the process composition root owns it.
        self.store = store  # Borrowed, likewise.
        self.admin_key = secret_env(self.config.capture.api_key_env)
        self.platform_key = secret_env(self.config.capture.platform_key_env)
        self.root = self.config.artifact_directory / self.config.run_id / "capture"
        write_immutable(self.root / "run.json", canonical_bytes(self.config))
        self.sessions: dict[str, LiveSession] = {}
        self.attempts: dict[str, str] = {}
        self._create_lock = asyncio.Lock()
        self.transport = BoundTransport(self.config, client, self.sessions)
        # Built here, never passed in: the registry's matcher is part of capture correctness.
        registry = capture_registry(self.config, tokenizer)
        self.core = SessionCore(self.transport, registry, session_config(self.config), self.config.run_id)
        self.app = FastAPI()
        self._routes()

    def _admin(self, request: Request) -> None:
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {self.admin_key}"):
            raise HTTPException(401, "Invalid capture control credential")

    def _rollout_session(self, request: Request, platform_rollout_id: str) -> str:
        """The platform's rollout route. One run per attempt, so the run ID is the attempt ID
        and its only rollout is ``<attempt id>-rollout-0``."""
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {self.platform_key}"):
            raise HTTPException(401, "Invalid platform credential")
        attempt_id = platform_rollout_id.removesuffix(ROLLOUT_SUFFIX)
        session_id = self.attempts.get(attempt_id) if platform_rollout_id.endswith(ROLLOUT_SUFFIX) else None
        if session_id is None or session_id not in self.sessions:
            raise HTTPException(404, "No live capture session for this rollout; regenerate the attempt")
        return session_id

    def _directory(self, session_id: str) -> Path:
        # Session IDs are generated here, never caller-controlled filesystem paths.
        if len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
            raise HTTPException(404, "Unknown session")
        return self.root / "sessions" / session_id

    async def _seal(self, session_id: str) -> CaptureReceipt:
        directory = self._directory(session_id)
        if (directory / "receipt.json").exists():
            return CaptureReceipt.model_validate_json((directory / "receipt.json").read_bytes())
        entry = self.sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, "Unsealed session was lost")
        async with entry.lock:
            if (directory / "receipt.json").exists():
                return CaptureReceipt.model_validate_json((directory / "receipt.json").read_bytes())
            entry.sealed = True
            session = self.core.registry.get_session(session_id)
            if not session.records or len(session.token_ids) > entry.attempt.sampling.max_sequence_tokens:
                raise HTTPException(422, "Missing capture or sequence budget exceeded")
            # Do not truncate a trajectory after its grade was earned.
            response = await self.core.collect_samples(session_id, max_seq_len=None)
            if response.status_code != 200:
                raise HTTPException(422, "TITO sample assembly failed")
            payload = bytes(response.body)
            receipt = CaptureReceipt(
                session_id=session_id,
                request_sha256=digest(entry.attempt),
                policy=entry.attempt.policy,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                num_calls=len(session.records),
                num_tokens=len(session.token_ids),
            )
            write_immutable(directory / "samples.safetensors", payload)
            write_immutable(directory / "attempt.json", canonical_bytes(entry.attempt))
            write_immutable(directory / "receipt.json", canonical_bytes(receipt))
            return receipt

    async def _create_session(self, attempt: Attempt, request: Request) -> dict[str, str]:
        self._admin(request)
        if (
            attempt.run_id != self.config.run_id
            or attempt.harness != self.config.harness
            or attempt.sampling != self.config.research.sampling
            or attempt.dataset_sha256 != pinned_dataset(self.config.dataset).sha256
            or attempt.task not in pinned_dataset(self.config.dataset).tasks
            or await self.store.policy(attempt.policy.version) != attempt.policy
        ):
            raise HTTPException(409, "Attempt is outside this run's dataset/harness/policy contract")
        async with self._create_lock:
            index = self.root / "attempts" / f"{attempt.attempt_id}.json"
            session_id = self.attempts.get(attempt.attempt_id)
            if session_id is None:
                if index.exists():
                    raise HTTPException(410, "Attempt was lost or released; create a new execution identity")
                if len(self.sessions) >= self.config.max_in_flight_samples:
                    raise HTTPException(429, "Capture session capacity reached")
                session_id = self.core.registry.create_session()
                write_immutable(
                    index, json.dumps({"session_id": session_id, "request_sha256": digest(attempt)}).encode()
                )
                self.sessions[session_id] = LiveSession(attempt)
                self.attempts[attempt.attempt_id] = session_id
            entry = self.sessions.get(session_id)
            if entry is None:
                raise HTTPException(410, "Attempt was released; create a new execution attempt")
            if entry.attempt != attempt:
                raise HTTPException(409, "Attempt identity reused with different inputs")
            return {
                "session_id": session_id,
                # What the platform's registry derives for this run; informational here.
                "base_url": f"{self.config.capture.url}/rollouts/{rollout_id(attempt.attempt_id)}/v1",
                "request_sha256": digest(attempt),
            }

    async def _chat_completion(self, session_id: str, request: Request) -> Response:
        entry = self.sessions[session_id]
        if request.url.query:
            raise HTTPException(422, "Query parameters are not supported")
        async with entry.lock:
            if entry.sealed:
                raise HTTPException(409, "Session is sealed")
            body = orjson.loads(await request.body())  # OpenAI SDK boundary, not a domain contract.
            if not isinstance(body, dict) or body.get("model") != self.config.base_model.name:
                raise HTTPException(422, "Wrong model for this session")
            allowed = {
                "model",
                "messages",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "stream",
                "stream_options",
                "temperature",
                "top_p",
                "top_k",
                "n",
                "max_tokens",
                # Sent by agent-px's Chat Completions adapter on every call.
                "max_completion_tokens",
                "reasoning_effort",
                "prompt_cache_key",
                "prompt_cache_retention",
            }
            if set(body) - allowed:
                raise HTTPException(422, f"Unsupported training request fields: {sorted(set(body) - allowed)}")
            normalize_agent_request(body, reasoning_effort=self.config.model_protocol.reasoning_effort)
            if body.get("tool_choice", "auto") not in ("auto", "none"):
                raise HTTPException(422, "Constrained tool selection changes the sampling distribution")
            for message in body.get("messages", []):
                if not isinstance(message, dict) or (
                    message.get("content") is not None and not isinstance(message["content"], str)
                ):
                    raise HTTPException(422, "First pass supports text-only messages")
            body["chat_template_kwargs"] = {"enable_thinking": self.config.enable_thinking}
            sampling = entry.attempt.sampling
            for key, value in {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
            }.items():
                if key in body and body[key] != value:
                    raise HTTPException(422, f"Sampling conflict: {key}")
                body[key] = value
            if body.get("n", 1) != 1:
                raise HTTPException(422, "One completion per request is required")
            # The harness's cap is a ceiling from the platform's model family; the training
            # contract's per-turn budget is authoritative, so the smaller one applies.
            budget = body.get("max_tokens", sampling.max_tokens)  # Normalized from max_completion_tokens.
            if type(budget) is not int or budget <= 0:
                raise HTTPException(422, "Invalid per-turn token budget")
            body["max_tokens"] = min(budget, sampling.max_tokens)
            return await self.core.chat_completions(
                session_id, method="POST", query="", headers={}, body=orjson.dumps(body)
            )

    def _routes(self) -> None:
        app = self.app

        @app.exception_handler(SessionError)
        async def session_error(request: Request, exc: SessionError) -> Response:
            return Response(status_code=exc.status_code, content=str(exc))

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "run_id": self.config.run_id}

        @app.post("/sessions")
        async def create(attempt: Attempt, request: Request) -> dict[str, str]:
            return await self._create_session(attempt, request)

        @app.post("/rollouts/{platform_rollout_id}/v1/chat/completions")
        async def chat(platform_rollout_id: str, request: Request) -> Response:
            return await self._chat_completion(self._rollout_session(request, platform_rollout_id), request)

        @app.post("/sessions/{session_id}/seal")
        async def seal(session_id: str, request: Request) -> CaptureReceipt:
            self._admin(request)
            return await self._seal(session_id)

        @app.get("/sessions/{session_id}/samples")
        async def samples(session_id: str, request: Request) -> Response:
            self._admin(request)
            directory = self._directory(session_id)
            if not (directory / "receipt.json").exists():
                raise HTTPException(409, "Capture is not sealed")
            return Response((directory / "samples.safetensors").read_bytes(), media_type="application/octet-stream")

        @app.delete("/sessions/{session_id}", status_code=204)
        async def release(session_id: str, request: Request) -> Response:
            self._admin(request)
            self._directory(session_id)
            entry = self.sessions.get(session_id)
            if entry is not None:
                async with entry.lock:
                    await self.core.delete_session(session_id)
                    self.sessions.pop(session_id)
                    self.attempts.pop(entry.attempt.attempt_id, None)
            return Response(status_code=204)
