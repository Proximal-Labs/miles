"""Authenticated, policy-bound HTTP adapter around Miles's real linear TITO core.

One CPU process owns live sessions. Sealed samples survive its restart on disk;
unfinished sessions are explicitly lost. This app exposes no unrecorded proxy.

The composing process decides where capture runs. On a serving replica (see
``serve_replica``) it sends turns to that replica's gateway in-process and verifies a
session's policy by admitting its adapter; next to the trainer (the local Stage A
harness) it calls the pool over the network and checks the rollout store.
"""

import asyncio
import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.core import ProxyRequest, SessionCore
from miles.rollout.session.errors import SessionError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles_plugins.proximal.call_timing import CallTimingMiddleware, mark, note
from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization, secret_env
from miles_plugins.proximal.contracts import (
    Attempt,
    CaptureReceipt,
    ROLLOUT_SUFFIX,
    Policy,
    RunConfig,
    canonical_bytes,
    digest,
    pinned_dataset,
    platform_rollout_id,
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


def fixed_chat_template(tito_model: str) -> tuple[str | None, dict[str, Any]]:
    """The family's fixed chat template and the kwargs it requires, as Miles's own
    argument resolution applies them for a named ``--tito-model``.

    Capture renders every turn incrementally (the new messages after a stand-in
    prefix). Native templates are not built for that: Qwen3.8's refuses to render a
    conversation without a user message, so every turn after the first fails.
    """
    from miles.utils.chat_template_utils import resolve_fixed_chat_template

    return resolve_fixed_chat_template(tito_model)


def capture_tokenizer(tokenizer_path: str | Path, tito_model: str) -> Any:
    """The one way to load capture's tokenizer: the base model's, with its family's fixed template."""
    from miles.utils.processing_utils import load_tokenizer

    template_path, _ = fixed_chat_template(tito_model)
    return load_tokenizer(
        str(tokenizer_path), chat_template_path=template_path, local_files_only=True, trust_remote_code=False
    )


def _template_kwargs(config: RunConfig) -> dict[str, Any]:
    _, fixed_kwargs = fixed_chat_template(config.tito_model)
    return {"enable_thinking": config.enable_thinking, **fixed_kwargs}


def capture_registry(config: RunConfig, tokenizer: Any) -> SessionRegistry:
    """The one way to build the capture session registry: TITO renderer + matcher."""
    from miles.utils.chat_template_utils import get_tito_tokenizer
    from miles.utils.chat_template_utils.message_matcher_hub import resolve_session_message_matcher

    check_tito_protocol(config)
    template_path, _ = fixed_chat_template(config.tito_model)
    if template_path is not None and tokenizer.chat_template != Path(template_path).read_text():
        raise ValueError(
            f"Capture needs the {config.tito_model} family's fixed chat template; load the tokenizer "
            "with capture_tokenizer"
        )
    tito = get_tito_tokenizer(tokenizer, config.tito_model, chat_template_kwargs=_template_kwargs(config))
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
        chat_template_path=fixed_chat_template(config.tito_model)[0],
        tito_model=config.tito_model,
        apply_chat_template_kwargs=_template_kwargs(config),
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

# Whether a session's policy is one the trainer published; checked when the session opens.
PolicyCheck = Callable[[Policy], Awaitable[bool]]


def committed_in(store: RolloutStore) -> PolicyCheck:
    """Next to the trainer: the rollout store is the policy authority."""

    async def check(policy: Policy) -> bool:
        return await store.policy(policy.version) == policy

    return check


def capture_root(config: RunConfig) -> Path:
    """Where capture keeps its state next to the trainer (the run's artifact directory)."""
    return config.artifact_directory / config.run_id / "capture"


@dataclass(frozen=True)
class EngineEndpoint:
    """Where capture sends rendered turns: a pool gateway's chat route and its credentials."""

    client: httpx.AsyncClient  # Borrowed: the composing process owns it.
    url: str
    headers: dict[str, str]

    @classmethod
    def pool(cls, config: RunConfig, client: httpx.AsyncClient) -> "EngineEndpoint":
        """The serving pool over the network, with the run config's inference credentials."""
        headers = {name: secret_env(env) for name, env in config.inference_header_env.items()}
        return cls(client=client, url=config.inference_url, headers=headers)


class BoundTransport:
    def __init__(self, config: RunConfig, engine: EngineEndpoint, sessions: dict[str, LiveSession]):
        self.config, self.engine, self.sessions = config, engine, sessions

    async def _post_engine(self, outbound: bytes, policy_sha256: str) -> httpx.Response:
        """One engine call, retried when the connection fails.

        Safe to retry: nothing is recorded until a reply is accepted, so a lost request or
        reply only costs a repeated generation. Retrying on transport failures does not
        depend on what was sampled, so it does not bias which completions are trained.
        """
        for attempt in range(ENGINE_ATTEMPTS):
            try:
                return await self.engine.client.post(
                    f"{self.engine.url}/v1/chat/completions",
                    content=outbound,
                    headers={
                        **self.engine.headers,
                        "Content-Type": "application/json",
                        "X-Proximal-Policy-Sha256": policy_sha256,
                    },
                    follow_redirects=False,
                )
            except httpx.TransportError:
                note(engine_retries=attempt + 1)
                if attempt == ENGINE_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(0.2 * 2**attempt)
        raise AssertionError("unreachable")

    async def do_proxy(
        self, request: ProxyRequest, path: str, *, body: bytes, headers: dict[str, str]
    ) -> dict[str, object]:
        if request.session_id is None or path != "v1/chat/completions" or request.method != "POST" or request.query:
            raise ValueError("Only bound recorded chat requests may reach inference")
        mark("proxy_start")  # SessionCore has rendered the turn's token IDs.
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
        note(
            input_tokens=len(payload["input_ids"]),
            max_tokens=payload["max_tokens"],
            engine_request_bytes=len(outbound),
        )
        mark("engine_sent")
        response = await self._post_engine(outbound, attempt.policy.snapshot.sha256)
        mark("engine_done")
        note(
            engine_status=response.status_code,
            engine_response_bytes=len(response.content),
            gateway_timing=response.headers.get("server-timing"),
        )
        if response.status_code == 200:
            if response.headers.get("x-proximal-policy-sha256") != attempt.policy.snapshot.sha256:
                raise ValueError("Inference response lacks verified immutable adapter identity")
            if response.headers.get("x-proximal-base-revision") != attempt.policy.base_model.revision:
                raise ValueError("Inference response base revision differs")
            result = orjson.loads(response.content)
            if result.get("model") != payload["model"] or len(result.get("choices", [])) != 1:
                raise ValueError("Inference response model/choice count differs")
            meta = result["choices"][0]["meta_info"]
            note(
                response_id=result.get("id"),
                output_tokens=len(meta.get("output_token_logprobs") or []),
                engine_meta={
                    k: meta[k]
                    for k in ("prompt_tokens", "cached_tokens", "e2e_latency", "completion_tokens")
                    if k in meta
                },
            )
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
        mark("proxy_end")
        return {
            "request_body": outbound,
            "response_body": content,
            "status_code": response.status_code,
            "headers": {"content-type": "application/json"},
        }


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
        engine: EngineEndpoint,
        policy_known: PolicyCheck,
        root: Path,
    ):
        self.config = require_authorization(authorization)
        self.policy_known = policy_known
        self.admin_key = secret_env(self.config.capture.api_key_env)
        self.platform_key = secret_env(self.config.capture.platform_key_env)
        self.root = root
        write_immutable(self.root / "run.json", canonical_bytes(self.config))
        self.sessions: dict[str, LiveSession] = {}
        self.attempts: dict[str, str] = {}
        self._create_lock = asyncio.Lock()
        self.transport = BoundTransport(self.config, engine, self.sessions)
        # Built here, never passed in: the registry's matcher is part of capture correctness.
        registry = capture_registry(self.config, tokenizer)
        self.core = SessionCore(self.transport, registry, session_config(self.config), self.config.run_id)
        self.app = FastAPI()
        # One JSON line per chat call: where each call's time goes (see call_timing).
        self.app.add_middleware(CallTimingMiddleware, log_path=self.root / "call-timing.jsonl")
        self._routes()

    @classmethod
    def beside_trainer(
        cls, authorization: AuthorizedRun, *, tokenizer: Any, client: httpx.AsyncClient, store: RolloutStore
    ) -> "CaptureServer":
        """Capture next to the trainer (the local Stage A harness): the pool over the
        network, the rollout store as policy authority, state in the run's artifacts."""
        config = require_authorization(authorization)
        return cls(
            authorization,
            tokenizer=tokenizer,
            engine=EngineEndpoint.pool(config, client),
            policy_known=committed_in(store),
            root=capture_root(config),
        )

    def _admin(self, request: Request) -> None:
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {self.admin_key}"):
            raise HTTPException(401, "Invalid capture control credential")

    def _rollout_session(self, request: Request, rollout: str) -> str:
        """The platform's rollout route. One run per attempt, so the run ID is the attempt ID
        and its only rollout is ``<attempt id>-rollout-0``."""
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {self.platform_key}"):
            raise HTTPException(401, "Invalid platform credential")
        attempt_id = rollout.removesuffix(ROLLOUT_SUFFIX)
        session_id = self.attempts.get(attempt_id) if rollout.endswith(ROLLOUT_SUFFIX) else None
        if session_id is None or session_id not in self.sessions:
            # Also what a call routed to a different replica than its session gets: the
            # rollout fails loudly rather than continuing without its token history.
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
            or not await self.policy_known(attempt.policy)
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
                "rollout_id": platform_rollout_id(attempt.attempt_id),
                "base_url": f"{self.config.capture.url}/rollouts/{platform_rollout_id(attempt.attempt_id)}/v1",
                "request_sha256": digest(attempt),
            }

    async def _chat_completion(self, session_id: str, request: Request) -> Response:
        mark("handler_start")
        entry = self.sessions[session_id]
        if request.url.query:
            raise HTTPException(422, "Query parameters are not supported")
        async with entry.lock:
            mark("session_locked")
            if entry.sealed:
                raise HTTPException(409, "Session is sealed")
            raw = await request.body()
            note(agent_request_bytes=len(raw), session_id=session_id)
            body = orjson.loads(raw)  # OpenAI SDK boundary, not a domain contract.
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
            note(messages=len(body.get("messages", [])))
            mark("validated")
            response = await self.core.chat_completions(
                session_id, method="POST", query="", headers={}, body=orjson.dumps(body)
            )
            mark("core_done")  # SessionCore has recorded the turn and built the reply.
            return response

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

        @app.post("/rollouts/{rollout}/v1/chat/completions")
        async def chat(rollout: str, request: Request) -> Response:
            return await self._chat_completion(self._rollout_session(request, rollout), request)

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
