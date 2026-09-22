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
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.core import ProxyRequest, SessionCore
from miles.rollout.session.errors import SessionError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization, secret_env
from miles_plugins.proximal.contracts import Attempt, CaptureReceipt, RunConfig, canonical_bytes, digest
from miles_plugins.proximal.storage import write_immutable
from miles_plugins.proximal.store import RolloutStore


@dataclass
class LiveSession:
    attempt: Attempt
    token: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sealed: bool = False


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
        session_message_matcher="strict",
        pause_generation_mode=None,
        session_sample_picker_path=None,
        session_sample_postprocessor_path=None,
    )


class BoundTransport:
    def __init__(self, config: RunConfig, client: httpx.AsyncClient, sessions: dict[str, LiveSession]):
        self.config, self.client, self.sessions = config, client, sessions
        self.headers = {name: secret_env(env) for name, env in config.inference_header_env.items()}

    async def do_proxy(
        self, request: ProxyRequest, path: str, *, body: bytes, headers: dict[str, str]
    ) -> dict[str, object]:
        if request.session_id is None or path != "v1/chat/completions" or request.method != "POST" or request.query:
            raise ValueError("Only bound recorded chat requests may reach inference")
        attempt = self.sessions[request.session_id].attempt
        payload = json.loads(body)  # SDK boundary; SessionCore already rendered and validated input_ids.
        remaining = attempt.sampling.max_sequence_tokens - len(payload["input_ids"])
        if remaining <= 0:
            raise HTTPException(422, "Sequence token budget exhausted")
        payload["max_tokens"] = min(payload["max_tokens"], remaining)
        payload["model"] = f"{self.config.base_model.name}:miles-{attempt.policy.snapshot.sha256}"
        outbound = json.dumps(payload).encode()
        response = await self.client.post(
            f"{self.config.inference_url}/v1/chat/completions",
            content=outbound,
            headers={
                **self.headers,
                "Content-Type": "application/json",
                "X-Proximal-Policy-Sha256": attempt.policy.snapshot.sha256,
            },
            follow_redirects=False,
        )
        if response.status_code == 200:
            if response.headers.get("x-proximal-policy-sha256") != attempt.policy.snapshot.sha256:
                raise ValueError("Inference response lacks verified immutable adapter identity")
            if response.headers.get("x-proximal-base-revision") != attempt.policy.base_model.revision:
                raise ValueError("Inference response base revision differs")
            result = response.json()
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
            content = json.dumps(result).encode()
        else:
            content = response.content
        return {
            "request_body": outbound,
            "response_body": content,
            "status_code": response.status_code,
            "headers": {"content-type": "application/json"},
        }


class CaptureServer:
    def __init__(
        self,
        authorization: AuthorizedRun,
        *,
        registry: SessionRegistry,
        client: httpx.AsyncClient,
        store: RolloutStore,
    ):
        self.config = require_authorization(authorization)
        self.client = client  # Borrowed: the process composition root owns it.
        self.store = store  # Borrowed, likewise.
        self.admin_key = secret_env(self.config.capture.api_key_env)
        self.root = self.config.artifact_directory / self.config.run_id / "capture"
        write_immutable(self.root / "run.json", canonical_bytes(self.config))
        self.sessions: dict[str, LiveSession] = {}
        self.attempts: dict[str, str] = {}
        self._create_lock = asyncio.Lock()
        self.transport = BoundTransport(self.config, client, self.sessions)
        self.core = SessionCore(self.transport, registry, session_config(self.config), self.config.run_id)
        self.app = FastAPI()
        self._routes()

    def _admin(self, request: Request) -> None:
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {self.admin_key}"):
            raise HTTPException(401, "Invalid capture control credential")

    def _live(self, request: Request, session_id: str) -> LiveSession:
        entry = self.sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, "Live session was lost or released; regenerate the attempt")
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {entry.token}"):
            raise HTTPException(401, "Invalid session credential")
        return entry

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
            or attempt.dataset_sha256 != digest(self.config.dataset)
            or attempt.task not in self.config.dataset.tasks
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
                self.sessions[session_id] = LiveSession(attempt, secrets.token_urlsafe(32))
                self.attempts[attempt.attempt_id] = session_id
            entry = self.sessions.get(session_id)
            if entry is None:
                raise HTTPException(410, "Attempt was released; create a new execution attempt")
            if entry.attempt != attempt:
                raise HTTPException(409, "Attempt identity reused with different inputs")
            return {
                "session_id": session_id,
                "base_url": f"{self.config.capture.url}/sessions/{session_id}",
                "api_key": entry.token,
                "request_sha256": digest(attempt),
            }

    async def _chat_completion(self, session_id: str, request: Request) -> Response:
        entry = self._live(request, session_id)
        if request.url.query:
            raise HTTPException(422, "Query parameters are not supported")
        async with entry.lock:
            if entry.sealed:
                raise HTTPException(409, "Session is sealed")
            body = await request.json()  # OpenAI SDK boundary, not a domain contract.
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
            }
            if set(body) - allowed:
                raise HTTPException(422, "Unsupported training request fields")
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
            budget = body.get("max_tokens", sampling.max_tokens)
            if type(budget) is not int or not 0 < budget <= sampling.max_tokens:
                raise HTTPException(422, "Invalid per-turn token budget")
            body["max_tokens"] = budget
            return await self.core.chat_completions(
                session_id, method="POST", query="", headers={}, body=json.dumps(body).encode()
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

        @app.post("/sessions/{session_id}/v1/chat/completions")
        async def chat(session_id: str, request: Request) -> Response:
            return await self._chat_completion(session_id, request)

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
