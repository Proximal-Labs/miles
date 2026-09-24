"""Replica-local policy admission in front of SGLang; mount in every fleet replica.

All requests use an immutable named LoRA. Adapter slots are bounded; active
requests pin their slot, and an idle LRU entry can be evicted and reloaded later.
"""

import asyncio
import hmac
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from miles_plugins.proximal.authorization import secret_env
from miles_plugins.proximal.contracts import Contract, PolicyEvidence, Positive
from miles_plugins.proximal.replica import RegisteredAdapter, ReplicaConfig, ReplicaLoRALoader
from miles_plugins.proximal.snapshot import BaseModelIdentity, SnapshotReference


class GatewayConfig(Contract):
    replica: ReplicaConfig
    api_key_env: str
    engine_model_path: str
    max_loaded_adapters: Positive


class PreparePolicy(Contract):
    snapshot: SnapshotReference
    base_model: BaseModelIdentity


@dataclass
class Slot:
    adapter: RegisteredAdapter
    active: int


class ReplicaGateway:
    def __init__(self, config: GatewayConfig, *, loader: ReplicaLoRALoader, client: httpx.AsyncClient):
        self.config, self.loader, self.client = config, loader, client
        self._key = secret_env(config.api_key_env)
        self._slots: OrderedDict[str, Slot] = OrderedDict()
        self._condition = asyncio.Condition()
        self._healthy = True
        self.app = FastAPI()
        self._routes()

    async def validate_engine(self) -> None:
        reply = await self.client.get(f"{self.config.replica.backend_url}/get_model_info", follow_redirects=False)
        reply.raise_for_status()
        if reply.json().get("model_path") != self.config.engine_model_path:
            raise ValueError("SGLang loaded a different base model path")

    def _authorize(self, request: Request) -> None:
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {self._key}"):
            raise HTTPException(401, "Invalid replica gateway credential")

    @asynccontextmanager
    async def _admit(self, reference: SnapshotReference) -> AsyncIterator[RegisteredAdapter]:
        async with self._condition:
            if not self._healthy:
                raise HTTPException(503, "Adapter state is uncertain; replace this replica")
            while reference.sha256 not in self._slots and len(self._slots) >= self.config.max_loaded_adapters:
                if not self._healthy:
                    raise HTTPException(503, "Adapter state is uncertain; replace this replica")
                idle = next((key for key, slot in self._slots.items() if slot.active == 0), None)
                if idle is None:
                    await self._condition.wait()
                    continue
                await self._evict_slot(idle)
            if not self._healthy:
                raise HTTPException(503, "Adapter state is uncertain; replace this replica")
            if reference.sha256 not in self._slots:
                await self._load_slot(reference)
            slot = self._slots[reference.sha256]
            slot.active += 1
            self._slots.move_to_end(reference.sha256)
        try:
            yield slot.adapter
        finally:
            async with self._condition:
                slot.active -= 1
                self._condition.notify_all()

    async def _load_slot(self, reference: SnapshotReference) -> None:
        # A cancelled HTTP waiter cannot cancel the underlying sync engine call.
        # Finish and reconcile it while retaining the admission lock.
        task = asyncio.create_task(asyncio.to_thread(self.loader.ensure_loaded, reference))
        cancelled = False
        try:
            try:
                adapter = await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                adapter = await task
        except BaseException:
            self._healthy = False
            self._condition.notify_all()
            raise
        self._slots[reference.sha256] = Slot(adapter, 0)
        if cancelled:
            raise asyncio.CancelledError()

    async def _evict_slot(self, key: str) -> None:
        task = asyncio.create_task(asyncio.to_thread(self.loader.unload_idle, self._slots[key].adapter.snapshot))
        cancelled = False
        try:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                await task
        except BaseException:
            self._healthy = False
            self._condition.notify_all()
            raise
        del self._slots[key]
        self._condition.notify_all()
        if cancelled:
            raise asyncio.CancelledError()

    def _routes(self) -> None:
        @self.app.get("/health")
        async def health() -> Response:
            return Response("ready" if self._healthy else "replace replica", status_code=200 if self._healthy else 503)

        @self.app.post("/policies/prepare")
        async def prepare(body: PreparePolicy, request: Request) -> PolicyEvidence:
            self._authorize(request)
            if body.base_model != self.config.replica.base_model:
                raise HTTPException(409, "Requested base differs from replica")
            async with self._admit(body.snapshot) as adapter:
                return PolicyEvidence(
                    snapshot=adapter.snapshot,
                    base_model=self.config.replica.base_model,
                    request_model=adapter.request_model,
                )

        @self.app.post("/v1/chat/completions")
        async def chat(request: Request) -> Response:
            self._authorize(request)
            if request.url.query:
                raise HTTPException(422, "Query parameters are unsupported")
            reference = SnapshotReference(sha256=request.headers.get("x-proximal-policy-sha256", ""))
            body = await request.json()
            if not isinstance(body, dict) or body.get("stream") is not False or "lora_path" in body:
                raise HTTPException(422, "Gateway accepts captured non-streaming chat only")
            started = time.perf_counter()
            async with self._admit(reference) as adapter:
                admitted = time.perf_counter()
                if body.get("model") != adapter.request_model:
                    raise HTTPException(409, "Request model does not name the verified adapter")
                response = await self.client.post(
                    f"{self.config.replica.backend_url}/v1/chat/completions", json=body, follow_redirects=False
                )
                done = time.perf_counter()
                return Response(
                    response.content,
                    status_code=response.status_code,
                    media_type="application/json",
                    headers={
                        "X-Proximal-Policy-Sha256": reference.sha256,
                        "X-Proximal-Base-Revision": self.config.replica.base_model.revision,
                        # Where the gateway's time went: adapter admission, then SGLang.
                        "Server-Timing": (
                            f"admit;dur={(admitted - started) * 1000:.1f}, upstream;dur={(done - admitted) * 1000:.1f}"
                        ),
                    },
                )
