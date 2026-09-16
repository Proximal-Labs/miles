"""
Utilities for the OpenAI endpoint
"""

import asyncio
import json
import logging
import random
from argparse import Namespace

import httpx

from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    SamplesReply,
    decode_samples_and_merge_input_sample,
)
from miles.utils.http_utils import post, post_bytes_no_retry
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120


class OpenAIEndpointTracer:
    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
        samples_wire_fields: tuple[str, ...] = COMPUTED_FIELDS,
    ):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id
        # The samples-wire allowlist must match the server's encode: v1 default,
        # extended under --use-session-server v2 (create() selects from args;
        # direct constructions keep v1).
        self.samples_wire_fields = samples_wire_fields

    @property
    def session_server_id(self) -> str:
        """``ip:port`` of the instance owning this session, as recorded in sample metadata."""
        return self.router_url.removeprefix("http://")

    @staticmethod
    async def create(args: Namespace):
        session_addrs = getattr(args, "session_server_addrs", None)
        if not session_addrs:
            raise RuntimeError(
                "session_server_addrs is not set. Pass --use-session-server to start the session server."
            )
        # The only routing decision in the system: pick the owning instance once
        # per session; every later touch of the session reuses this URL.
        session_addr = random.choice(session_addrs)
        session_url = f"http://{session_addr}"
        instance_ids = getattr(args, "session_server_instance_ids", None) or {}
        session_server_instance_id = instance_ids.get(session_addr)
        response = await post(f"{session_url}/sessions", {}, action="post")
        session_id = response["session_id"]
        use_v2 = getattr(args, "use_session_server", None) == "v2"
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
            samples_wire_fields=COMPUTED_FIELDS_V2 if use_v2 else COMPUTED_FIELDS,
        )

    async def collect_samples(
        self,
        input_sample: Sample,
        *,
        max_seq_len: int | None,
        agent_metadata: dict | None = None,
        producer_finished: bool = True,
    ) -> SamplesReply:
        """Collect after the agent has joined its children and tool work."""
        body: dict = {"max_seq_len": max_seq_len}
        if agent_metadata is not None:
            body["metadata"] = agent_metadata
        if self.samples_wire_fields == COMPUTED_FIELDS_V2:
            return await self._collect_finalized(input_sample, body, producer_finished=producer_finished)
        try:
            # Timeouts and transport errors propagate after cleanup, for `generate` to handle.
            payload = await post_bytes_no_retry(
                f"{self.base_url}/samples",
                body,
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        finally:
            await self._release()

        return decode_samples_and_merge_input_sample(payload, input_sample, fields=self.samples_wire_fields)

    async def _collect_finalized(self, input_sample: Sample, body: dict, *, producer_finished: bool) -> SamplesReply:
        finished = json.loads(await self._post_finalized("finish", {"producer_finished": producer_finished}))
        body = {**body, "snapshot_id": finished["snapshot_id"]}
        payload = await self._post_finalized("samples", body)
        reply = decode_samples_and_merge_input_sample(payload, input_sample, fields=self.samples_wire_fields)
        if reply.empty_reason != "incomplete":
            await self._release()
        else:
            logger.warning("Incomplete session retained for inspection: %s", self.base_url)
        return reply

    async def _post_finalized(self, operation: str, body: dict) -> bytes:
        # finish and sealed exports are idempotent, including after a lost response
        for attempt in range(2):
            try:
                return await post_bytes_no_retry(
                    f"{self.base_url}/{operation}", body, timeout=_SESSION_REQUEST_TIMEOUT
                )
            except (TimeoutError, httpx.TransportError):
                if attempt:
                    logger.warning("Session retained after %s failed: %s", operation, self.base_url)
                    raise

    async def _release(self) -> None:
        try:
            await asyncio.wait_for(post(self.base_url, {}, action="delete"), timeout=_SESSION_REQUEST_TIMEOUT)
        except Exception as exc:
            logger.warning("Failed to delete session %s after collecting samples: %s", self.session_id, exc)
