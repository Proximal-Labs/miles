"""Discover the Modal serving fleet and deliver each rank's bytes to every replica."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization
from miles_plugins.proximal.contracts import PolicyEvidence, RunConfig
from miles_plugins.proximal.http_sync import REPLICA_HEADER, UploadTarget, upload_headers, upload_request
from miles_plugins.proximal.sharded_snapshot import ShardedManifest

UPSTREAM_HEADER = "modal-flash-upstream"
logger = logging.getLogger(__name__)


def discover_replicas(config: RunConfig) -> tuple[str, ...]:
    """List actual containers, not affinity guesses; refuse a partial warm pool."""
    # Only the Modal launcher uses discovery; optional SDK imports stay at this boundary.
    import modal
    from modal.client import _Client
    from modal_proto import api_pb2

    pool = config.weight_sync_pool
    if pool is None:
        raise ValueError("HTTP sync requires the launcher's weight_sync_pool")

    async def discover() -> tuple[str, ...]:
        client = await _Client.from_env()
        environment = config.volume.environment_name
        server = modal.Server.from_name(pool.app_name, "Replica", environment_name=environment)
        url = await server.get_url.aio()
        if url is None or url.rstrip("/") != config.inference_url.rstrip("/"):
            raise ValueError("Discovered Modal serving URL differs from inference_url")
        function = await client.stub.FunctionGet(
            api_pb2.FunctionGetRequest(app_name=pool.app_name, object_tag="Replica", environment_name=environment)
        )
        reply = await client.stub.FlashContainerList(
            api_pb2.FlashContainerListRequest(function_id=function.function_id)
        )
        hosts = []
        for container in reply.containers:
            parsed = urlsplit("https://" + container.host.removeprefix("https://"))
            if not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username:
                raise ValueError("Modal returned an invalid replica host")
            hosts.append(f"{parsed.hostname}:{parsed.port or 443}")
        if len(set(hosts)) != len(hosts) or len(hosts) < pool.min_replicas:
            raise ValueError(f"HTTP sync needs at least {pool.min_replicas} distinct replicas, found {len(hosts)}")
        return tuple(sorted(hosts))

    return asyncio.run(discover())


@dataclass(frozen=True)
class ReplicaUpload:
    upstream: str
    target: UploadTarget


def _client(authorization: AuthorizedRun, upstream: str) -> httpx.Client:
    config = require_authorization(authorization)
    return httpx.Client(
        base_url=config.inference_url,
        headers=upload_headers(authorization) | {UPSTREAM_HEADER: upstream},
        timeout=config.request_timeout_seconds,
        trust_env=False,
    )


def begin_uploads(
    authorization: AuthorizedRun, manifest: ShardedManifest, upstreams: tuple[str, ...]
) -> tuple[ReplicaUpload, ...]:
    # Handshakes are small; retain every acknowledged target for cleanup on partial failure.
    targets = []
    try:
        for upstream in upstreams:
            with _client(authorization, upstream) as client:
                client.headers["Content-Type"] = "application/json"
                response = upload_request(
                    client, "POST", "/policies/uploads", content=manifest.model_dump_json().encode()
                )
                target = UploadTarget.model_validate_json(response.content)
                targets.append(ReplicaUpload(upstream, target))
                if target.snapshot != manifest.snapshot or len({t.target.replica_id for t in targets}) != len(targets):
                    raise ValueError("HTTP discovery reached a duplicate replica or another snapshot")
        return tuple(targets)
    except BaseException:
        cancel_uploads(authorization, manifest, tuple(targets))
        raise


def send_to_replicas(
    authorization: AuthorizedRun,
    manifest: ShardedManifest,
    targets: tuple[ReplicaUpload, ...],
    parts: list[tuple[int, bytes]] | None,
) -> None:
    """Each rank fans its disjoint shards out; rank zero later verifies every receiver."""
    config = require_authorization(authorization)
    prefix = f"/policies/uploads/{manifest.snapshot.sha256}"

    def send(replica: ReplicaUpload) -> None:
        started = time.monotonic()
        with _client(authorization, replica.upstream) as client:
            client.headers[REPLICA_HEADER] = replica.target.replica_id
            client.headers["Content-Type"] = "application/octet-stream"
            if parts is not None:
                for index, data in parts:
                    upload_request(client, "PUT", f"{prefix}/{index}", content=data)
            else:
                response = upload_request(client, "POST", f"{prefix}/complete", content=b"")
                evidence = PolicyEvidence.model_validate_json(response.content)
                if (
                    evidence.snapshot != manifest.snapshot
                    or evidence.base_model != config.base_model
                    or evidence.request_model != f"{config.base_model.name}:miles-{manifest.snapshot.sha256}"
                ):
                    raise ValueError("Replica did not verify the published policy")
                logger.info(
                    "HTTP adapter loaded: snapshot=%s replica=%s upstream=%s prepare_seconds=%.3f",
                    manifest.snapshot.sha256,
                    replica.target.replica_id,
                    replica.upstream,
                    time.monotonic() - started,
                )

    if not targets:
        raise ValueError("Cannot publish to an empty HTTP fleet")
    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as workers:
        list(workers.map(send, targets))


def cancel_uploads(
    authorization: AuthorizedRun, manifest: ShardedManifest, targets: tuple[ReplicaUpload, ...]
) -> None:
    for replica in targets:
        with suppress(httpx.HTTPError), _client(authorization, replica.upstream) as client:
            client.headers[REPLICA_HEADER] = replica.target.replica_id
            upload_request(client, "DELETE", f"/policies/uploads/{manifest.snapshot.sha256}", content=b"")
