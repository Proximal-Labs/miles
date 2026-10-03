"""Full-snapshot HTTP shards, pinned to one replica for an entire upload."""

import asyncio
import hashlib
import os
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization, secret_env
from miles_plugins.proximal.contracts import PolicyEvidence
from miles_plugins.proximal.replica import ReplicaLoRALoader
from miles_plugins.proximal.sharded_snapshot import (
    ShardedManifest,
    part_relative_path,
    restore_sharded,
    sharded_relative_path,
)
from miles_plugins.proximal.snapshot import SnapshotReference

REPLICA_HEADER = "X-Proximal-Upload-Replica"
MAX_SNAPSHOT_BYTES = 8 * 2**30


class UploadTarget(FrozenStrictBaseModel):
    replica_id: str
    snapshot: SnapshotReference


def upload_headers(authorization: AuthorizedRun) -> dict[str, str]:
    config = require_authorization(authorization)
    return {name: secret_env(env) for name, env in config.inference_header_env.items()}


def upload_request(client: httpx.Client, method: str, path: str, *, content: bytes) -> httpx.Response:
    response = client.request(method, path, content=content, follow_redirects=False)
    response.raise_for_status()
    return response


def add_upload_routes(
    app: FastAPI,
    *,
    loader: ReplicaLoRALoader,
    authorize: Callable[[Request], None],
    prepare: Callable[[SnapshotReference], Awaitable[PolicyEvidence]],
) -> None:
    replica_id = uuid.uuid4().hex
    active: ShardedManifest | None = None
    root = loader.upload_directory
    lock = asyncio.Lock()

    def check(request: Request, sha: str) -> ShardedManifest:
        authorize(request)
        if request.headers.get(REPLICA_HEADER) != replica_id:
            raise HTTPException(409, "Upload routed to another replica; retry the whole publication")
        if active is None or active.snapshot.sha256 != sha:
            raise HTTPException(409, "No matching upload")
        return active

    @app.post("/policies/uploads")
    async def begin(body: ShardedManifest, request: Request) -> UploadTarget:
        nonlocal active
        authorize(request)
        if sum(part.size_bytes for parts in body.files.values() for part in parts) > MAX_SNAPSHOT_BYTES:
            raise HTTPException(413, "Snapshot exceeds upload limit")
        async with lock:
            if active is not None and active != body:
                raise HTTPException(409, "Another snapshot upload is pending")
            (root / sharded_relative_path(body.snapshot)).mkdir(parents=True, exist_ok=True)
            active = body
        return UploadTarget(replica_id=replica_id, snapshot=body.snapshot)

    @app.put("/policies/uploads/{sha}/{index}")
    async def receive(sha: str, index: int, request: Request) -> None:
        manifest = check(request, sha)
        parts = [(name, part) for name, group in manifest.files.items() for part in group]
        if not 0 <= index < len(parts):
            raise HTTPException(404, "Unknown part")
        name, part = parts[index]
        destination = root / sharded_relative_path(manifest.snapshot) / part_relative_path(name, part)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                digest, size = hashlib.sha256(), 0
                async for block in request.stream():
                    size += len(block)
                    if size > part.size_bytes:
                        raise HTTPException(413, "Part exceeds declared size")
                    digest.update(block)
                    await asyncio.to_thread(stream.write, block)
                if size != part.size_bytes or digest.hexdigest() != part.sha256:
                    raise HTTPException(409, "Part checksum or size mismatch")
                stream.flush()
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    with destination.open("rb") as existing:
                        if hashlib.file_digest(existing, "sha256").hexdigest() != part.sha256:
                            raise HTTPException(409, "Conflicting part") from None
            finally:
                temporary.unlink(missing_ok=True)

    @app.post("/policies/uploads/{sha}/complete")
    async def complete(sha: str, request: Request) -> PolicyEvidence:
        nonlocal active
        async with lock:
            manifest = check(request, sha)
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                directory = Path(temporary) / "snapshot"
                try:
                    await asyncio.to_thread(restore_sharded, root, manifest, directory)
                    await asyncio.to_thread(loader.install_uploaded, directory, manifest.snapshot)
                except (ValueError, OSError) as exc:
                    raise HTTPException(409, "Incomplete or invalid snapshot") from exc
            evidence = await prepare(manifest.snapshot)
            shutil.rmtree(root / sharded_relative_path(manifest.snapshot))
            active = None
            return evidence

    @app.delete("/policies/uploads/{sha}")
    async def cancel(sha: str, request: Request) -> None:
        nonlocal active
        async with lock:
            manifest = check(request, sha)
            shutil.rmtree(root / sharded_relative_path(manifest.snapshot))
            active = None
