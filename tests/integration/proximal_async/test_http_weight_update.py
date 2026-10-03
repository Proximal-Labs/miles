"""Real HTTP, four Gloo uploader ranks and real snapshot/store recovery; engine/Volume are boundaries."""

import asyncio
import json
import socket
import threading
import time
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import modal
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import uvicorn
from fastapi.testclient import TestClient
from tests.fast.proximal_publication.test_publication import FakeVolume
from tests.integration.proximal_async.test_weight_update import (
    CpuAdapterIterator,
    current_version,
    make_updater,
    transfer_args,
)

from miles.utils import distributed_utils
from miles_plugins.proximal import http_fleet, policy_archive
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.contracts import HTTPServingPool, Policy, RunStateArtifacts
from miles_plugins.proximal.gateway import GatewayConfig, ReplicaGateway
from miles_plugins.proximal.http_sync import REPLICA_HEADER
from miles_plugins.proximal.replica import ReplicaConfig, ReplicaLoRALoader, authorize_replica_load
from miles_plugins.proximal.sharded_snapshot import prepare_sharded
from miles_plugins.proximal.snapshot import SnapshotMetadata, prepare_snapshot, read_snapshot
from miles_plugins.proximal.state_writer import StateWriter


@contextmanager
def receiver(config, root, monkeypatch, *, allow_volume=True):
    monkeypatch.setenv("GATEWAY_HTTP_TEST_KEY", "fleet-secret")
    loaded = {}

    def engine(request):
        body = json.loads(request.content)
        name = body["lora_name"]
        assert Path(body["lora_path"], "manifest.json").is_file()
        loaded[name] = body["lora_path"]
        return httpx.Response(200, json={"success": True, "loaded_adapters": loaded})

    def reload_volume():
        assert allow_volume, "HTTP publication must not read weights from Volume"

    replica = ReplicaConfig(
        base_model=config.base_model, served_model_name=config.base_model.name, backend_url="http://127.0.0.1:1"
    )
    with httpx.Client(transport=httpx.MockTransport(engine)) as client:
        loader = ReplicaLoRALoader(
            authorize_replica_load(replica, yes_load=True),
            volume_mount=root / "volume",
            local_cache=root / "cache",
            reload_volume=reload_volume,
            client=client,
        )
        inference = httpx.AsyncClient()
        gateway = ReplicaGateway(
            GatewayConfig(
                replica=replica, api_key_env="GATEWAY_HTTP_TEST_KEY", engine_model_path="/base", max_loaded_adapters=4
            ),
            loader=loader,
            client=inference,
        )
        try:
            yield gateway, loaded
        finally:
            asyncio.run(inference.aclose())


def test_http_auth_integrity_routing_and_cold_replica_recovery(config, tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}')
    (adapter / "adapter_model.bin").write_bytes(b"weight-data" * 100)
    full = prepare_snapshot(
        adapter,
        metadata=SnapshotMetadata(run_id=config.run_id, checkpoint_iteration=0, base_model=config.base_model),
        output_root=config.artifact_directory / config.run_id / "publication",
    )
    plan = prepare_sharded(full.directory, full.reference, world_size=4)
    with receiver(config, tmp_path / "first", monkeypatch) as (gateway, loaded), TestClient(gateway.app) as client:
        assert client.post("/policies/uploads", json=plan.model_dump(mode="json")).status_code == 401
        client.headers["Authorization"] = "Bearer fleet-secret"
        for key, value in (("offset", 1), ("rank", 4)):
            invalid = plan.model_dump(mode="json")
            invalid["files"]["adapter_model.bin"][0][key] = value
            assert client.post("/policies/uploads", json=invalid).status_code == 422
        start = client.post("/policies/uploads", json=plan.model_dump(mode="json"))
        assert start.status_code == 200, start.text
        prefix = f"/policies/uploads/{full.reference.sha256}"
        client.headers[REPLICA_HEADER] = "other-replica"
        assert client.put(prefix + "/0", content=b"bad").status_code == 409
        client.headers[REPLICA_HEADER] = start.json()["replica_id"]
        assert client.post(prefix + "/complete").status_code == 409
        assert not loaded
        assert client.put(prefix + "/0", content=b"bad").status_code == 409
        index = 0
        for name, parts in plan.files.items():
            for part in parts:
                data = (full.directory / name).read_bytes()[part.offset : part.offset + part.size_bytes]
                for _ in range(2):
                    reply = client.put(f"{prefix}/{index}", content=data)
                    assert reply.status_code == 200, reply.text
                index += 1
        reply = client.post(prefix + "/complete")
        assert reply.status_code == 200, reply.text
        assert reply.json()["snapshot"] == full.reference.model_dump()
        assert len(loaded) == 1
    authorization = authorize_run(config, yes_rollouts=True, yes_publish=True)
    policy = Policy(run_id=config.run_id, version=1, snapshot=full.reference, base_model=config.base_model)
    policy_archive.enqueue(authorization, policy)
    assert current_version(config) is None
    volume = FakeVolume()
    monkeypatch.setattr(modal.Volume, "from_name", lambda *a, **kw: volume)
    volume.fail_after = 1
    with pytest.raises(ConnectionError):
        policy_archive.publish_pending(authorization)
    assert current_version(config) is None and policy_archive.pending_path(config).exists()
    volume.fail_after = None
    # The same writer used for native recovery publishes the serving archive and policy.
    writer = StateWriter(
        dsn="unused",
        run_id=config.run_id,
        artifacts=config.artifact_directory,
        snapshot_root=tmp_path / "state",
        commit=lambda: None,
        publish_policy=lambda: policy_archive.publish_pending(authorization),
    )
    from miles_plugins.proximal import state_artifacts

    monkeypatch.setattr(state_artifacts, "backlog", lambda *a: (0, 0, 0))
    monkeypatch.setattr(state_artifacts, "publish_pending", lambda **kw: 0)
    writer.close()
    assert current_version(config) == 1 and not policy_archive.pending_path(config).exists()
    for name, data in volume.files.items():
        path = tmp_path / "replacement" / "volume" / name.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with receiver(config, tmp_path / "replacement", monkeypatch) as (gateway, loaded), TestClient(
        gateway.app
    ) as client:
        reply = client.post(
            "/policies/prepare",
            headers={"Authorization": "Bearer fleet-secret"},
            json={"snapshot": full.reference.model_dump(), "base_model": config.base_model.model_dump()},
        )
        assert reply.status_code == 200, reply.text
        recovered = read_snapshot(Path(next(iter(loaded.values()))), full.reference)
        assert recovered.manifest == full.manifest


def rank_worker(rank, world, args, root, failure):
    torch.set_num_threads(1)
    root = Path(root)
    original = http_fleet.upload_request
    fail = failure is not None
    upstreams = ("replica-0:443", "replica-1:443", "replica-2:443")
    discoveries = 0

    def discover(config):
        nonlocal discoveries
        discoveries += 1
        if fail and failure == "churn" and discoveries == 2:
            return upstreams[:-1]
        return upstreams

    http_fleet.discover_replicas = discover

    def upload(client, method, path, *, content):
        if (
            fail
            and failure == "complete"
            and path.endswith("/complete")
            and client.headers[http_fleet.UPSTREAM_HEADER] == "replica-1:443"
        ):
            raise httpx.ConnectError("Injected receiver load failure")
        if method == "PUT":
            if fail and (
                (failure == "rank" and rank == 2)
                or (failure == "receiver" and client.headers[http_fleet.UPSTREAM_HEADER] == "replica-1:443")
            ):
                raise httpx.ConnectError("Injected upload failure")
            with (root / f"uploads-{rank}").open("a") as log:
                log.write(client.headers[http_fleet.UPSTREAM_HEADER] + " " + path + "\n")
        return original(client, method, path, content=content)

    http_fleet.upload_request = upload
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=world, timeout=timedelta(seconds=45)
    )
    distributed_utils.GLOO_GROUP = dist.group.WORLD
    try:
        updater = make_updater(args, CpuAdapterIterator)
        updater.connect_rollout_engines([])
        if failure is not None:
            with pytest.raises(RuntimeError, match="HTTP (shard upload|fleet preparation) failed"):
                updater.update_weights()
            assert updater.weight_version == 0
            if rank == 0:
                assert not policy_archive.pending_path(updater.protocol.config).exists()
                assert current_version(updater.protocol.config) is None
            fail = False
        updater.update_weights()
        assert updater.weight_version == 1
        # No background writer is running: HTTP finished without waiting for Volume persistence.
        if rank == 0:
            assert policy_archive.pending_path(updater.protocol.config).exists()
            assert current_version(updater.protocol.config) is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", [None, "rank", "receiver", "churn", "complete"])
def test_four_ranks_upload_every_shard_to_three_replicas(config, tmp_path, monkeypatch, failure):
    assert config.weight_sync_transport == "http"
    monkeypatch.setenv("FLEET_TEST_KEY", "Bearer fleet-secret")
    with socket.socket() as sock, ExitStack() as stack:
        sock.bind(("127.0.0.1", 0))
        config = config.model_copy(
            update={
                "inference_url": f"http://127.0.0.1:{sock.getsockname()[1]}",
                "artifact_storage": RunStateArtifacts(kind="run_state"),
                "weight_sync_pool": HTTPServingPool(app_name="test-fleet", min_replicas=3),
            }
        )
        args = transfer_args(config, tmp_path)
        replicas = {
            f"replica-{i}:443": stack.enter_context(
                receiver(config, tmp_path / str(i), monkeypatch, allow_volume=False)
            )
            for i in range(3)
        }

        async def edge(scope, receive, send):
            if scope["type"] == "lifespan":
                await next(iter(replicas.values()))[0].app(scope, receive, send)
                return
            # Emulate the Modal edge: no affinity/default route may hide a missing target.
            upstream = dict(scope["headers"])[http_fleet.UPSTREAM_HEADER.encode()].decode()
            await replicas[upstream][0].app(scope, receive, send)

        server = uvicorn.Server(uvicorn.Config(edge, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            mp.spawn(rank_worker, args=(4, args, str(tmp_path), failure), nprocs=4, join=True)
            assert all(len(loaded) == 1 for _, loaded in replicas.values())
            assert len({next(iter(loaded)) for _, loaded in replicas.values()}) == 1
            for rank in range(4):
                uploads = (tmp_path / f"uploads-{rank}").read_text().splitlines()
                by_replica = [
                    {line.split(" ", 1)[1] for line in uploads if line.startswith(host + " ")} for host in replicas
                ]
                assert by_replica[0] and by_replica[0] == by_replica[1] == by_replica[2]
            assert current_version(config) is None
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def test_discovery_validates_pool_url_and_complete_membership(config, monkeypatch):
    from modal.client import _Client
    from modal_proto import api_pb2

    config = config.model_copy(update={"weight_sync_pool": HTTPServingPool(app_name="serving-test", min_replicas=2)})
    reply = api_pb2.FlashContainerListResponse()
    reply.containers.add(host="replica-b")
    reply.containers.add(host="replica-a")
    stub = SimpleNamespace(
        FunctionGet=AsyncMock(return_value=SimpleNamespace(function_id="fu-serving")),
        FlashContainerList=AsyncMock(return_value=reply),
    )
    monkeypatch.setattr(_Client, "from_env", AsyncMock(return_value=SimpleNamespace(stub=stub)))
    get_url = AsyncMock(return_value=config.inference_url)

    def server(app, name, *, environment_name):
        assert (app, name, environment_name) == ("serving-test", "Replica", config.volume.environment_name)
        return SimpleNamespace(get_url=SimpleNamespace(aio=get_url))

    monkeypatch.setattr(modal.Server, "from_name", server)
    assert http_fleet.discover_replicas(config) == ("replica-a:443", "replica-b:443")
    assert stub.FlashContainerList.call_args.args[0].function_id == "fu-serving"
    get_url.return_value = "https://another-pool.example"
    with pytest.raises(ValueError, match="URL differs"):
        http_fleet.discover_replicas(config)
    get_url.return_value = config.inference_url
    reply.containers[1].host = "replica-b"
    with pytest.raises(ValueError, match="distinct replicas"):
        http_fleet.discover_replicas(config)
    del reply.containers[1:]
    with pytest.raises(ValueError, match="distinct replicas"):
        http_fleet.discover_replicas(config)
    del reply.containers[:]
    with pytest.raises(ValueError, match="distinct replicas"):
        http_fleet.discover_replicas(config)


def test_edge_ignoring_direct_routing_cannot_acknowledge_the_fleet(config, tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}')
    (adapter / "adapter_model.bin").write_bytes(b"weights")
    full = prepare_snapshot(
        adapter,
        metadata=SnapshotMetadata(run_id=config.run_id, checkpoint_iteration=0, base_model=config.base_model),
        output_root=tmp_path / "snapshots",
    )
    manifest = prepare_sharded(full.directory, full.reference, world_size=2)
    authorization = authorize_run(config, yes_rollouts=True, yes_publish=True)
    with receiver(config, tmp_path / "receiver", monkeypatch) as (gateway, loaded), TestClient(
        gateway.app
    ) as app_client:

        def route(request):
            reply = app_client.request(
                request.method, request.url.path, headers=request.headers, content=request.content
            )
            return httpx.Response(reply.status_code, content=reply.content)

        def client(*args):
            return httpx.Client(
                base_url="http://testserver",
                transport=httpx.MockTransport(route),
                headers={"Authorization": "Bearer fleet-secret"},
            )

        monkeypatch.setattr(http_fleet, "_client", client)
        with pytest.raises(ValueError, match="duplicate replica"):
            http_fleet.begin_uploads(authorization, manifest, ("replica-a:443", "replica-b:443"))
        assert not loaded
        # Failure cleaned up the acknowledged receiver so a fresh upload can begin.
        assert len(http_fleet.begin_uploads(authorization, manifest, ("replica-a:443",))) == 1
