"""Real HTTP, four Gloo uploader ranks and real snapshot/store recovery; engine/Volume are boundaries."""

import asyncio
import json
import socket
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

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
from miles_plugins.proximal import policy_archive, weight_update
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.contracts import Policy, RunStateArtifacts
from miles_plugins.proximal.gateway import GatewayConfig, ReplicaGateway
from miles_plugins.proximal.http_sync import REPLICA_HEADER
from miles_plugins.proximal.replica import ReplicaConfig, ReplicaLoRALoader, authorize_replica_load
from miles_plugins.proximal.sharded_snapshot import prepare_sharded
from miles_plugins.proximal.snapshot import SnapshotMetadata, prepare_snapshot, read_snapshot
from miles_plugins.proximal.state_writer import StateWriter


@contextmanager
def receiver(config, root, monkeypatch):
    monkeypatch.setenv("GATEWAY_HTTP_TEST_KEY", "fleet-secret")
    loaded = {}

    def engine(request):
        body = json.loads(request.content)
        name = body["lora_name"]
        assert Path(body["lora_path"], "manifest.json").is_file()
        loaded[name] = body["lora_path"]
        return httpx.Response(200, json={"success": True, "loaded_adapters": loaded})

    replica = ReplicaConfig(
        base_model=config.base_model, served_model_name=config.base_model.name, backend_url="http://127.0.0.1:1"
    )
    with httpx.Client(transport=httpx.MockTransport(engine)) as client:
        loader = ReplicaLoRALoader(
            authorize_replica_load(replica, yes_load=True),
            volume_mount=root / "volume",
            local_cache=root / "cache",
            reload_volume=lambda: None,
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


def rank_worker(rank, world, args, root, fail_rank):
    torch.set_num_threads(1)
    root = Path(root)
    original = weight_update.upload_request
    fail = rank == fail_rank

    def upload(client, method, path, *, content):
        if method == "PUT":
            if fail:
                raise httpx.ConnectError("Injected failed rank")
            with (root / f"uploads-{rank}").open("a") as log:
                log.write(path + "\n")
        return original(client, method, path, content=content)

    weight_update.upload_request = upload
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=world, timeout=timedelta(seconds=45)
    )
    distributed_utils.GLOO_GROUP = dist.group.WORLD
    try:
        updater = make_updater(args, CpuAdapterIterator)
        updater.connect_rollout_engines([])
        if fail_rank is not None:
            with pytest.raises(RuntimeError, match="HTTP shard upload failed"):
                updater.update_weights()
            assert updater.weight_version == 0
            fail = False
        updater.update_weights()
        assert updater.weight_version == 1
        # No background writer is running: HTTP finished without waiting for Volume persistence.
        if rank == 0:
            assert policy_archive.pending_path(updater.protocol.config).exists()
            assert current_version(updater.protocol.config) is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_rank", [None, 2])
def test_four_real_ranks_send_http_and_failed_rank_can_retry(config, tmp_path, monkeypatch, fail_rank):
    monkeypatch.setenv("FLEET_TEST_KEY", "Bearer fleet-secret")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        config = config.model_copy(
            update={
                "inference_url": f"http://127.0.0.1:{sock.getsockname()[1]}",
                "weight_sync_transport": "http",
                "artifact_storage": RunStateArtifacts(kind="run_state"),
            }
        )
        args = transfer_args(config, tmp_path)
        with receiver(config, tmp_path / "receiver", monkeypatch) as (gateway, loaded):
            server = uvicorn.Server(uvicorn.Config(gateway.app, log_level="error"))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 10
                while not server.started:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                mp.spawn(rank_worker, args=(4, args, str(tmp_path), fail_rank), nprocs=4, join=True)
                assert len(loaded) == 1
                assert all((tmp_path / f"uploads-{rank}").read_text() for rank in range(4))
                assert current_version(config) is None
            finally:
                server.should_exit = True
                thread.join(timeout=10)
