import asyncio
import json
import threading

import httpx
import pytest

from miles_plugins.proximal.gateway import GatewayConfig, ReplicaGateway
from miles_plugins.proximal.replica import ReplicaConfig, ReplicaLoRALoader, authorize_replica_load
from miles_plugins.proximal.snapshot import SnapshotMetadata, prepare_snapshot


async def test_replica_gateway_pins_active_adapter_and_evicts_idle_only(config, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}')
    (adapter / "adapter_model.bin").write_bytes(b"one")
    metadata = SnapshotMetadata(run_id=config.run_id, checkpoint_iteration=0, base_model=config.base_model)
    mount = tmp_path / "volume"
    one = prepare_snapshot(adapter, metadata=metadata, output_root=mount)
    (adapter / "adapter_model.bin").write_bytes(b"two")
    two = prepare_snapshot(adapter, metadata=metadata, output_root=mount)
    loaded = {}
    events = []
    blocked_method = [None]
    control_started = threading.Event()
    control_release = threading.Event()
    active = asyncio.Event()
    release = asyncio.Event()

    def control(request):
        body = json.loads(request.content)
        name = body["lora_name"]
        if request.url.path == blocked_method[0]:
            control_started.set()
            assert control_release.wait(timeout=3)
        if request.url.path == "/load_lora_adapter":
            loaded[name] = body["lora_path"]
            events.append(("load", name))
        else:
            assert request.url.path == "/unload_lora_adapter"
            loaded.pop(name)
            events.append(("unload", name))
        return httpx.Response(200, json={"success": True, "loaded_adapters": loaded})

    async def inference(request):
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/base"})
        body = json.loads(request.content)
        name = body["model"].split(":")[1]
        assert name in loaded
        active.set()
        await release.wait()
        assert name in loaded  # No eviction while its generation is running.
        return httpx.Response(200, json={"model": body["model"]})

    replica = ReplicaConfig(
        base_model=config.base_model, served_model_name=config.base_model.name, backend_url="http://127.0.0.1:9000"
    )
    with httpx.Client(transport=httpx.MockTransport(control)) as control_client:
        loader = ReplicaLoRALoader(
            authorize_replica_load(replica, yes_load=True),
            volume_mount=mount,
            local_cache=tmp_path / "cache",
            reload_volume=lambda: None,
            client=control_client,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(inference)) as inference_client:
            gateway = ReplicaGateway(
                GatewayConfig(
                    replica=replica, api_key_env="FLEET_TEST_KEY", engine_model_path="/base", max_loaded_adapters=1
                ),
                loader=loader,
                client=inference_client,
            )
            await gateway.validate_engine()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway.app), base_url="http://replica"
            ) as client:
                headers = {"Authorization": "Bearer fleet-secret"}
                generation = asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        headers=headers | {"X-Proximal-Policy-Sha256": one.reference.sha256},
                        json={"model": f"{config.base_model.name}:miles-{one.reference.sha256}", "stream": False},
                    )
                )
                await asyncio.wait_for(active.wait(), 2)
                prepare = asyncio.create_task(
                    client.post(
                        "/policies/prepare",
                        headers=headers,
                        json={"snapshot": two.reference.model_dump(), "base_model": config.base_model.model_dump()},
                    )
                )
                await asyncio.sleep(0.01)
                assert not prepare.done()
                assert len(events) == 1
                release.set()
                reply = await generation
                assert reply.headers["x-proximal-policy-sha256"] == one.reference.sha256
                assert (await prepare).status_code == 200
                assert [event[0] for event in events] == ["load", "unload", "load"]
                # An old rollout can reload its immutable version after idle eviction.
                assert (
                    await client.post(
                        "/policies/prepare",
                        headers=headers,
                        json={"snapshot": one.reference.model_dump(), "base_model": config.base_model.model_dump()},
                    )
                ).status_code == 200
                assert (await client.post("/policies/prepare", json={})).status_code in (401, 422)

                # Disconnecting the requesting client must still reconcile an
                # engine mutation already running in the loader's sync thread.
                for operation in ("unload_lora_adapter", "load_lora_adapter"):
                    control_started.clear()
                    control_release.clear()
                    blocked_method[0] = f"/{operation}"
                    interrupted = asyncio.create_task(
                        client.post(
                            "/policies/prepare",
                            headers=headers,
                            json={
                                "snapshot": two.reference.model_dump(),
                                "base_model": config.base_model.model_dump(),
                            },
                        )
                    )
                    assert await asyncio.to_thread(control_started.wait, 3)
                    interrupted.cancel()
                    await asyncio.sleep(0)
                    assert not interrupted.done()
                    control_release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await interrupted
                    blocked_method[0] = None
                    assert (await client.get("/health")).status_code == 200
                    assert (
                        await client.post(
                            "/policies/prepare",
                            headers=headers,
                            json={
                                "snapshot": one.reference.model_dump(),
                                "base_model": config.base_model.model_dump(),
                            },
                        )
                    ).status_code == 200
