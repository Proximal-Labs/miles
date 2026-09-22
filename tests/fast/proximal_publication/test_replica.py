import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from miles_plugins.proximal.replica import ReplicaConfig, ReplicaLoRALoader, authorize_replica_load
from miles_plugins.proximal.snapshot import BaseModelIdentity, prepare_snapshot, read_snapshot


@contextmanager
def _engine():
    state = {"loads": [], "registered": {}, "mode": "success"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/load_lora_adapter"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name, path = request["lora_name"], request["lora_path"]
            assert Path(path, "adapter_model.bin").is_file()
            state["loads"].append(request)
            if name in state["registered"]:
                status, body = 400, {"success": False, "loaded_adapters": state["registered"]}
            elif state["mode"] == "missing":
                status, body = 200, {"success": True}
            elif state["mode"] == "mismatch":
                status, body = 200, {"success": True, "loaded_adapters": {name: "/wrong/adapter"}}
            elif state["mode"] == "failure":
                status, body = 200, {"success": False, "loaded_adapters": {}}
            else:
                state["registered"][name] = path
                status, body = 200, {"success": True, "loaded_adapters": state["registered"], "error_message": ""}
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _loader(config, mount, cache, reload_volume, client):
    return ReplicaLoRALoader(
        authorize_replica_load(config, yes_load=True),
        volume_mount=mount,
        local_cache=cache,
        reload_volume=reload_volume,
        client=client,
    )


def test_two_replica_loaders_register_two_versions_with_independent_http_engines(tmp_path, adapter, metadata):
    mount = tmp_path / "shared-volume"
    first = prepare_snapshot(adapter, metadata=metadata, output_root=mount)
    (adapter / "adapter_model.bin").write_bytes(b"version B")
    second = prepare_snapshot(adapter, metadata=metadata, output_root=mount)
    reloads = []
    with _engine() as (url1, state1), _engine() as (url2, state2), httpx.Client(trust_env=False, timeout=5) as client:
        replicas = [
            _loader(
                ReplicaConfig(base_model=metadata.base_model, served_model_name="served", backend_url=url),
                mount,
                tmp_path / f"replica-{index}",
                lambda: reloads.append(True),
                client,
            )
            for index, url in enumerate((url1, url2))
        ]
        for replica in replicas:
            a = replica.ensure_loaded(first.reference)
            b = replica.ensure_loaded(second.reference)
            assert a.adapter_name != b.adapter_name
            assert a.request_model == f"served:{a.adapter_name}"
            assert replica.ensure_loaded(first.reference) == a
        assert len(state1["loads"]) == len(state2["loads"]) == 2
        assert state1["loads"][0]["lora_path"] != state2["loads"][0]["lora_path"]
        assert len(reloads) == 4
        assert (first.directory / "adapter_model.bin").read_bytes() == b"adapter version A"
        assert read_snapshot(first.directory, first.reference) == first


def test_concurrent_calls_register_once_and_replacement_process_reloads(tmp_path, adapter, metadata):
    mount = tmp_path / "volume"
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=mount)
    reloads = []
    with _engine() as (url, state), httpx.Client(trust_env=False, timeout=5) as client:
        config = ReplicaConfig(base_model=metadata.base_model, served_model_name="served", backend_url=url)
        loader = _loader(config, mount, tmp_path / "cache", lambda: reloads.append(True), client)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(loader.ensure_loaded, [snapshot.reference] * 4))
        assert all(result == results[0] for result in results)
        assert len(state["loads"]) == 1
        # The serving owner constructs a new loader for the new engine lifetime.
        state["registered"].clear()
        replacement = _loader(config, mount, tmp_path / "cache", lambda: reloads.append(True), client)
        assert replacement.ensure_loaded(snapshot.reference) == results[0]
        assert len(state["loads"]) == 2
        assert len(reloads) == 1  # The verified immutable local cache can survive engine restart.


@pytest.mark.parametrize("mode", ["missing", "mismatch", "failure"])
def test_bad_registration_reply_does_not_enter_ready_cache(tmp_path, adapter, metadata, mode):
    mount = tmp_path / "volume"
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=mount)
    with _engine() as (url, state), httpx.Client(trust_env=False, timeout=5) as client:
        config = ReplicaConfig(base_model=metadata.base_model, served_model_name="served", backend_url=url)
        loader = _loader(config, mount, tmp_path / "cache", lambda: None, client)
        state["mode"] = mode
        with pytest.raises(ValueError):
            loader.ensure_loaded(snapshot.reference)
        state["mode"] = "success"
        loader.ensure_loaded(snapshot.reference)
        assert len(state["loads"]) == 2


@pytest.mark.parametrize("failure", ["base", "integrity", "reload"])
def test_invalid_snapshot_or_refresh_failure_prevents_engine_call(tmp_path, adapter, metadata, failure):
    mount = tmp_path / "volume"
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=mount)
    base = metadata.base_model
    if failure == "base":
        base = BaseModelIdentity(name="other/base", revision="b" * 40)
    if failure == "integrity":
        (snapshot.directory / "adapter_model.bin").write_bytes(b"corrupt")

    def reload():
        if failure == "reload":
            raise RuntimeError("volume busy")

    with _engine() as (url, state), httpx.Client(trust_env=False, timeout=5) as client:
        config = ReplicaConfig(base_model=base, served_model_name="served", backend_url=url)
        loader = _loader(config, mount, tmp_path / "cache", reload, client)
        with pytest.raises((ValueError, RuntimeError)):
            loader.ensure_loaded(snapshot.reference)
        assert state["loads"] == []


@pytest.mark.parametrize("url", ["https://fleet.modal.direct", "http://127.0.0.1/v1", "http://user:secret@localhost"])
def test_fleet_endpoints_and_embedded_credentials_are_rejected(metadata, url):
    with pytest.raises(ValidationError, match="replica-local"):
        ReplicaConfig(base_model=metadata.base_model, served_model_name="served", backend_url=url)


def test_explicit_authorization_and_separate_cache_are_required(tmp_path, metadata):
    config = ReplicaConfig(
        base_model=metadata.base_model, served_model_name="served", backend_url="http://127.0.0.1:1"
    )
    with pytest.raises(PermissionError, match="yes-load"):
        authorize_replica_load(config, yes_load=False)
    with httpx.Client(trust_env=False) as client:
        with pytest.raises(ValueError, match="separate directory"):
            _loader(config, tmp_path, tmp_path / "cache", lambda: pytest.fail("Must not reload"), client)
