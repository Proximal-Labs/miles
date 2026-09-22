import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.contracts import Attempt, Policy, RunConfig, digest
from miles_plugins.proximal.snapshot import SnapshotReference


@pytest.fixture(scope="session")
def postgres_server():
    """A throwaway local Postgres on a Unix socket; no network needed.

    Set PROXIMAL_TEST_POSTGRES_DSN to use an existing server instead. A missing
    server is a failure, not a skip: the store is part of the tested contract.
    """
    if dsn := os.environ.get("PROXIMAL_TEST_POSTGRES_DSN"):
        yield dsn
        return
    binaries = sorted(Path("/usr/lib/postgresql").glob("*/bin"))
    if not binaries:
        raise RuntimeError("Postgres server binaries are required; use the proximal_async test image")
    bindir = binaries[-1]
    # Not under pytest's tmp root: the unprivileged server user must traverse it.
    root = Path(tempfile.mkdtemp(prefix="proximal-pg-"))
    data, socket = root / "data", root / "socket"
    socket.mkdir()
    as_postgres: list[str] = []
    if os.geteuid() == 0:  # initdb refuses to run as root.
        shutil.chown(root, "postgres")
        shutil.chown(socket, "postgres")
        as_postgres = ["runuser", "-u", "postgres", "--"]
    subprocess.run(
        [*as_postgres, str(bindir / "initdb"), "-D", str(data), "-U", "postgres", "--auth=trust"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            *as_postgres,
            str(bindir / "pg_ctl"),
            "-D",
            str(data),
            "-l",
            str(root / "server.log"),
            "-o",
            f"-k {socket} -c listen_addresses=''",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
    )
    try:
        yield f"host={socket} user=postgres dbname=postgres"
    finally:
        subprocess.run([*as_postgres, str(bindir / "pg_ctl"), "-D", str(data), "-m", "fast", "stop"], check=False)
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def store_dsn(postgres_server, monkeypatch):
    """A fresh database per test, exposed through the run's DSN variable."""
    import psycopg

    name = f"t_{uuid.uuid4().hex}"
    with psycopg.connect(postgres_server, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    dsn = postgres_server.replace("dbname=postgres", f"dbname={name}")
    monkeypatch.setenv("STORE_TEST_DSN", dsn)
    return dsn


@pytest.fixture
def config(tmp_path, monkeypatch, store_dsn):
    monkeypatch.setenv("PX_TEST_KEY", "platform-secret")
    monkeypatch.setenv("CAPTURE_TEST_KEY", "capture-secret")
    monkeypatch.setenv("FLEET_TEST_KEY", "fleet-secret")
    data = {
        "run_id": "test-run",
        "base_model": {"name": "test-qwen3", "revision": "a" * 40},
        "dataset": {"project_id": 12, "tasks": [{"environment_id": 7, "image_id": 8, "source_commit_sha": "b" * 40}]},
        "harness": {
            "agent_type": "native",
            "revision": "c" * 40,
            "max_turns": 4,
            "timeout_seconds": 60,
            "p2p_enforce": True,
        },
        "research": {
            "behavior_correction": "rollout_logprobs",
            "lora": {
                "rank": 8,
                "alpha": 16,
                "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            },
            "sampling": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 64,
                "max_sequence_tokens": 4096,
                "logprob_semantics": "untransformed",
                "budget_policy": "cap_to_remaining_context",
            },
            "group_size": 2,
            "max_policy_lag": 1,
            "unused_groups": "retry",
            "max_consecutive_failed_groups": 2,
        },
        "platform": {"url": "http://127.0.0.1:9010", "api_key_env": "PX_TEST_KEY"},
        "capture": {"url": "http://127.0.0.1:9011", "api_key_env": "CAPTURE_TEST_KEY"},
        "inference_url": "http://127.0.0.1:9012",
        "inference_header_env": {"Authorization": "FLEET_TEST_KEY"},
        "volume": {"volume_name": "adapters", "environment_name": "dev"},
        "artifact_directory": str(tmp_path / "artifacts"),
        "artifact_storage": {"kind": "shared_disk"},
        "store_dsn_env": "STORE_TEST_DSN",
        "tokenizer_path": str(tmp_path),
        "tito_model": "qwen3",
        "enable_thinking": True,
        "model_protocol": {"reasoning_parser": "qwen3", "tool_call_parser": "qwen25"},
        "max_in_flight_samples": 4,
        "completed_group_capacity": 2,
        "request_timeout_seconds": 10,
        "poll_interval_seconds": 0.01,
    }
    return RunConfig.model_validate_json(json.dumps(data))


@pytest.fixture
def authorization(config):
    return authorize_run(config, yes_rollouts=True, yes_publish=True)


@pytest.fixture
def policy(config):
    return Policy(
        run_id=config.run_id, version=1, snapshot=SnapshotReference(sha256="d" * 64), base_model=config.base_model
    )


@pytest.fixture
def attempt(config, policy):
    return Attempt(
        attempt_id="attempt-1",
        run_id=config.run_id,
        group_id="group-1",
        sample_index=0,
        dataset_sha256=digest(config.dataset),
        task=config.dataset.tasks[0],
        harness=config.harness,
        policy=policy,
        sampling=config.research.sampling,
    )


@pytest.fixture
async def store(config):
    from miles_plugins.proximal.store import open_store

    opened = await open_store(config)
    yield opened
    await opened.close()
