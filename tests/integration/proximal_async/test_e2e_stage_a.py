"""Stage A offline: every Stage A process for real on loopback, only training faked.

Stub platform, fake serving pool and the real capture service run as real HTTP
servers; the fake trainer drives Miles's real producer, buffer, store and
publication steps against them. Replacing the fake pool with the Modal serving pool
is the paid Stage A run described in examples/proximal/e2e/README.md.
"""

import asyncio
import os
import socket
from argparse import Namespace
from pathlib import Path

import httpx
import pytest
import uvicorn
from transformers import AutoTokenizer

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.capture_server import CaptureServer
from miles_plugins.proximal.contracts import Service
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.e2e import fake_trainer
from miles_plugins.proximal.e2e.adapters import DenseDecoderShape, write_adapter
from miles_plugins.proximal.e2e.fake_pool import FakePool
from miles_plugins.proximal.e2e.stub_platform import StubPlatform
from miles_plugins.proximal.store import open_store


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def serve(app, port):
    # Bind ourselves with SO_REUSEADDR: the resume run rebinds ports the first run just freed.
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.01)
    return server, task


@pytest.fixture
def stage_a(config, tmp_path):
    ports = [free_port() for _ in range(3)]
    tokenizer_path = Path(os.environ["PROXIMAL_TEST_TOKENIZER"])
    run = config.model_copy(
        update={
            "platform": Service(url=f"http://127.0.0.1:{ports[0]}", api_key_env="PX_TEST_KEY"),
            "capture": Service(url=f"http://127.0.0.1:{ports[1]}", api_key_env="CAPTURE_TEST_KEY"),
            "inference_url": f"http://127.0.0.1:{ports[2]}",
            "tokenizer_path": tokenizer_path,
            "max_in_flight_samples": 4,
        }
    )
    path = tmp_path / "stage-a.json"
    path.write_text(run.model_dump_json())
    shape = DenseDecoderShape.model_validate_json((tokenizer_path / "config.json").read_bytes())
    for seed in range(2):
        write_adapter(run, shape, tmp_path / "adapters" / f"adapter-{seed}", seed=seed)
    return run, path, ports


async def run_stage_a(run, path, ports, tmp_path, **overrides):
    tokenizer = AutoTokenizer.from_pretrained(str(run.tokenizer_path), local_files_only=True)
    registry = SessionRegistry(
        tokenizer,
        tito_tokenizer=get_tito_tokenizer(tokenizer, "qwen3", chat_template_kwargs={"enable_thinking": True}),
    )
    authorization = authorize_run(run, yes_rollouts=True, yes_publish=True)
    store = await open_store(run)
    servers = []
    async with httpx.AsyncClient(timeout=30) as backend, httpx.AsyncClient(timeout=30) as agent_http:
        pool = FakePool(run, tokenizer=tokenizer, api_key="fleet-secret")
        capture = CaptureServer(authorization, registry=registry, client=backend, store=store)
        stub = StubPlatform(run, api_key="platform-secret", turns=3, reward="mixed", client=agent_http)
        try:
            for app, port in ((stub.app, ports[0]), (capture.app, ports[1]), (pool.app, ports[2])):
                servers.append(await serve(app, port))
            args = Namespace(
                config=path,
                adapters=tmp_path / "adapters",
                checkpoints=tmp_path / "checkpoints",
                steps=4,
                batch_size=2,
                publish_every=2,
                publish="local",
                resume_step=None,
                yes_rollouts=True,
                yes_publish=True,
            )
            for key, value in overrides.items():
                setattr(args, key, value)
            report = await asyncio.wait_for(fake_trainer.run(args), 120)
            return report, stub, pool
        finally:
            for server, task in servers:
                server.should_exit = True
                await task
            await store.close()


async def test_stage_a_offline_trains_on_captured_platform_rollouts(config, tmp_path, stage_a):
    run, path, ports = stage_a
    report, stub, pool = await run_stage_a(run, path, ports, tmp_path)

    assert [entry["step"] for entry in report] == [0, 1, 2, 3]
    trained = [group for entry in report for group in entry["groups"]]
    assert len(trained) == 8 and len({group["group_id"] for group in trained}) == 8
    for entry in report:
        assert len(entry["groups"]) == 2
        for group in entry["groups"]:
            lag = entry["trainer_version"] - group["policy_version"]
            assert 0 <= lag <= run.research.max_policy_lag
            assert set(group["rewards"]) <= {0.0, 1.0}
            assert all(tokens > 0 for tokens in group["loss_tokens"])
            # The scripted agent: a bash tool call, a tool result, then DONE.
            assert group["model_calls"] == [2, 2]
    # Version 3 is published after the last step; the store checks it below.
    assert {entry["trainer_version"] for entry in report} == {1, 2}
    assert {group["policy_version"] for group in trained} >= {1, 2}
    assert {0.0, 1.0} <= {reward for group in trained for reward in group["rewards"]}
    assert len(stub.runs) >= 16 and pool.requests >= 32
    assert all(state.error is None for state in stub.runs.values())

    # The durable side agrees: live policy history and the checkpointed ledger.
    store = await open_store(run)
    try:
        assert (await store.current_policy()).version == 3
    finally:
        await store.close()
    source = PlatformTaskSource(fake_trainer.miles_args(path, run, tmp_path / "checkpoints", batch_size=2))
    source.load(3)
    ledger = {entry.group_id for entry in source.consumed.snapshot()}
    fresh = {group["group_id"] for group in trained if group["policy_version"] >= 3 - run.research.max_policy_lag}
    assert fresh <= ledger

    # Resume from step 1: steps 2-3 retrain, never on groups consumed by steps 0-1.
    resumed, _, _ = await run_stage_a(run, path, ports, tmp_path, resume_step=1)
    assert [entry["step"] for entry in resumed] == [2, 3]
    before = {group["group_id"] for entry in report[:2] for group in entry["groups"]}
    after = {group["group_id"] for entry in resumed for group in entry["groups"]}
    assert not before & after
