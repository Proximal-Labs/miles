import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from miles_plugins.inkling_eval.config import EvalConfig, evaluation_due, summarize
from miles_plugins.inkling_eval.platform import Platform
from miles_plugins.inkling_eval.serving import server_command


def test_named_sets_and_disabled_cadence():
    config = EvalConfig(platform_url="https://api.example.com", sets={"coding": [1, 2], "heldout": [3, 4]})
    assert config.sets == {"coding": [1, 2], "heldout": [3, 4]}
    assert [step for step in range(21) if evaluation_due(step, 5, 2)] == [10, 20]
    assert not any(evaluation_due(step, 5, 0) for step in range(21))
    assert [step for step in range(20) if evaluation_due(step, 190, 1, batch_size=32)] == [6, 12, 18]
    assert [step for step in range(25) if evaluation_due(step, 190, 2, batch_size=32)] == [12, 24]
    with pytest.raises(ValueError, match="distinct"):
        EvalConfig(platform_url="https://api.example.com", sets={"coding": [1, 1]})


def test_zero_rewards_are_scored_and_failures_are_separate():
    assert summarize([{"reward": 1.0}, {"reward": 0.0}, {"reward": None}]) == {
        "rollouts": 3,
        "scored": 2,
        "failures": 1,
        "mean_reward": 0.5,
        "pass_rate": 0.5,
    }
    assert summarize([{"reward": None}]) == {"rollouts": 1, "scored": 0, "failures": 1}


@pytest.fixture
def platform_server(monkeypatch):
    state = {
        "registry": {
            "version": 1,
            "models": {
                "modal/inkling-small": {
                    "defaultEndpoint": "production",
                    "endpoints": {"production": {"mode": "dedicated"}},
                }
            },
        },
        "runs": {},
        "images": [{"id": 11, "digest": "sha256:one", "commitHash": "abc", "pushedAt": "100"}],
        "blocked": None,
    }

    def handle(request):
        body = json.loads(request.content)
        method = request.url.path.split("/")[-1]
        if method == "ListImages":
            return httpx.Response(200, json={"images": state["images"]})
        if method == "GetLiveConfig":
            return httpx.Response(200, json={"config": {"valueJson": json.dumps(state["registry"]), "revision": 1}})
        if method == "SetLiveConfig":
            state["registry"] = json.loads(body["valueJson"])
            return httpx.Response(200, json={})
        if method == "CreateEnvironmentRun":
            state["runs"].setdefault(body["runId"], body)
            return httpx.Response(200, json={"runId": body["runId"], "instancesStarted": 1})
        if method == "GetEnvironmentRunContainers":
            if state["blocked"]:
                assert state["blocked"].wait(10), "test did not unblock evaluation"
            return httpx.Response(
                200,
                json={
                    "containers": [
                        {
                            "status": "ROLLOUT_CONTAINER_STATUS_COMPLETED",
                            "rewardScored": True,
                        }
                    ]
                },
            )
        raise AssertionError(method)

    real_client = httpx.Client
    monkeypatch.setattr("miles_plugins.inkling_eval.platform.time.sleep", lambda seconds: None)
    monkeypatch.setenv("PROXIMAL_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    return state


def test_platform_pins_images_routes_endpoint_and_keeps_zero_reward(platform_server):
    client = Platform(EvalConfig(platform_url="https://api.example.com", sets={"coding": [1]}))
    pins = client.resolve_suite()
    assert pins == {"1": {"environmentId": 1, "imageId": 11, "sourceCommitSha": "abc", "imageDigest": "sha256:one"}}
    endpoint = {"mode": "dedicated", "baseURL": "https://eval.modal.direct/v1", "model": "snapshot"}
    client.endpoint("epoch-1", endpoint)
    assert platform_server["registry"]["models"]["modal/inkling-small"]["defaultEndpoint"] == "production"
    result = client.rollout(pins["1"], identity="run:1:1:0", endpoint="epoch-1")
    assert result["reward"] == 0.0
    payload = platform_server["runs"][result["run_id"]]
    assert payload["imageId"] == 11 and payload["sourceCommitSha"] == "abc"
    assert payload["config"]["agents"][0]["endpointName"] == "epoch-1"
    client.rollout(pins["1"], identity="run:1:1:0", endpoint="epoch-1")
    assert len(platform_server["runs"]) == 1
    client.endpoint("epoch-1", None)
    assert set(platform_server["registry"]["models"]["modal/inkling-small"]["endpoints"]) == {"production"}
    client.close()


def test_serving_always_selects_fixed_adapter_and_bf16():
    argv = server_command({"base": "/base", "tp": 8, "context_length": 262144, "rank": 32, "adapter": "/snapshots/step1", "concurrency": 4})
    assert argv[argv.index("--lora-paths") + 1] == "snapshot=/snapshots/step1"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert "--quantization" not in argv


def test_adapter_snapshot_roundtrip_and_integrity(tmp_path):
    import safetensors.torch
    import torch

    from miles_plugins.inkling_eval.export import _write_adapter
    from miles_plugins.inkling_eval.serving import _verify_snapshot

    args = SimpleNamespace(lora_rank=2, lora_alpha=4, hf_checkpoint="/base")
    tensors = {
        "model.layers.0.attn.wq_du.lora_A.weight": torch.arange(8).reshape(2, 4).float(),
        "model.layers.0.attn.wq_du.lora_B.weight": torch.arange(12).reshape(6, 2).float(),
        "model.layers.0.moe.w1.lora_A.weight": torch.ones(1, 2, 4),
        "model.layers.0.moe.w1.lora_B.weight": torch.ones(8, 6, 2),
    }
    _write_adapter(args, list(tensors.items()), tmp_path)
    _verify_snapshot(tmp_path)
    saved = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")
    assert saved.keys() == tensors.keys()
    for name in tensors:
        torch.testing.assert_close(saved[name], tensors[name])
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["target_modules"]) == (2, 4, "all-linear")
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        _verify_snapshot(tmp_path)
    with pytest.raises(ValueError, match="LoRA pair"):
        _write_adapter(args, list(tensors.items())[:-1], tmp_path)


@pytest.mark.parametrize("rollouts", [None, 3])
def test_baseline_async_named_sets_and_resume(tmp_path, monkeypatch, platform_server, rollouts):
    from miles.utils.tracking_utils import tracking
    from miles_plugins.inkling_eval import serving
    from miles_plugins.inkling_eval.runner import EvaluationRunner

    metrics, definitions, exports = [], [], []
    monkeypatch.setattr(tracking, "log", lambda args, values, step_key: metrics.append((values, step_key)))
    monkeypatch.setattr(tracking, "define_step_key_metric_group", lambda *args: definitions.append(args))
    monkeypatch.setattr(serving, "deploy", lambda *args, **kwargs: {"app_id": "test", "url": "https://eval.modal.direct"})
    monkeypatch.setattr(serving, "wait_ready", lambda *args: None)
    monkeypatch.setattr(serving, "stop", lambda *args: None)
    monkeypatch.setattr(serving, "commit_volume", lambda *args: None)
    config = tmp_path / "eval.json"
    config.write_text(json.dumps({"platform_url": "https://api.example.com", "sets": {"coding": [1], "heldout": [2]}}))
    args = SimpleNamespace(
        inkling_eval_config=str(config),
        save=str(tmp_path / "run"),
        hf_checkpoint="/base",
        lora_rank=32,
        lora_alpha=32,
        inkling_eval_image="image@sha256:fixed",
        inkling_eval_environment="main",
        inkling_eval_every_n_epochs=2,
        start_rollout_id=0,
        rollout_batch_size=1,
        inkling_eval_rollouts_per_env=rollouts,
    )

    class Actor:
        async def export_hf(self, step, path, adapter_only):
            exports.append(step)

    async def exercise():
        runner = EvaluationRunner(args, Actor(), 5)
        await runner.start()
        assert exports == [-1]
        assert len(platform_server["runs"]) == 2 * (rollouts or 1)
        assert definitions == [("eval/coding", "eval/coding/epoch"), ("eval/heldout", "eval/heldout/epoch")]
        for step in range(1, 10):
            await runner.after_step(step)
        assert exports == [-1]
        blocked = threading.Event()
        platform_server["blocked"] = blocked
        await asyncio.wait_for(runner.after_step(10), timeout=2)
        assert exports == [-1, 9]
        # A normal subsequent training step is independent of the blocked rollout.
        await asyncio.wait_for(runner.after_step(11), timeout=2)
        blocked.set()
        await runner.finish()
        assert len(platform_server["runs"]) == 4 * (rollouts or 1)
        assert {key for _, key in metrics} == {"eval/coding/epoch", "eval/heldout/epoch"}
        assert {values[key] for values, key in metrics} == {0, 2}
        assert all(values[key.replace("/epoch", "/mean_reward")] == 0 for values, key in metrics)
        # Changing the platform's latest image cannot change a resumed suite.
        platform_server["images"] = [{"id": 99, "digest": "sha256:two", "commitHash": "def", "pushedAt": "200"}]
        args.start_rollout_id = 10
        resumed = EvaluationRunner(args, Actor(), 5)
        await resumed.start()
        await resumed.finish()
        assert resumed.suite["environments"]["1"]["imageId"] == 11
        assert len(platform_server["runs"]) == 4 * (rollouts or 1)
        assert exports == [-1, 9]

    asyncio.run(exercise())
