"""The sweep cannot silently change source data or lose an update's eval export."""

import json
from pathlib import Path

import pytest
import torch

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.contracts import RunConfig
from miles_plugins.proximal.e2e.batch_sweep_artifacts import finalize_step, publish_node_files
from miles_plugins.proximal.e2e.batch_sweep_inputs import SweepPhase, SweepPlan, phase_command
from miles_plugins.proximal.snapshot import SnapshotReference, read_snapshot
from miles_plugins.proximal.state_checkpoints import NativeCompletion

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def source():
    return RunConfig.model_validate_json((REPO / "examples/proximal/e2e/run.stage-a.json").read_bytes())


@pytest.fixture
def plan(source):
    return SweepPlan(
        experiment_id="two-configs",
        batch_path="run/assembled/batch",
        batch_sha256="a" * 64,
        samples=1024,
        nodes=1,
        phases=(
            SweepPhase(name="mlp", updates=2, target_modules=("linear_fc1", "linear_fc2")),
            SweepPhase(name="full", updates=2, target_modules=("linear_qkv", "linear_fc1", "linear_fc2")),
        ),
        recipe=(
            "--optimizer",
            "adam",
            "--lr",
            "4e-5",
            "--seed",
            "42",
            "--num-rollout",
            "20",
            "--save",
            "/old",
            "--use-wandb",
            "--wandb-project",
            "old",
        ),
    )


def test_two_configurations_start_fresh_with_identical_data_and_preserve_source(source, plan):
    before = source.model_dump_json()
    commands = [
        phase_command(plan, phase, source, bundle=Path("/batch"), save=Path("/output") / phase.name)
        for phase in plan.phases
    ]
    for phase, command in zip(plan.phases, commands, strict=True):

        def value(flag, command=command):
            assert command.count(flag) == 1
            return command[command.index(flag) + 1]

        assert value("--num-rollout") == "2"
        assert value("--start-rollout-id") == "0"
        assert value("--verification-batch") == "/batch"
        assert value("--load") == str(source.tokenizer_path)
        assert value("--save-interval") == "1"
        assert value("--target-modules") == ",".join(phase.target_modules)
        assert "--lora-adapter-path" not in command and "--use-wandb" not in command
    assert source.model_dump_json() == before


@pytest.mark.parametrize(
    "flag", ["--lora-adapter-path", "--no-save-optim", "--enable-mtp-training", "--load-debug-rollout-data"]
)
def test_incompatible_recipe_is_refused_before_resources(source, plan, flag):
    invalid = plan.model_copy(update={"recipe": (*plan.recipe, flag)})
    with pytest.raises(ValueError, match="Unsupported"):
        phase_command(invalid, plan.phases[0], source, bundle=Path("/batch"), save=Path("/output"))


def write_shards(local, source, ranks):
    local.mkdir(parents=True)
    for rank in ranks:
        torch.save({"weight": torch.ones(2)}, local / f"adapter_megatron_rank{rank}.pt")
        torch.save({"optimizer": {"step": 1}}, local / f"training_state_rank{rank}.pt")
    if 0 in ranks:
        r = source.research.lora.rank
        weights = {
            f"model.layers.0.mlp.{leaf}.lora_{side}.weight": torch.ones((r, 4) if side == "A" else (4, r))
            for leaf in ("gate_proj", "up_proj", "down_proj")
            for side in ("A", "B")
        }
        torch.save(weights, local / "adapter_model.bin")
        (local / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": r,
                    "lora_alpha": source.research.lora.alpha,
                    "target_modules": ["gate_proj", "up_proj", "down_proj"],
                }
            )
        )


def native():
    return NativeCompletion(
        schema_version=1,
        iteration=0,
        world_size=8,
        layout={"tensor_model_parallel_size": 4},
        optimizer=True,
        scheduler=True,
        rng=True,
    )


def test_every_native_shard_and_eval_export_commit_before_completion(source, plan, tmp_path):
    authorization = authorize_run(source, yes_rollouts=True, yes_publish=True)
    local, root = tmp_path / "local", tmp_path / "volume"
    destination = root / "mlp/checkpoints/iter_0000000/adapter"
    write_shards(local, source, range(8))
    events = []

    def commit():
        events.append(
            (destination.joinpath("native_checkpoint.json").exists(), root.joinpath("mlp/step-1.json").exists())
        )

    files = publish_node_files(authorization, local=local, destination=destination, commit=commit)
    assert events == [(False, False)]
    result = finalize_step(
        authorization,
        plan=plan,
        phase=plan.phases[0],
        step=0,
        native=native(),
        receipts=[files],
        root=root,
        commit=commit,
    )
    assert events == [(False, False), (True, False), (True, True)]
    snapshot = read_snapshot(root / result["eval_path"], SnapshotReference.model_validate(result["snapshot"]))
    assert snapshot.manifest.metadata.checkpoint_iteration == 0
    assert len(snapshot.manifest.files) == 2
    assert len(list(destination.glob("training_state_rank*.pt"))) == 8


def test_incomplete_or_corrupted_shards_cannot_claim_durability(source, plan, tmp_path):
    authorization = authorize_run(source, yes_rollouts=True, yes_publish=True)
    local, root = tmp_path / "local", tmp_path / "volume"
    destination = root / "mlp/checkpoints/iter_0000000/adapter"
    write_shards(local, source, range(7))
    files = publish_node_files(authorization, local=local, destination=destination, commit=lambda: None)
    with pytest.raises(ValueError, match="Missing native"):
        finalize_step(
            authorization,
            plan=plan,
            phase=plan.phases[0],
            step=0,
            native=native(),
            receipts=[files],
            root=root,
            commit=lambda: None,
        )
    assert not (destination / "native_checkpoint.json").exists()
    assert not (root / "mlp/step-1.json").exists()


def test_failed_volume_commit_cannot_publish_completion(source, plan, tmp_path):
    authorization = authorize_run(source, yes_rollouts=True, yes_publish=True)
    local, root = tmp_path / "local", tmp_path / "volume"
    destination = root / "mlp/checkpoints/iter_0000000/adapter"
    write_shards(local, source, range(8))
    files = publish_node_files(authorization, local=local, destination=destination, commit=lambda: None)

    def fail():
        raise OSError("commit failed")

    with pytest.raises(OSError, match="commit failed"):
        finalize_step(
            authorization,
            plan=plan,
            phase=plan.phases[0],
            step=0,
            native=native(),
            receipts=[files],
            root=root,
            commit=fail,
        )
    assert not (root / "mlp/step-1.json").exists()
