"""The sweep cannot silently change source data or lose an update's eval export."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.contracts import RunConfig, sampling_args
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
            SweepPhase(name="mlp", updates=2, resume=None, target_modules=("linear_fc1", "linear_fc2")),
            SweepPhase(name="full", updates=2, resume=None, target_modules=("linear_qkv", "linear_fc1", "linear_fc2")),
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
        # The trainer scores tokens under the batch's sampling (Miles replay follows top-p/top-k).
        for name, expected in sampling_args(source.research.sampling).items():
            assert value("--" + name.replace("_", "-")) == str(expected)
        assert "--lora-adapter-path" not in command and "--use-wandb" not in command
    assert source.model_dump_json() == before


@pytest.mark.parametrize(
    "flag", ["--lora-adapter-path", "--no-save-optim", "--enable-mtp-training", "--load-debug-rollout-data"]
)
def test_incompatible_recipe_is_refused_before_resources(source, plan, flag):
    invalid = plan.model_copy(update={"recipe": (*plan.recipe, flag)})
    with pytest.raises(ValueError, match="Unsupported"):
        phase_command(invalid, plan.phases[0], source, bundle=Path("/batch"), save=Path("/output"))


def write_shards(local, source, ranks, export="adapter_model.safetensors"):
    """A node's checkpoint directory as the trainer leaves it: native shards, and on rank 0 the PEFT export."""
    local.mkdir(parents=True)
    for rank in ranks:
        torch.save({"weight": torch.ones(2)}, local / f"adapter_megatron_rank{rank}.pt")
        torch.save({"optimizer": {"step": 1}}, local / f"training_state_rank{rank}.pt")
    if 0 in ranks:
        r = source.research.lora.rank
        modules = [f"model.language_model.layers.0.mlp.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj")]
        weights = {
            f"base_model.model.{module}.lora_{side}.weight": torch.ones((r, 4) if side == "A" else (4, r))
            for module in modules
            for side in ("A", "B")
        }
        if export.endswith(".safetensors"):
            save_file(weights, str(local / export))
        else:
            torch.save(weights, local / export)
        # SnapshotPublisher.write_adapter lists the adapted module paths, not serving leaf names.
        (local / "adapter_config.json").write_text(
            json.dumps(
                {"peft_type": "LORA", "r": r, "lora_alpha": source.research.lora.alpha, "target_modules": modules}
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


@pytest.mark.parametrize("export", ["adapter_model.safetensors", "adapter_model.bin"])
def test_every_native_shard_and_eval_export_commit_before_completion(source, plan, tmp_path, export):
    authorization = authorize_run(source, yes_rollouts=True, yes_publish=True)
    local, root = tmp_path / "local", tmp_path / "volume"
    destination = root / "mlp/checkpoints/iter_0000000/adapter"
    write_shards(local, source, range(8), export)
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
    assert len(list(destination.glob("training_state_rank*.pt"))) == 8
    # The evaluation snapshot is the form replicas serve, whatever the trainer's export looked like.
    assert {file.name for file in snapshot.manifest.files} == {"adapter_config.json", "adapter_model.safetensors"}
    served = json.loads((snapshot.directory / "adapter_config.json").read_text())
    assert served["target_modules"] == ["gate_proj", "up_proj", "down_proj"]
    assert served["base_model_name_or_path"] == source.base_model.name
    assert len(load_file(str(snapshot.directory / "adapter_model.safetensors"))) == 6
    assert export in {file["path"] for file in result["files"]}


def test_update_without_a_serving_export_cannot_claim_durability(source, plan, tmp_path):
    authorization = authorize_run(source, yes_rollouts=True, yes_publish=True)
    local, root = tmp_path / "local", tmp_path / "volume"
    destination = root / "mlp/checkpoints/iter_0000000/adapter"
    write_shards(local, source, range(8))
    (local / "adapter_model.safetensors").unlink()
    files = publish_node_files(authorization, local=local, destination=destination, commit=lambda: None)
    with pytest.raises(ValueError, match="adapter_model export"):
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
    assert not (root / "mlp/step-1.json").exists()


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


@pytest.fixture
def resumable(source, plan, tmp_path):
    from miles_plugins.proximal.state_artifacts import describe

    authorization = authorize_run(source, yes_rollouts=True, yes_publish=True)
    local, root = tmp_path / "local", tmp_path / "old"
    destination = root / "mlp/checkpoints/iter_0000000/adapter"
    write_shards(local, source, range(8))
    files = publish_node_files(authorization, local=local, destination=destination, commit=lambda: None)
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
    (root / "plan.json").write_text(plan.model_dump_json())
    phase = plan.phases[0].model_copy(
        update={"resume": describe(root / "mlp/step-1.json", relative="old/mlp/step-1.json")}
    )
    return phase, destination


def test_resume_stages_native_optimizer_and_starts_only_second_update(source, plan, resumable, tmp_path):
    from miles_plugins.proximal.e2e.batch_sweep_recovery import stage_resume

    phase, original = resumable
    local = stage_resume(tmp_path, plan, phase, tmp_path / "staged")
    assert len(list(local.glob("training_state_rank*.pt"))) == 8
    # Simulate the Volume disappearing during publication. The training command
    # and every native load continue to use verified local files.
    original.rename(original.with_name("temporarily-unmounted"))
    command = phase_command(
        plan, phase, source, bundle=tmp_path / "local-batch", save=tmp_path / "new", resume_adapter=local
    )
    assert command[command.index("--start-rollout-id") + 1] == "1"
    assert command[command.index("--num-rollout") + 1] == "2"
    assert command[command.index("--lora-adapter-path") + 1] == str(local)
    assert (local / "adapter_megatron_rank7.pt").is_file()
    assert "--no-load-optim" not in command and "--no-load-rng" not in command
    with pytest.raises(ValueError, match="verified"):
        phase_command(plan, phase, source, bundle=tmp_path, save=tmp_path / "other")


@pytest.mark.parametrize("change", ["targets", "recipe", "corrupt", "missing", "receipt"])
def test_resume_fails_before_training_for_incompatible_or_damaged_state(plan, resumable, tmp_path, change):
    from miles_plugins.proximal.e2e.batch_sweep_recovery import stage_resume

    phase, original = resumable
    if change == "targets":
        phase = phase.model_copy(update={"target_modules": ("linear_qkv",)})
    elif change == "recipe":
        plan = plan.model_copy(update={"recipe": (*plan.recipe, "--lr", "1e-3")})
    elif change == "corrupt":
        (original / "training_state_rank3.pt").write_bytes(b"damaged")
    elif change == "missing":
        (original / "adapter_megatron_rank7.pt").unlink()
    elif change == "receipt":
        (tmp_path / phase.resume.path).write_text("{}")
    with pytest.raises(ValueError):
        stage_resume(tmp_path, plan, phase, tmp_path / "staged")
    assert not (tmp_path / "staged/native_checkpoint.json").exists()


def test_metric_rows_do_not_double_count_an_optimizer_update():
    from miles_plugins.proximal.e2e.batch_sweep_recovery import completed_training_steps

    rows = [{"rollout": 0, "rollout/num_training_samples": 1024}, {"rollout": 0, "perf/actor_train_time": 1201.0}]
    assert completed_training_steps(rows) == {0}
    rows.append({"rollout": 1, "perf/actor_train_time": 1100.0})
    assert completed_training_steps(rows) == {0, 1}


def coordinator(tmp_path, source, plan, outcomes):
    """Exercise the actual coordinator with CPU-only subprocess/publication results.

    Load just its function to avoid constructing the configured Modal GPU image.
    Each fake result certifies a completed native save, independently of metrics.
    """
    import ast
    import shutil
    from functools import partial
    from types import SimpleNamespace
    from miles_plugins.proximal.e2e.batch_sweep_recovery import completed_training_steps
    from miles_plugins.proximal.state_artifacts import describe
    from miles_plugins.proximal.storage import write_atomic

    path = REPO / "miles_plugins/proximal/e2e/batch_sweep.py"
    function = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_run_phases"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    state = {}
    calls, ray_stops = [], []
    root = tmp_path / "experiment"

    def run(command, *, label, log, **kwargs):
        phase, attempt = label.split("-attempt-")
        attempt = int(attempt)
        start = int(command[command.index("--start-rollout-id") + 1])
        calls.append((phase, attempt, start, "--lora-adapter-path" in command))
        code, steps = outcomes[(phase, attempt)]
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("retained diagnostic log")
        for step in steps:
            receipt = root / f"attempt-{attempt}" / phase / f"step-{step+1}.json"
            write_atomic(receipt, b"{}")
            state[f"save/{phase}/{attempt}/{step}"] = True
            state[f"durable/{phase}/{step}"] = {"receipt_path": str(receipt.relative_to(tmp_path))}
        state[f"scanned/{phase}/{attempt}/0"] = True
        return {"exit_code": code, "perf": [{"rollout": s, "perf/actor_train_time": 1.0} for s in steps]}

    def wait(predicate, *_):
        assert predicate(), "Coordinator waited for missing recovery evidence"

    class State(dict):
        scope = "owned-gang"

        @property
        def store(self):
            return self

    state = State()
    cluster = SimpleNamespace(run_command=run, _wait=wait, _State=lambda *args: state, _start_ray=lambda *args: None)
    import threading

    namespace = dict(
        partial=partial,
        _all_keys=lambda state, keys: all(state.get(k) for k in keys),
        cluster=cluster,
        require_authorization=lambda auth: None,
        LOCAL=tmp_path / "local",
        MOUNT=tmp_path,
        RUN=source,
        SweepPhase=SweepPhase,
        describe=describe,
        json=json,
        shutil=shutil,
        write_atomic=write_atomic,
        phase_command=phase_command,
        completed_training_steps=completed_training_steps,
        _VOLUME_LOCK=threading.Lock(),
        VOLUME=SimpleNamespace(reload=lambda: None, commit=lambda: None),
        stage_resume=lambda mount, plan, phase, target: target if phase.resume else None,
        subprocess=SimpleNamespace(run=lambda *args, **kwargs: ray_stops.append(args)),
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return (
        lambda: namespace["_run_phases"](object(), plan, state, root, tmp_path / "batch", 0, ["node"]),
        calls,
        ray_stops,
    )


def test_failed_phase_retries_from_committed_update_without_reapplying_it(tmp_path, source, plan):
    run, calls, stops = coordinator(
        tmp_path, source, plan, {("mlp", 0): (1, [0]), ("mlp", 1): (0, [1]), ("full", 0): (0, [0, 1])}
    )
    result = run()
    assert calls == [("mlp", 0, 0, False), ("mlp", 1, 1, True), ("full", 0, 0, False)]
    assert len(stops) == 1 and len(result) == 3


def test_independent_full_phase_runs_after_mlp_exhausts_attempts(tmp_path, source, plan):
    run, calls, stops = coordinator(
        tmp_path, source, plan, {("mlp", 0): (1, []), ("mlp", 1): (1, []), ("full", 0): (0, [0, 1])}
    )
    with pytest.raises(RuntimeError, match="holding allocation"):
        run()
    assert [p for p, *_ in calls] == ["mlp", "mlp", "full"]
    assert len(stops) == 2  # Ray workers only; the Modal allocation is untouched.


def test_committed_final_update_is_not_retried_after_subprocess_cleanup_failure(tmp_path, source, plan):
    # A failed subprocess after its last save must not cause an extra update.
    run, calls, stops = coordinator(tmp_path, source, plan, {("mlp", 0): (1, [0, 1]), ("full", 0): (0, [0, 1])})
    run()
    assert [p for p, *_ in calls] == ["mlp", "full"]


@pytest.mark.parametrize("release", [True, False])
def test_control_plane_outage_does_not_bypass_allocation_hold(plan, release):
    import ast
    from types import SimpleNamespace

    path = REPO / "miles_plugins/proximal/e2e/batch_sweep.py"
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_hold_failed_cluster"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    clock = [0]

    def sleep(seconds):
        clock[0] += seconds

    class State:
        def get(self, key):
            if key == "release" and release and clock[0] >= 10:
                return True
            raise ConnectionError("control plane temporarily unavailable")

    namespace = {"time": SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    namespace["_hold_failed_cluster"](
        State(), 0, plan.model_copy(update={"failure_hold_seconds": 20}), 0, RuntimeError("failed phase")
    )
    assert clock[0] == (10 if release else 20)
