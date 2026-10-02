"""A batch chain applies one update per batch, continues only verified native state and never repeats a step."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.contracts import Policy, RunConfig, sampling_args
from miles_plugins.proximal.e2e import batch_chain_inputs
from miles_plugins.proximal.e2e.batch_chain_artifacts import (
    ChainStepReceipt,
    finalize_chain_step,
    read_previous,
    stage_previous,
)
from miles_plugins.proximal.e2e.batch_chain_coordination import (
    ChainPublisher,
    ChainRuntime,
    hold_failed_cluster,
    publish_ready,
    resume_confirmed,
    run_chain,
    train_metrics,
)
from miles_plugins.proximal.e2e.batch_chain_inputs import (
    CHAIN_REPLAY,
    ChainArm,
    ChainBatch,
    ChainPlan,
    step_command,
    validate_batches,
)
from miles_plugins.proximal.e2e.batch_sweep_artifacts import publish_node_files
from miles_plugins.proximal.snapshot import SnapshotReference
from miles_plugins.proximal.state_artifacts import StateFile, describe
from miles_plugins.proximal.state_checkpoints import NativeCompletion

REPO = Path(__file__).resolve().parents[3]
MLP = ("linear_fc1", "linear_fc2")
FULL = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")


@pytest.fixture
def source():
    return RunConfig.model_validate_json((REPO / "examples/proximal/e2e/run.stage-a.json").read_bytes())


@pytest.fixture
def plan():
    return ChainPlan(
        experiment_id="six-steps",
        samples=1024,
        nodes=1,
        batches=tuple(ChainBatch(path=f"run/batches/{c}", sha256=c * 64) for c in "abc"),
        arms=(ChainArm(name="mlp", target_modules=MLP), ChainArm(name="full", target_modules=FULL)),
        recipe=("--optimizer", "adam", "--lr", "4e-5", "--seed", "42", "--num-rollout", "9", "--use-wandb"),
        gate="manual",
    )


def policy(source):
    return Policy(
        run_id=source.run_id,
        version=1,
        snapshot=SnapshotReference(sha256="d" * 64),
        base_model=source.base_model,
    )


@pytest.mark.parametrize(
    "update, message",
    [
        ({"arms": (ChainArm(name="mlp", target_modules=MLP), ChainArm(name="mlp", target_modules=FULL))}, "unique"),
        ({"arms": (ChainArm(name="a", target_modules=MLP), ChainArm(name="b", target_modules=MLP[::-1]))}, "differ"),
        ({"batches": (ChainBatch(path="x", sha256="a" * 64), ChainBatch(path="y", sha256="a" * 64))}, "replay"),
        ({"batches": (ChainBatch(path="x", sha256="a" * 64), ChainBatch(path="x", sha256="b" * 64))}, "replay"),
    ],
)
def test_plan_refuses_duplicate_arms_and_replayed_batches(plan, update, message):
    with pytest.raises(ValueError, match=message):
        ChainPlan.model_validate({**plan.model_dump(), **update})


def test_each_step_is_one_update_on_its_own_batch_and_continues_native_state(plan, source):
    before = source.model_dump_json()
    for arm in plan.arms:
        for step in range(3):
            adapter = Path(f"/local/{arm.name}/{step}/resume") if step else None
            command = step_command(
                plan, arm, step, source, bundle=Path(f"/batch-{step}"), save=Path("/out"), resume_adapter=adapter
            )

            def value(flag, command=command):
                assert command.count(flag) == 1
                return command[command.index(flag) + 1]

            assert value("--num-rollout") == str(step + 1)
            assert value("--start-rollout-id") == str(step)
            assert value("--chain-batch") == f"/batch-{step}"
            assert value("--rollout-function-path") == CHAIN_REPLAY
            assert value("--target-modules") == ",".join(arm.target_modules)
            assert value("--global-batch-size") == "1024"
            for name, expected in sampling_args(source.research.sampling).items():
                assert value("--" + name.replace("_", "-")) == str(expected)
            assert ("--lora-adapter-path" in command) == (step > 0)
            assert "--use-wandb" not in command and "--no-load-optim" not in command
    assert source.model_dump_json() == before


@pytest.mark.parametrize("step, adapter", [(0, Path("/resume")), (1, None)])
def test_continuation_requires_exactly_the_staged_predecessor(plan, source, step, adapter):
    with pytest.raises(ValueError, match="continues the previous step"):
        step_command(plan, plan.arms[0], step, source, bundle=Path("/b"), save=Path("/o"), resume_adapter=adapter)


def fake_batches(tmp_path, monkeypatch, source, *, groups, policies=None):
    """Bundles whose manifests hash as pinned; validation returns the described batches."""
    pinned, described = [], {}
    for step, ids in enumerate(groups):
        bundle = tmp_path / f"b{step}"
        bundle.mkdir()
        (bundle / "batch.json").write_text(json.dumps({"step": step}))
        sha = hashlib.sha256((bundle / "batch.json").read_bytes()).hexdigest()
        pinned.append(ChainBatch(path=f"b{step}", sha256=sha))
        described[bundle] = SimpleNamespace(
            source=source,
            policy=(policies or [policy(source)] * len(groups))[step],
            num_samples=4,
            groups=tuple(SimpleNamespace(header=SimpleNamespace(group_id=g)) for g in ids),
        )
    monkeypatch.setattr("miles_plugins.proximal.offline_batch.validate_batch", lambda bundle: described[bundle])
    monkeypatch.setattr(batch_chain_inputs, "verify_base_policy", lambda *args: None)
    return tuple(pinned)


def chain_plan(plan, batches):
    return plan.model_copy(update={"batches": batches, "samples": 4})


def test_batches_validate_inside_the_staleness_window(tmp_path, monkeypatch, source, plan):
    source = source.model_copy(update={"research": source.research.model_copy(update={"max_policy_lag": 2})})
    batches = fake_batches(tmp_path, monkeypatch, source, groups=[("g0",), ("g1",), ("g2",)])
    assert len(validate_batches(tmp_path, chain_plan(plan, batches), source)) == 3


@pytest.mark.parametrize("problem", ["lag", "policy", "shared", "tampered"])
def test_batches_outside_the_contract_are_refused_before_resources(tmp_path, monkeypatch, source, plan, problem):
    lag = 1 if problem == "lag" else 2
    source = source.model_copy(update={"research": source.research.model_copy(update={"max_policy_lag": lag})})
    groups = [("g0",), ("g0",) if problem == "shared" else ("g1",), ("g2",)]
    other = policy(source).model_copy(update={"version": 2})
    policies = [policy(source), other, policy(source)] if problem == "policy" else None
    batches = fake_batches(tmp_path, monkeypatch, source, groups=groups, policies=policies)
    if problem == "tampered":
        (tmp_path / "b1/batch.json").write_text("{}")
    with pytest.raises(
        ValueError,
        match={"lag": "max_policy_lag", "policy": "one exact", "shared": "share", "tampered": "pinned"}[problem],
    ):
        validate_batches(tmp_path, chain_plan(plan, batches), source)


def write_shards(local, source, ranks):
    local.mkdir(parents=True)
    for rank in ranks:
        torch.save({"weight": torch.ones(2)}, local / f"adapter_megatron_rank{rank}.pt")
        torch.save({"optimizer": {"step": 1}}, local / f"training_state_rank{rank}.pt")
    r = source.research.lora.rank
    modules = [f"model.language_model.layers.0.mlp.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj")]
    weights = {
        f"base_model.model.{module}.lora_{side}.weight": torch.ones((r, 4) if side == "A" else (4, r))
        for module in modules
        for side in ("A", "B")
    }
    save_file(weights, str(local / "adapter_model.safetensors"))
    (local / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": source.research.lora.alpha, "target_modules": modules})
    )


def native(step):
    return NativeCompletion(
        schema_version=1,
        iteration=step,
        world_size=8,
        layout={"tensor_model_parallel_size": 4},
        optimizer=True,
        scheduler=True,
        rng=True,
    )


@pytest.fixture
def committed(tmp_path, source, plan):
    """The mlp arm's first step, durable on a fake Volume mounted at tmp_path."""
    authorization = authorize_run(source, yes_rollouts=True, yes_publish=True)
    root = tmp_path / "run/experiments/six-steps"
    root.mkdir(parents=True)
    (root / "plan.json").write_text(plan.model_dump_json())
    arm = plan.arms[0]
    write_shards(tmp_path / "local", source, range(8))
    destination = root / "mlp/step-1/attempt-0/checkpoints/iter_0000000/adapter"
    files = publish_node_files(authorization, local=tmp_path / "local", destination=destination, commit=lambda: None)
    events = []
    receipt = finalize_chain_step(
        authorization,
        plan=plan,
        arm=arm,
        step=0,
        attempt=0,
        behavior_policy=policy(source),
        previous=None,
        native=native(0),
        receipts=[files],
        root=root,
        commit=lambda: events.append((root / "mlp/step-1/attempt-0/receipt.json").exists()),
    )
    path = root / "mlp/step-1/attempt-0/receipt.json"
    return root, receipt, describe(path, relative=str(path.relative_to(tmp_path))), events, destination


def test_receipt_commits_last_and_records_behavior_policy_and_lag(source, committed):
    root, receipt, file, events, _ = committed
    assert events == [False, True]
    stored = ChainStepReceipt.model_validate_json((root / "mlp/step-1/attempt-0/receipt.json").read_bytes())
    assert stored == receipt
    assert (stored.step, stored.policy_lag, stored.previous, stored.behavior_policy.version) == (1, 0, None, 1)


def test_next_step_stages_every_verified_native_file(tmp_path, plan, committed):
    root, _, file, _, original = committed
    staged = stage_previous(tmp_path, root, plan, plan.arms[0], 1, file, tmp_path / "staged")
    assert len(list(staged.glob("training_state_rank*.pt"))) == 8
    assert NativeCompletion.model_validate_json((staged / "native_checkpoint.json").read_bytes()) == native(0)


@pytest.mark.parametrize("change", ["arm", "step", "plan", "corrupt", "missing", "receipt"])
def test_continuation_fails_before_training_for_incompatible_or_damaged_state(tmp_path, plan, committed, change):
    root, _, file, _, original = committed
    arm, step = plan.arms[0], 1
    if change == "arm":
        arm = plan.arms[1]
    elif change == "step":
        step = 2
    elif change == "plan":
        plan = plan.model_copy(update={"recipe": (*plan.recipe, "--lr", "1e-3")})
    elif change == "corrupt":
        (original / "training_state_rank3.pt").write_bytes(b"damaged")
    elif change == "missing":
        (original / "adapter_megatron_rank7.pt").unlink()
    elif change == "receipt":
        (tmp_path / file.path).write_text("{}")
    with pytest.raises(ValueError):
        read_previous(tmp_path, root, plan, arm, step, file)


def test_logs_yield_metrics_and_proof_of_continuation():
    log = (
        "x] log_utils.py:554 - step 1: {'train/loss': -0.01, 'train/tis': 1.0, 'train/train_rollout_kl': 0.0013}\n"
        "x] model.py:876 - step 1: {'train/loss': -0.01, 'train/tis': 1.0, 'train/train_rollout_kl': 0.0013}\n"
    )
    assert train_metrics(log) == [{"step": 1, "train/loss": -0.01, "train/tis": 1.0, "train/train_rollout_kl": 0.0013}]
    resumed = "Restored optimizer state from LoRA checkpoint\nResuming LoRA training from iteration 1\n"
    assert resume_confirmed(resumed, 2)
    assert not resume_confirmed(resumed, 3)
    assert not resume_confirmed(resumed + "Training will start with freshly initialized adapter weights.", 2)


class FakeCluster:
    """One node whose publisher scans whenever the coordinator waits, as the real thread would."""

    def __init__(self, tmp_path, plan, outcomes, gates=None):
        self.tmp, self.plan, self.outcomes, self.gates = tmp_path, plan, outcomes, dict(gates or {})
        self.state, self.calls, self.staged, self.resets, self.finalized = {}, [], [], [], []
        self.copied, self.finished = set(), set()
        self.publisher = ChainPublisher(
            rank=0,
            state=self.state,
            local_adapter=lambda arm, step, attempt: tmp_path / "local" / arm.name / f"{step}-{attempt}",
            publish_node=lambda arm, step, attempt, local: (
                StateFile(path=f"{arm.name}-{step}-{attempt}", size_bytes=1, sha256="e" * 64),
            ),
            finalize=self.finalize,
        )
        self.runtime = ChainRuntime(
            rank=0,
            state=self.state,
            wait=self.wait,
            stage=self.stage,
            reset_ray=self.resets.append,
            gate=lambda arm, step: self.gates.get((arm.name, step), "go"),
            run_step=self.run_step,
            persist=lambda *args: None,
            describe_receipt=lambda path: StateFile(path=path, size_bytes=1, sha256="f" * 64),
        )

    def wait(self, predicate, timeout, what):
        for _ in range(3):
            if predicate():
                return
            publish_ready(self.plan, self.publisher, self.copied, self.finished)
        raise AssertionError(f"Coordinator waited for missing evidence: {what}")

    def stage(self, arm, step, attempt, previous):
        self.staged.append((arm.name, step, attempt, previous.path if previous else None))
        return self.tmp / f"batch-{step}", (self.tmp / "resume") if previous else None

    def finalize(self, arm, step, attempt, completion, receipts):
        self.finalized.append((arm.name, step, attempt))
        return f"run/{arm.name}/step-{step + 1}/attempt-{attempt}/receipt.json"

    def run_step(self, arm, step, attempt, bundle, adapter):
        self.calls.append((arm.name, step, attempt, adapter is not None))
        code, saves, *resumed = self.outcomes.get((arm.name, step, attempt), (0, True))
        if saves:
            local = self.publisher.local_adapter(arm, step, attempt)
            local.mkdir(parents=True)
            (local / "native_checkpoint.json").write_text(native(step).model_dump_json())
        result = {"exit_code": code, "perf": []}
        if adapter is not None:
            result["resume_confirmed"] = resumed[0] if resumed else True
        return result


def test_both_arms_take_every_step_in_order_continuing_their_own_receipts(tmp_path, plan):
    fake = FakeCluster(tmp_path, plan, {})
    results = run_chain(plan, fake.runtime)
    assert [(a, s) for a, s, *_ in fake.calls] == [(a.name, s) for a in plan.arms for s in range(3)]
    assert [c[3] for c in fake.calls] == [False, True, True] * 2
    assert fake.staged[1][3] == "run/mlp/step-1/attempt-0/receipt.json"
    assert fake.staged[4][3] == "run/full/step-1/attempt-0/receipt.json"
    assert len(results) == 6 and fake.resets == []


def test_operator_stop_ends_the_chain_without_holding(tmp_path, plan):
    fake = FakeCluster(tmp_path, plan, {}, gates={("mlp", 2): "stop"})
    results = run_chain(plan, fake.runtime)
    assert [(a, s) for a, s, *_ in fake.calls] == [("mlp", 0), ("mlp", 1)]
    assert len(results) == 2 and fake.state["stop-all"]


def test_failed_attempt_retries_from_the_same_receipt_on_fresh_ray(tmp_path, plan):
    fake = FakeCluster(tmp_path, plan, {("mlp", 1, 0): (1, False)})
    run_chain(plan, fake.runtime)
    assert [c for c in fake.calls if c[0] == "mlp"] == [
        ("mlp", 0, 0, False),
        ("mlp", 1, 0, True),
        ("mlp", 1, 1, True),
        ("mlp", 2, 0, True),
    ]
    assert fake.staged[1][3] == fake.staged[2][3] and len(fake.resets) == 1


def test_committed_update_is_never_applied_twice_after_cleanup_failure(tmp_path, plan):
    fake = FakeCluster(tmp_path, plan, {("mlp", 0, 0): (1, True)})
    run_chain(plan, fake.runtime)
    assert [(a, s, t) for a, s, t, _ in fake.calls if a == "mlp"] == [("mlp", 0, 0), ("mlp", 1, 0), ("mlp", 2, 0)]


def test_continuation_without_proof_of_resumed_state_holds(tmp_path, plan):
    fake = FakeCluster(tmp_path, plan, {("mlp", 1, 0): (0, True, False)})
    with pytest.raises(RuntimeError, match="without proof"):
        run_chain(plan, fake.runtime)
    assert fake.state["stop-all"]


def test_exhausted_arm_holds_after_the_independent_arm_runs(tmp_path, plan):
    fake = FakeCluster(tmp_path, plan, {("mlp", 1, 0): (1, False), ("mlp", 1, 1): (1, False)})
    with pytest.raises(RuntimeError, match="holding allocation"):
        run_chain(plan, fake.runtime)
    assert [(a, s) for a, s, *_ in fake.calls if a == "full"] == [("full", 0), ("full", 1), ("full", 2)]
    assert ("mlp", 2) not in [(a, s) for a, s, *_ in fake.calls]


@pytest.mark.parametrize("release", [True, False])
def test_control_plane_outage_does_not_bypass_allocation_hold(release):
    clock = [0.0]

    class State:
        def get(self, key):
            if key == "release" and release and clock[0] >= 10:
                return True
            raise ConnectionError("control plane temporarily unavailable")

        def __getitem__(self, key):
            raise ConnectionError("control plane temporarily unavailable")

        def __setitem__(self, key, value):
            raise ConnectionError("control plane temporarily unavailable")

    def sleep(seconds):
        clock[0] += seconds

    hold_failed_cluster(
        State(), 0, hold_seconds=20, entered=0, exc=RuntimeError("failed"), clock=lambda: clock[0], sleep=sleep
    )
    assert clock[0] == (10 if release else 20)
