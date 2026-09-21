# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ACTOR_CELL_TYPE
from tests.e2e.ft.conftest_ft.modes import DENSE_MODEL_HF_REPO, DENSE_MODEL_NAME, DENSE_MODEL_TYPE, FTTestMode
from tests.e2e.ft.conftest_ft.scaling import (
    LANDING_LAG_ROLLOUTS,
    ScalingDriver,
    ScalingStep,
    assert_counts_follow_schedule,
    assert_observed_cells,
    run_scaling_scenario,
)

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.test_utils.comparisons.inference_engine_checksums import assert_engine_count
from miles.utils.test_utils.comparisons.metrics import assert_gradients_nonzero
from miles.utils.test_utils.reconfigure_assertions import ReconfigureInfo, load_reconfigure_events

app: typer.Typer = typer.Typer()

TEST_NAME: str = "trainer_scaling"
NUM_ROLLOUTS: int = 12
SCHEDULE: tuple[ScalingStep, ...] = (
    ScalingStep(at_rollout=2, replicas=3, moment="generating"),
    ScalingStep(at_rollout=7, replicas=2, moment="generating"),
)

_MODE: FTTestMode = FTTestMode(
    model_name=DENSE_MODEL_NAME,
    model_hf_repo=DENSE_MODEL_HF_REPO,
    megatron_model_type=DENSE_MODEL_TYPE,
    num_cells=2,
    train_gpus_per_node=4,
    rollout_num_engines=2,
    rollout_gpus_per_engine=1,
    ft_components=("train",),
    parallel_args="--context-parallel-size 2",
)


@app.command(name="run")
def run_ci() -> None:
    run_scaling_scenario(
        test_name=TEST_NAME,
        mode=_MODE,
        num_rollouts=NUM_ROLLOUTS,
        schedule=SCHEDULE,
        initial_replicas=_MODE.num_cells,
        pool_id=compute_trainer_pool_id(ACTOR_ROLE),
        cell_type=ACTOR_CELL_TYPE,
        assert_scaled=_assert_trainer_cells_scaled,
    )


def _assert_trainer_cells_scaled(driver: ScalingDriver, dump_dir: str) -> None:
    events_dir = Path(dump_dir) / EVENTS_DIRNAME

    assert_counts_follow_schedule(
        _read_num_trained_cells_of_rollout(events_dir),
        initial=_MODE.num_cells,
        schedule=SCHEDULE,
        num_rollouts=NUM_ROLLOUTS,
        what="the number of trainer cells that trained a step",
    )
    _assert_reconfigured_once_per_resize(events_dir)
    assert_observed_cells(
        driver.event_log,
        cell_type=ACTOR_CELL_TYPE,
        initial=_MODE.num_cells,
        schedule=SCHEDULE,
        counts=lambda info: info.alive,
    )

    assert_engine_count(side=TEST_NAME, dump_dir=dump_dir, expected=_MODE.rollout_num_engines)
    assert_gradients_nonzero(side=TEST_NAME, dump_dir=dump_dir, min_trained_rollouts=NUM_ROLLOUTS)


def _read_num_trained_cells_of_rollout(events_dir: Path) -> dict[int, int]:
    final_of_rollout: dict[int, TrainGroupStepEndEvent] = {}
    for event in read_events(events_dir):
        if isinstance(event, TrainGroupStepEndEvent) and (
            (final := final_of_rollout.get(event.rollout_id)) is None or event.attempt > final.attempt
        ):
            final_of_rollout[event.rollout_id] = event

    for rollout_id, event in sorted(final_of_rollout.items()):
        assert all(
            outcomes != "error" and all(one is TrainStepOutcome.NORMAL for one in outcomes)
            for outcomes in event.cell_outcomes.values()
        ), (
            f"the final attempt {event.attempt} of rollout {rollout_id} ended with {event.cell_outcomes}, so a "
            f"cell that a resize added or removed took the step down with it instead of joining or leaving cleanly"
        )
    return {rollout_id: len(event.cell_outcomes) for rollout_id, event in final_of_rollout.items()}


def _assert_reconfigured_once_per_resize(events_dir: Path) -> None:
    actual = [ReconfigureInfo.from_event(event) for event in load_reconfigure_events(events_dir)]
    assert len(actual) == len(SCHEDULE), (
        f"a resize of the trainer pool reconfigures the quorum exactly once, and {len(SCHEDULE)} resize(s) left "
        f"{len(actual)} CellReconfigureEvent(s): {actual}"
    )

    alive = list(range(_MODE.num_cells))
    for step, info in zip(SCHEDULE, actual, strict=True):
        assert step.at_rollout <= info.rollout_id <= step.at_rollout + LANDING_LAG_ROLLOUTS, (
            f"the resize fired while generating rollout {step.at_rollout} reconfigured the quorum at rollout "
            f"{info.rollout_id}, outside the {LANDING_LAG_ROLLOUTS}-rollout landing window: {info}"
        )
        will_alive = list(range(step.replicas))
        expected = ReconfigureInfo(
            rollout_id=info.rollout_id,
            src_cell_index=0 if step.replicas > len(alive) else None,
            healed_cell_indices=[index for index in will_alive if index not in alive],
            alive_cell_indices_after=will_alive,
        )
        assert (
            info == expected
        ), f"the quorum was reconfigured as {info}, and the resize to {step.replicas} cells means {expected}"
        alive = will_alive

    print(f"the trainer quorum was reconfigured once per resize: {actual}")


if __name__ == "__main__":
    app()
