# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import asyncio
from functools import partial
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import (
    ModeOption,
    NumStepsOption,
    RolloutCrashIntervalSecondsOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
    materialize_cyclic_debug_rollout_data,
    prepare,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode
from tests.e2e.ft.conftest_ft.training_launcher import TrainingLaunchSpec, execute_session
from tests.utils.soak.checks.ft import assert_healing
from tests.utils.soak.checks.tail import assert_tail_complete
from tests.utils.soak.checks.weights import assert_published_weight_checksums
from tests.utils.soak.cli_options import SeedOption
from tests.utils.soak.config import create_policy, create_tail_policy
from tests.utils.soak.entrypoint import API_SERVER_PORT, create_soak_session
from tests.utils.soak.fault_forms import compute_mean_interval_seconds_of_cell_type, create_cell_fault_forms
from tests.utils.soak.state import event_source
from tests.utils.soak.teardown import teardown_run
from tests.utils.soak.utils import create_soak_config, evidence_directory, get_api_server_args

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.external_utils import command_utils

app: typer.Typer = typer.Typer()

TEST_NAME: str = "random_crash"

DEFAULT_SEED: int = 42
DEFAULT_NUM_STEPS: int = 60
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 120.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 240.0


@app.command(name="run")
def run_ci(
    mode: ModeOption,
    seed: SeedOption = DEFAULT_SEED,
    num_steps: NumStepsOption = DEFAULT_NUM_STEPS,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
) -> None:
    """Random failure soak test, for whichever components the mode enables ft on.

    Runs an async session that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    tail_policy = create_tail_policy(num_rollout=num_steps)

    config = create_soak_config(command_utils.default_config())
    test_name: str = TEST_NAME
    dump_dir: str = resolve_dump_dir(f"{test_name}_{mode}", run_id=config.run_id)
    print(f"Dump directory: {dump_dir}")
    mean_interval_seconds_of_cell_type: dict[str, float] = compute_mean_interval_seconds_of_cell_type(
        ft_mode.ft_components,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
    )
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection intervals: {mean_interval_seconds_of_cell_type}")
    print(f"FT components: {ft_mode.ft_components}, cluster backend: {config.cluster_backend.value}")

    prepare(ft_mode, config=config)

    debug_rollout_data_dir = None if ft_mode.has_real_rollout else materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(
            ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=debug_rollout_data_dir
        )
        + get_ft_args(
            ft_mode,
            api_server_args=get_api_server_args(config),
        )
        + "--mini-ft-controller-enable "
    )
    base_url = f"http://{config.create_backend().api_server_host(config)}:{API_SERVER_PORT}"
    evidence_dir = evidence_directory(Path(dump_dir))
    cell_fault_forms = create_cell_fault_forms(base_url=base_url, config=config)
    assert not Path(dump_dir).exists() or not any(
        Path(dump_dir).iterdir()
    ), f"Soak dump directory contains existing artifacts: {dump_dir}; choose a new run_id"
    injector = create_soak_session(
        tail_policy=tail_policy,
        policy=create_policy(
            expected_cells={
                kind: count
                for kind, count in {"actor": ft_mode.num_cells, "rollout": ft_mode.rollout_num_engines}.items()
                if kind in mean_interval_seconds_of_cell_type
            },
        ),
        evidence_path=evidence_dir / "events.jsonl",
        sources={"training_events": Path(dump_dir) / EVENTS_DIRNAME},
        config=config,
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=cell_fault_forms,
    )

    asyncio.run(
        injector.run(
            execute_session(
                spec=TrainingLaunchSpec(
                    train_args=train_args,
                    mode=ft_mode,
                    extra_env_vars={},
                    config=config,
                    train_script="train.py",
                ),
                injector=injector,
                log_path=evidence_dir / "launcher-initial.log",
            ),
            teardown=partial(teardown_run, config=config, event_log=injector.event_log, evidence_dir=evidence_dir),
        )
    )

    assert_tail_complete(injector.event_log.events)
    if ft_mode.has_real_rollout:
        assert_published_weight_checksums(
            read_events(
                event_source(
                    injector.event_log.events, name="training_events", fallback=Path(dump_dir) / EVENTS_DIRNAME
                )
            )
        )
    assert_healing(
        ft_mode.ft_components,
        events=injector.event_log.events,
        forms=injector.cell_fault_forms,
        event_dir=Path(dump_dir) / EVENTS_DIRNAME,
        context=f"{test_name} {mode}",
    )

    print(f"Random failure soak test PASSED ({test_name}, mode={mode}, seed={seed}, steps={num_steps})")


if __name__ == "__main__":
    app()
