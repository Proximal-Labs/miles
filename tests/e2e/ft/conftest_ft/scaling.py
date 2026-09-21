# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.execution import (
    get_api_server_args,
    get_common_train_args,
    get_ft_args,
    get_train_env_vars_arg,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.fault_injection.core import list_cells
from tests.e2e.ft.conftest_ft.fault_injection.entrypoint import API_SERVER_PORT
from tests.e2e.ft.conftest_ft.fault_injection.state import CellInfo, EventLog, ObservationsEvent
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.fast.cluster_backends import create_backend_for_run

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, MetricEvent
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.comparisons.inference_engine_checksums import read_inference_engine_checksum_events
from miles.utils.test_utils.kubectl_reads import (
    KUBECTL_TIMEOUT_SECONDS,
    LEADER_WORKER_SET_KIND,
    patch_replicas,
    read_replicas,
)
from miles.utils.test_utils.polling_worker import PollingWorker, poll_until_stopped
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 5.0
STEP_TIMEOUT_SECONDS: float = 3600.0
JOIN_TIMEOUT_SECONDS: float = 300.0
LANDING_LAG_ROLLOUTS: int = 3
TRAIN_STEP_METRIC_KEY: str = "train/grad_norm"

Moment = Literal["generating", "training"]
Standing = Literal["before", "at", "after"]


# =============================== the schedule =================================


@dataclass(frozen=True)
class ScalingStep:
    at_rollout: int
    replicas: int
    moment: Moment


def assert_schedule_leaves_room(
    schedule: tuple[ScalingStep, ...], *, initial_replicas: int, num_rollouts: int
) -> None:
    assert schedule, "an empty schedule resizes nothing, and this scenario is about a pool that changes size"
    replicas = initial_replicas
    previous_at = -LANDING_LAG_ROLLOUTS - 1
    for step in schedule:
        assert step.replicas != replicas, f"{step} keeps the pool at {replicas} replica(s), so it scales nothing"
        assert step.at_rollout > previous_at + LANDING_LAG_ROLLOUTS, (
            f"{step} fires while the step before it may still be landing (up to {LANDING_LAG_ROLLOUTS} rollouts "
            f"after rollout {previous_at}), so the two sizes could not be told apart"
        )
        replicas = step.replicas
        previous_at = step.at_rollout
    assert previous_at + LANDING_LAG_ROLLOUTS < num_rollouts - 1, (
        f"the last step fires at rollout {previous_at} and may land up to {LANDING_LAG_ROLLOUTS} rollouts later, "
        f"leaving no rollout of the {num_rollouts} to train at the final size"
    )


# ============================== running a scenario ============================


def run_scaling_scenario(
    *,
    test_name: str,
    mode: FTTestMode,
    num_rollouts: int,
    schedule: tuple[ScalingStep, ...],
    initial_replicas: int,
    pool_id: str,
    cell_type: str,
    assert_scaled: Callable[["ScalingDriver", str], None],
) -> None:
    config = command_utils.default_config()
    assert config.cluster_backend is ClusterBackend.KUBERNETES, (
        f"a scaling scenario resizes the LeaderWorkerSet a pool is deployed as, which only the "
        f"{ClusterBackend.KUBERNETES.value} backend has, and this environment declares the "
        f"{config.cluster_backend.value} backend"
    )
    assert config.namespace, "resizing a pool needs the namespace the run is installed into"
    U = create_backend_for_run(config)
    assert_schedule_leaves_room(schedule, initial_replicas=initial_replicas, num_rollouts=num_rollouts)

    dump_dir: str = resolve_dump_dir(test_name, run_id=config.run_id)
    print(f"Dump directory: {dump_dir}")
    prepare(mode, config=config)

    train_args = (
        get_common_train_args(mode, dump_dir=dump_dir, num_steps=num_rollouts, enable_dumper=False)
        + get_ft_args(mode)
        + get_api_server_args(config)
        + "--mini-ft-controller-enable "
        + get_train_env_vars_arg(mode, deterministic=False)
    )
    release = ReleaseName(
        run_id=config.run_id,
        deploy_component=config.deploy_component,
        deploy_instance_id=config.deploy_instance_id,
    ).serialize()

    driver = ScalingDriver(
        dump_dir=dump_dir,
        namespace=config.namespace,
        workload=component_name(release, pool_id),
        base_url=f"http://{U.api_server_host(config)}:{API_SERVER_PORT}",
        cell_type=cell_type,
        schedule=schedule,
    )
    driver.start()
    try:
        run_training(train_args=train_args, mode=mode, dump_dir=dump_dir, config=config)
    finally:
        driver.stop_and_join()

    driver.assert_all_steps_applied(initial_replicas=initial_replicas)
    assert_scaled(driver, dump_dir)
    print(f"Scaling test PASSED ({test_name}, rollouts={num_rollouts}, schedule={schedule})")


# ============================ how far the run has come =========================


@dataclass(frozen=True)
class RunProgress:
    last_generated_rollout_id: int | None
    last_trained_rollout_id: int | None
    last_weight_updated_rollout_id: int | None


def read_run_progress(dump_dir: str) -> RunProgress:
    events_dir = Path(dump_dir) / EVENTS_DIRNAME
    events = read_events(events_dir) if events_dir.is_dir() else []
    metric_events = [event for event in events if isinstance(event, MetricEvent) and event.rollout_id is not None]
    return RunProgress(
        last_generated_rollout_id=max(
            (event.rollout_id for event in metric_events if event.source.component == "rollout_executor"),
            default=None,
        ),
        last_trained_rollout_id=max(
            (event.rollout_id for event in metric_events if TRAIN_STEP_METRIC_KEY in event.metrics), default=None
        ),
        last_weight_updated_rollout_id=max(
            (event.rollout_id for event in events if isinstance(event, InferenceEngineWeightChecksumEvent)),
            default=None,
        ),
    )


def read_engine_counts_of_rollout(dump_dir: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for event in read_inference_engine_checksum_events(Path(dump_dir)):
        assert event.rollout_id not in counts, f"{dump_dir} holds two weight updates for rollout {event.rollout_id}"
        counts[event.rollout_id] = len(event.engine_checksums)
    return counts


def compute_standing(progress: RunProgress, *, step: ScalingStep) -> Standing:
    if step.moment == "training":
        opened, closed = progress.last_generated_rollout_id, progress.last_trained_rollout_id
        opens_at = step.at_rollout
    else:
        opened, closed = progress.last_weight_updated_rollout_id, progress.last_generated_rollout_id
        opens_at = step.at_rollout - 1

    if _reached(closed, step.at_rollout):
        return "after"
    return "at" if _reached(opened, opens_at) else "before"


def _reached(rollout_id: int | None, threshold: int) -> bool:
    return rollout_id is not None and rollout_id >= threshold


# ================================ the driver ==================================


@dataclass
class ScalingDriver:
    dump_dir: str
    namespace: str
    workload: str
    base_url: str
    cell_type: str
    schedule: tuple[ScalingStep, ...]
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS
    step_timeout_seconds: float = STEP_TIMEOUT_SECONDS
    replicas_before: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.event_log = EventLog()
        self._failures: list[BaseException] = []
        self._worker = PollingWorker(name="ft-scaling-driver", run=self._drive)

    def start(self) -> None:
        result = run_process(
            ["kubectl", "auth", "can-i", "patch", LEADER_WORKER_SET_KIND, "--namespace", self.namespace],
            capture_output=True,
            check=False,
            timeout=KUBECTL_TIMEOUT_SECONDS,
        )
        assert result.stdout.strip() == "yes", (
            f"this account may not patch {LEADER_WORKER_SET_KIND} in namespace {self.namespace}, so no pool of the "
            f"run can be resized: {result.stderr.strip() or result.stdout.strip()}"
        )
        self._worker.start()

    def stop_and_join(self) -> None:
        self._worker.stop_and_join(timeout_seconds=JOIN_TIMEOUT_SECONDS)
        self._worker.assert_not_running(
            message=(
                f"the scaling driver was still working {JOIN_TIMEOUT_SECONDS}s after being asked to stop, so "
                f"reading what it collected would race it"
            )
        )
        self._observe()

    def assert_all_steps_applied(self, *, initial_replicas: int) -> None:
        assert not self._failures, "the scaling driver failed:\n" + "\n".join(f"  - {one!r}" for one in self._failures)
        assert len(self.replicas_before) == len(self.schedule), (
            f"the run ended after {len(self.replicas_before)} of {len(self.schedule)} resize(s), so the pool never "
            f"went through every size the schedule names"
        )
        assert self.replicas_before[0] == initial_replicas, (
            f"{self.workload} was deployed with {self.replicas_before[0]} replica(s), not the {initial_replicas} "
            f"the mode declares, so the schedule resized a pool nobody described"
        )

    def _drive(self, stop_event: threading.Event) -> None:
        try:
            for index, step in enumerate(self.schedule):
                if not self._wait_until_standing_at(stop_event, index=index, step=step):
                    return
                self._apply(index, step=step)
            poll_until_stopped(stop_event, tick=self._observe, poll_interval_seconds=self.poll_interval_seconds)
        except BaseException as e:
            logger.warning("The scaling driver stopped before every resize had been applied", exc_info=True)
            self._failures.append(e)

    def _wait_until_standing_at(self, stop_event: threading.Event, *, index: int, step: ScalingStep) -> bool:
        deadline = time.monotonic() + self.step_timeout_seconds
        while not stop_event.is_set():
            self._observe()
            progress = read_run_progress(self.dump_dir)
            standing = compute_standing(progress, step=step)
            assert standing != "after", (
                f"resize {index} was to fire while the run was {step.moment} rollout {step.at_rollout}, and the "
                f"run stands at {progress}: the driver missed the moment, so where the resize landed would be raced"
            )
            if standing == "at":
                return True
            assert time.monotonic() < deadline, (
                f"resize {index} waited {self.step_timeout_seconds}s for the run to be {step.moment} rollout "
                f"{step.at_rollout}, and the run only reached {progress}"
            )
            stop_event.wait(timeout=self.poll_interval_seconds)
        return False

    def _apply(self, index: int, *, step: ScalingStep) -> None:
        before = read_replicas(namespace=self.namespace, workload=self.workload)
        patch_replicas(namespace=self.namespace, workload=self.workload, replicas=step.replicas)
        after = read_replicas(namespace=self.namespace, workload=self.workload)
        assert (
            after == step.replicas
        ), f"{self.workload} reads {after} replica(s) right after being resized to {step.replicas}"

        self.replicas_before.append(before)
        logger.info(
            f"Resize {index}: {self.workload} {before} -> {after} while the run was {step.moment} rollout {step.at_rollout}"
        )

    def _observe(self) -> None:
        if (cells := list_cells(base_url=self.base_url, cell_types={self.cell_type})) is not None:
            self.event_log.observe(cells)


# =============================== the witnesses ================================


def assert_counts_follow_schedule(
    counts_of_rollout: dict[int, int],
    *,
    initial: int,
    schedule: tuple[ScalingStep, ...],
    num_rollouts: int,
    what: str,
) -> None:
    assert sorted(counts_of_rollout) == list(range(num_rollouts)), (
        f"{what} is known for rollouts {sorted(counts_of_rollout)}, not for each of the {num_rollouts} rollouts "
        f"exactly once"
    )

    sizes = [initial, *(step.replicas for step in schedule)]
    phase = 0
    landed_at: list[int] = []
    for rollout_id in range(num_rollouts):
        count = counts_of_rollout[rollout_id]
        if phase < len(schedule) and rollout_id >= schedule[phase].at_rollout and count == sizes[phase + 1]:
            phase += 1
            landed_at.append(rollout_id)
        assert count == sizes[phase], (
            f"{what} was {count} at rollout {rollout_id}, and the schedule {schedule} (from {initial}) has it at "
            f"{sizes[phase]} there; the whole series is {counts_of_rollout}"
        )

    assert phase == len(schedule), (
        f"only {phase} of the {len(schedule)} resize(s) ever showed in {what}: the series {counts_of_rollout} "
        f"never reached {sizes[phase + 1]}"
    )
    for step, rollout_id in zip(schedule, landed_at, strict=True):
        assert rollout_id <= step.at_rollout + LANDING_LAG_ROLLOUTS, (
            f"the resize fired at rollout {step.at_rollout} showed in {what} only at rollout {rollout_id}, more "
            f"than {LANDING_LAG_ROLLOUTS} rollout(s) later"
        )

    print(f"{what} followed the schedule: {sizes} landing at rollouts {landed_at}")


def assert_observed_cells(
    event_log: EventLog,
    *,
    cell_type: str,
    initial: int,
    schedule: tuple[ScalingStep, ...],
    counts: Callable[[CellInfo], bool],
) -> None:
    peak = max(initial, *(step.replicas for step in schedule))
    final = schedule[-1].replicas
    observed = [
        sum(1 for info in event.cell_infos.values() if info.cell_type == cell_type and counts(info))
        for event in event_log.events
        if isinstance(event, ObservationsEvent)
    ]
    assert observed, f"the api server was never read, so nothing here says how many {cell_type} cells the run saw"
    assert max(observed) == peak, (
        f"the api server listed at most {max(observed)} {cell_type} cell(s) at once, not the {peak} the pool "
        f"was resized to: {observed}"
    )
    assert observed[-1] == final, (
        f"the api server listed {observed[-1]} {cell_type} cell(s) when the run ended, not the {final} the "
        f"pool was resized back to: {observed}"
    )
    print(f"the api server listed up to {peak} and finally {final} {cell_type} cell(s) across {len(observed)} reads")
