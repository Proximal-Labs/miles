# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import requests
from tests.utils.soak.fault_forms import BaseFaultForm, CellFaultForms
from tests.utils.soak.state import (
    Event,
    EventLog,
    ObservationsEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakObservation,
    SoakScheduleEvent,
    cell_is_alive,
    cell_type_of,
)
from tests.utils.soak.views import compute_successful_form_names

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 2.0
QUIESCENT_POLLS_REQUIRED: int = 60


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


def run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    stop_event: threading.Event,
    event_log: EventLog,
    cell_fault_forms: CellFaultForms,
    get_virtual_cells: Callable[[], list[dict]] | None = None,
    injection_enabled: Callable[[], bool] | None = None,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
) -> None:
    rng = random.Random(seed)
    observer = _SynchronousObserver(
        base_url=base_url, cell_types=set(mean_interval_seconds_of_cell_type), get_virtual_cells=get_virtual_cells
    )
    scheduler = SoakActionScheduler(
        rng=rng,
        mean_intervals=mean_interval_seconds_of_cell_type,
        forms=cell_fault_forms,
        injection_enabled=injection_enabled,
        quiescent_polls_required=quiescent_polls_required,
    )
    event_log.note_schedule(scheduler.initial_schedule())

    while not stop_event.is_set():
        if stop_event.wait(timeout=poll_interval_seconds):
            break

        cells = observer.observe()
        if cells is None:
            continue

        # Record every poll so the post-run witnesses see the same stream the injector saw.
        event_log.observe(cells)

        if stop_event.is_set():
            break

        if (action := scheduler.choose(events=event_log.events, now=time.monotonic())) is not None:
            _execute_action(action=action, forms=cell_fault_forms, rng=rng, event_log=event_log)


@dataclass(frozen=True)
class _SynchronousObserver:
    base_url: str
    cell_types: set[str]
    get_virtual_cells: Callable[[], list[dict]] | None = None

    def observe(self) -> list[dict] | None:
        cells = list_cells(base_url=self.base_url, cell_types=self.cell_types)
        if cells is None:
            return None
        if self.get_virtual_cells is not None:
            cells.extend(self.get_virtual_cells())
        return cells


class SoakActionScheduler:
    def __init__(
        self,
        *,
        rng: random.Random,
        mean_intervals: dict[str, float],
        forms: CellFaultForms,
        injection_enabled: Callable[[], bool] | None = None,
        quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
    ) -> None:
        self._rng = rng
        self._mean_intervals = mean_intervals
        self._forms = forms
        self._injection_enabled = injection_enabled
        self._quiescent_polls_required = quiescent_polls_required

    def initial_schedule(self) -> SoakScheduleEvent:
        return SoakScheduleEvent(
            due_of_type={
                cell_type: _compute_next_injection_time(self._rng, mean_interval_seconds)
                for cell_type, mean_interval_seconds in sorted(self._mean_intervals.items())
            }
        )

    def choose(self, *, events: list[Event], now: float) -> SoakActionRequest | None:
        due_of_type: dict[str, float] = {}
        # Quiescence is derived, not remembered: the largest replica count a kind ever showed, and
        # how many consecutive polls it has looked settled since its last injection attempt.
        max_num_cells_of_type: dict[str, int] = dict.fromkeys(self._mean_intervals, 0)
        quiescent_polls_of_type: dict[str, int] = dict.fromkeys(self._mean_intervals, 0)
        landed_request_ids = {
            event.request_id for event in events if isinstance(event, SoakActionResultEvent) and event.returned
        }
        observation = None
        for event in events:
            if isinstance(event, SoakScheduleEvent):
                due_of_type.update(event.due_of_type)
            elif isinstance(event, SoakActionRequestedEvent):
                # M38 moves the deadline only once an injection lands, and clears the streak on
                # every attempt, so a failed one leaves the kind due again on the next poll.
                if event.request.next_due_at is not None and event.request.request_id in landed_request_ids:
                    due_of_type[cell_type_of(event.request.target)] = event.request.next_due_at
                quiescent_polls_of_type[cell_type_of(event.request.target)] = 0
            elif isinstance(event, (ObservationsEvent, SoakObservation)):
                observation = event
                if event.cells is not None:
                    polled_of_type: dict[str, list[dict]] = {cell_type: [] for cell_type in self._mean_intervals}
                    for cell in event.cells:
                        polled_of_type[cell_type_of(cell)].append(cell)
                    for cell_type, kind_cells in sorted(polled_of_type.items()):
                        max_num_cells_of_type[cell_type] = max(max_num_cells_of_type[cell_type], len(kind_cells))
                        if _kind_is_quiescent(kind_cells, expected_num_cells=max_num_cells_of_type[cell_type]):
                            quiescent_polls_of_type[cell_type] += 1
                        else:
                            quiescent_polls_of_type[cell_type] = 0
        if observation is None or observation.cells is None:
            return None
        cells_of_type: dict[str, list[dict]] = {cell_type: [] for cell_type in self._mean_intervals}
        for cell in observation.cells:
            cells_of_type[cell_type_of(cell)].append(cell)
        due_types = sorted(kind for kind, due_at in due_of_type.items() if now >= due_at)
        if not due_types:
            return None

        # Inject only at a quiescent point: every replica of the kind present and alive for long
        # enough that the readings cannot all be stale. A due kind that is still recovering (or has
        # no spare replica to survive the kill) waits for a later poll.
        ready_types = [
            kind
            for kind in due_types
            if quiescent_polls_of_type[kind] >= self._quiescent_polls_required and len(cells_of_type[kind]) > 1
        ]
        if not ready_types:
            logger.info(
                "Deferring injection: no due cell kind is quiescent with a spare replica (due %s, "
                "quiescent polls %s, replicas %s)",
                due_types,
                {kind: quiescent_polls_of_type[kind] for kind in due_types},
                {kind: len(cells_of_type[kind]) for kind in due_types},
            )
            return None

        cell_type = self._rng.choice(ready_types)
        target = self._rng.choice(cells_of_type[cell_type])
        form = _draw_form(self._forms[cell_type], events=events, cell_type=cell_type, rng=self._rng)
        if self._injection_enabled is not None and not self._injection_enabled():
            return None
        candidates = None
        if isinstance(observation, SoakObservation) and form.name in {"delete_pod", "exec_sigkill"}:
            candidates = observation.pods_of_cell.get(target["metadata"]["name"], [])
            if not candidates:
                return None
        next_due_at = _compute_next_injection_time(self._rng, self._mean_intervals[cell_type])
        pod = self._rng.choice(candidates) if candidates is not None else None
        return SoakActionRequest(
            target=deepcopy(target),
            form_name=form.name,
            harms_cell=form.harms_cell,
            next_due_at=next_due_at,
            pod=pod,
        )


def _execute_action(
    *, action: SoakActionRequest, forms: CellFaultForms, rng: random.Random, event_log: EventLog
) -> None:
    matching = [form for form in forms[cell_type_of(action.target)] if form.name == action.form_name]
    assert len(matching) == 1, f"Expected one form named {action.form_name}, found {len(matching)}"
    form = matching[0]
    cell_name = action.target["metadata"]["name"]
    request = action
    event_log.note_action_requested(request)
    try:
        form.inject(action.target, rng)
    except Exception as error:
        event_log.note_action_result(
            SoakActionResultEvent(request_id=request.request_id, returned=False, error=repr(error))
        )
        logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
        return

    event_log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=True))
    logger.info("Injected fault %s into %s", form.name, cell_name)


def _kind_is_quiescent(kind_cells: list[dict], *, expected_num_cells: int) -> bool:
    if not kind_cells or len(kind_cells) < expected_num_cells:
        return False
    return all(cell_is_alive(cell) for cell in kind_cells)


def _draw_form(
    forms: list[BaseFaultForm], *, events: list[Event], cell_type: str, rng: random.Random
) -> BaseFaultForm:
    worked = compute_successful_form_names(events, cell_type=cell_type)
    unproven = [form for form in forms if form.name not in worked]
    return rng.choice(unproven or forms)


def list_cells(*, base_url: str, cell_types: set[str]) -> list[dict] | None:
    try:
        resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
        resp.raise_for_status()
        return [c for c in resp.json()["items"] if cell_type_of(c) in cell_types]
    except Exception:
        logger.info("Failed to list cells from api server", exc_info=True)
        return None
