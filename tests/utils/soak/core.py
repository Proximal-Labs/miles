# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import requests
from tests.utils.soak.fault_forms import BaseFaultForm, CellFaultForms
from tests.utils.soak.state import (
    Event,
    EventLog,
    SoakActionRequest,
    SoakActionResultEvent,
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
    observer = SoakObserver(
        base_url=base_url, cell_types=set(mean_interval_seconds_of_cell_type), get_virtual_cells=get_virtual_cells
    )
    scheduler = SoakActionScheduler(
        rng=rng,
        mean_intervals=mean_interval_seconds_of_cell_type,
        forms=cell_fault_forms,
        injection_enabled=injection_enabled,
        quiescent_polls_required=quiescent_polls_required,
    )

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

        if (action := scheduler.choose(cells=cells, events=event_log.events)) is not None:
            injected = _execute_action(action=action, rng=rng, event_log=event_log)
            scheduler.note_attempt(cell_type_of(action.target), injected=injected)


@dataclass(frozen=True)
class SoakObserver:
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
        self._next_due = {
            cell_type: _compute_next_injection_time(rng, mean_interval_seconds)
            for cell_type, mean_interval_seconds in sorted(mean_intervals.items())
        }
        self._quiescent_polls_required = quiescent_polls_required
        self._quiescent_polls = dict.fromkeys(self._next_due, 0)
        self._max_num_cells = dict.fromkeys(self._next_due, 0)

    def choose(self, *, cells: list[dict], events: list[Event]) -> "_SelectedAction | None":
        cells_of_type: dict[str, list[dict]] = {cell_type: [] for cell_type in self._next_due}
        for cell in cells:
            cells_of_type[cell_type_of(cell)].append(cell)
        for cell_type, kind_cells in sorted(cells_of_type.items()):
            self._max_num_cells[cell_type] = max(self._max_num_cells[cell_type], len(kind_cells))
            if _kind_is_quiescent(kind_cells, expected_num_cells=self._max_num_cells[cell_type]):
                self._quiescent_polls[cell_type] += 1
            else:
                self._quiescent_polls[cell_type] = 0

        now: float = time.monotonic()
        due_types = sorted(kind for kind, due_at in self._next_due.items() if now >= due_at)
        if not due_types:
            return None

        # Inject only at a quiescent point: every replica of the kind present and alive for long
        # enough that the readings cannot all be stale. A due kind that is still recovering (or has
        # no spare replica to survive the kill) waits for a later poll.
        ready_types = [
            kind
            for kind in due_types
            if self._quiescent_polls[kind] >= self._quiescent_polls_required and len(cells_of_type[kind]) > 1
        ]
        if not ready_types:
            logger.info(
                "Deferring injection: no due cell kind is quiescent with a spare replica (due %s, "
                "quiescent polls %s, replicas %s)",
                due_types,
                {kind: self._quiescent_polls[kind] for kind in due_types},
                {kind: len(cells_of_type[kind]) for kind in due_types},
            )
            return None

        cell_type = self._rng.choice(ready_types)
        target = self._rng.choice(cells_of_type[cell_type])
        form = _draw_form(self._forms[cell_type], events=events, cell_type=cell_type, rng=self._rng)
        if self._injection_enabled is not None and not self._injection_enabled():
            return None
        return _SelectedAction(target=target, form=form)

    def note_attempt(self, cell_type: str, *, injected: bool) -> None:
        self._quiescent_polls[cell_type] = 0
        if injected:
            self._next_due[cell_type] = _compute_next_injection_time(self._rng, self._mean_intervals[cell_type])


@dataclass(frozen=True)
class _SelectedAction:
    target: dict
    form: BaseFaultForm


def _execute_action(*, action: _SelectedAction, rng: random.Random, event_log: EventLog) -> bool:
    form = action.form
    cell_name = action.target["metadata"]["name"]
    request = SoakActionRequest(target=action.target, form_name=form.name, harms_cell=form.harms_cell)
    event_log.note_action_requested(request)
    try:
        form.inject(action.target, rng)
    except Exception as error:
        event_log.note_action_result(
            SoakActionResultEvent(request_id=request.request_id, returned=False, error=repr(error))
        )
        logger.info("Failed to inject fault %s into %s", form.name, cell_name, exc_info=True)
        return False

    event_log.note_action_result(SoakActionResultEvent(request_id=request.request_id, returned=True))
    logger.info("Injected fault %s into %s", form.name, cell_name)
    return True


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
