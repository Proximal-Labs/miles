from tests.utils.soak.recovery import compute_recovery_episodes
from tests.utils.soak.state import SoakAdmissionClosedEvent, SoakDeploymentTarget, SoakEvent, SoakObservation
from tests.utils.soak.views import project_actions

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


def assert_tail_complete(events: list[SoakEvent], *, trainer_id: str = "actor") -> None:
    closed = next((event for event in events if isinstance(event, SoakAdmissionClosedEvent)), None)
    assert closed is not None, "Soak injection admission never closed"
    observations = [event for event in events if isinstance(event, SoakObservation)]
    reconfigurations = [
        training_event
        for observation in observations
        for training_event in observation.training_events
        if isinstance(training_event, CellReconfigureEvent)
    ]
    episodes = compute_recovery_episodes(events, reconfigurations=reconfigurations)
    unresolved = [episode.request_ids for episode in episodes if episode.recovered_at is None]
    assert not unresolved, f"Soak tail ended before recovery: {unresolved}"

    actions = project_actions(events)
    unknown = [request_id for request_id, action in actions.items() if action.applied is None]
    assert not unknown, f"Soak tail has actions without confirmed effects: {unknown}"
    unfinished = [
        request_id
        for request_id, action in actions.items()
        if not isinstance(action.requested.request.target, SoakDeploymentTarget) and action.result is None
    ]
    assert not unfinished, f"Soak tail has unfinished actions: {unfinished}"

    recovered_at = max([closed.timestamp, *(episode.recovered_at for episode in episodes)])
    progress = [
        training_event
        for observation in observations
        for training_event in observation.training_events
        if isinstance(training_event, TrainGroupStepEndEvent)
        and isinstance(training_event.source, TrainerControllerProcessIdentity)
        and training_event.source.trainer_id == trainer_id
    ]
    before_close = max((event.rollout_id for event in progress if event.timestamp <= closed.timestamp), default=-1)
    assert any(
        event.timestamp > recovered_at
        and event.rollout_id > before_close
        and any(
            isinstance(outcomes, list) and TrainStepOutcome.NORMAL in outcomes
            for outcomes in event.cell_outcomes.values()
        )
        for event in progress
    ), "Soak tail has no successful training progress after admission closed and faults recovered"
