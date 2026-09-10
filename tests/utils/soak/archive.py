import asyncio
import json
import sys
from pathlib import Path
from typing import get_args

from tests.utils.soak.action import run_command
from tests.utils.soak.state import EventLog, SoakCollectionClosedEvent, SoakEvent

_COLLECTION_TIMEOUT_SECONDS = 180.0


def collect_events(*, path: Path, events: list[SoakEvent]) -> list[SoakEvent]:
    payload = json.dumps(
        {
            "path": str(path),
            "events": [
                {"event_type": type(event).__name__, "event": event.model_dump(mode="json")} for event in events
            ],
        }
    )
    try:
        result = asyncio.run(
            run_command(
                [sys.executable, "-m", "tests.utils.soak.archive"],
                timeout_seconds=_COLLECTION_TIMEOUT_SECONDS,
                stdin_data=payload,
                check=False,
            )
        )
    except TimeoutError as error:
        raise TimeoutError(f"Soak evidence collection exceeded {_COLLECTION_TIMEOUT_SECONDS}s: {path}") from error
    if result.returncode != 0:
        raise RuntimeError(f"Soak evidence collection failed ({result.returncode}): {result.stderr}")
    collected = _parse_events(json.loads(result.stdout))
    assert collected[: len(events)] == events, "Collector changed the recorded soak events"
    assert collected and isinstance(collected[-1], SoakCollectionClosedEvent), "Collector did not close evidence"
    return collected


def _main() -> None:
    payload = json.load(sys.stdin)
    event_log = EventLog()
    event_log._path = Path(payload["path"])
    event_log._events = _parse_events(payload["events"])
    event_log._finish_collection()
    json.dump(
        [{"event_type": type(event).__name__, "event": event.model_dump(mode="json")} for event in event_log.events],
        sys.stdout,
    )


def _parse_events(values: list[dict]) -> list[SoakEvent]:
    event_types = {event_type.__name__: event_type for event_type in get_args(SoakEvent)}
    return [event_types[value["event_type"]].model_validate(value["event"]) for value in values]


if __name__ == "__main__":
    _main()
