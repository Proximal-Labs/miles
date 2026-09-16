from miles.utils.types import Sample


def keep_all(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Keep every leaf: sibling order cannot distinguish retries from parallel work."""
    return list(leaf_samples)
