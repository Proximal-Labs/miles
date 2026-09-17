from miles.rollout.session.v2.postprocessor_hub.default_postprocess import finalize_samples
from miles.utils.types import Sample


def exclude_superseded(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Keep superseded generations as context while excluding their training loss."""
    excluded = frozenset(
        node["supersedes"] for node in session_metadata["tree"]["nodes"] if node.get("supersedes") is not None
    )
    processed = finalize_samples(leaf_samples, session_metadata, excluded_generation_ids=excluded)
    trainable = [sample for sample in processed if any(sample.loss_mask)]
    session_metadata["selection"] = {
        "policy": "exclude_superseded",
        "excluded_generation_ids": sorted(excluded),
        "trainable_tokens": sum(sum(sample.loss_mask) for sample in trainable),
    }
    return trainable
