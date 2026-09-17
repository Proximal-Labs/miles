from miles.utils.types import Sample

_SERVER_OWNED_METADATA_KEYS = ("accumulated_token_ids", "tito_session_mismatch", "leaf")


def check_input_metadata(agent_metadata: dict) -> dict:
    return {key: value for key, value in agent_metadata.items() if key not in _SERVER_OWNED_METADATA_KEYS}


def assign_reward(samples: list[Sample], trajectory_reward: float) -> None:
    for sample in samples:
        sample.reward = trajectory_reward


def default_postprocess(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Assign shared-token ownership after selection and preserve picker order."""
    return finalize_samples(leaf_samples, session_metadata, excluded_generation_ids=frozenset())


def finalize_samples(
    leaf_samples: list[Sample], session_metadata: dict, *, excluded_generation_ids: frozenset[str]
) -> list[Sample]:
    nodes_by_id = {node["id"]: node for node in session_metadata["tree"]["nodes"]}
    agent_metadata = session_metadata.get("agent") or {}
    trajectory_reward = agent_metadata.get("reward")
    agent_sample_metadata = check_input_metadata(agent_metadata)

    owned_tokens: dict[int, set[int]] = {}
    for sample in sorted(leaf_samples, key=lambda sample: sample.metadata["leaf"]["node_id"]):
        leaf = sample.metadata["leaf"]
        response_start = len(sample.tokens) - sample.response_length
        # Spans index all tokens; `loss_mask` indexes only the response.
        # Clamp when early-stop or truncation shortens the sample.
        for node_id in leaf["path_node_ids"]:
            node = nodes_by_id[node_id]
            completion_start, completion_end = node["completion_span"]
            mask_start = max(completion_start - response_start, 0)
            mask_end = min(completion_end - response_start, sample.response_length)
            owned = owned_tokens.setdefault(node_id, set())
            excluded = node.get("generation_id") in excluded_generation_ids
            for position in range(mask_start, mask_end):
                token_offset = position + response_start - completion_start
                if excluded or token_offset in owned:
                    sample.loss_mask[position] = 0
                elif sample.loss_mask[position]:
                    owned.add(token_offset)
        server_metadata = {key: sample.metadata[key] for key in _SERVER_OWNED_METADATA_KEYS if key in sample.metadata}
        sample.metadata = {**sample.metadata, **agent_sample_metadata, **server_metadata}
    # Skip `assign_reward` to score downstream, or replace the scalar
    # `agent_metadata["reward"]` for finer-grained assignment.
    if trajectory_reward is not None:
        assign_reward(leaf_samples, trajectory_reward)
    return leaf_samples
