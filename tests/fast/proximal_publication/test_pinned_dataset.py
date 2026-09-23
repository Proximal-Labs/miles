import time

from miles_plugins.proximal.contracts import Task, TaskDataset, digest, pinned_dataset

SHA = "0123456789abcdef0123456789abcdef01234567"


def _dataset(count: int) -> TaskDataset:
    return TaskDataset(
        project_id=1,
        tasks=tuple(Task(environment_id=i + 1, image_id=1, source_commit_sha=SHA) for i in range(count)),
    )


def test_pinned_dataset_matches_digest_and_membership():
    dataset = _dataset(3)
    pinned = pinned_dataset(dataset)
    assert pinned.sha256 == digest(dataset)
    assert Task(environment_id=2, image_id=1, source_commit_sha=SHA) in pinned.tasks
    assert Task(environment_id=4, image_id=1, source_commit_sha=SHA) not in pinned.tasks


def test_pinned_dataset_is_computed_once_per_dataset_object():
    dataset = _dataset(7473)
    first = pinned_dataset(dataset)
    started = time.perf_counter()
    for _ in range(1000):
        assert pinned_dataset(dataset) is first
    # Per-attempt checks run thousands of times per step; re-serializing a 7,473-task
    # dataset each time (~8 ms) saturated the rollout executor.
    assert time.perf_counter() - started < 0.5
    assert pinned_dataset(_dataset(7473)) is not first  # A different object gets its own entry.
