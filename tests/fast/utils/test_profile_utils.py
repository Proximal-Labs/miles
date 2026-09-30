from types import SimpleNamespace

import pytest

from miles.utils import profile_utils


class _FakeProfiler:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def step(self) -> None:
        self.events.append("step")

    def stop(self) -> None:
        self.events.append("stop")


def _args(*, enabled: bool = True, targets: tuple[str, ...] = ("train_actor",)) -> SimpleNamespace:
    return SimpleNamespace(use_pytorch_profiler=enabled, profile_target=list(targets))


@pytest.fixture
def profiler(monkeypatch: pytest.MonkeyPatch) -> _FakeProfiler:
    fake = _FakeProfiler()
    monkeypatch.setattr(profile_utils, "_create_torch_profiler", lambda args, name: fake)
    return fake


def test_profile_microbatches_steps_between_microbatch_forwards(profiler: _FakeProfiler) -> None:
    calls = []

    def forward_step(data_iterator, model):
        calls.append((data_iterator, model))
        profiler.events.append("forward")
        return data_iterator

    with profile_utils.profile_microbatches(forward_step, _args(), name="train_actor") as wrapped:
        assert [wrapped(i, "model") for i in range(3)] == [0, 1, 2]

    assert calls == [(0, "model"), (1, "model"), (2, "model")]
    # Micro-batch k ends (its backward included) when micro-batch k + 1 begins its forward.
    assert profiler.events == ["start", "forward", "step", "forward", "step", "forward", "stop"]


def test_profile_microbatches_stops_when_the_pass_raises(profiler: _FakeProfiler) -> None:
    with pytest.raises(RuntimeError):
        with profile_utils.profile_microbatches(lambda: None, _args(), name="train_actor"):
            raise RuntimeError("out of memory")

    assert profiler.events == ["start", "stop"]


@pytest.mark.parametrize("args", [_args(enabled=False), _args(targets=("train_overall",))])
def test_profile_microbatches_passes_forward_step_through_when_not_profiled(
    profiler: _FakeProfiler, args: SimpleNamespace
) -> None:
    def forward_step():
        return None

    with profile_utils.profile_microbatches(forward_step, args, name="train_actor") as wrapped:
        assert wrapped is forward_step

    assert profiler.events == []
