import inspect
from types import SimpleNamespace

import pytest

from miles.utils.checkpoint_write_diagnostics import trace_checkpoint_writes


def test_write_diagnostics_preserves_chained_exception_and_signature(caplog):
    def write_item(stream, data, *, serialization_format):
        try:
            raise OSError(28, "No space left on device")
        except OSError as error:
            raise RuntimeError("unexpected pos 704 vs 598") from error

    writer = SimpleNamespace(_write_item=write_item)
    trace_checkpoint_writes(writer)
    assert inspect.signature(writer._write_item) == inspect.signature(write_item)
    with pytest.raises(RuntimeError, match="unexpected pos") as caught:
        writer._write_item(None, None, serialization_format="torch_save")
    assert caught.value.__cause__.errno == 28
    assert "No space left on device" in caplog.text
    assert "unexpected pos" in caplog.text


def test_write_diagnostics_does_not_change_successful_writes():
    calls = []

    def write_item(*args, **kwargs):
        calls.append((args, kwargs))
        return "saved"

    writer = SimpleNamespace(_write_item=write_item)
    trace_checkpoint_writes(writer)
    assert writer._write_item("stream", "tensor", serialization_format="torch_save") == "saved"
    assert calls == [(("stream", "tensor"), {"serialization_format": "torch_save"})]
