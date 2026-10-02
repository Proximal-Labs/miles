"""Eight real CPU/Gloo ranks; only Modal and serving HTTP are replaced."""

import json
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import httpx
import modal
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.integration.proximal_async.test_weight_update import (
    CpuAdapterIterator,
    current_version,
    make_updater,
    transfer_args,
)

from miles.utils import distributed_utils
from miles_plugins.proximal import weight_update
from miles_plugins.proximal.adapter_delta import materialize_snapshot
from miles_plugins.proximal.contracts import RunConfig
from miles_plugins.proximal.snapshot import SnapshotReference


class SharedFakeVolume:
    def __init__(self, root: Path, rank: int, fail: bool):
        self.root, self.rank, self.fail = root, rank, fail

    def read_file(self, path):
        yield (self.root / path.lstrip("/")).read_bytes()

    @contextmanager
    def batch_upload(self, force=False):
        assert force is False
        pending = []

        class Upload:
            def put_file(self, source, destination):
                pending.append((source, destination))

        yield Upload()
        if self.fail:
            raise ConnectionError("Injected rank upload failure")
        for source, destination in pending:
            path = self.root / destination.lstrip("/")
            path.parent.mkdir(parents=True, exist_ok=True)
            data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
            with path.open("xb") as stream:
                stream.write(data)
            with (self.root / f"rank-{self.rank}.jsonl").open("a") as log:
                log.write(json.dumps(destination) + "\n")


class ChangedAdapterIterator(CpuAdapterIterator):
    value = 1

    def iter_hf_weights(self, weights, **kwargs):
        for bucket in super().iter_hf_weights(weights, **kwargs):
            yield [
                (name, torch.full((2, 8192) if ".lora_A." in name else (8192, 2), self.value / 16))
                for name, _ in bucket
            ]


def rank_worker(rank, world, args, root, fail_rank, use_delta):
    torch.set_num_threads(1)
    config = RunConfig.model_validate_json(Path(args.proximal_config).read_bytes())
    root = Path(root)
    volume = SharedFakeVolume(root / "volume", rank, rank == fail_rank)
    volume.root.mkdir(exist_ok=True)
    modal.Volume.from_name = lambda *a, **kw: volume
    client_cls = httpx.AsyncClient

    def serving(request):
        body = json.loads(request.content)
        reference = SnapshotReference.model_validate(body["snapshot"])
        materialize_snapshot(volume.root, root / "replica", reference, config.base_model)
        return httpx.Response(
            200,
            json={
                "snapshot": body["snapshot"],
                "base_model": body["base_model"],
                "request_model": f"{config.base_model.name}:miles-{reference.sha256}",
            },
        )

    weight_update.httpx.AsyncClient = lambda **kwargs: client_cls(transport=httpx.MockTransport(serving))
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=world, timeout=timedelta(seconds=45)
    )
    distributed_utils.GLOO_GROUP = dist.group.WORLD
    try:
        updater = make_updater(args, ChangedAdapterIterator)
        updater.connect_rollout_engines([])
        if fail_rank is None:
            updater.update_weights()
            ChangedAdapterIterator.value = 2
            updater.update_weights()
            assert updater.weight_version == 2
            if rank == 0:
                assert updater.protocol._delta_depth == (1 if use_delta else 0)
        else:
            with pytest.raises(RuntimeError, match="shard upload failed"):
                updater.update_weights()
            assert updater.weight_version == 0
            assert not list(volume.root.glob("sharded/*/parts.json"))
            # Retry the same update after restoring the external boundary.
            volume.fail = False
            updater.update_weights()
            assert updater.weight_version == 1
        (root / f"done-{rank}").write_text(str(updater.weight_version))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("use_delta", "fail_rank"), [(False, None), (True, None), (True, 3)])
def test_all_eight_ranks_upload_before_policy_commit(config, tmp_path, use_delta, fail_rank):
    config = config.model_copy(update={"lora_sharded_upload": True, "lora_delta_sync": use_delta})
    args = transfer_args(config, tmp_path)
    mp.spawn(rank_worker, args=(8, args, str(tmp_path), fail_rank, use_delta), nprocs=8, join=True)
    expected = 2 if fail_rank is None else 1
    assert current_version(config) == expected
    assert [int((tmp_path / f"done-{rank}").read_text()) for rank in range(8)] == [expected] * 8
    logs = [
        [json.loads(line) for line in (tmp_path / "volume" / f"rank-{rank}.jsonl").read_text().splitlines()]
        for rank in range(8)
    ]
    assert all(any("/parts/" in path for path in paths) for paths in logs)
    assert all(len(paths) == len(set(paths)) for paths in logs)
    assert sum(path.endswith("parts.json") for paths in logs for path in paths) == expected
    assert all(not path.endswith("parts.json") for paths in logs[1:] for path in paths)
    assert logs[0][-1].endswith("parts.json")
    assert not list((tmp_path / "volume" / "snapshots").glob("*"))
