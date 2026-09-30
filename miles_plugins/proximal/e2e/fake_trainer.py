"""Drive Miles's real platform rollout path with a fake optimizer, for Stage A.

Real: PlatformTaskSource (cursor + consumption ledger checkpoint), PlatformRolloutFn
and Miles's fully async producer, PlatformDataBuffer with the durable store and batch
query, the capture service, snapshot preparation, serving-pool verification, and the
store's policy registry. Fake: training. Every ``publish_every`` steps the driver
publishes the next pre-made adapter as a new policy version, as ModalVolumeTransfer
would after an optimizer step; ``--resume-step`` restores a checkpoint and rewinds
policy history like a resumed trainer.

    python -m miles_plugins.proximal.e2e.fake_trainer --config run.json \\
        --adapters /tmp/adapters --checkpoints /tmp/ckpt --steps 4 --publish local \\
        --yes-rollouts --yes-publish
"""

import argparse
import asyncio
import json
from argparse import Namespace
from pathlib import Path
from typing import Literal, TypedDict

import httpx

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.utils.types import Sample
from miles_plugins.proximal.authorization import AuthorizedRun, authorize_run
from miles_plugins.proximal.buffer import accepted
from miles_plugins.proximal.clients import ServingPoolClient
from miles_plugins.proximal.contracts import Policy, RunConfig, read_run_config
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.modal_volume import authorize_volume_publication, modal_publish_snapshot
from miles_plugins.proximal.rollout import PlatformRolloutFn
from miles_plugins.proximal.snapshot import SnapshotMetadata, prepare_snapshot
from miles_plugins.proximal.store import RolloutStore, open_store

PublishMode = Literal["local", "modal"]


class GroupReport(TypedDict):
    group_id: str
    policy_version: int
    policy_sha256: str
    rewards: list[float]
    tokens: list[int]
    loss_tokens: list[int]
    model_calls: list[int]


class StepReport(TypedDict):
    step: int
    trainer_version: int
    groups: list[GroupReport]


def version_after(step: int, publish_every: int) -> int:
    """Policy version current after ``step`` completes: v1 at startup, +1 per publication."""
    return 1 + (step + 1) // publish_every


class Publisher:
    """The real publication steps minus Megatron: snapshot, upload, verify, commit."""

    def __init__(
        self,
        authorization: AuthorizedRun,
        http: httpx.AsyncClient,
        store: RolloutStore,
        *,
        adapters: list[Path],
        mode: PublishMode,
    ):
        self.authorization, self.http, self.store, self.adapters, self.mode = (
            authorization,
            http,
            store,
            adapters,
            mode,
        )
        self.config: RunConfig = authorization.config

    async def publish(self, version: int) -> Policy:
        adapter = self.adapters[(version - 1) % len(self.adapters)]
        snapshot = prepare_snapshot(
            adapter,
            metadata=SnapshotMetadata(
                run_id=self.config.run_id, checkpoint_iteration=version - 1, base_model=self.config.base_model
            ),
            output_root=self.config.artifact_directory / self.config.run_id / "publication",
        )
        if self.mode == "modal":
            authorization = authorize_volume_publication(self.config.volume, yes_publish=True)
            await asyncio.to_thread(modal_publish_snapshot, authorization, snapshot)
        policy = Policy(
            run_id=self.config.run_id, version=version, snapshot=snapshot.reference, base_model=self.config.base_model
        )
        await ServingPoolClient(self.authorization, self.http).prepare(policy)
        await self.store.commit_policy(policy)
        return policy


def miles_args(config_path: Path, config: RunConfig, checkpoints: Path, *, batch_size: int) -> Namespace:
    """The subset of Miles's parsed arguments the fully async rollout path reads."""
    return Namespace(
        proximal_config=str(config_path),
        proximal_yes_rollouts=True,
        proximal_yes_publish=True,
        rollout_submission_granularity="group",
        n_samples_per_prompt=config.research.group_size,
        async_unused_samples_handler=config.research.unused_groups,
        rollout_sample_filter_path=None,
        rollout_batch_size=batch_size,
        rollout_global_dataset=True,
        async_max_concurrent_samples=config.max_in_flight_samples,
        custom_async_data_buffer_path="miles_plugins.proximal.buffer.PlatformDataBuffer",
        dynamic_sampling_filter_path=None,
        reward_key=None,
        save=str(checkpoints),
        load=str(checkpoints),
    )


def summarize(step: int, version: int, groups: list[list[Sample]]) -> StepReport:
    rows: list[GroupReport] = []
    for group in groups:
        evidence = [accepted(sample) for sample in group]
        rows.append(
            {
                "group_id": evidence[0].attempt.group_id,
                "policy_version": evidence[0].attempt.policy.version,
                "policy_sha256": evidence[0].attempt.policy.snapshot.sha256,
                "rewards": [float(sample.reward) for sample in group],  # type: ignore[arg-type]
                "tokens": [len(sample.tokens) for sample in group],
                "loss_tokens": [sum(sample.loss_mask or []) for sample in group],
                "model_calls": [proof.capture.num_calls for proof in evidence],
            }
        )
    return {"step": step, "trainer_version": version, "groups": rows}


async def run(args: argparse.Namespace) -> list[StepReport]:
    config = read_run_config(args.config)
    authorization = authorize_run(config, yes_rollouts=args.yes_rollouts, yes_publish=args.yes_publish)
    adapters = sorted(path for path in args.adapters.iterdir() if (path / "adapter_config.json").exists())
    if not adapters:
        raise ValueError(f"No adapters under {args.adapters}; run miles_plugins.proximal.e2e.adapters first")
    miles = miles_args(args.config, config, args.checkpoints, batch_size=args.batch_size)
    source = PlatformTaskSource(miles)
    start = 0 if args.resume_step is None else args.resume_step + 1
    source.load(-1 if args.resume_step is None else args.resume_step)
    report = []
    async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as http:
        store = await open_store(config)
        try:
            publisher = Publisher(authorization, http, store, adapters=adapters, mode=args.publish)
            version = 1 if args.resume_step is None else version_after(args.resume_step, args.publish_every)
            # Like the trainer's first publication after (re)start: abandon later history.
            await store.rewind(keep_through=version - 1)
            await publisher.publish(version)
            rollout = PlatformRolloutFn(RolloutFnConstructorInput(args=miles, data_source=source))
            try:
                for step in range(start, args.steps):
                    output = await rollout(RolloutFnTrainInput(rollout_id=step, weight_version=version))
                    assert isinstance(output, RolloutFnTrainOutput)
                    groups = [[sample for sample in group if isinstance(sample, Sample)] for group in output.samples]
                    entry = summarize(step, version, groups)
                    report.append(entry)
                    print(json.dumps(entry), flush=True)
                    source.save(step)  # The checkpoint's task/consumption state.
                    if version_after(step, args.publish_every) > version:
                        version += 1
                        await publisher.publish(version)
            finally:
                await rollout.close()
        finally:
            await store.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2, help="Groups per training batch")
    parser.add_argument("--publish-every", type=int, default=2)
    parser.add_argument("--publish", choices=["local", "modal"], required=True)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--yes-rollouts", action="store_true")
    parser.add_argument("--yes-publish", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
