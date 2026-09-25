"""Publish retained sealed captures independently of rollout generation and training."""

import argparse
import asyncio
import logging
from pathlib import Path

import httpx

from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.clients import PlatformClient
from miles_plugins.proximal.contracts import AcceptedAttempt, read_run_config
from miles_plugins.proximal.store import RolloutStore, open_store

logger = logging.getLogger(__name__)


async def publish_pending(store: RolloutStore, platform: PlatformClient) -> int:
    pending = await store.pending_captures()

    async def publish(entry: tuple[AcceptedAttempt, Path]) -> None:
        evidence, path = entry
        attempt_id = evidence.attempt.attempt_id
        try:
            if not path.exists():
                await asyncio.to_thread(store.sync.reload)
            await platform.archive_capture(evidence, path)
            await store.finish_capture_upload(attempt_id)
            logger.info("Capture archive published for attempt %s (%s bytes)", attempt_id, path.stat().st_size)
        except Exception as exc:
            # Never include signed URLs/headers in diagnostics.
            logger.warning("Capture archive pending for attempt %s (%s)", attempt_id, type(exc).__name__)
            await store.retry_capture_upload(attempt_id, type(exc).__name__)

    await asyncio.gather(*(publish(entry) for entry in pending))
    return len(pending)


async def publish_loop(store: RolloutStore, platform: PlatformClient) -> None:
    while True:
        try:
            await publish_pending(store, platform)
        except Exception as exc:
            logger.error("Capture archive worker will retry (%s)", type(exc).__name__)
        await asyncio.sleep(2)


async def replay(config_path: str, *, yes_rollouts: bool, yes_publish: bool, watch: bool) -> int:
    config = read_run_config(config_path)
    authorization = authorize_run(config, yes_rollouts=yes_rollouts, yes_publish=yes_publish)
    store = await open_store(config)
    try:
        async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
            platform = PlatformClient(authorization, client)
            if watch:
                await publish_loop(store, platform)
            else:
                while await publish_pending(store, platform):
                    pass
            remaining = await store.pending_capture_count()
            logger.info("Capture archive replay finished with %s uploads still pending", remaining)
            return remaining
    finally:
        await store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay retained capture uploads; never starts rollouts or training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--yes-rollouts", action="store_true", help="Authorize access to the existing run")
    parser.add_argument("--yes-publish", action="store_true", help="Authorize archive publication")
    parser.add_argument("--watch", action="store_true", help="Continue retrying pending uploads with backoff")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    remaining = asyncio.run(
        replay(args.config, yes_rollouts=args.yes_rollouts, yes_publish=args.yes_publish, watch=args.watch)
    )
    raise SystemExit(1 if remaining else 0)
