"""Durable evaluation orchestration; only snapshot export blocks training."""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

from miles_plugins.inkling_eval import serving
from miles_plugins.inkling_eval.config import EvalConfig, evaluation_due, summarize, write_json
from miles_plugins.inkling_eval.platform import Platform

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(self, args, actor, samples_per_epoch):
        self.args = args
        self.actor = actor
        self.samples_per_epoch = samples_per_epoch
        self.config = EvalConfig.read(args.inkling_eval_config)
        if getattr(args, "inkling_eval_rollouts_per_env", None) is not None:
            self.config = replace(self.config, rollouts_per_environment=args.inkling_eval_rollouts_per_env)
        self.root = Path(args.save) / "evaluation"
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.suite = None

    async def start(self):
        from miles.utils.tracking_utils import tracking

        for name in self.config.sets:
            tracking.define_step_key_metric_group(f"eval/{name}", f"eval/{name}/epoch")
        await asyncio.to_thread(self._resolve)
        points = sorted(self.root.glob("step_*/point.json"))
        for path in points:
            point = json.loads(path.read_text())
            if point["step"] > self.args.start_rollout_id:
                raise ValueError("Resume checkpoint predates an evaluation snapshot; resume a newer checkpoint or use a new run")
            if point["status"] == "complete":
                self._log(point)
                if not point.get("cleaned_up", False):
                    await asyncio.to_thread(self._cleanup, point, path)
            else:
                await self._settle()
                self.pending = self.executor.submit(self._evaluate, point)
        if self.args.start_rollout_id == 0:
            baseline = self.root / "step_00000000" / "point.json"
            if not baseline.exists():
                await self._submit(0)
            await self._settle()
        elif not (self.root / "step_00000000" / "point.json").exists():
            raise ValueError("No baseline evaluation exists for this resumed run")

    def _resolve(self):
        path = self.root / "suite.json"
        contract = {
            "config": self.config.to_dict(),
            "base": self.args.hf_checkpoint,
            "rank": self.args.lora_rank,
            "alpha": self.args.lora_alpha,
            "image": self.args.inkling_eval_image,
            "samples_per_epoch": self.samples_per_epoch,
            "batch_size": self.args.rollout_batch_size,
        }
        if path.exists():
            self.suite = json.loads(path.read_text())
            if self.suite["contract"] != contract:
                raise ValueError("Evaluation configuration changed on resume; use a new run ID")
        else:
            platform = Platform(self.config)
            try:
                self.suite = {"contract": contract, "environments": platform.resolve_suite()}
                write_json(path, self.suite)
            finally:
                platform.close()

    async def after_step(self, completed_steps):
        if self.pending is not None and self.pending.done():
            await self._settle()
        if self.due(completed_steps):
            await self._settle()
            await self._submit(completed_steps)

    def due(self, completed_steps):
        return evaluation_due(
            completed_steps,
            self.samples_per_epoch,
            self.args.inkling_eval_every_n_epochs,
            batch_size=self.args.rollout_batch_size,
        )

    async def _submit(self, step):
        directory = self.root / f"step_{step:08d}"
        adapter = directory / "adapter"
        await self.actor.export_hf(step - 1, str(adapter), adapter_only=True)
        point = {
            "step": step,
            "epoch": step * self.args.rollout_batch_size / self.samples_per_epoch,
            "status": "pending",
            "adapter": str(adapter),
            "results": {},
        }
        write_json(directory / "point.json", point)
        await asyncio.to_thread(serving.commit_volume, self.args.inkling_eval_environment)
        self.pending = self.executor.submit(self._evaluate, point)

    async def _settle(self):
        if self.pending is None:
            return
        future, self.pending = self.pending, None
        point = await asyncio.wrap_future(future)
        self._log(point)

    def _log(self, point):
        from miles.utils.tracking_utils import tracking

        for name, results in point["results"].items():
            prefix = f"eval/{name}"
            metrics = {f"{prefix}/{k}": v for k, v in summarize(results).items()}
            metrics[f"{prefix}/epoch"] = point["epoch"]
            metrics[f"{prefix}/checkpoint_step"] = point["step"]
            for environment_id in self.config.sets[name]:
                subset = [r for r in results if r["environment_id"] == environment_id]
                metrics.update({f"{prefix}/environment_{environment_id}/{k}": v for k, v in summarize(subset).items()})
            tracking.log(self.args, metrics, step_key=f"{prefix}/epoch")
            if getattr(self.args, "use_wandb", False):
                self._log_rollout_table(prefix, point["epoch"], results)
        logger.info("Evaluation at epoch %s: %s", point["epoch"], point["results"])

    def _log_rollout_table(self, prefix, epoch, results):
        # W&B tables are optional and cannot be sent to the scalar tracking backends.
        import wandb

        columns = ["environment_id", "replica", "run_id", "run_url", "reward", "status"]
        table = wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in results])
        wandb.log({f"{prefix}/epoch": epoch, f"{prefix}/rollout_results": table})

    async def finish(self):
        try:
            await self._settle()
        finally:
            self.executor.shutdown(wait=True)

    def _evaluate(self, point):
        path = self.root / f"step_{point['step']:08d}" / "point.json"
        identity = f"{self.args.save}:{point['step']}"
        name = "inkling-eval-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        platform = Platform(self.config)
        try:
            deployment = point.get("deployment")
            if deployment is None:
                deployment = serving.deploy(
                    {
                        "base": self.args.hf_checkpoint,
                        "adapter": point["adapter"],
                        "rank": self.args.lora_rank,
                        "tp": self.config.serving_tp,
                        "context_length": self.config.context_length,
                        "concurrency": self.config.max_concurrent_rollouts,
                    },
                    name=name,
                    image=self.args.inkling_eval_image,
                    environment=self.args.inkling_eval_environment,
                    gpu=self.config.serving_gpu,
                )
                point["deployment"] = deployment
                write_json(path, point)
            serving.wait_ready(deployment["url"])
            platform.endpoint(name, {"mode": "dedicated", "baseURL": deployment["url"] + "/v1", "model": serving.WIRE_MODEL})
            self._run_suite(platform, point, path, identity, name)
            point["status"] = "complete"
            write_json(path, point)
            self._cleanup(point, path)
            return point
        except Exception:
            # Keep the endpoint/snapshot available for retry; never kill a live rollout's model.
            logger.exception("Evaluation failed; durable state is at %s", path)
            raise
        finally:
            platform.close()

    def _cleanup(self, point, path):
        name = "inkling-eval-" + hashlib.sha256(f"{self.args.save}:{point['step']}".encode()).hexdigest()[:24]
        platform = Platform(self.config)
        try:
            platform.endpoint(name, None)
            serving.stop(point["deployment"]["app_id"], self.args.inkling_eval_environment)
            point["cleaned_up"] = True
            write_json(path, point)
            serving.commit_volume(self.args.inkling_eval_environment)
        finally:
            platform.close()

    def _run_suite(self, platform, point, path, identity, endpoint):
        work = []
        for name, ids in self.config.sets.items():
            results = point["results"].setdefault(name, [])
            done = {(r["environment_id"], r["replica"]) for r in results}
            for environment_id in ids:
                for replica in range(self.config.rollouts_per_environment):
                    if (environment_id, replica) not in done:
                        work.append((name, environment_id, replica))

        def run(item):
            name, environment_id, replica = item
            result = platform.rollout(
                self.suite["environments"][str(environment_id)],
                identity=f"{identity}:{name}:{environment_id}:{replica}",
                endpoint=endpoint,
            )
            if self.config.platform_ui_url:
                result["run_url"] = f"{self.config.platform_ui_url.rstrip('/')}/environment/{environment_id}/run/{result['run_id']}"
            return name, {**result, "environment_id": environment_id, "replica": replica}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_concurrent_rollouts) as pool:
            futures = [pool.submit(run, item) for item in work]
            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                point["results"][name].append(result)
                write_json(path, point)
