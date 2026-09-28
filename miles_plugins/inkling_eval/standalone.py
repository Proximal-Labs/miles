"""Prepare and evaluate saved Inkling checkpoints without a training job."""

import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from miles_plugins.inkling_eval.config import EvalConfig, summarize, write_json
from miles_plugins.inkling_eval.runner import EvaluationRunner
from miles_plugins.inkling_eval.serving import _verify_snapshot, commit_volume


def checkpoint_steps(epochs, samples_per_epoch, batch_size):
    if samples_per_epoch <= 0 or batch_size <= 0 or any(epoch < 0 for epoch in epochs):
        raise ValueError('Epochs must be nonnegative and dataset/batch sizes positive')
    return [math.ceil(epoch * samples_per_epoch / batch_size) for epoch in epochs]


def prepare(plan):
    # Only preparation needs Torch; the evaluation coordinator uses no model tensors.
    from miles_plugins.inkling_eval.offline import export_native_adapter

    source = Path(plan['source_run'])
    root = Path(plan['output_dir'])
    config = EvalConfig(**plan['evaluation'])
    suite = json.loads((source / 'evaluation/suite.json').read_text())
    launch = json.loads((source / 'launch.json').read_text())
    samples = suite['contract']['samples_per_epoch']
    batch = suite['contract']['batch_size']
    steps = checkpoint_steps(plan['epochs'], samples, batch)
    contract = {'plan': plan, 'environments': suite['environments'], 'steps': steps, 'samples_per_epoch': samples}
    path = root / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Evaluation plan changed; use a new output directory')
    if any(str(i) not in suite['environments'] for ids in config.sets.values() for i in ids):
        raise ValueError('Evaluation environments are absent from the original pinned suite')
    # Validate the CPU mapping against a collective export from this exact run.
    reference_step = plan['validation_step']
    reference = source / f'evaluation/step_{reference_step:08d}/adapter'
    _verify_snapshot(reference)
    export_native_adapter(source, reference_step - 1, root / 'validation_adapter', reference=reference)
    for epoch, step in zip(plan['epochs'], steps, strict=True):
        directory = root / f'evaluation/step_{step:08d}'
        adapter = None
        if step:
            existing = source / f'evaluation/step_{step:08d}/adapter'
            if (existing / '.complete').exists():
                _verify_snapshot(existing)
                adapter = str(existing)
            else:
                adapter = export_native_adapter(source, step - 1, directory / 'adapter')
        point_path = directory / 'point.json'
        if not point_path.exists():
            write_json(point_path, {'step': step, 'epoch': step * batch / samples,
                                   'requested_epoch': epoch, 'adapter': adapter,
                                   'status': 'pending', 'results': {}})
        print(f'Prepared epoch {epoch}: step {step}, adapter={adapter}', flush=True)
    write_json(root / 'evaluation-config.json', config.to_dict())
    write_json(root / 'source-launch.json', launch)
    write_json(path, contract)
    commit_volume(plan['environment'])
    return str(path)


def evaluate(plan):
    import wandb

    root = Path(plan['output_dir'])
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['plan'] != plan:
        raise ValueError('Prepare this exact evaluation plan before launching')
    launch = json.loads((root / 'source-launch.json').read_text())
    args = SimpleNamespace(save=str(root), inkling_eval_config=str(root / 'evaluation-config.json'),
                           hf_checkpoint=str(Path(launch['model_dir']) / 'Inkling-Small'),
                           lora_rank=launch['lora_rank'], lora_alpha=launch['lora_alpha'],
                           inkling_eval_image=launch['image'], inkling_eval_environment=plan['environment'])
    runner = EvaluationRunner(args, actor=None, samples_per_epoch=manifest['samples_per_epoch'])
    runner.suite = {'environments': manifest['environments']}
    points = [json.loads((runner.root / f'step_{step:08d}/point.json').read_text())
              for step in manifest['steps']]
    run = wandb.init(entity=launch['wandb_entity'], project=launch['wandb_project'],
                     id=hashlib.sha256(str(root).encode()).hexdigest()[:12], resume='allow',
                     name=root.name, job_type='checkpoint-evaluation', config=plan)
    print(f'W&B: {run.url}', flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=runner.config.max_concurrent_evaluations) as pool:
            futures = {pool.submit(_evaluate_point, runner, point): point for point in points}
            for future in concurrent.futures.as_completed(futures):
                point = future.result()
                metrics = {name: summarize(results) for name, results in point['results'].items()}
                label = f"epoch_{point['requested_epoch']}"
                run.summary[label] = metrics
                run.log({'checkpoint_step': point['step'], 'epoch': point['epoch'],
                         **{f'{label}/{name}/{key}': value for name, result in metrics.items()
                            for key, value in result.items()}})
                print(f'{label}: {metrics}', flush=True)
        write_json(root / 'results.json', {str(point['requested_epoch']): point for point in points})
        commit_volume(plan['environment'])
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    finally:
        runner.executor.shutdown(wait=True)


def _evaluate_point(runner, point):
    if point['status'] == 'complete':
        if not point.get('cleaned_up', False):
            runner._cleanup(point, runner.root / f"step_{point['step']:08d}/point.json")
        return point
    return runner._evaluate(point)
