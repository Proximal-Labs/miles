import json
from types import SimpleNamespace

import torch

from miles_plugins.inkling_eval.offline import _merge_parameter
from miles_plugins.inkling_eval.serving import BASE_MODEL, server_command
from miles_plugins.inkling_eval.standalone import checkpoint_steps, evaluate


def test_requested_epochs_use_consumed_samples():
    assert checkpoint_steps([0, 2, 5], 190, 32) == [0, 12, 30]


def test_base_serving_has_no_adapter_flags():
    settings = dict(base='/base', adapter=None, tp=8, context_length=1048576, concurrency=128)
    command = server_command(settings)
    assert not any('lora' in token for token in command)
    assert command[command.index('--served-model-name') + 1] == BASE_MODEL


def test_fused_gate_up_export_keeps_global_gate_before_up():
    name = 'module.module.decoder.layers.0.mlp.lora_adapter.fc1_B'
    shards = [{name: torch.tensor([[1], [2], [11], [12]])},
              {name: torch.tensor([[3], [4], [13], [14]])}]
    target, tensor = _merge_parameter(name, shards, 21, 100)
    assert target == 'language_model.layers.21.mlp.gate_up_proj.lora_B.weight'
    assert tensor.flatten().tolist() == [1, 2, 3, 4, 11, 12, 13, 14]


def test_shared_experts_export_keeps_experts_before_tp_shards():
    name = 'module.decoder.layers.0.mlp.shared_experts.lora_adapter.w1_B'
    shards = [{name: torch.tensor([[[1], [2]], [[11], [12]]])},
              {name: torch.tensor([[[3], [4]], [[13], [14]]])}]
    _, tensor = _merge_parameter(name, shards, 0, 100)
    assert tensor.flatten().tolist() == [1, 2, 3, 4, 11, 12, 13, 14]


def test_standalone_dispatches_only_selected_snapshots(tmp_path, monkeypatch):
    import sys
    from miles_plugins.inkling_eval import standalone
    from miles_plugins.inkling_eval.config import write_json

    plan = {'output_dir': str(tmp_path), 'environment': 'main'}
    write_json(tmp_path / 'manifest.json', {'plan': plan, 'steps': [0, 12, 30],
                                          'samples_per_epoch': 190, 'environments': {}})
    write_json(tmp_path / 'source-launch.json', dict(model_dir='/models', lora_rank=32,
               lora_alpha=32, image='image', wandb_entity='entity', wandb_project='project'))
    write_json(tmp_path / 'evaluation-config.json', dict(platform_url='https://api.example.com',
               sets={'test_set_50': [1]}, rollouts_per_environment=8))
    for epoch, step in zip([0, 2, 5], [0, 12, 30]):
        write_json(tmp_path / f'evaluation/step_{step:08d}/point.json',
                   dict(step=step, requested_epoch=epoch, epoch=step * 32 / 190,
                        adapter=None if step == 0 else f'/adapter/{step}', status='pending', results={}))
    seen = []
    def run_point(self, point):
        seen.append((point['step'], point['adapter'], self.config.rollouts_per_environment))
        point['results'] = {'test_set_50': [{'reward': 1}] * 8}
        point['status'] = 'complete'
        return point
    monkeypatch.setattr(standalone.EvaluationRunner, '_evaluate', run_point)
    monkeypatch.setattr(standalone, 'commit_volume', lambda env: None)
    run = SimpleNamespace(url='wandb', summary={}, log=lambda row: None, finish=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kwargs: run))
    evaluate(plan)
    assert sorted(seen) == [(0, None, 8), (12, '/adapter/12', 8), (30, '/adapter/30', 8)]
    assert set(json.loads((tmp_path / 'results.json').read_text())) == {'0', '2', '5'}
    assert set(run.summary) == {'epoch_0', 'epoch_2', 'epoch_5'}


def test_base_evaluation_routes_to_base_model(tmp_path, monkeypatch):
    from miles_plugins.inkling_eval import runner as module
    from miles_plugins.inkling_eval.config import write_json

    config = tmp_path / 'eval.json'
    write_json(config, dict(platform_url='https://api.example.com', sets={'test': [1]}))
    args = SimpleNamespace(inkling_eval_config=str(config), save=str(tmp_path), hf_checkpoint='/base',
                           lora_rank=32, inkling_eval_image='image', inkling_eval_environment='main')
    seen = {}
    class Platform:
        def __init__(self, config):
            pass
        def endpoint(self, name, value):
            seen['model'] = value['model']
        def close(self):
            pass
    monkeypatch.setattr(module, 'Platform', Platform)
    monkeypatch.setattr(module.serving, 'deploy', lambda settings, **kwargs: {'app_id': 'app', 'url': 'https://eval'})
    monkeypatch.setattr(module.serving, 'wait_ready', lambda url, **kwargs: seen.update(kwargs))
    runner = module.EvaluationRunner(args, actor=None, samples_per_epoch=190)
    monkeypatch.setattr(runner, '_run_suite', lambda *args: None)
    monkeypatch.setattr(runner, '_cleanup', lambda *args: None)
    try:
        result = runner._evaluate(dict(step=0, adapter=None, results={}))
    finally:
        runner.executor.shutdown()
    assert result['status'] == 'complete'
    assert seen == {'model_id': BASE_MODEL, 'model': BASE_MODEL}
