from copy import deepcopy

import pytest

from miles_plugins.inkling_eval.standalone import evaluation_environments


def test_separate_suite_uses_exact_pins_without_changing_training_suite():
    original = {'1': {'environmentId': 1, 'imageId': 100, 'sourceCommitSha': 'a' * 40,
                      'imageDigest': 'sha256:' + 'b' * 64}}
    before = deepcopy(original)
    pin = dict(environmentId=49784, imageId=1002839, sourceCommitSha='c' * 40,
               imageDigest='sha256:' + 'd' * 64)
    plan = {'pinned_environments': {'49784': pin}}
    assert evaluation_environments(plan, original, {'deepswe': [49784]}) == {'49784': pin}
    assert original == before
    assert evaluation_environments({}, original, {'original': [1]}) == before
    with pytest.raises(ValueError, match='original pinned suite'):
        evaluation_environments({}, original, {'deepswe': [49784]})


@pytest.mark.parametrize('override', [
    None, {},
    {'49784': {}},
    {'49784': {'environmentId': 49784, 'imageId': 1002839}},
    {'49784': dict(environmentId=49785, imageId=1002839, sourceCommitSha='a' * 40,
                   imageDigest='sha256:' + 'b' * 64)},
    {'49784': dict(environmentId=49784, imageId=0, sourceCommitSha='a' * 40,
                   imageDigest='sha256:' + 'b' * 64)},
    {'49784': dict(environmentId=49784, imageId='1002839', sourceCommitSha='a' * 40,
                   imageDigest='sha256:' + 'b' * 64)},
    {'49784': dict(environmentId=49784, imageId=1002839, sourceCommitSha='main',
                   imageDigest='sha256:' + 'b' * 64)},
    {'49784': dict(environmentId=49784, imageId=1002839, sourceCommitSha='a' * 40,
                   imageDigest='latest')},
    {'49784': dict(environmentId=49784, imageId=1002839, sourceCommitSha='a' * 40,
                   imageDigest='sha256:' + 'b' * 64, deploymentConfig={'modal': {}})},
])
def test_separate_suite_rejects_incomplete_or_unpinned_inputs(override):
    with pytest.raises(ValueError):
        evaluation_environments({'pinned_environments': override}, {}, {'deepswe': [49784]})


def test_prepare_refuses_changed_pins_in_existing_output(tmp_path):
    from miles_plugins.inkling_eval.config import write_json
    from miles_plugins.inkling_eval.standalone import prepare

    source = tmp_path / 'training'
    output = tmp_path / 'evaluation'
    original_suite = {'contract': {'samples_per_epoch': 190, 'batch_size': 32}, 'environments': {}}
    write_json(source / 'evaluation/suite.json', original_suite)
    write_json(source / 'launch.json', {})
    pin = dict(environmentId=49784, imageId=1002839, sourceCommitSha='a' * 40,
               imageDigest='sha256:' + 'b' * 64)
    plan = dict(source_run=str(source), output_dir=str(output), epochs=[2, 5],
                evaluation={'platform_url': 'https://api.example.com', 'sets': {'deepswe': [49784]}},
                pinned_environments={'49784': pin})
    write_json(output / 'manifest.json', dict(plan=plan, environments={'49784': pin},
                                             steps=[12, 30], samples_per_epoch=190))
    manifest_before = (output / 'manifest.json').read_bytes()
    training_before = (source / 'evaluation/suite.json').read_bytes()
    changed_plan = {**plan, 'pinned_environments': {'49784': {**pin, 'imageId': 1002840}}}
    with pytest.raises(ValueError, match='new output directory'):
        prepare(changed_plan)
    assert (output / 'manifest.json').read_bytes() == manifest_before
    assert (source / 'evaluation/suite.json').read_bytes() == training_before
