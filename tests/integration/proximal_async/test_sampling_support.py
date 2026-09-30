"""Sampling-support replay (Miles TITO on CPU; only the engine is scripted).

A run whose contract declares ``sampling_support`` logprobs samples with top-p/top-k.
Capture asks SGLang for each generated token's surviving set, seals it with the
sample, and every later reader keeps it, so the trainer renormalizes over the same set.
"""

import json
import os

import httpx
import pytest
from tests.integration.proximal_async.test_buffer import entry, sample_for
from tests.integration.proximal_async.test_capture import PLATFORM, scripted_engine

from miles.rollout.session.samples.codec import COMPUTED_FIELDS, decode_samples_and_merge_input_sample
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample
from miles_plugins.proximal.buffer import validate_sample
from miles_plugins.proximal.capture_server import CaptureServer, EngineEndpoint, capture_tokenizer
from miles_plugins.proximal.clients import CaptureClient
from miles_plugins.proximal.contracts import RunConfig, Sampling, replays_sampling_support
from miles_plugins.proximal.preflight import canary
from miles_plugins.proximal.store import sample_fields

TOP_P, TOP_K = 0.95, 20
# sample_for's two generated tokens (2 and 3), each with a two-token support.
SUPPORTS = [[2, 5], [3, 6]]


def with_sampling(base: Sampling, **update: object) -> Sampling:
    return Sampling.model_validate({**base.model_dump(mode="json"), **update})


def supports(mask: RolloutSamplingMask) -> list[list[int]]:
    ids, offsets = mask._as_tensors()
    return [ids[offsets[i] : offsets[i + 1]].tolist() for i in range(len(mask))]


@pytest.fixture
def config(config: RunConfig) -> RunConfig:
    """The shared test run, sampling with top-p/top-k and replaying the support."""
    data = json.loads(config.model_dump_json())
    data["research"]["sampling"].update(top_p=TOP_P, top_k=TOP_K, logprob_semantics="sampling_support")
    return RunConfig.model_validate_json(json.dumps(data))


@pytest.fixture
def tokenizer(config):
    return capture_tokenizer(os.environ["PROXIMAL_TEST_TOKENIZER"], config.tito_model)


def with_support(engine):
    """SGLang's reply to ``return_sampling_mask``: each generated token's surviving set and its logprob."""

    def reply(request: httpx.Request) -> httpx.Response:
        response = engine(request)
        if request.url.path == "/policies/prepare" or not json.loads(request.content).get("return_sampling_mask"):
            return response
        body = response.json()
        meta = body["choices"][0]["meta_info"]
        ids = [item[1] for item in meta["output_token_logprobs"]]
        meta["output_token_sampling_mask"] = [[token, token + 1] for token in ids]
        meta["output_token_sampling_logprobs"] = [-0.1] * len(ids)
        headers = {name: response.headers[name] for name in ("x-proximal-policy-sha256", "x-proximal-base-revision")}
        return httpx.Response(200, headers=headers, json=body)

    return reply


def replica(authorization, policy, tokenizer, root, engine) -> CaptureServer:
    """A capture composed like a replica's, in front of ``engine``."""

    async def admits(candidate):
        return candidate.snapshot == policy.snapshot

    return CaptureServer(
        authorization,
        tokenizer=tokenizer,
        engine=EngineEndpoint(
            client=httpx.AsyncClient(transport=httpx.MockTransport(engine)),
            url="http://replica-gateway",
            headers={"Authorization": "Bearer g"},
        ),
        policy_known=admits,
        root=root,
    )


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"top_k": -1}, "positive top_k"),
        ({"temperature": 0.7}, "temperature=1"),
        ({"logprob_semantics": "untransformed"}, "top_p=1, top_k=-1"),
    ],
)
def test_the_contract_pins_what_a_logprob_means(config, update, error):
    replay = config.research.sampling
    assert replays_sampling_support(replay)
    assert replays_sampling_support(with_sampling(replay, top_p=1.0))  # top-k alone also filters.
    assert not replays_sampling_support(with_sampling(replay, top_p=1.0, top_k=-1, logprob_semantics="untransformed"))
    with pytest.raises(ValueError, match=error):
        with_sampling(replay, **update)


async def test_capture_seals_each_generated_tokens_support(config, authorization, policy, attempt, tokenizer, store):
    requests = []
    engine = with_support(scripted_engine(config, policy, tokenizer, requests))
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as backend:
        server = CaptureServer.beside_trainer(authorization, tokenizer=tokenizer, client=backend, store=store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
            client = CaptureClient(authorization, http)
            await store.commit_policy(policy)
            handle = await client.create(attempt)
            url = handle.base_url + "/chat/completions"
            messages = [{"role": "user", "content": "Inspect this feature."}]
            first = await http.post(
                url, headers=PLATFORM, json={"model": config.base_model.name, "messages": messages}
            )
            assert first.status_code == 200, first.text
            # The agent gets an ordinary reply; the support stays with capture.
            assert "output_token_sampling_mask" not in first.text
            messages += [first.json()["choices"][0]["message"], {"role": "user", "content": "Verify the result."}]
            second = await http.post(
                url, headers=PLATFORM, json={"model": config.base_model.name, "messages": messages}
            )
            assert second.status_code == 200, second.text
            _, payload = await client.collect(handle, attempt)
            await client.release(handle)

    assert len(requests) == 2
    for sent in requests:
        assert (sent["return_sampling_mask"], sent["temperature"], sent["top_p"], sent["top_k"]) == (
            True,
            1.0,
            TOP_P,
            TOP_K,
        )
    fields = sample_fields(COMPUTED_FIELDS, attempt.sampling)
    [sample] = decode_samples_and_merge_input_sample(payload, Sample(), fields=fields).samples
    assert sample.rollout_sampling_mask is not None
    rows = supports(sample.rollout_sampling_mask)
    assert len(rows) == sample.response_length
    response = sample.tokens[-sample.response_length :]
    for position, (token, kept) in enumerate(zip(response, sample.loss_mask, strict=True)):
        if kept:  # Generated: the engine's support, and the logprob renormalized over it.
            assert rows[position] == [token, token + 1] and sample.rollout_log_probs[position] == -0.1
        else:  # The user turn between the calls was not sampled: a singleton support.
            assert rows[position] == [token]


async def test_an_engine_without_support_fails_the_seal_with_its_reason(
    config, authorization, policy, attempt, tokenizer, store
):
    engine = scripted_engine(config, policy, tokenizer, [])
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as backend:
        server = CaptureServer.beside_trainer(authorization, tokenizer=tokenizer, client=backend, store=store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
            client = CaptureClient(authorization, http)
            await store.commit_policy(policy)
            handle = await client.create(attempt)
            messages = [{"role": "user", "content": "Inspect this feature."}]
            reply = await http.post(
                handle.base_url + "/chat/completions",
                headers=PLATFORM,
                json={"model": config.base_model.name, "messages": messages},
            )
            assert reply.status_code == 200, reply.text
            with pytest.raises(httpx.HTTPStatusError, match="output_token_sampling_mask"):
                await client.collect(handle, attempt)


@pytest.mark.parametrize("returns_support", [True, False])
async def test_the_canary_requires_the_engine_to_return_the_support(
    authorization, policy, tokenizer, tmp_path, config, returns_support
):
    engine = scripted_engine(config, policy, tokenizer, [])
    server = replica(
        authorization, policy, tokenizer, tmp_path / "capture", with_support(engine) if returns_support else engine
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
        if returns_support:
            assert (await canary(authorization, http, policy))["sealed_tokens"] > 0
        else:
            with pytest.raises(httpx.HTTPStatusError, match="output_token_sampling_mask"):
                await canary(authorization, http, policy)


async def test_stored_groups_keep_the_support(attempt, policy, store):
    await store.commit_policy(policy)
    group = entry(attempt, policy, group="g").group
    for sample in group:
        sample.rollout_sampling_mask = RolloutSamplingMask.from_mask_list(SUPPORTS)
    await store.add_group("g", policy, group)
    [row] = await store.select(min_version=1, max_version=1, exclude=[], limit=1, filter_path=None)
    _, loaded = await store.load(row)
    assert [supports(sample.rollout_sampling_mask) for sample in loaded] == [SUPPORTS, SUPPORTS]


def test_a_sample_carries_the_support_exactly_when_its_contract_replays_it(attempt):
    sample, proof = sample_for(attempt)
    with pytest.raises(ValueError, match="Sampling support presence"):
        validate_sample(sample, proof)
    sample.rollout_sampling_mask = RolloutSamplingMask.from_mask_list(SUPPORTS)
    validate_sample(sample, proof)

    unfiltered = with_sampling(attempt.sampling, top_p=1.0, top_k=-1, logprob_semantics="untransformed")
    sample, proof = sample_for(attempt.model_copy(update={"sampling": unfiltered}))
    validate_sample(sample, proof)
    sample.rollout_sampling_mask = RolloutSamplingMask.from_mask_list(SUPPORTS)
    with pytest.raises(ValueError, match="Sampling support presence"):
        validate_sample(sample, proof)
