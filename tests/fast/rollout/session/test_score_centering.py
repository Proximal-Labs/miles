"""Session configuration must distinguish training calls from evaluation."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.core import SessionCore, prepare_chat_request
from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2


@pytest.mark.parametrize(
    "registry_type,core_type", [(SessionRegistry, SessionCore), (SessionRegistryV2, SessionCoreV2)]
)
def test_score_centering_training_and_evaluation_sessions(registry_type: type, core_type: type) -> None:
    config = make_session_server_config(loss_type="score_centering", rollout_temperature=0.7)
    tokenizer = SimpleNamespace(create_comparator=lambda: None, chat_template_kwargs={})
    registry = registry_type(tokenizer=None, tito_tokenizer=tokenizer)
    core = core_type(None, registry, config)
    for evaluation in (False, True):
        response = asyncio.run(core.create_session(evaluation=evaluation))
        session = registry.get_session(json.loads(response.body)["session_id"])
        assert session.evaluation is evaluation
        if evaluation:
            request, _, _ = prepare_chat_request(
                b'{"temperature":0}', config, tokenizer, evaluation=session.evaluation
            )
            assert request["temperature"] == 0 and "top_logprobs" not in request
        else:
            request, _, _ = prepare_chat_request(b"{}", config, tokenizer, evaluation=session.evaluation)
            assert request["top_logprobs"] == 128 and request["temperature"] == 0.7
            with pytest.raises(MessageValidationError, match="temperature"):
                prepare_chat_request(b'{"temperature":0}', config, tokenizer, evaluation=session.evaluation)
