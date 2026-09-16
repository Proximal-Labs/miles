import pytest

from miles.rollout.session.context_budget import fit_completion_budget


@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens"])
def test_long_history_reduces_output_budget_without_changing_prompt(key):
    request = {key: 65536, "messages": [{"role": "user", "content": "unchanged"}]}
    result = fit_completion_budget(request, prompt_length=50000, context_length=81920)
    assert result[key] == 31920
    assert result["messages"] is request["messages"]
    assert request[key] == 65536


def test_short_prompt_preserves_requested_limit():
    assert fit_completion_budget({"max_tokens": 65536}, prompt_length=1000, context_length=81920)["max_tokens"] == 65536


def test_no_space_rejects_instead_of_truncating_input():
    with pytest.raises(ValueError, match="Input already fills"):
        fit_completion_budget({}, prompt_length=81920, context_length=81920)


def test_budget_is_opt_in():
    request = {"max_tokens": 65536}
    assert fit_completion_budget(request, prompt_length=99999, context_length=None) is request
