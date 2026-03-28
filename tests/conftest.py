"""
tests/conftest.py — shared fixtures for all test files.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_litellm_response():
    """Factory fixture: returns a function that builds mock LiteLLM responses."""
    def _make(content="test response", model="gpt-4o-mini",
               prompt_tokens=50, completion_tokens=30):
        resp = MagicMock()
        resp.model = model
        resp.choices[0].message.content = content
        resp.usage.prompt_tokens = prompt_tokens
        resp.usage.completion_tokens = completion_tokens
        return resp
    return _make


@pytest.fixture
def cheap_mock_patches():
    """
    Patch the three external calls that need API keys.
    Provides: completion_cost=0.000025, cost_per_token=(tiny, tiny), token_counter=50.
    Usage: use as context manager or inject via autouse.
    """
    with patch("baar.core.budget.completion_cost", return_value=0.000025), \
         patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002)), \
         patch("baar.router.token_counter", return_value=50):
        yield


@pytest.fixture
def router_no_api():
    """
    A BAARRouter with:
    - use_llm_router=False  (heuristic only, no OpenAI calls for routing)
    - All LiteLLM calls mocked out
    """
    from baar import BAARRouter

    resp = MagicMock()
    resp.model = "gpt-4o-mini"
    resp.choices[0].message.content = "mock answer"
    resp.usage.prompt_tokens = 50
    resp.usage.completion_tokens = 30

    with patch("baar.router.litellm.completion", return_value=resp), \
         patch("baar.core.budget.completion_cost", return_value=0.000025), \
         patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002)), \
         patch("baar.router.token_counter", return_value=50):
        yield BAARRouter(budget=1.00, use_llm_router=False)
