"""
tests/test_review_followups.py — two review follow-ups.

  - A call that loses the budget to a concurrent call between its check and its
    reservation raises BudgetExhausted (like every other kill-switch refusal), not
    a raw BudgetExceeded; no network call is made.
  - preflight / truncate / continue on a model with no LiteLLM pricing warn once per
    model that the budget can't be enforced for it.
"""

import asyncio
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from baar import BAARRouter, BudgetExceeded, BudgetExhausted

PATHS = ["chat", "stream_chat", "achat", "astream_chat"]


def call_path(router, path, task="hello", **kw):
    if path == "chat":
        return router.chat(task, **kw)
    if path == "stream_chat":
        return "".join(router.stream_chat(task, **kw))
    if path == "achat":
        return asyncio.run(router.achat(task, **kw))

    async def _collect():
        return "".join([t async for t in router.astream_chat(task, **kw)])
    return asyncio.run(_collect())


def response(model="gpt-4o-mini"):
    r = MagicMock()
    r.model = model
    r.choices[0].message.content = "ok"
    r.choices[0].finish_reason = "stop"
    r.usage.prompt_tokens = 10
    r.usage.completion_tokens = 10
    return r


@pytest.fixture
def cheap():
    with patch("baar.core.budget.cost_per_token", return_value=(1e-6, 1e-6)), \
         patch("baar.core.budget.completion_cost", return_value=2e-6), \
         patch("baar.router.token_counter", return_value=10):
        yield


@pytest.mark.parametrize("path", PATHS)
def test_losing_the_reservation_race_raises_budget_exhausted(cheap, path):
    router = BAARRouter(budget=1.0, use_llm_router=False)
    race = BudgetExceeded(requested=0.01, remaining=0.0, model="(reservation)")
    with patch.object(router._tracker, "check_and_reserve", side_effect=race), \
         patch("baar.router.litellm.completion") as sync_call, \
         patch("baar.router.litellm.acompletion") as async_call:
        with pytest.raises(BudgetExhausted, match="concurrent") as exc:
            call_path(router, path)
    assert exc.value.__cause__ is race
    sync_call.assert_not_called()
    async_call.assert_not_called()


def _unpriced_router(mode):
    return BAARRouter(budget=1.0, use_llm_router=False, small_model="mystery-small",
                      big_model="mystery-big", on_output_overrun=mode, max_output_tokens=500)


@pytest.mark.parametrize("mode", ["preflight", "truncate", "continue"])
def test_unpriced_model_warns_once_in_enforcing_modes(mode):
    router = _unpriced_router(mode)
    with patch("baar.router.litellm.completion", return_value=response("mystery-small")), \
         patch("baar.router.token_counter", return_value=10):
        with pytest.warns(UserWarning, match="can't enforce"):
            router.chat("hello")
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            router.chat("hello")
    assert not [w for w in seen if "can't enforce" in str(w.message)]


def test_unpriced_model_in_estimate_mode_has_no_extra_warning():
    router = _unpriced_router("estimate")
    with patch("baar.router.litellm.completion", return_value=response("mystery-small")), \
         patch("baar.router.token_counter", return_value=10), \
         warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        router.chat("hello")
    assert not [w for w in seen if "can't enforce" in str(w.message)]


def test_priced_model_has_no_extra_warning(cheap):
    router = BAARRouter(budget=1.0, use_llm_router=False, on_output_overrun="preflight", max_output_tokens=500)
    with patch("baar.router.litellm.completion", return_value=response()), \
         warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        router.chat("hello")
    assert not [w for w in seen if "can't enforce" in str(w.message)]
