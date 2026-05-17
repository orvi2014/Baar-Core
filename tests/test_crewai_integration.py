"""
tests/test_crewai_integration.py — Unit tests for baar/integrations/crewai.py

All tests mock the crewai package so they run without crewai installed.
"""

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from baar import BAARRouter
from baar.core.exceptions import BudgetExhausted


# ── crewai stub — must be injected before importing the module under test ──────

class _FakeCrewAILLM:
    """Minimal stand-in for crewai.LLM so BaarCrewLLM has a real base class."""
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


_mock_crewai = MagicMock()
_mock_crewai.LLM = _FakeCrewAILLM
sys.modules.setdefault("crewai", _mock_crewai)

# Now safe to import the integration
from baar.integrations.crewai import BaarCrewLLM, BaarStepCallback, _last_user_content  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return BAARRouter(budget=1.00, small_model="gpt-4o-mini", big_model="gpt-4o")


@pytest.fixture
def llm(router):
    return BaarCrewLLM(router=router)


# ── _last_user_content ────────────────────────────────────────────────────────

class TestLastUserContent:
    def test_returns_last_user_message(self):
        msgs = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second question"},
        ]
        assert _last_user_content(msgs) == "Second question"

    def test_falls_back_to_last_message_when_no_user_role(self):
        msgs = [{"role": "system", "content": "Instructions"}]
        assert _last_user_content(msgs) == "Instructions"

    def test_empty_messages_returns_empty_string(self):
        assert _last_user_content([]) == ""

    def test_non_string_content_is_coerced(self):
        msgs = [{"role": "user", "content": {"text": "complex"}}]
        result = _last_user_content(msgs)
        assert isinstance(result, str)
        assert "complex" in result

    def test_missing_content_key_returns_empty_string(self):
        msgs = [{"role": "user"}]
        assert _last_user_content(msgs) == ""


# ── BaarCrewLLM init ──────────────────────────────────────────────────────────

class TestBaarCrewLLMInit:
    def test_model_defaults_to_small_model(self, router):
        llm = BaarCrewLLM(router=router)
        assert llm.model == "gpt-4o-mini"

    def test_explicit_model_overrides_default(self, router):
        llm = BaarCrewLLM(router=router, model="gpt-4o")
        assert llm.model == "gpt-4o"

    def test_router_is_stored(self, router):
        llm = BaarCrewLLM(router=router)
        assert llm._baar_router is router


# ── BaarCrewLLM budget properties ─────────────────────────────────────────────

class TestBaarCrewLLMBudgetProperties:
    def test_spent_delegates_to_router(self, llm, router):
        with patch.object(type(router), "spent", new_callable=lambda: property(lambda self: 0.42)):
            assert llm.spent == 0.42

    def test_remaining_delegates_to_router(self, llm, router):
        with patch.object(type(router), "remaining", new_callable=lambda: property(lambda self: 0.58)):
            assert llm.remaining == 0.58

    def test_budget_delegates_to_router(self, llm, router):
        assert llm.budget == 1.00

    def test_log_delegates_to_router(self, llm, router):
        sentinel = object()
        with patch.object(type(router), "log", new_callable=lambda: property(lambda self: sentinel)):
            assert llm.log is sentinel


# ── BaarCrewLLM.call (sync) ───────────────────────────────────────────────────

class TestBaarCrewLLMCall:
    def test_routes_last_user_message_as_task(self, llm, router):
        msgs = [
            {"role": "system", "content": "You are a researcher."},
            {"role": "user", "content": "Compare agent frameworks"},
        ]
        with patch.object(router, "chat", return_value="Result text") as mock_chat:
            result = llm.call(msgs)
        mock_chat.assert_called_once_with("Compare agent frameworks", messages=msgs)
        assert result == "Result text"

    def test_passes_full_messages_list_to_router(self, llm, router):
        msgs = [{"role": "user", "content": "Hello"}]
        with patch.object(router, "chat", return_value="Hi") as mock_chat:
            llm.call(msgs)
        _, kwargs = mock_chat.call_args
        assert kwargs["messages"] is msgs

    def test_propagates_budget_exhausted(self, llm, router):
        msgs = [{"role": "user", "content": "Expensive task"}]
        with patch.object(router, "chat", side_effect=BudgetExhausted("No budget", remaining=0.0)):
            with pytest.raises(BudgetExhausted):
                llm.call(msgs)

    def test_tools_and_callbacks_accepted_without_error(self, llm, router):
        msgs = [{"role": "user", "content": "Task"}]
        with patch.object(router, "chat", return_value="ok"):
            result = llm.call(
                msgs,
                tools=[{"name": "search"}],
                callbacks=[MagicMock()],
                available_functions={"search": lambda: None},
            )
        assert result == "ok"

    def test_empty_messages_sends_empty_task(self, llm, router):
        with patch.object(router, "chat", return_value="ok") as mock_chat:
            llm.call([])
        mock_chat.assert_called_once_with("", messages=[])


# ── BaarCrewLLM.acall (async) ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestBaarCrewLLMAcall:
    async def test_routes_last_user_message_async(self, llm, router):
        msgs = [{"role": "user", "content": "Async task"}]
        mock_achat = AsyncMock(return_value="Async result")
        with patch.object(router, "achat", mock_achat):
            result = await llm.acall(msgs)
        mock_achat.assert_awaited_once_with("Async task", messages=msgs)
        assert result == "Async result"

    async def test_propagates_budget_exhausted_async(self, llm, router):
        msgs = [{"role": "user", "content": "Task"}]
        mock_achat = AsyncMock(side_effect=BudgetExhausted("No budget", remaining=0.0))
        with patch.object(router, "achat", mock_achat):
            with pytest.raises(BudgetExhausted):
                await llm.acall(msgs)

    async def test_tools_and_callbacks_accepted_async(self, llm, router):
        msgs = [{"role": "user", "content": "Task"}]
        mock_achat = AsyncMock(return_value="done")
        with patch.object(router, "achat", mock_achat):
            result = await llm.acall(
                msgs,
                tools=[{"name": "search"}],
                callbacks=[MagicMock()],
                available_functions={"search": lambda: None},
            )
        assert result == "done"

    async def test_system_only_messages_falls_back_to_last(self, llm, router):
        msgs = [{"role": "system", "content": "Instructions only"}]
        mock_achat = AsyncMock(return_value="ok")
        with patch.object(router, "achat", mock_achat):
            await llm.acall(msgs)
        mock_achat.assert_awaited_once_with("Instructions only", messages=msgs)


# ── BaarStepCallback ──────────────────────────────────────────────────────────

class TestBaarStepCallback:
    def test_does_nothing_when_budget_remains(self, router):
        guard = BaarStepCallback(router)
        # remaining > 0 — should not raise
        with patch.object(type(router), "remaining", new_callable=lambda: property(lambda self: 0.50)):
            guard(step_output="some output")  # no exception

    def test_raises_budget_exhausted_when_remaining_is_zero(self, router):
        guard = BaarStepCallback(router)
        with patch.object(type(router), "remaining", new_callable=lambda: property(lambda self: 0.0)):
            with pytest.raises(BudgetExhausted):
                guard(step_output="step")

    def test_raises_budget_exhausted_when_remaining_is_negative(self, router):
        guard = BaarStepCallback(router)
        with patch.object(type(router), "remaining", new_callable=lambda: property(lambda self: -0.01)):
            with pytest.raises(BudgetExhausted):
                guard(step_output="step")

    def test_error_message_includes_spent_and_budget(self, router):
        guard = BaarStepCallback(router)
        with patch.object(type(router), "remaining", new_callable=lambda: property(lambda self: 0.0)):
            with patch.object(type(router), "spent", new_callable=lambda: property(lambda self: 1.00)):
                with pytest.raises(BudgetExhausted, match=r"\$1\.0000 spent of \$1\.0000"):
                    guard(step_output="step")

    def test_accepts_any_step_output_type(self, router):
        guard = BaarStepCallback(router)
        with patch.object(type(router), "remaining", new_callable=lambda: property(lambda self: 0.50)):
            guard(step_output=None)
            guard(step_output={"key": "val"})
            guard(step_output=42)

    def test_spent_property_delegates_to_router(self, router):
        guard = BaarStepCallback(router)
        with patch.object(type(router), "spent", new_callable=lambda: property(lambda self: 0.77)):
            assert guard.spent == 0.77

    def test_remaining_property_delegates_to_router(self, router):
        guard = BaarStepCallback(router)
        with patch.object(type(router), "remaining", new_callable=lambda: property(lambda self: 0.23)):
            assert guard.remaining == 0.23


# ── Integration: BaarCrewLLM + BaarStepCallback together ─────────────────────

class TestIntegrationBaarCrewLLMWithStepCallback:
    def test_llm_and_callback_share_router_state(self, router):
        llm  = BaarCrewLLM(router=router)
        guard = BaarStepCallback(router)
        assert llm._baar_router is guard._router

    def test_separate_routers_have_independent_budgets(self):
        router_a = BAARRouter(budget=1.00)
        router_b = BAARRouter(budget=2.00)
        llm_a = BaarCrewLLM(router=router_a)
        llm_b = BaarCrewLLM(router=router_b)
        assert llm_a.budget == 1.00
        assert llm_b.budget == 2.00
