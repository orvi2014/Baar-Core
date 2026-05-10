"""
Tests for baar.integrations.langchain — no real API calls, no OPENAI_API_KEY needed.
"""

import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration, ChatResult

from baar import BAARRouter, BudgetExhausted
from baar.integrations.langchain import (
    BaarCallbackHandler,
    BaarChatModel,
    _messages_to_dicts,
    _last_human_content,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_router(budget: float = 0.10) -> BAARRouter:
    return BAARRouter(budget=budget, use_llm_router=False)


# ── _messages_to_dicts ────────────────────────────────────────────────────────

def test_messages_to_dicts_basic():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    dicts = _messages_to_dicts(msgs)
    assert dicts == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def test_last_human_content():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hello")]
    assert _last_human_content(msgs) == "hello"


def test_last_human_content_no_human():
    msgs = [SystemMessage(content="sys")]
    assert _last_human_content(msgs) == "sys"


# ── BaarCallbackHandler ───────────────────────────────────────────────────────

class TestBaarCallbackHandler:
    def test_instantiation(self):
        handler = BaarCallbackHandler(make_router())
        assert handler._router is not None

    def test_on_llm_start_passes_when_budget_ok(self):
        router = make_router(budget=10.0)
        handler = BaarCallbackHandler(router)
        # Should not raise with a healthy budget
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o-mini"}},
            prompts=["Hello world"],
        )

    def test_on_llm_start_raises_when_budget_exhausted(self):
        router = make_router(budget=0.000001)
        handler = BaarCallbackHandler(router)
        with pytest.raises(BudgetExhausted):
            handler.on_llm_start(
                serialized={"kwargs": {"model_name": "gpt-4o-mini"}},
                prompts=["Hello world"],
            )

    def test_on_chat_model_start_passes_when_budget_ok(self):
        router = make_router(budget=10.0)
        handler = BaarCallbackHandler(router)
        handler.on_chat_model_start(
            serialized={"kwargs": {"model_name": "gpt-4o-mini"}},
            messages=[[HumanMessage(content="hi")]],
        )

    def test_on_llm_end_records_spend(self):
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        handler._pending_model = "gpt-4o-mini"

        result = LLMResult(
            generations=[],
            llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        )
        before = router.spent
        handler.on_llm_end(result)
        assert router.spent > before

    def test_on_llm_end_noop_without_usage(self):
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        result = LLMResult(generations=[], llm_output={})
        before = router.spent
        handler.on_llm_end(result)
        assert router.spent == before


# ── BaarChatModel ─────────────────────────────────────────────────────────────

class TestBaarChatModel:
    def test_instantiation(self):
        router = make_router()
        llm = BaarChatModel(router=router)
        assert llm._llm_type == "baar"
        assert llm.small_model == router.small_model
        assert llm.budget == router._tracker.total_budget

    def test_identifying_params(self):
        router = make_router(budget=0.05)
        llm = BaarChatModel(router=router)
        params = llm._identifying_params
        assert params["budget_usd"] == 0.05
        assert "remaining_usd" in params

    def test_generate_calls_router_chat(self):
        router = make_router()
        llm = BaarChatModel(router=router)
        messages = [HumanMessage(content="What is 2+2?")]

        with patch.object(router, "chat", return_value="4") as mock_chat:
            result = llm._generate(messages)
            mock_chat.assert_called_once()
            assert result.generations[0].message.content == "4"

    def test_generate_raises_budget_exhausted(self):
        router = make_router(budget=0.000001)
        llm = BaarChatModel(router=router)
        messages = [HumanMessage(content="hi")]
        with pytest.raises(BudgetExhausted):
            llm._generate(messages)

    @pytest.mark.asyncio
    async def test_agenerate_calls_router_achat(self):
        router = make_router()
        llm = BaarChatModel(router=router)
        messages = [HumanMessage(content="ping")]

        with patch.object(router, "achat", return_value="pong") as mock_achat:
            result = await llm._agenerate(messages)
            mock_achat.assert_called_once()
            assert result.generations[0].message.content == "pong"
