"""
Tests for baar.integrations.langchain — no real API calls, no OPENAI_API_KEY needed.
"""

import threading
import uuid
import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from unittest.mock import patch
from langchain_core.callbacks.manager import handle_event
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import LLMResult

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

def make_run_id() -> uuid.UUID:
    return uuid.uuid4()


# ── _messages_to_dicts ────────────────────────────────────────────────────────

def test_messages_to_dicts_basic():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    dicts = _messages_to_dicts(msgs)
    assert dicts == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_messages_to_dicts_with_ai_message():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello"), HumanMessage(content="bye")]
    dicts = _messages_to_dicts(msgs)
    assert dicts[1] == {"role": "assistant", "content": "hello"}


def test_last_human_content():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hello")]
    assert _last_human_content(msgs) == "hello"


def test_last_human_content_picks_last():
    msgs = [HumanMessage(content="first"), AIMessage(content="reply"), HumanMessage(content="second")]
    assert _last_human_content(msgs) == "second"


def test_last_human_content_no_human():
    msgs = [SystemMessage(content="sys")]
    assert _last_human_content(msgs) == "sys"


def test_last_human_content_empty():
    assert _last_human_content([]) == ""


# ── BaarCallbackHandler — Bug 1: raise_error must be True ─────────────────────

class TestRaiseError:
    def test_raise_error_is_true(self):
        # Bug 1: default raise_error=False means LangChain swallows BudgetExhausted.
        # The handler MUST set raise_error=True so the kill-switch actually fires.
        handler = BaarCallbackHandler(make_router())
        assert handler.raise_error is True

    def test_kill_switch_propagates_through_langchain(self):
        # Verify that BudgetExhausted is NOT swallowed by handle_event.
        router = make_router(budget=0.000001)
        handler = BaarCallbackHandler(router)
        with pytest.raises(BudgetExhausted):
            handle_event(
                [handler],
                "on_llm_start",
                "ignore_llm",
                {"kwargs": {"model_name": "gpt-4o-mini"}},
                ["hello"],
                run_id=make_run_id(),
            )

    def test_on_chat_model_start_propagates_through_langchain(self):
        router = make_router(budget=0.000001)
        handler = BaarCallbackHandler(router)
        with pytest.raises(BudgetExhausted):
            handle_event(
                [handler],
                "on_chat_model_start",
                "ignore_chat_model",
                {"kwargs": {"model_name": "gpt-4o-mini"}},
                [[HumanMessage(content="hi")]],
                run_id=make_run_id(),
            )


# ── BaarCallbackHandler — Bug 2: run_id-keyed model tracking ─────────────────

class TestRunIdTracking:
    def test_on_llm_end_uses_correct_model_per_run(self):
        # Each run_id must track its own model name independently.
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        run_a = make_run_id()
        run_b = make_run_id()

        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}}, ["prompt a"], run_id=run_a
        )
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o"}}, ["prompt b"], run_id=run_b
        )

        # Both run_ids should be tracked separately
        assert handler._pending_models[run_a] == "gpt-4o-mini"
        assert handler._pending_models[run_b] == "gpt-4o"

    def test_on_llm_end_clears_run_id(self):
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        run_id = make_run_id()

        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}}, ["prompt"], run_id=run_id
        )
        assert run_id in handler._pending_models

        handler.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=run_id)
        assert run_id not in handler._pending_models

    def test_on_llm_error_clears_run_id(self):
        # run_id must be cleaned up even when the LLM call fails.
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        run_id = make_run_id()

        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}}, ["prompt"], run_id=run_id
        )
        assert run_id in handler._pending_models

        handler.on_llm_error(Exception("provider error"), run_id=run_id)
        assert run_id not in handler._pending_models

    def test_concurrent_calls_track_independently(self):
        # Ten concurrent threads each start and end with their own run_id.
        # No thread should see another thread's model name in on_llm_end.
        router = make_router(budget=100.0)
        handler = BaarCallbackHandler(router)
        errors = []

        def run_call(model_name: str):
            run_id = make_run_id()
            handler.on_llm_start(
                {"kwargs": {"model_name": model_name}}, ["p"], run_id=run_id
            )
            # Small sleep to maximise interleaving
            import time; time.sleep(0.005)
            recorded = handler._pending_models.get(run_id)
            if recorded != model_name:
                errors.append(f"expected {model_name}, got {recorded}")
            handler.on_llm_end(
                LLMResult(generations=[], llm_output={}), run_id=run_id
            )

        with patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000001)):
            threads = [
                threading.Thread(target=run_call, args=(f"model-{i}",))
                for i in range(10)
            ]
            for t in threads: t.start()
            for t in threads: t.join()

        assert errors == [], f"Race conditions detected: {errors}"

    def test_on_llm_end_records_spend(self):
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        run_id = make_run_id()

        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}}, ["prompt"], run_id=run_id
        )
        before = router.spent
        handler.on_llm_end(
            LLMResult(
                generations=[],
                llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}},
            ),
            run_id=run_id,
        )
        assert router.spent > before

    def test_on_llm_end_noop_without_usage(self):
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        run_id = make_run_id()

        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}}, ["prompt"], run_id=run_id
        )
        before = router.spent
        handler.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=run_id)
        assert router.spent == before

    def test_on_llm_start_passes_when_budget_ok(self):
        router = make_router(budget=10.0)
        handler = BaarCallbackHandler(router)
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}},
            ["Hello world"],
            run_id=make_run_id(),
        )

    def test_on_llm_start_raises_when_budget_exhausted(self):
        router = make_router(budget=0.000001)
        handler = BaarCallbackHandler(router)
        with pytest.raises(BudgetExhausted):
            handler.on_llm_start(
                {"kwargs": {"model_name": "gpt-4o-mini"}},
                ["Hello world"],
                run_id=make_run_id(),
            )

    def test_on_chat_model_start_passes_when_budget_ok(self):
        router = make_router(budget=10.0)
        handler = BaarCallbackHandler(router)
        handler.on_chat_model_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}},
            [[HumanMessage(content="hi")]],
            run_id=make_run_id(),
        )


# ── BaarChatModel — Bug 3: _astream must be implemented ──────────────────────

class TestAstream:
    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self):
        # _astream must yield ChatGenerationChunks, not silently fall back.
        router = make_router()
        llm = BaarChatModel(router=router)
        messages = [HumanMessage(content="hi")]

        with patch.object(router, "stream_chat", return_value=iter(["hello", " world"])):
            chunks = []
            async for chunk in llm._astream(messages):
                chunks.append(chunk.message.content)

        assert chunks == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_astream_raises_budget_exhausted(self):
        router = make_router(budget=0.000001)
        llm = BaarChatModel(router=router)
        messages = [HumanMessage(content="hi")]

        with pytest.raises(BudgetExhausted):
            async for _ in llm._astream(messages):
                pass

    def test_stream_yields_chunks(self):
        router = make_router()
        llm = BaarChatModel(router=router)
        messages = [HumanMessage(content="hi")]

        with patch.object(router, "stream_chat", return_value=iter(["foo", "bar"])):
            chunks = [c.message.content for c in llm._stream(messages)]

        assert chunks == ["foo", "bar"]


# ── BaarChatModel — general ───────────────────────────────────────────────────

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
        with pytest.raises(BudgetExhausted):
            llm._generate([HumanMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_agenerate_calls_router_achat(self):
        router = make_router()
        llm = BaarChatModel(router=router)

        with patch.object(router, "achat", return_value="pong") as mock_achat:
            result = await llm._agenerate([HumanMessage(content="ping")])
            mock_achat.assert_called_once()
            assert result.generations[0].message.content == "pong"
