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

        handler.on_llm_end(
            LLMResult(
                generations=[],
                llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 5}},
            ),
            run_id=run_id,
        )
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
                LLMResult(
                    generations=[],
                    llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 5}},
                ),
                run_id=run_id,
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

        before = router.spent  # capture before preflight reservation
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}}, ["prompt"], run_id=run_id
        )
        handler.on_llm_end(
            LLMResult(
                generations=[],
                llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}},
            ),
            run_id=run_id,
        )
        # Net spend after cancel_reservation + record_manual must be positive.
        assert router.spent > before

    def test_preflight_reserves_atomically(self):
        # Two concurrent on_llm_start calls on a budget that covers only one call
        # must not both pass — check_and_reserve is atomic, so the second is blocked.
        router = make_router(budget=0.0001)
        handler = BaarCallbackHandler(router)
        passed = []
        failed = []
        result_lock = threading.Lock()

        def run_preflight():
            run_id = make_run_id()
            try:
                handler.on_llm_start(
                    {"kwargs": {"model_name": "gpt-4o-mini"}},
                    ["hello world"],
                    run_id=run_id,
                )
                with result_lock:
                    passed.append(run_id)
            except BudgetExhausted:
                with result_lock:
                    failed.append(run_id)

        # Patch outside threads — per-thread patching is not thread-safe.
        with patch("baar.core.budget.cost_per_token", return_value=(0.00006, 0.0)):
            threads = [threading.Thread(target=run_preflight) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Budget ($0.0001) covers exactly one $0.00006 estimate — only 1 should pass.
        assert len(passed) <= 1, f"TOCTOU: {len(passed)} preflights passed, expected at most 1"
        assert len(failed) >= 3

    def test_on_llm_end_retains_estimate_without_usage(self):
        # Provider omits token usage → preflight estimate must be kept as spend
        # (not cancelled) so the budget isn't silently inflated.
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        run_id = make_run_id()

        before = router.spent
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}}, ["prompt"], run_id=run_id
        )
        with pytest.warns(UserWarning, match="no token usage"):
            handler.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=run_id)
        # Estimate is retained — net spend must be positive.
        assert router.spent > before

    def test_on_llm_end_no_usage_emits_warning(self):
        router = make_router(budget=1.0)
        handler = BaarCallbackHandler(router)
        run_id = make_run_id()
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}}, ["prompt"], run_id=run_id
        )
        with pytest.warns(UserWarning, match="no token usage"):
            handler.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=run_id)

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

    def test_extract_model_warns_on_fallback(self):
        # When serialized dict has no recognisable model key, a UserWarning must
        # be emitted so the caller knows cost estimation is using a fallback.
        router = make_router(budget=10.0)
        handler = BaarCallbackHandler(router)
        with pytest.warns(UserWarning, match="could not extract model name"):
            handler.on_llm_start(
                {},  # no kwargs → no model_name key
                ["hello"],
                run_id=make_run_id(),
            )

    def test_on_chat_model_start_includes_non_string_content(self):
        # Tool result messages with non-string content must not be silently skipped —
        # omitting them underestimates tokens for agent/tool chains.
        from langchain_core.messages import ToolMessage
        router = make_router(budget=10.0)
        handler = BaarCallbackHandler(router)

        tool_msg = ToolMessage(content={"result": "42"}, tool_call_id="call_1")
        human_msg = HumanMessage(content="what is 6*7?")

        with patch("baar.integrations.langchain.token_counter") as mock_tc:
            mock_tc.return_value = 10
            handler.on_chat_model_start(
                {"kwargs": {"model_name": "gpt-4o-mini"}},
                [[human_msg, tool_msg]],
                run_id=make_run_id(),
            )
            # token_counter must have been called with a string that includes
            # the non-string tool message content (converted via str()).
            called_text = mock_tc.call_args[1]["text"]
            assert "result" in called_text or "42" in called_text


# ── BaarChatModel — Bug 3: _astream must be implemented ──────────────────────

class TestAstream:
    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self):
        # _astream must use astream_chat (native async generator), not stream_chat (sync).
        router = make_router()
        llm = BaarChatModel(router=router)
        messages = [HumanMessage(content="hi")]

        async def fake_astream_chat(task, *, messages=None):
            yield "hello"
            yield " world"

        with patch.object(router, "astream_chat", fake_astream_chat):
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
