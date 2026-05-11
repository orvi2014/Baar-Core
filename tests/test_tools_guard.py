"""
tests/test_tools_guard.py — baar_guard decorator unit tests

Runs entirely without API keys. Every test is deterministic and fast.
"""

import pytest
from unittest.mock import MagicMock, patch

from baar import BAARRouter
from baar.core.exceptions import BudgetExhausted
from baar.integrations.tools import baar_guard, ToolCallLimitExceeded


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_router(budget: float = 1.0) -> BAARRouter:
    return BAARRouter(budget=budget, use_llm_router=False)


# ── ToolCallLimitExceeded ─────────────────────────────────────────────────────

class TestToolCallLimitExceeded:
    def test_attributes(self):
        exc = ToolCallLimitExceeded(calls=5, max_calls=5, tool_name="search")
        assert exc.calls == 5
        assert exc.max_calls == 5
        assert exc.tool_name == "search"

    def test_message_includes_tool_name(self):
        exc = ToolCallLimitExceeded(calls=3, max_calls=3, tool_name="fetch_page")
        assert "fetch_page" in str(exc)

    def test_is_runtime_error(self):
        exc = ToolCallLimitExceeded(calls=1, max_calls=1, tool_name="tool")
        assert isinstance(exc, RuntimeError)


# ── Call limit ────────────────────────────────────────────────────────────────

class TestCallLimit:
    def test_raises_after_max_calls(self):
        router = make_router()

        @baar_guard(router=router, max_calls=2)
        def my_tool():
            return "ok"

        my_tool()
        my_tool()
        with pytest.raises(ToolCallLimitExceeded) as exc_info:
            my_tool()
        assert exc_info.value.calls == 2
        assert exc_info.value.max_calls == 2

    def test_call_count_increments(self):
        router = make_router()

        @baar_guard(router=router, max_calls=10)
        def my_tool():
            return "ok"

        my_tool()
        my_tool()
        assert my_tool._call_count == 2

    def test_no_limit_when_max_calls_none(self):
        router = make_router()

        @baar_guard(router=router, max_calls=None)
        def my_tool():
            return "ok"

        for _ in range(20):
            my_tool()
        assert my_tool._call_count == 20

    def test_function_not_called_when_limit_exceeded(self):
        router = make_router()
        inner = MagicMock(return_value="ok")

        @baar_guard(router=router, max_calls=1)
        def my_tool():
            return inner()

        my_tool()
        try:
            my_tool()
        except ToolCallLimitExceeded:
            pass

        assert inner.call_count == 1

    def test_tool_name_in_exception_matches_function(self):
        router = make_router()

        @baar_guard(router=router, max_calls=1)
        def search_web(query):
            return "result"

        search_web("hello")
        with pytest.raises(ToolCallLimitExceeded) as exc_info:
            search_web("world")
        assert exc_info.value.tool_name == "search_web"


# ── Budget gate ───────────────────────────────────────────────────────────────

class TestBudgetGate:
    def test_raises_when_below_min_budget(self):
        router = make_router(budget=0.001)

        @baar_guard(router=router, min_budget=0.05)
        def my_tool():
            return "ok"

        with pytest.raises(BudgetExhausted):
            my_tool()

    def test_passes_when_budget_sufficient(self):
        router = make_router(budget=1.0)

        @baar_guard(router=router, min_budget=0.05)
        def my_tool():
            return "ok"

        assert my_tool() == "ok"

    def test_no_budget_check_when_min_budget_none(self):
        router = make_router(budget=0.000001)

        @baar_guard(router=router, min_budget=None)
        def my_tool():
            return "ok"

        # Should not raise — budget gate is disabled
        assert my_tool() == "ok"

    def test_function_not_called_when_budget_exhausted(self):
        router = make_router(budget=0.001)
        inner = MagicMock(return_value="ok")

        @baar_guard(router=router, min_budget=0.05)
        def my_tool():
            return inner()

        try:
            my_tool()
        except BudgetExhausted:
            pass

        inner.assert_not_called()

    def test_call_limit_checked_before_budget(self):
        # Even with exhausted budget, step limit fires first.
        router = make_router(budget=0.000001)

        @baar_guard(router=router, max_calls=1, min_budget=0.05)
        def my_tool():
            return "ok"

        my_tool._call_count = 1  # already at limit
        with pytest.raises(ToolCallLimitExceeded):
            my_tool()


# ── Cost deduction ────────────────────────────────────────────────────────────

class TestCostDeduction:
    def test_deducts_cost_per_call_from_budget(self):
        router = make_router(budget=1.0)

        @baar_guard(router=router, cost_per_call=0.01)
        def my_tool():
            return "ok"

        before = router.remaining
        my_tool()
        assert router.remaining == pytest.approx(before - 0.01)

    def test_no_deduction_when_cost_per_call_none(self):
        router = make_router(budget=1.0)

        @baar_guard(router=router, cost_per_call=None)
        def my_tool():
            return "ok"

        before = router.remaining
        my_tool()
        assert router.remaining == pytest.approx(before)

    def test_no_deduction_when_function_raises(self):
        router = make_router(budget=1.0)

        @baar_guard(router=router, cost_per_call=0.01)
        def my_tool():
            raise ValueError("tool failed")

        before = router.remaining
        with pytest.raises(ValueError):
            my_tool()
        assert router.remaining == pytest.approx(before)

    def test_multiple_calls_accumulate_cost(self):
        router = make_router(budget=1.0)

        @baar_guard(router=router, cost_per_call=0.05)
        def my_tool():
            return "ok"

        my_tool()
        my_tool()
        my_tool()
        assert router.remaining == pytest.approx(1.0 - 0.15)


# ── Async support ─────────────────────────────────────────────────────────────

class TestAsyncSupport:
    @pytest.mark.asyncio
    async def test_async_function_works(self):
        router = make_router()

        @baar_guard(router=router, max_calls=5)
        async def async_tool():
            return "async ok"

        result = await async_tool()
        assert result == "async ok"

    @pytest.mark.asyncio
    async def test_async_call_count_increments(self):
        router = make_router()

        @baar_guard(router=router, max_calls=5)
        async def async_tool():
            return "ok"

        await async_tool()
        await async_tool()
        assert async_tool._call_count == 2

    @pytest.mark.asyncio
    async def test_async_raises_after_max_calls(self):
        router = make_router()

        @baar_guard(router=router, max_calls=1)
        async def async_tool():
            return "ok"

        await async_tool()
        with pytest.raises(ToolCallLimitExceeded):
            await async_tool()

    @pytest.mark.asyncio
    async def test_async_raises_budget_exhausted(self):
        router = make_router(budget=0.001)

        @baar_guard(router=router, min_budget=0.05)
        async def async_tool():
            return "ok"

        with pytest.raises(BudgetExhausted):
            await async_tool()

    @pytest.mark.asyncio
    async def test_async_deducts_cost_per_call(self):
        router = make_router(budget=1.0)

        @baar_guard(router=router, cost_per_call=0.02)
        async def async_tool():
            return "ok"

        before = router.remaining
        await async_tool()
        assert router.remaining == pytest.approx(before - 0.02)

    @pytest.mark.asyncio
    async def test_async_no_deduction_on_failure(self):
        router = make_router(budget=1.0)

        @baar_guard(router=router, cost_per_call=0.02)
        async def async_tool():
            raise RuntimeError("fail")

        before = router.remaining
        with pytest.raises(RuntimeError):
            await async_tool()
        assert router.remaining == pytest.approx(before)


# ── Function preservation ─────────────────────────────────────────────────────

class TestFunctionPreservation:
    def test_name_preserved(self):
        router = make_router()

        @baar_guard(router=router, max_calls=5)
        def search_web(query: str) -> str:
            """Searches the web."""
            return "result"

        assert search_web.__name__ == "search_web"

    def test_docstring_preserved(self):
        router = make_router()

        @baar_guard(router=router, max_calls=5)
        def search_web(query: str) -> str:
            """Searches the web."""
            return "result"

        assert search_web.__doc__ == "Searches the web."

    def test_return_value_passed_through(self):
        router = make_router()

        @baar_guard(router=router, max_calls=5)
        def my_tool(x, y):
            return x + y

        assert my_tool(3, 4) == 7

    def test_arguments_passed_through(self):
        router = make_router()
        inner = MagicMock(return_value="ok")

        @baar_guard(router=router, max_calls=5)
        def my_tool(a, b, *, c=None):
            return inner(a, b, c=c)

        my_tool(1, 2, c=3)
        inner.assert_called_once_with(1, 2, c=3)
