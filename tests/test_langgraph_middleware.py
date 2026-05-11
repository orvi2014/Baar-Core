"""
tests/test_langgraph_middleware.py — BaarMiddleware unit tests

Runs entirely without API keys or LangGraph installed.
Uses mock graphs. Every test is deterministic and fast.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from baar import BAARRouter
from baar.core.exceptions import BudgetExhausted
from baar.integrations.langgraph import BaarMiddleware, StepLimitExceeded


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_router(budget: float = 1.0) -> BAARRouter:
    return BAARRouter(budget=budget, use_llm_router=False)


def make_mock_graph(return_value=None):
    """Build a mock compiled LangGraph graph."""
    graph = MagicMock()
    graph.invoke.return_value = return_value or {"messages": ["response"]}
    graph.stream.return_value = iter([{"step": 1}, {"step": 2}])
    graph.ainvoke = AsyncMock(return_value=return_value or {"messages": ["response"]})

    async def _astream(*args, **kwargs):
        yield {"step": 1}
        yield {"step": 2}

    graph.astream = _astream
    return graph


# ── Initialisation ────────────────────────────────────────────────────────────

class TestBaarMiddlewareInit:
    def test_basic_instantiation(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router())
        assert mw._max_steps is None
        assert mw._step_count == 0

    def test_custom_max_steps(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router(), max_steps=5)
        assert mw._max_steps == 5

    def test_default_recursion_limit(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router())
        assert mw._recursion_limit == 25

    def test_custom_recursion_limit(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router(), recursion_limit=10)
        assert mw._recursion_limit == 10

    def test_steps_property(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router())
        assert mw.steps == 0

    def test_remaining_property(self):
        router = make_router(budget=0.50)
        mw = BaarMiddleware(graph=make_mock_graph(), router=router)
        assert mw.remaining == pytest.approx(0.50)


# ── Step limit enforcement ────────────────────────────────────────────────────

class TestStepLimit:
    def test_raises_after_max_steps_exceeded(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router(), max_steps=2)
        mw.invoke({"messages": ["hello"]})
        mw.invoke({"messages": ["hello"]})
        with pytest.raises(StepLimitExceeded) as exc_info:
            mw.invoke({"messages": ["hello"]})
        assert exc_info.value.steps == 2
        assert exc_info.value.max_steps == 2

    def test_step_count_increments_on_invoke(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router())
        mw.invoke({"messages": ["hi"]})
        assert mw.steps == 1
        mw.invoke({"messages": ["hi"]})
        assert mw.steps == 2

    def test_no_limit_when_max_steps_is_none(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router(), max_steps=None)
        for _ in range(10):
            mw.invoke({"messages": ["hi"]})
        assert mw.steps == 10

    def test_step_limit_exceeded_attributes(self):
        exc = StepLimitExceeded(steps=5, max_steps=5)
        assert exc.steps == 5
        assert exc.max_steps == 5

    def test_stream_increments_step_count(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router())
        list(mw.stream({"messages": ["hi"]}))
        assert mw.steps == 1

    def test_stream_raises_after_max_steps(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router(), max_steps=1)
        list(mw.stream({"messages": ["hi"]}))
        with pytest.raises(StepLimitExceeded):
            list(mw.stream({"messages": ["hi"]}))

    def test_graph_not_called_when_step_limit_exceeded(self):
        graph = make_mock_graph()
        mw = BaarMiddleware(graph=graph, router=make_router(), max_steps=1)
        mw.invoke({"messages": ["hi"]})
        try:
            mw.invoke({"messages": ["hi"]})
        except StepLimitExceeded:
            pass
        assert graph.invoke.call_count == 1


# ── Budget enforcement ────────────────────────────────────────────────────────

class TestBudgetEnforcement:
    def test_raises_budget_exhausted_when_below_threshold(self):
        router = make_router(budget=0.000001)
        mw = BaarMiddleware(graph=make_mock_graph(), router=router, max_budget=0.01)
        with pytest.raises(BudgetExhausted):
            mw.invoke({"messages": ["hi"]})

    def test_passes_when_budget_sufficient(self):
        router = make_router(budget=1.0)
        mw = BaarMiddleware(graph=make_mock_graph(), router=router, max_budget=0.01)
        result = mw.invoke({"messages": ["hi"]})
        assert result is not None

    def test_no_budget_check_when_max_budget_none(self):
        router = make_router(budget=0.000001)
        mw = BaarMiddleware(graph=make_mock_graph(), router=router, max_budget=None)
        # No BudgetExhausted — middleware-level budget gate is disabled
        mw.invoke({"messages": ["hi"]})

    def test_graph_not_called_when_budget_exhausted(self):
        graph = make_mock_graph()
        router = make_router(budget=0.000001)
        mw = BaarMiddleware(graph=graph, router=router, max_budget=0.01)
        try:
            mw.invoke({"messages": ["hi"]})
        except BudgetExhausted:
            pass
        graph.invoke.assert_not_called()

    def test_step_limit_checked_before_budget(self):
        # Step limit fires first — budget error should not be raised.
        router = make_router(budget=0.000001)
        mw = BaarMiddleware(
            graph=make_mock_graph(),
            router=router,
            max_steps=1,
            max_budget=0.01,
        )
        mw._step_count = 1  # already at limit
        with pytest.raises(StepLimitExceeded):
            mw.invoke({"messages": ["hi"]})


# ── Config injection ──────────────────────────────────────────────────────────

class TestConfigInjection:
    def test_recursion_limit_injected(self):
        graph = make_mock_graph()
        mw = BaarMiddleware(graph=graph, router=make_router(), recursion_limit=10)
        mw.invoke({"messages": ["hi"]})
        _, call_kwargs = graph.invoke.call_args
        assert call_kwargs["config"]["recursion_limit"] == 10

    def test_existing_config_keys_preserved(self):
        graph = make_mock_graph()
        mw = BaarMiddleware(graph=graph, router=make_router(), recursion_limit=15)
        mw.invoke({"messages": ["hi"]}, config={"tags": ["test"]})
        _, call_kwargs = graph.invoke.call_args
        assert call_kwargs["config"]["tags"] == ["test"]
        assert call_kwargs["config"]["recursion_limit"] == 15

    def test_recursion_limit_not_overridden_if_caller_sets_it(self):
        # Caller explicitly sets recursion_limit — middleware should not override.
        graph = make_mock_graph()
        mw = BaarMiddleware(graph=graph, router=make_router(), recursion_limit=25)
        mw.invoke({"messages": ["hi"]}, config={"recursion_limit": 5})
        _, call_kwargs = graph.invoke.call_args
        assert call_kwargs["config"]["recursion_limit"] == 5


# ── Passthrough ───────────────────────────────────────────────────────────────

class TestPassthrough:
    def test_invoke_returns_graph_result(self):
        graph = make_mock_graph(return_value={"messages": ["hello world"]})
        mw = BaarMiddleware(graph=graph, router=make_router())
        result = mw.invoke({"messages": ["hi"]})
        assert result == {"messages": ["hello world"]}

    def test_stream_yields_graph_chunks(self):
        graph = make_mock_graph()
        graph.stream.return_value = iter([{"a": 1}, {"b": 2}])
        mw = BaarMiddleware(graph=graph, router=make_router())
        chunks = list(mw.stream({"messages": ["hi"]}))
        assert chunks == [{"a": 1}, {"b": 2}]

    @pytest.mark.asyncio
    async def test_ainvoke_returns_graph_result(self):
        graph = make_mock_graph(return_value={"messages": ["async result"]})
        mw = BaarMiddleware(graph=graph, router=make_router())
        result = await mw.ainvoke({"messages": ["hi"]})
        assert result == {"messages": ["async result"]}

    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router())
        chunks = []
        async for chunk in mw.astream({"messages": ["hi"]}):
            chunks.append(chunk)
        assert chunks == [{"step": 1}, {"step": 2}]

    @pytest.mark.asyncio
    async def test_ainvoke_increments_step_count(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router())
        await mw.ainvoke({"messages": ["hi"]})
        assert mw.steps == 1

    @pytest.mark.asyncio
    async def test_ainvoke_raises_after_max_steps(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router(), max_steps=1)
        await mw.ainvoke({"messages": ["hi"]})
        with pytest.raises(StepLimitExceeded):
            await mw.ainvoke({"messages": ["hi"]})

    @pytest.mark.asyncio
    async def test_astream_increments_step_count(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router())
        async for _ in mw.astream({"messages": ["hi"]}):
            pass
        assert mw.steps == 1

    @pytest.mark.asyncio
    async def test_astream_raises_after_max_steps(self):
        mw = BaarMiddleware(graph=make_mock_graph(), router=make_router(), max_steps=1)
        async for _ in mw.astream({"messages": ["hi"]}):
            pass
        with pytest.raises(StepLimitExceeded):
            async for _ in mw.astream({"messages": ["hi"]}):
                pass
