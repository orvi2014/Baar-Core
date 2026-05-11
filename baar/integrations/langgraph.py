"""
baar/integrations/langgraph.py — LangGraph middleware adapter.

BaarMiddleware wraps any compiled LangGraph graph and enforces:
  - Step limit  : raises StepLimitExceeded after max_steps invocations
  - Budget gate : raises BudgetExhausted if remaining < max_budget before each step
  - Recursion   : injects recursion_limit into LangGraph config automatically

Usage::

    from baar import BAARRouter
    from baar.integrations.langgraph import BaarMiddleware

    router = BAARRouter(budget=0.50)
    graph  = workflow.compile()           # your compiled LangGraph graph

    protected = BaarMiddleware(
        graph=graph,
        router=router,
        max_steps=10,       # stop after 10 outer invocations
        max_budget=0.05,    # block if less than $0.05 remaining
        recursion_limit=25, # LangGraph internal node recursion cap
    )

    result = protected.invoke({"messages": [...]})

    # Async
    result = await protected.ainvoke({"messages": [...]})

    # Streaming
    for chunk in protected.stream({"messages": [...]}):
        print(chunk)

    async for chunk in protected.astream({"messages": [...]}):
        print(chunk)

Install:
    pip install baar-core[langchain]
    # LangGraph is included transitively via langchain-core
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, Optional

from baar.router import BAARRouter
from baar.core.exceptions import BudgetExhausted


class StepLimitExceeded(RuntimeError):
    """Raised when BaarMiddleware has been invoked more than max_steps times."""

    def __init__(self, steps: int, max_steps: int):
        self.steps = steps
        self.max_steps = max_steps
        super().__init__(
            f"LangGraph step limit reached: {steps}/{max_steps} steps used. "
            "Increase max_steps or reduce the number of graph invocations."
        )


class BaarMiddleware:
    """
    Wraps a compiled LangGraph graph with Baar's execution governance.

    Enforces two hard limits before each invocation:

      1. Step limit  — total number of times invoke/stream/ainvoke/astream
                       may be called on this middleware instance.
      2. Budget gate — blocks the call if router.remaining < max_budget,
                       preventing a graph step from starting when the budget
                       is already too low to be meaningful.

    Also injects ``recursion_limit`` into every LangGraph config so the graph's
    internal node loop is capped automatically — no need to remember to set it
    in every ``invoke`` call.

    Args:
        graph:           Compiled LangGraph graph (StateGraph.compile() result).
        router:          BAARRouter instance — budget state is read from here.
        max_steps:       Maximum outer invocation count. None = unlimited.
        max_budget:      Minimum remaining budget (USD) required to start a step.
                         None = no middleware-level budget gate (router's own
                         pre-flight checks still apply inside each node call).
        recursion_limit: LangGraph internal recursion cap injected into config.
                         Default 25 matches LangGraph's own default.
    """

    def __init__(
        self,
        graph: Any,
        router: BAARRouter,
        max_steps: Optional[int] = None,
        max_budget: Optional[float] = None,
        recursion_limit: int = 25,
    ) -> None:
        self._graph = graph
        self._router = router
        self._max_steps = max_steps
        self._max_budget = max_budget
        self._recursion_limit = recursion_limit
        self._step_count: int = 0

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def steps(self) -> int:
        """Number of invocations completed so far."""
        return self._step_count

    @property
    def remaining(self) -> float:
        """Remaining budget from the underlying router."""
        return self._router.remaining

    # ── Pre-flight ─────────────────────────────────────────────────────────────

    def _preflight(self) -> None:
        """Check step limit then budget gate. Step limit is checked first."""
        if self._max_steps is not None and self._step_count >= self._max_steps:
            raise StepLimitExceeded(steps=self._step_count, max_steps=self._max_steps)

        if self._max_budget is not None and self._router.remaining < self._max_budget:
            raise BudgetExhausted(
                f"Baar middleware: remaining budget ${self._router.remaining:.6f} "
                f"is below the required minimum ${self._max_budget:.6f}. "
                "No graph step was started.",
                remaining=self._router.remaining,
            )

    def _inject_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge recursion_limit into config without overriding caller-set values."""
        merged = dict(config or {})
        merged.setdefault("recursion_limit", self._recursion_limit)
        return merged

    # ── Sync ──────────────────────────────────────────────────────────────────

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self._preflight()
        self._step_count += 1
        return self._graph.invoke(input, config=self._inject_config(config), **kwargs)

    def stream(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        self._preflight()
        self._step_count += 1
        yield from self._graph.stream(input, config=self._inject_config(config), **kwargs)

    # ── Async ─────────────────────────────────────────────────────────────────

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self._preflight()
        self._step_count += 1
        return await self._graph.ainvoke(input, config=self._inject_config(config), **kwargs)

    async def astream(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self._preflight()
        self._step_count += 1
        async for chunk in self._graph.astream(input, config=self._inject_config(config), **kwargs):
            yield chunk
