"""
baar/integrations/tools.py — Tool execution guards.

baar_guard wraps any callable (sync or async) with:
  - Call limit   : raises ToolCallLimitExceeded after max_calls invocations
  - Budget gate  : raises BudgetExhausted if router.remaining < min_budget
  - Cost tracking: deducts cost_per_call from budget after each successful call

Usage::

    from baar import BAARRouter
    from baar.integrations.tools import baar_guard

    router = BAARRouter(budget=0.50)

    @baar_guard(router=router, max_calls=5)
    def search_web(query: str) -> str:
        return requests.get(f"https://api.search.com?q={query}").text

    @baar_guard(router=router, max_calls=3, min_budget=0.01, cost_per_call=0.005)
    async def generate_image(prompt: str) -> bytes:
        ...

Args:
    router:        BAARRouter instance — budget state is read and written here.
    max_calls:     Maximum number of times the tool may be called. None = unlimited.
    min_budget:    Minimum remaining budget (USD) required before each call.
                   None = no budget gate.
    cost_per_call: Fixed USD amount deducted from budget after each successful call.
                   None = no deduction (use this for tools whose cost is tracked
                   via LLM calls inside them — avoid double-counting).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from baar.router import BAARRouter
from baar.core.exceptions import BudgetExhausted


class ToolCallLimitExceeded(RuntimeError):
    """Raised when a baar_guard-decorated tool exceeds its max_calls limit."""

    def __init__(self, calls: int, max_calls: int, tool_name: str):
        self.calls = calls
        self.max_calls = max_calls
        self.tool_name = tool_name
        super().__init__(
            f"Tool '{tool_name}' call limit reached: {calls}/{max_calls} calls used. "
            "Increase max_calls or reduce tool invocations."
        )


def baar_guard(
    router: BAARRouter,
    max_calls: Optional[int] = None,
    min_budget: Optional[float] = None,
    cost_per_call: Optional[float] = None,
) -> Callable:
    """
    Decorator factory that wraps a tool function with call and budget limits.

    Works on both sync and async functions. The decorated function gets a
    ``_call_count`` attribute so callers can inspect usage without catching
    exceptions.

    Checks are applied in this order (first failure wins):
      1. Call limit  — ToolCallLimitExceeded
      2. Budget gate — BudgetExhausted
      3. Execute function
      4. Cost deduction (only on success)
    """

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _preflight(async_wrapper, fn.__name__, router, max_calls, min_budget)
                try:
                    result = await fn(*args, **kwargs)
                except Exception:
                    raise
                else:
                    async_wrapper._call_count += 1
                    _deduct(router, cost_per_call)
                    return result

            async_wrapper._call_count = 0
            return async_wrapper

        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                _preflight(sync_wrapper, fn.__name__, router, max_calls, min_budget)
                try:
                    result = fn(*args, **kwargs)
                except Exception:
                    raise
                else:
                    sync_wrapper._call_count += 1
                    _deduct(router, cost_per_call)
                    return result

            sync_wrapper._call_count = 0
            return sync_wrapper

    return decorator


# ── Internal helpers ──────────────────────────────────────────────────────────

def _preflight(
    wrapper: Callable,
    name: str,
    router: BAARRouter,
    max_calls: Optional[int],
    min_budget: Optional[float],
) -> None:
    """Check call limit then budget gate. Call limit is checked first."""
    if max_calls is not None and wrapper._call_count >= max_calls:
        raise ToolCallLimitExceeded(
            calls=wrapper._call_count,
            max_calls=max_calls,
            tool_name=name,
        )

    if min_budget is not None and router.remaining < min_budget:
        raise BudgetExhausted(
            f"Baar guard: remaining budget ${router.remaining:.6f} is below "
            f"the required minimum ${min_budget:.6f} for tool '{name}'. "
            "No tool call was made.",
            remaining=router.remaining,
        )


def _deduct(router: BAARRouter, cost_per_call: Optional[float]) -> None:
    """Deduct fixed cost from budget after a successful call."""
    if cost_per_call:
        router._tracker._store.add_spent(float(cost_per_call))
