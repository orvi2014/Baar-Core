"""
examples/tool_guard.py — @baar_guard for tool execution governance.

Wraps any callable — sync or async — with hard limits:
  - max_calls:     tool can be called at most N times per session
  - min_budget:    block if remaining budget drops below threshold
  - cost_per_call: deduct a fixed cost from budget on each successful call

Works with any agent framework: LangChain, LangGraph, Claude, OpenAI, plain Python.

Run:
    pip install baar-core
    python examples/tool_guard.py
"""

import asyncio

from baar import BAARRouter, BudgetExhausted
from baar.integrations.tools import baar_guard, ToolCallLimitExceeded

router = BAARRouter(budget=0.50)


# ── Define guarded tools ──────────────────────────────────────────────────────

@baar_guard(router=router, max_calls=5, cost_per_call=0.001)
def web_search(query: str) -> str:
    """Search the web. Max 5 calls, $0.001 per call."""
    return f"[mock] Results for '{query}'"


@baar_guard(router=router, max_calls=3, min_budget=0.05)
def database_query(sql: str) -> list:
    """Run a DB query. Max 3 calls, requires $0.05 remaining."""
    return [{"id": 1, "value": "mock result"}]


@baar_guard(router=router, max_calls=2, cost_per_call=0.01)
def image_generator(prompt: str) -> str:
    """Generate an image. Expensive — max 2 calls, $0.01 per call."""
    return f"https://cdn.example.com/{hash(prompt)}.png"


@baar_guard(router=router, max_calls=10)
async def async_fetch(url: str) -> str:
    """Async HTTP fetch. Max 10 calls, no fixed cost."""
    await asyncio.sleep(0)  # simulate async I/O
    return f"[mock] Content from {url}"


# ── Pattern 1 — Basic call counting ──────────────────────────────────────────

def pattern_1_call_counting():
    print("Pattern 1 — Call counting")
    print("-" * 40)

    for i in range(6):
        try:
            result = web_search(f"query {i+1}")
            print(f"  Call {i+1}: {result} (calls used: {web_search._call_count}/5)")
        except ToolCallLimitExceeded as e:
            print(f"  Call {i+1}: BLOCKED — '{e.tool_name}' limit reached ({e.calls}/{e.max_calls})")

    print(f"  Budget after: ${router.remaining:.5f} (deducted 5 × $0.001 = $0.005)")


# ── Pattern 2 — Budget gate ───────────────────────────────────────────────────

def pattern_2_budget_gate():
    print("\nPattern 2 — Budget gate")
    print("-" * 40)

    # Drain budget close to $0.05 threshold
    router._tracker._spent = router.budget - 0.03

    try:
        database_query("SELECT * FROM users LIMIT 10")
        print("  Query succeeded")
    except BudgetExhausted as e:
        print(f"  BLOCKED — remaining ${e.remaining:.5f} < $0.05 threshold")

    # Restore
    router._tracker._spent = 0.0


# ── Pattern 3 — Cost deduction ────────────────────────────────────────────────

def pattern_3_cost_deduction():
    print("\nPattern 3 — Fixed cost deduction")
    print("-" * 40)

    before = router.remaining
    image_generator("a futuristic city at sunset")
    after  = router.remaining

    print(f"  Before: ${before:.5f}")
    print(f"  After:  ${after:.5f}")
    print(f"  Deducted: ${before - after:.5f} (expected $0.01)")


# ── Pattern 4 — Async tool ────────────────────────────────────────────────────

async def pattern_4_async():
    print("\nPattern 4 — Async tool")
    print("-" * 40)

    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]

    results = await asyncio.gather(*[async_fetch(url) for url in urls])
    for url, result in zip(urls, results):
        print(f"  {url} → {result[:40]}")
    print(f"  Calls used: {async_fetch._call_count}/10")


# ── Pattern 5 — Safe dispatcher (agent integration) ──────────────────────────

def safe_dispatch(tool_name: str, **kwargs) -> str:
    """
    Dispatcher pattern for agent frameworks — catches guard exceptions
    and returns natural language so the agent can reason about limits.
    """
    tools = {
        "web_search": web_search,
        "database_query": database_query,
        "image_generator": image_generator,
    }
    fn = tools.get(tool_name)
    if not fn:
        return f"Unknown tool: {tool_name}"
    try:
        return str(fn(**kwargs))
    except ToolCallLimitExceeded as e:
        return (
            f"Tool '{e.tool_name}' has reached its call limit "
            f"({e.calls}/{e.max_calls} calls). Cannot use this tool again."
        )
    except BudgetExhausted as e:
        return (
            f"Insufficient budget for '{tool_name}'. "
            f"Remaining: ${e.remaining:.6f}. Stopping to avoid overspend."
        )


def pattern_5_safe_dispatch():
    print("\nPattern 5 — Safe dispatcher (agent-friendly error messages)")
    print("-" * 40)

    # Exhaust web_search calls
    for _ in range(5):
        web_search(f"query")

    # Now dispatch will return a string instead of raising
    result = safe_dispatch("web_search", query="one more search")
    print(f"  Dispatch result: {result}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pattern_1_call_counting()
    pattern_2_budget_gate()
    pattern_3_cost_deduction()
    asyncio.run(pattern_4_async())
    pattern_5_safe_dispatch()

    print(f"\nFinal budget: ${router.remaining:.5f} / ${router.budget:.2f}")
