"""
examples/langgraph_middleware.py — BaarMiddleware with LangGraph agents.

BaarMiddleware wraps a compiled LangGraph graph and enforces:
  - max_steps:       stop after N outer invocations (prevents runaway loops)
  - max_budget:      block if remaining budget drops below threshold
  - recursion_limit: cap LangGraph's internal node recursion automatically

Three patterns shown:
  1. Basic agent with step limit
  2. Multi-turn loop with budget gate
  3. Async streaming agent

Run:
    pip install baar-core[langchain] langgraph langchain-openai
    export OPENAI_API_KEY=sk-...
    python examples/langgraph_middleware.py
"""

from baar import BAARRouter, BudgetExhausted
from baar.integrations.langgraph import BaarMiddleware, StepLimitExceeded

# ── Pattern 1 — Basic agent with step limit ───────────────────────────────────

def pattern_1_step_limit():
    """Wrap a compiled LangGraph graph — stop after 10 outer invocations."""
    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
    except ImportError:
        print("Pattern 1 skipped — install: pip install langgraph langchain-openai")
        return

    router = BAARRouter(budget=0.50)

    @tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"Sunny, 22°C in {city}"

    llm   = ChatOpenAI(model="gpt-4o-mini")
    graph = create_react_agent(model=llm, tools=[get_weather])

    # Wrap with BaarMiddleware — max 10 outer calls, LangGraph recurse cap = 15
    protected = BaarMiddleware(
        graph=graph,
        router=router,
        max_steps=10,
        recursion_limit=15,
    )

    try:
        result = protected.invoke({
            "messages": [{"role": "user", "content": "What's the weather in Tokyo and Paris?"}]
        })
        print("Pattern 1 result:", result["messages"][-1].content)
        print(f"Steps used: {protected.steps}/10 | Remaining: ${protected.remaining:.5f}")
    except StepLimitExceeded as e:
        print(f"Step limit hit: {e.steps}/{e.max_steps} steps used")
    except BudgetExhausted as e:
        print(f"Budget exhausted. Remaining: ${e.remaining:.6f}")


# ── Pattern 2 — Multi-turn loop with budget gate ──────────────────────────────

def pattern_2_budget_gate():
    """Run an agent loop — stop automatically when budget drops below $0.05."""
    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
    except ImportError:
        print("Pattern 2 skipped — install: pip install langgraph langchain-openai")
        return

    router = BAARRouter(budget=0.50)

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Error: {e}"

    llm   = ChatOpenAI(model="gpt-4o-mini")
    graph = create_react_agent(model=llm, tools=[calculator])

    protected = BaarMiddleware(
        graph=graph,
        router=router,
        max_steps=20,
        max_budget=0.05,   # block if less than $0.05 remaining
        recursion_limit=25,
    )

    tasks = [
        "What is 1337 * 42?",
        "Calculate the compound interest on $10,000 at 5% for 3 years.",
        "What is the 10th Fibonacci number?",
    ]

    for i, task in enumerate(tasks):
        try:
            result = protected.invoke({
                "messages": [{"role": "user", "content": task}]
            })
            print(f"Task {i+1}: {result['messages'][-1].content[:80]}")
            print(f"  Budget: ${router.remaining:.5f} remaining | Steps: {protected.steps}")
        except StepLimitExceeded as e:
            print(f"Step limit reached after {e.steps} steps.")
            break
        except BudgetExhausted as e:
            print(f"Budget gate triggered. Remaining ${e.remaining:.6f} < $0.05 threshold.")
            break


# ── Pattern 3 — Async streaming agent ────────────────────────────────────────

async def pattern_3_async_stream():
    """Async streaming with BaarMiddleware — budget-safe event loop."""
    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
    except ImportError:
        print("Pattern 3 skipped — install: pip install langgraph langchain-openai")
        return

    router = BAARRouter(budget=0.50)

    @tool
    def web_search(query: str) -> str:
        """Search the web."""
        return f"[mock] Results for '{query}': article1, article2, article3"

    llm   = ChatOpenAI(model="gpt-4o-mini")
    graph = create_react_agent(model=llm, tools=[web_search])

    protected = BaarMiddleware(
        graph=graph,
        router=router,
        max_steps=5,
        max_budget=0.02,
        recursion_limit=20,
    )

    try:
        print("Streaming chunks:")
        async for chunk in protected.astream({
            "messages": [{"role": "user", "content": "Search for recent AI breakthroughs"}]
        }):
            # LangGraph streams node outputs as dicts
            for node, state in chunk.items():
                if "messages" in state:
                    last = state["messages"][-1]
                    content = getattr(last, "content", "")
                    if content:
                        print(f"  [{node}] {str(content)[:80]}")
    except StepLimitExceeded as e:
        print(f"Step limit: {e.steps}/{e.max_steps}")
    except BudgetExhausted as e:
        print(f"Budget exhausted: ${e.remaining:.6f} remaining")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("Pattern 1 — Step limit")
    print("=" * 60)
    pattern_1_step_limit()

    print("\n" + "=" * 60)
    print("Pattern 2 — Budget gate (multi-turn loop)")
    print("=" * 60)
    pattern_2_budget_gate()

    print("\n" + "=" * 60)
    print("Pattern 3 — Async streaming")
    print("=" * 60)
    asyncio.run(pattern_3_async_stream())
