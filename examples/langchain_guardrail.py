"""
examples/langchain_guardrail.py

Two ways to add Baar's hard budget kill-switch to LangChain chains and
LangGraph agents.

─────────────────────────────────────────────────────────────────────────────
Pattern 1 — BaarCallbackHandler
  Plug into an *existing* chain via callbacks=[...].
  Raises BudgetExhausted before any API call when budget is too low.
  No changes to your chain or LLM required.

Pattern 2 — BaarChatModel
  Drop-in BaseChatModel with full Baar routing (semantic routing + budget
  aware downgrade + hard kill-switch).
  Use anywhere a ChatOpenAI / ChatAnthropic would be used.

Pattern 3 — LangGraph agent
  Pass BaarChatModel as the model to create_react_agent.
  Every tool-call iteration is budget-checked before the LLM is invoked.
─────────────────────────────────────────────────────────────────────────────

Install:
    pip install baar-core[langchain] langchain-openai langgraph

Run:
    export OPENAI_API_KEY=sk-...
    python examples/langchain_guardrail.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baar import BAARRouter, BudgetExhausted
from baar.integrations.langchain import BaarCallbackHandler, BaarChatModel


# ── Pattern 1: BaarCallbackHandler on an existing chain ───────────────────────

def demo_callback_handler() -> None:
    print("\n" + "=" * 60)
    print("Pattern 1 — BaarCallbackHandler")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        print("Skipped: pip install langchain-openai")
        return

    router  = BAARRouter(budget=0.05)
    handler = BaarCallbackHandler(router)

    llm   = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    questions = [
        "What is the capital of France?",
        "Explain the difference between TCP and UDP.",
    ]

    for q in questions:
        try:
            reply = chain.invoke({"question": q})
            print(f"Q: {q}")
            print(f"A: {reply[:120]}")
            print(f"   spent=${router.spent:.5f}  remaining=${router.remaining:.5f}\n")
        except BudgetExhausted as e:
            print(f"Kill-switch triggered — no API call made. Remaining: ${e.remaining:.6f}")
            break


# ── Pattern 2: BaarChatModel as a drop-in ChatModel ───────────────────────────

def demo_chat_model() -> None:
    print("\n" + "=" * 60)
    print("Pattern 2 — BaarChatModel (full routing)")
    print("=" * 60)

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        print("Skipped: pip install langchain-core")
        return

    router = BAARRouter(
        budget=0.10,
        small_model="gpt-4o-mini",
        big_model="gpt-4o",
        use_llm_router=True,
        complexity_threshold=0.75,
    )
    llm = BaarChatModel(router=router)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    tasks = [
        ("simple",  "What is 12 * 8?"),
        ("complex", "Design a distributed rate-limiter for 100k RPS. Include trade-offs."),
    ]

    for label, q in tasks:
        try:
            reply = chain.invoke({"question": q})
            print(f"[{label}] Q: {q}")
            print(f"        A: {reply[:140]}")
            print(f"        spent=${router.spent:.5f}  remaining=${router.remaining:.5f}\n")
        except BudgetExhausted as e:
            print(f"[{label}] Kill-switch — remaining: ${e.remaining:.6f}")


# ── Pattern 3: LangGraph agent with BaarChatModel ─────────────────────────────

def demo_langgraph_agent() -> None:
    print("\n" + "=" * 60)
    print("Pattern 3 — LangGraph agent with budget guardrail")
    print("=" * 60)

    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool
    except ImportError:
        print("Skipped: pip install langgraph")
        return

    router = BAARRouter(budget=0.05, use_llm_router=False)
    llm    = BaarChatModel(router=router)

    @tool
    def get_weather(city: str) -> str:
        """Return the current weather for a city (stub)."""
        return f"Sunny, 22°C in {city}."

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a simple arithmetic expression safely."""
        try:
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    agent = create_react_agent(model=llm, tools=[get_weather, calculate])

    try:
        result = agent.invoke({
            "messages": [("human", "What's the weather in Tokyo and what is 42 * 7?")]
        })
        final = result["messages"][-1].content
        print(f"Agent answer: {final}")
        print(f"spent=${router.spent:.5f}  remaining=${router.remaining:.5f}")
    except BudgetExhausted as e:
        print(f"Agent stopped by kill-switch — remaining: ${e.remaining:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run these examples.")
        sys.exit(1)

    demo_callback_handler()
    demo_chat_model()
    demo_langgraph_agent()
