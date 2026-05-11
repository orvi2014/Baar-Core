"""
examples/openai_tool_guard.py — @baar_guard with OpenAI / Codex tool calling.

Identical guard pattern to claude_tool_guard.py — only the client differs.
@baar_guard fires before the tool executes, regardless of which AI called it.

Run:
    pip install baar-core openai
    export OPENAI_API_KEY=sk-...
    python examples/openai_tool_guard.py
"""

import json
import openai

from baar import BAARRouter, BudgetExhausted
from baar.integrations.tools import baar_guard, ToolCallLimitExceeded

# ── Budget setup ──────────────────────────────────────────────────────────────

router = BAARRouter(budget=0.10)

# ── Guarded tools ─────────────────────────────────────────────────────────────

@baar_guard(router=router, max_calls=5, cost_per_call=0.001)
def web_search(query: str) -> str:
    """Search the web. Max 5 calls, $0.001 per call."""
    return f"[mock] Top results for '{query}': result1, result2, result3"


@baar_guard(router=router, max_calls=3, min_budget=0.02)
def code_interpreter(code: str) -> str:
    """Execute Python code. Max 3 calls, requires $0.02 remaining."""
    try:
        result = {}
        exec(code, {"__builtins__": {}}, result)
        return str(result.get("result", "executed"))
    except Exception as e:
        return f"Error: {e}"


# ── OpenAI tool schema ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Execute Python code and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        },
    },
]

# ── Tool dispatcher ───────────────────────────────────────────────────────────

TOOL_MAP = {
    "web_search": web_search,
    "code_interpreter": code_interpreter,
}


def call_tool(name: str, arguments: str) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return str(fn(**json.loads(arguments)))
    except ToolCallLimitExceeded as e:
        return (
            f"Tool '{e.tool_name}' call limit reached "
            f"({e.calls}/{e.max_calls}). Cannot use this tool again."
        )
    except BudgetExhausted as e:
        return (
            f"Insufficient budget for '{name}'. "
            f"Remaining: ${e.remaining:.6f}. Stopping."
        )


# ── Agent loop ────────────────────────────────────────────────────────────────

def run(task: str) -> str:
    client = openai.OpenAI()
    messages = [{"role": "user", "content": task}]

    print(f"\nTask: {task}")
    print(f"Budget: ${router.budget:.2f}")
    print("-" * 50)

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            tools=TOOLS,
            messages=messages,
        )

        choice = response.choices[0]
        messages.append({"role": "assistant", "content": choice.message.content,
                          "tool_calls": choice.message.tool_calls})

        # No tool calls — final answer
        if choice.finish_reason == "stop" or not choice.message.tool_calls:
            answer = choice.message.content or ""
            print(f"\nAnswer: {answer}")
            break

        # Process tool calls
        stop_loop = False
        for tc in choice.message.tool_calls:
            print(f"→ {tc.function.name}({tc.function.arguments[:80]})")
            result = call_tool(tc.function.name, tc.function.arguments)
            print(f"← {result[:100]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

            if result.startswith("Insufficient budget"):
                stop_loop = True

        print(f"   Remaining: ${router.remaining:.5f}")

        if stop_loop:
            print("\nStopping — budget too low.")
            break

    print(f"\nSpent: ${router.budget - router.remaining:.6f} / ${router.budget:.2f}")
    return ""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        "Search for Python async best practices, then write a short "
        "async example script and explain how it works."
    )
