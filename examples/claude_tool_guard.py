"""
examples/claude_tool_guard.py — @baar_guard with Claude tool use.

Shows how to wrap tools that Claude calls with hard limits:
  - max_calls: tool can only be called N times per session
  - min_budget: block if remaining budget is too low
  - cost_per_call: deduct fixed cost per tool execution

Works identically with OpenAI/Codex function calling — just swap the client.

Run:
    pip install baar-core anthropic
    export ANTHROPIC_API_KEY=sk-...
    python examples/claude_tool_guard.py
"""

import json
import anthropic

from baar import BAARRouter, BudgetExhausted
from baar.integrations.tools import baar_guard, ToolCallLimitExceeded

# ── Budget setup ──────────────────────────────────────────────────────────────

router = BAARRouter(budget=0.10)  # $0.10 hard cap for this session

# ── Tools — wrapped with @baar_guard ─────────────────────────────────────────

@baar_guard(router=router, max_calls=5, cost_per_call=0.001)
def web_search(query: str) -> str:
    """Search the web. Capped at 5 calls, $0.001 deducted per call."""
    # Replace with your actual search API
    return f"[mock] Top results for '{query}': result1, result2, result3"


@baar_guard(router=router, max_calls=3, min_budget=0.02)
def code_interpreter(code: str) -> str:
    """Execute Python code. Capped at 3 calls, requires $0.02 remaining."""
    try:
        result = {}
        exec(code, {}, result)
        return str(result.get("result", "executed successfully"))
    except Exception as e:
        return f"Error: {e}"


@baar_guard(router=router, max_calls=2, cost_per_call=0.005)
def image_generator(prompt: str) -> str:
    """Generate an image. Capped at 2 calls — expensive tool."""
    # Replace with your actual image API
    return f"[mock] Image generated for: '{prompt}' → https://cdn.example.com/img.png"


# ── Tool schema for Claude ────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "code_interpreter",
        "description": "Execute Python code and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    },
    {
        "name": "image_generator",
        "description": "Generate an image from a text prompt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Image description"}
            },
            "required": ["prompt"],
        },
    },
]

# ── Tool dispatcher ───────────────────────────────────────────────────────────

TOOL_MAP = {
    "web_search": web_search,
    "code_interpreter": code_interpreter,
    "image_generator": image_generator,
}


def call_tool(name: str, inputs: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return fn(**inputs)
    except ToolCallLimitExceeded as e:
        return (
            f"[LIMIT] Tool '{e.tool_name}' has reached its call limit "
            f"({e.calls}/{e.max_calls}). Cannot call again this session."
        )
    except BudgetExhausted as e:
        return (
            f"[BUDGET] Insufficient budget for tool '{name}'. "
            f"Remaining: ${e.remaining:.6f}."
        )


# ── Agent loop ────────────────────────────────────────────────────────────────

def run(task: str) -> str:
    """Run Claude with budget-guarded tools until it finishes or budget runs out."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task}]

    print(f"\nTask: {task}")
    print(f"Budget: ${router.budget:.2f}")
    print("-" * 50)

    while True:
        try:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            print(f"Claude API error: {e}")
            break

        # Collect assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nFinal answer: {block.text}")
                    return block.text
            break

        if response.stop_reason != "tool_use":
            break

        # Process tool calls
        tool_results = []
        stop_loop = False

        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"→ Tool call: {block.name}({json.dumps(block.input)})")
            result = call_tool(block.name, block.input)
            print(f"← Result: {result[:120]}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

            # Stop the loop if budget is exhausted
            if result.startswith("[BUDGET]"):
                stop_loop = True

        messages.append({"role": "user", "content": tool_results})

        print(f"   Budget remaining: ${router.remaining:.6f} | "
              f"web_search calls: {web_search._call_count}/{web_search._call_count + (5 - web_search._call_count)}")

        if stop_loop:
            print("\nStopping — budget too low to continue.")
            break

    print(f"\nSession complete. Spent: ${router.budget - router.remaining:.6f} / ${router.budget:.2f}")
    return ""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        "Search for the latest news on LLM cost reduction techniques, "
        "then write a short Python script that estimates token costs "
        "and show me an image prompt for a dashboard visualizing the results."
    )
