"""
examples/agent_loop.py

Autonomous agent loop with a hard budget kill-switch.

The agent runs tool-use iterations until:
  (a) the task is complete, or
  (b) the budget is exhausted — caught cleanly, no runaway spend.

This is the pattern that prevents the "I left it running overnight" bill.
The loop never overshoots: BudgetExhausted is raised *before* the API call,
so the final spent amount is always <= the configured cap.

Run:
    pip install baar-core
    export OPENAI_API_KEY=sk-...
    python examples/agent_loop.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baar import BAARRouter, BudgetExhausted

TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for current information.",
        "args": ["query: str"],
    },
    {
        "name": "read_file",
        "description": "Read a local file by path.",
        "args": ["path: str"],
    },
    {
        "name": "write_file",
        "description": "Write content to a local file.",
        "args": ["path: str", "content: str"],
    },
    {
        "name": "run_python",
        "description": "Execute a Python snippet and return stdout.",
        "args": ["code: str"],
    },
    {
        "name": "finish",
        "description": "Return the final answer to the user.",
        "args": ["answer: str"],
    },
]


def _tool_list_str() -> str:
    lines = []
    for t in TOOLS:
        lines.append(f"  - {t['name']}({', '.join(t['args'])}): {t['description']}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are an autonomous research agent. You have access to these tools:

{_tool_list_str()}

At each step, output ONLY one line in the format:
  TOOL: tool_name | ARGS: arg_value

When you have enough information, output:
  TOOL: finish | ARGS: <your final answer>

Do not explain yourself. Just output the tool call."""


def run_agent(goal: str, budget: float = 0.05, max_steps: int = 10) -> str:
    """
    Run a simple ReAct-style agent loop with a hard USD budget cap.
    Returns the final answer or a budget-exhausted message.
    """
    router = BAARRouter(
        budget=budget,
        use_llm_router=True,
        complexity_threshold=0.70,
        system_prompt=SYSTEM_PROMPT,
    )

    history = [f"Goal: {goal}"]
    print(f"\nAgent starting. Goal: {goal}")
    print(f"Budget: ${budget:.4f} | Max steps: {max_steps}\n")

    for step in range(1, max_steps + 1):
        context = "\n".join(history)
        prompt = f"{context}\n\nStep {step}: What tool do you call next?"

        try:
            raw = router.chat(prompt)
        except BudgetExhausted as e:
            msg = (
                f"[Budget exhausted after {step - 1} steps. "
                f"Remaining: ${e.remaining:.6f}. No API call was made.]"
            )
            print(msg)
            return msg

        print(f"Step {step}: {raw.strip()}")
        history.append(f"Step {step} output: {raw.strip()}")

        if "TOOL: finish" in raw:
            answer = raw.split("ARGS:", 1)[-1].strip() if "ARGS:" in raw else raw
            print(f"\nDone in {step} steps.")
            print(f"Total spent: ${router.spent:.6f} / ${budget:.4f}")
            return answer

    return f"[Max steps ({max_steps}) reached. Last output: {history[-1]}]"


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(1)

    answer = run_agent(
        goal="What is the latest Python version and when was it released?",
        budget=0.03,
        max_steps=6,
    )
    print(f"\nFinal answer: {answer}")
