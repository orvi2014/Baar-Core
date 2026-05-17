"""
examples/hermes_integration.py — Budget-safe Hermes Agent demo.

Demonstrates baar-core acting as a local OpenAI-compatible proxy that
Hermes Agent routes every LLM call through.  Hermes handles agentic work
(web search, multi-step reasoning, tool use); baar enforces a hard $1.00
spending cap pre-flight on every turn.

Install:
    pip install baar-core[vercel] hermes-agent pyyaml

Run:
    python examples/hermes_integration.py

The script runs three scenarios:

  1. Interactive session  — opens a budget-capped Hermes terminal
  2. Non-interactive task — runs a complex research task, prints results
  3. Kill-switch demo     — sets budget to $0.001 to show 402 enforcement

Set environment variables for your LLM provider before running:
    export OPENAI_API_KEY="sk-..."
    # or
    export ANTHROPIC_API_KEY="..."  (and use small_model="claude-haiku-4-5-20251001")
"""

import os
import shutil
import sys
import argparse

from baar.integrations.hermes import BaarHermesSession


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    width = 60
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)


def _print_budget(session: BaarHermesSession) -> None:
    bar_width = 30
    used_frac = session.spent / session.budget if session.budget > 0 else 0.0
    filled = int(bar_width * used_frac)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = used_frac * 100
    print(
        f"\n  Budget  [{bar}] {pct:5.1f}%"
        f"   ${session.spent:.4f} spent / ${session.remaining:.4f} remaining"
    )


# ── scenario 1: interactive session ──────────────────────────────────────────

def run_interactive(budget: float, port: int) -> None:
    _print_header("Scenario 1 — Interactive Hermes session (budget-capped)")
    print(f"  Budget:  ${budget:.2f}")
    print(f"  Proxy:   http://127.0.0.1:{port}/v1")
    print()
    print("  Every LLM call Hermes makes passes through baar-core.")
    print("  When the budget hits zero, Hermes receives a 402 and stops.")
    print()
    print("  Starting Hermes... (type /exit or Ctrl-C to quit)")

    with BaarHermesSession(budget=budget, port=port) as session:
        try:
            session.run_interactive()
        except KeyboardInterrupt:
            print("\n  [interrupted]")
        _print_budget(session)


# ── scenario 2: non-interactive task ──────────────────────────────────────────

def run_task_demo(budget: float, port: int) -> None:
    _print_header("Scenario 2 — Non-interactive research task")

    task = (
        "Research the top 3 open-source AI agent frameworks available today. "
        "For each one list: name, primary language, key capabilities, and GitHub stars. "
        "Present the results as a markdown table."
    )

    print(f"  Budget:  ${budget:.2f}")
    print(f"  Task:    {task[:80]}...")
    print()

    with BaarHermesSession(budget=budget, port=port) as session:
        print("  Running Hermes Agent... (this may take 30-120 seconds)\n")
        try:
            result = session.run_task(task, timeout=180)
            print(result)
        except Exception as exc:
            print(f"  Error: {exc}")
        _print_budget(session)

        if session.log.steps:
            print("\n  Routing decisions:")
            for step in session.log.steps:
                tier = step.decision.tier.upper() if hasattr(step.decision, 'tier') else "?"
                print(
                    f"    turn {step.step_num:2d}  [{tier:5s}]  "
                    f"{step.decision.model:<30s}  ${step.cost:.5f}"
                )


# ── scenario 3: kill-switch demo ──────────────────────────────────────────────

def run_killswitch_demo(port: int) -> None:
    _print_header("Scenario 3 — Kill-switch demo (budget = $0.001)")
    print("  A $0.001 budget is below the cost floor of any real call.")
    print("  baar fires the kill-switch before Hermes reaches the provider.")
    print()

    task = "What is the capital of France?"

    with BaarHermesSession(budget=0.001, port=port) as session:
        try:
            result = session.run_task(task, timeout=30)
            print(f"  Response: {result[:200]}")
        except Exception as exc:
            print(f"  Kill-switch fired: {exc}")
        _print_budget(session)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="baar-core × Hermes Agent integration demo"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "task", "killswitch", "all"],
        default="task",
        help="Which scenario to run (default: task)",
    )
    parser.add_argument("--budget", type=float, default=1.00, help="USD budget (default: 1.00)")
    parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")
    args = parser.parse_args()

    if not shutil.which("hermes"):
        print("ERROR: hermes CLI not found.")
        print("Install with:  pip install hermes-agent")
        print("           or: curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash")
        sys.exit(1)

    print("\nbaar-core × Hermes Agent Integration")
    print("======================================")
    print("baar-core is running as a local OpenAI-compatible proxy.")
    print("Hermes Agent routes all LLM calls through it for budget enforcement.\n")

    if args.mode in ("interactive", "all"):
        run_interactive(args.budget, args.port)

    if args.mode in ("task", "all"):
        run_task_demo(args.budget, args.port)

    if args.mode in ("killswitch", "all"):
        run_killswitch_demo(args.port)

    print("\nDone.")


if __name__ == "__main__":
    main()
