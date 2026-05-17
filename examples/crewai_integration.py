"""
examples/crewai_integration.py — baar-core × CrewAI budget-capped agent demo.

Requires:
    pip install baar-core[crewai] crewai

Usage:
    python examples/crewai_integration.py --mode research
    python examples/crewai_integration.py --mode killswitch
    python examples/crewai_integration.py --mode callback

Modes:
    research    Run a two-agent research crew with a $1.00 budget.
    killswitch  Set budget=$0.001 — kill-switch fires immediately on first call.
    callback    Use BaarStepCallback on an existing crew (not BaarCrewLLM).
"""

from __future__ import annotations

import argparse
import sys

try:
    from crewai import Agent, Crew, Task
except ImportError:
    print("crewai is not installed.  Run:  pip install crewai", file=sys.stderr)
    sys.exit(1)

from baar import BAARRouter
from baar.core.exceptions import BudgetExhausted
from baar.integrations.crewai import BaarCrewLLM, BaarStepCallback


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_budget(router: BAARRouter, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    bar_len = 30
    used = min(1.0, router.spent / router.budget) if router.budget else 0.0
    filled = int(bar_len * used)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = used * 100
    print(
        f"{prefix}Budget: [{bar}] {pct:5.1f}%  "
        f"${router.spent:.4f} spent / ${router.remaining:.4f} remaining"
    )


def _print_log(router: BAARRouter) -> None:
    log = router.log
    if not log.entries:
        print("  (no routing decisions recorded)")
        return
    print(f"  {'Task':<40} {'Model':<18} {'Cost':>8}")
    print(f"  {'-'*40} {'-'*18} {'-'*8}")
    for e in log.entries:
        task_preview = (e.task[:37] + "...") if len(e.task) > 40 else e.task
        print(f"  {task_preview:<40} {e.model:<18} ${e.cost:.6f}")
    print(f"\n  Total: ${router.spent:.6f}")


# ── modes ─────────────────────────────────────────────────────────────────────

def run_research(budget: float = 1.00) -> None:
    """Two-agent research + writing crew, budget-capped."""
    router = BAARRouter(budget=budget, small_model="gpt-4o-mini", big_model="gpt-4o")
    llm    = BaarCrewLLM(router=router)

    print(f"\n=== Research Crew  (budget=${budget:.2f}) ===\n")

    researcher = Agent(
        role="Research Analyst",
        goal="Find and summarise accurate technical information",
        backstory=(
            "You are a senior AI research analyst who specialises in "
            "open-source tooling and agent frameworks."
        ),
        llm=llm,
        verbose=True,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Produce clear, concise markdown documentation",
        backstory="You turn research notes into polished markdown tables and summaries.",
        llm=llm,
        verbose=True,
    )

    research_task = Task(
        description=(
            "Research the top 3 open-source agent frameworks (e.g. LangGraph, "
            "CrewAI, AutoGen). For each, note: language, license, key feature, "
            "and GitHub stars order-of-magnitude."
        ),
        agent=researcher,
        expected_output="A bullet-point list with the four data points per framework.",
    )
    write_task = Task(
        description=(
            "Using the researcher's notes, produce a concise markdown table "
            "comparing the three frameworks on: Language, License, Key Feature, Stars."
        ),
        agent=writer,
        expected_output="A markdown table with 4 columns and 3 data rows.",
    )

    crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task], verbose=True)

    try:
        result = crew.kickoff()
        print("\n--- Result ---")
        print(result.raw if hasattr(result, "raw") else result)
    except BudgetExhausted as exc:
        print(f"\nBudget exhausted: {exc}")

    print("\n--- Routing log ---")
    _print_log(router)
    _print_budget(router, "Final")


def run_killswitch(budget: float = 0.001) -> None:
    """Demonstrate the kill-switch firing immediately on the first call."""
    router = BAARRouter(budget=budget, small_model="gpt-4o-mini", big_model="gpt-4o")
    llm    = BaarCrewLLM(router=router)

    print(f"\n=== Kill-switch demo  (budget=${budget:.4f}) ===\n")

    agent = Agent(
        role="Test Agent",
        goal="Run any task",
        backstory="A test agent.",
        llm=llm,
        verbose=False,
    )
    task = Task(
        description="Explain quantum computing in one sentence.",
        agent=agent,
        expected_output="One sentence.",
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)

    try:
        crew.kickoff()
        print("(crew completed without hitting the kill-switch)")
    except BudgetExhausted as exc:
        print(f"Kill-switch fired: {exc}")
        _print_budget(router, "After kill-switch")


def run_callback(budget: float = 0.50) -> None:
    """BaarStepCallback on a crew that uses its own LLM."""
    router = BAARRouter(budget=budget, small_model="gpt-4o-mini", big_model="gpt-4o")
    guard  = BaarStepCallback(router)

    print(f"\n=== Step-callback demo  (budget=${budget:.2f}) ===\n")
    print("BaarStepCallback is installed as step_callback= on the Crew.")
    print("It checks the budget after each agent step and stops the crew if exhausted.\n")

    # The crew uses whatever LLM the Agent defaults to; baar only guards at step boundaries.
    llm = BaarCrewLLM(router=router)  # still route through baar for this demo
    agent = Agent(
        role="Researcher",
        goal="Find information",
        backstory="A helpful researcher.",
        llm=llm,
        verbose=True,
    )
    task = Task(
        description="List the top 5 programming languages in 2025.",
        agent=agent,
        expected_output="A numbered list of 5 languages.",
    )
    crew = Crew(agents=[agent], tasks=[task], step_callback=guard, verbose=True)

    try:
        result = crew.kickoff()
        print("\n--- Result ---")
        print(result.raw if hasattr(result, "raw") else result)
    except BudgetExhausted as exc:
        print(f"\nStep-callback stopped the crew: {exc}")

    _print_budget(router, "Final")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="baar-core × CrewAI demo")
    parser.add_argument(
        "--mode",
        choices=["research", "killswitch", "callback"],
        default="research",
        help="Demo mode (default: research)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Override the default budget for the chosen mode",
    )
    args = parser.parse_args()

    defaults = {"research": 1.00, "killswitch": 0.001, "callback": 0.50}
    budget = args.budget if args.budget is not None else defaults[args.mode]

    if args.mode == "research":
        run_research(budget)
    elif args.mode == "killswitch":
        run_killswitch(budget)
    elif args.mode == "callback":
        run_callback(budget)


if __name__ == "__main__":
    main()
