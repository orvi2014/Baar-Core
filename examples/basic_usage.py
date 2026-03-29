"""
examples/basic_usage.py

Copy-paste example for the README.
Run with: python examples/basic_usage.py
Requires: export OPENAI_API_KEY=sk-...
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baar import BAARRouter, BudgetExceeded


def example_single_chat():
    """Example 1: single routed call."""
    print("=" * 50)
    print("Example 1: Single routed chat call")
    print("=" * 50)

    router = BAARRouter(budget=0.05)
    response = router.chat("What is the capital of France?")

    print(f"Response: {response}")
    print(f"Spent:    ${router.spent:.6f}")
    print(f"Remaining: ${router.remaining:.6f}")


def example_multi_step():
    """Example 2: multi-step task with routing report."""
    print("\n" + "=" * 50)
    print("Example 2: Multi-step routing")
    print("=" * 50)

    router = BAARRouter(budget=0.20, complexity_threshold=0.80)

    tasks = [
        "What is 42 * 17?",
        "Translate 'good morning' to Spanish",
        "Write a Python function that implements binary search with full error handling and docstrings",
        "What does API stand for?",
        "Analyze the trade-offs between PostgreSQL and MongoDB for a real-time chat application with 1M users",
        "Convert 72°F to Celsius",
    ]

    log = router.run(tasks)
    log.print_report()

    summary = log.summary()
    savings = summary["savings_vs_always_big"]
    print(f"\nKey metric: BAAR saved {savings['savings_pct']}% vs using gpt-4o for everything")


def example_budget_exceeded():
    """Example 3: demonstrate hard budget enforcement."""
    print("\n" + "=" * 50)
    print("Example 3: Hard budget cap in action")
    print("=" * 50)

    router = BAARRouter(budget=0.001)  # Very tight budget

    tasks = [
        "Write a 2000-word essay on the history of artificial intelligence",
        "Analyze and refactor this entire codebase",
        "Generate a complete machine learning pipeline",
    ]

    log = router.run(tasks)

    print(f"\nCompleted {log.total_steps} of {len(tasks)} tasks before budget exhausted")
    print(f"Total spent: ${log.total_cost:.6f} / ${router.budget:.4f}")


def example_custom_models():
    """Example 4: use Claude models instead of OpenAI."""
    print("\n" + "=" * 50)
    print("Example 4: Custom model pair (Claude)")
    print("=" * 50)

    router = BAARRouter(
        budget=0.10,
        small_model="claude-haiku-3-5",
        big_model="claude-sonnet-4-5",
        complexity_threshold=0.70,
    )

    response = router.chat("Explain recursion with a practical example")
    print(f"Response (first 200 chars): {response[:200]}...")
    print(f"Spent: ${router.spent:.6f}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run examples with OpenAI models.")
        print("Set ANTHROPIC_API_KEY to run example 4 with Claude models.")
        sys.exit(1)

    example_single_chat()
    example_multi_step()
    example_budget_exceeded()
    # example_custom_models()  # uncomment if you have ANTHROPIC_API_KEY
