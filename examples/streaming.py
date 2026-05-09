"""
examples/streaming.py

Streaming responses with live budget tracking.

Demonstrates:
  - stream_chat() for synchronous token-by-token output
  - astream_chat() for async streaming
  - Budget is tracked accurately even during streaming
  - Kill-switch works identically — raised before the stream starts if unaffordable

Run:
    pip install baar-core
    export OPENAI_API_KEY=sk-...
    python examples/streaming.py
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baar import BAARRouter, BudgetExhausted


def example_sync_stream():
    print("=" * 55)
    print("Sync streaming — tokens print as they arrive")
    print("=" * 55)

    router = BAARRouter(budget=0.05, use_llm_router=False)

    print("\nPrompt: Explain how TCP handshake works\n")
    print("Response: ", end="", flush=True)

    for token in router.stream_chat("Explain how TCP handshake works in 3 steps"):
        print(token, end="", flush=True)

    print(f"\n\nSpent: ${router.spent:.6f} / Remaining: ${router.remaining:.6f}")


async def example_async_stream():
    print("\n" + "=" * 55)
    print("Async streaming — same API, fully async")
    print("=" * 55)

    router = BAARRouter(budget=0.05, use_llm_router=False)

    print("\nPrompt: What is a Python generator?\n")
    print("Response: ", end="", flush=True)

    async for token in router.astream_chat("What is a Python generator? One paragraph."):
        print(token, end="", flush=True)

    print(f"\n\nSpent: ${router.spent:.6f} / Remaining: ${router.remaining:.6f}")


def example_stream_kill_switch():
    print("\n" + "=" * 55)
    print("Kill-switch during streaming")
    print("=" * 55)

    router = BAARRouter(budget=0.00001, use_llm_router=False)

    try:
        for token in router.stream_chat("Write a long essay on distributed systems"):
            print(token, end="", flush=True)
    except BudgetExhausted as e:
        print(f"\nBlocked before stream started. Remaining: ${e.remaining:.8f}")
        print("Zero tokens were fetched from the provider.")


def example_multi_turn_stream():
    print("\n" + "=" * 55)
    print("Multi-turn streaming conversation")
    print("=" * 55)

    router = BAARRouter(budget=0.10, use_llm_router=True)
    turns = [
        "In one sentence: what is gradient descent?",
        "Now give a concrete Python example.",
        "What's the main drawback of vanilla gradient descent?",
    ]

    for i, prompt in enumerate(turns, 1):
        print(f"\nTurn {i}: {prompt}")
        print("Reply: ", end="", flush=True)
        try:
            for token in router.stream_chat(prompt):
                print(token, end="", flush=True)
            print(f"\n  [spent so far: ${router.spent:.6f}]")
        except BudgetExhausted:
            print("\n[Budget exhausted — no more calls will be made]")
            break


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(1)

    example_sync_stream()
    asyncio.run(example_async_stream())
    example_stream_kill_switch()
    example_multi_turn_stream()
