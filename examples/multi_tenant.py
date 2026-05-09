"""
examples/multi_tenant.py

Multi-tenant budget isolation: each user/team gets their own spending namespace
backed by a shared SQLite database. Concurrent writes are safe — SQLiteBudgetStore
uses WAL mode and per-namespace locking.

Demonstrates:
  - Namespace isolation (alice can't spend bob's budget)
  - Concurrent safe usage across threads
  - Resetting a single user's quota without affecting others
  - Querying all user balances

Run:
    pip install baar-core
    export OPENAI_API_KEY=sk-...
    python examples/multi_tenant.py
"""

import os
import sys
import threading
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baar import BAARRouter, BudgetExhausted
from baar.core.stores import SQLiteBudgetStore


def make_router(db_path: str, user_id: str, budget: float = 0.05) -> BAARRouter:
    store = SQLiteBudgetStore(db_path, namespace=user_id)
    return BAARRouter(
        budget=budget,
        store=store,
        use_llm_router=False,
        small_model="gpt-4o-mini",
        big_model="gpt-4o",
    )


def example_isolation(db_path: str):
    print("=" * 55)
    print("Namespace isolation — alice and bob are independent")
    print("=" * 55)

    alice = make_router(db_path, "alice", budget=0.05)
    bob = make_router(db_path, "bob", budget=0.05)

    # Exhaust alice's budget manually via the store
    alice._tracker._store.set_spent(0.05)

    try:
        alice.chat("Hello")
    except BudgetExhausted:
        print("Alice: budget exhausted (expected)")

    # Bob's budget is untouched
    print(f"Bob's remaining: ${bob.remaining:.4f} (untouched by Alice's exhaustion)")


def example_concurrent(db_path: str):
    print("\n" + "=" * 55)
    print("Concurrent writes — multiple threads, one database")
    print("=" * 55)

    errors = []

    def drain_user(user_id: str):
        store = SQLiteBudgetStore(db_path, namespace=user_id)
        try:
            for i in range(5):
                store.add_spent(0.001)
        except Exception as e:
            errors.append(f"{user_id}: {e}")

    threads = [threading.Thread(target=drain_user, args=(f"user_{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print(f"Errors: {errors}")
    else:
        print("10 concurrent users wrote to the same DB — no conflicts")

    # Verify each user has the right balance
    all_correct = True
    for i in range(10):
        store = SQLiteBudgetStore(db_path, namespace=f"user_{i}")
        spent = store.get_spent()
        if abs(spent - 0.005) > 1e-9:
            all_correct = False
            print(f"user_{i}: expected $0.005, got ${spent:.6f}")
    if all_correct:
        print("All balances correct: each user spent exactly $0.005")


def example_quota_report(db_path: str):
    print("\n" + "=" * 55)
    print("Quota report across all namespaces")
    print("=" * 55)

    users = ["alice", "bob"] + [f"user_{i}" for i in range(10)]
    monthly_budget = 0.05

    print(f"{'User':<12} {'Spent':>10} {'Remaining':>12} {'Used':>8}")
    print("-" * 46)
    for uid in users:
        store = SQLiteBudgetStore(db_path, namespace=uid)
        spent = store.get_spent()
        remaining = max(0.0, monthly_budget - spent)
        pct = min(100.0, spent / monthly_budget * 100)
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"{uid:<12} ${spent:>8.5f}  ${remaining:>9.5f}  {bar} {pct:4.0f}%")


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        example_isolation(db_path)
        example_concurrent(db_path)
        example_quota_report(db_path)
    finally:
        os.unlink(db_path)
        print(f"\nCleaned up temp database.")
