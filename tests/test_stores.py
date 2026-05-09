"""
tests/test_stores.py — Pluggable budget store backends.

Tests every public method of MemoryBudgetStore, FileBudgetStore, and
SQLiteBudgetStore. No API calls — pure Python + stdlib.
"""

import threading

import pytest
from baar.core.stores import MemoryBudgetStore, FileBudgetStore, SQLiteBudgetStore


# ─────────────────────────────────────────────────────────
# Shared contract — the same behaviour is required from all three backends
# ─────────────────────────────────────────────────────────

class StoreContractMixin:
    """
    Run against any BudgetStore implementation by overriding `make_store`.
    Subclasses that need a temp path receive it via `tmp_path` fixture — the
    mixin does not use it directly, so it stays out of the abstract method.
    """

    def make_store(self, namespace="default"):
        raise NotImplementedError

    # ── get_spent / add_spent ─────────────────────────────

    def test_initial_spent_is_zero(self):
        store = self.make_store()
        assert store.get_spent() == 0.0

    def test_add_spent_accumulates(self):
        store = self.make_store()
        store.add_spent(0.01)
        store.add_spent(0.02)
        assert store.get_spent() == pytest.approx(0.03)

    def test_add_spent_returns_new_total(self):
        store = self.make_store()
        result = store.add_spent(0.05)
        assert result == pytest.approx(0.05)
        result2 = store.add_spent(0.05)
        assert result2 == pytest.approx(0.10)

    # ── set_spent ─────────────────────────────────────────

    def test_set_spent_overwrites(self):
        store = self.make_store()
        store.add_spent(0.99)
        store.set_spent(0.01)
        assert store.get_spent() == pytest.approx(0.01)

    def test_set_spent_to_zero(self):
        store = self.make_store()
        store.add_spent(0.50)
        store.set_spent(0.0)
        assert store.get_spent() == 0.0

    # ── reset ─────────────────────────────────────────────

    def test_reset_clears_to_zero(self):
        store = self.make_store()
        store.add_spent(1.23)
        store.reset()
        assert store.get_spent() == 0.0

    def test_reset_on_fresh_store_is_a_noop(self):
        store = self.make_store()
        store.reset()
        assert store.get_spent() == 0.0

    # ── namespace isolation ───────────────────────────────

    def test_namespaces_are_isolated(self):
        a = self.make_store(namespace="alice")
        b = self.make_store(namespace="bob")
        a.add_spent(0.10)
        assert b.get_spent() == 0.0

    def test_namespace_reset_does_not_affect_other(self):
        a = self.make_store(namespace="alice")
        b = self.make_store(namespace="bob")
        a.add_spent(0.10)
        b.add_spent(0.20)
        a.reset()
        assert a.get_spent() == 0.0
        assert b.get_spent() == pytest.approx(0.20)


# ─────────────────────────────────────────────────────────
# MemoryBudgetStore
# ─────────────────────────────────────────────────────────

class TestMemoryBudgetStore(StoreContractMixin):
    def make_store(self, namespace="default"):
        return MemoryBudgetStore()

    def test_two_instances_are_independent(self):
        a = MemoryBudgetStore()
        b = MemoryBudgetStore()
        a.add_spent(0.50)
        assert b.get_spent() == 0.0


# ─────────────────────────────────────────────────────────
# FileBudgetStore
# ─────────────────────────────────────────────────────────

class TestFileBudgetStore(StoreContractMixin):
    @pytest.fixture(autouse=True)
    def _set_tmp(self, tmp_path):
        self._tmp_path = tmp_path

    def make_store(self, namespace="default"):
        path = str(self._tmp_path / "budgets.json")
        return FileBudgetStore(path, namespace=namespace)

    def test_data_persists_across_instances(self):
        path = str(self._tmp_path / "persist.json")
        s1 = FileBudgetStore(path, namespace="user1")
        s1.add_spent(0.07)
        # Discard s1 and open a new handle to the same file
        s2 = FileBudgetStore(path, namespace="user1")
        assert s2.get_spent() == pytest.approx(0.07)

    def test_file_created_if_absent(self):
        path = str(self._tmp_path / "new_dir" / "budgets.json")
        store = FileBudgetStore(path)
        store.add_spent(0.01)
        assert store.get_spent() == pytest.approx(0.01)

    def test_multiple_namespaces_in_same_file(self):
        path = str(self._tmp_path / "multi.json")
        a = FileBudgetStore(path, namespace="a")
        b = FileBudgetStore(path, namespace="b")
        a.add_spent(0.11)
        b.add_spent(0.22)
        assert FileBudgetStore(path, namespace="a").get_spent() == pytest.approx(0.11)
        assert FileBudgetStore(path, namespace="b").get_spent() == pytest.approx(0.22)


# ─────────────────────────────────────────────────────────
# SQLiteBudgetStore
# ─────────────────────────────────────────────────────────

class TestSQLiteBudgetStore(StoreContractMixin):
    @pytest.fixture(autouse=True)
    def _set_tmp(self, tmp_path):
        self._tmp_path = tmp_path

    def make_store(self, namespace="default"):
        db = str(self._tmp_path / "budgets.db")
        return SQLiteBudgetStore(db, namespace=namespace)

    def test_data_persists_across_instances(self):
        db = str(self._tmp_path / "persist.db")
        s1 = SQLiteBudgetStore(db, namespace="user1")
        s1.add_spent(0.07)
        s2 = SQLiteBudgetStore(db, namespace="user1")
        assert s2.get_spent() == pytest.approx(0.07)

    def test_db_created_if_absent(self):
        db = str(self._tmp_path / "sub" / "budgets.db")
        store = SQLiteBudgetStore(db)
        store.add_spent(0.01)
        assert store.get_spent() == pytest.approx(0.01)

    def test_multiple_namespaces_in_same_db(self):
        db = str(self._tmp_path / "multi.db")
        a = SQLiteBudgetStore(db, namespace="tenant:a")
        b = SQLiteBudgetStore(db, namespace="tenant:b")
        a.add_spent(0.15)
        b.add_spent(0.30)
        assert SQLiteBudgetStore(db, namespace="tenant:a").get_spent() == pytest.approx(0.15)
        assert SQLiteBudgetStore(db, namespace="tenant:b").get_spent() == pytest.approx(0.30)

    def test_daily_namespace_pattern(self):
        """Verify the recommended per-user-per-day namespace pattern works."""
        import datetime
        db = str(self._tmp_path / "quota.db")
        today = datetime.date.today().isoformat()
        store = SQLiteBudgetStore(db, namespace=f"user:42:{today}")
        store.add_spent(0.05)
        assert store.get_spent() == pytest.approx(0.05)


# ─────────────────────────────────────────────────────────
# BudgetTracker integration — store wired end-to-end
# ─────────────────────────────────────────────────────────

class TestBudgetTrackerWithStore:
    """Verify BudgetTracker delegates to the injected store correctly."""

    def test_default_tracker_uses_memory_store(self):
        from baar.core.budget import BudgetTracker
        tracker = BudgetTracker(total_budget=1.0)
        tracker._spent = 0.25
        assert tracker.spent == pytest.approx(0.25)
        assert tracker.remaining == pytest.approx(0.75)

    def test_tracker_with_memory_store(self):
        from baar.core.budget import BudgetTracker
        from baar.core.stores import MemoryBudgetStore
        store = MemoryBudgetStore()
        tracker = BudgetTracker(total_budget=1.0, store=store)
        tracker._spent = 0.50
        assert store.get_spent() == pytest.approx(0.50)

    def test_tracker_with_file_store_persists(self, tmp_path):
        from baar.core.budget import BudgetTracker
        from unittest.mock import patch
        path = str(tmp_path / "tracker.json")

        with patch("baar.core.budget.completion_cost", return_value=0.05):
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.model = "gpt-4o-mini"
            resp.usage.prompt_tokens = 100
            resp.usage.completion_tokens = 50

            tracker = BudgetTracker(
                total_budget=1.0,
                store=FileBudgetStore(path, namespace="session1"),
            )
            tracker.record(resp, "gpt-4o-mini")

        # New tracker, same file — should see the previous spend
        tracker2 = BudgetTracker(
            total_budget=1.0,
            store=FileBudgetStore(path, namespace="session1"),
        )
        assert tracker2.spent == pytest.approx(0.05)

    def test_tracker_spent_assignment_updates_store(self):
        from baar.core.budget import BudgetTracker
        from baar.core.stores import MemoryBudgetStore
        store = MemoryBudgetStore()
        tracker = BudgetTracker(total_budget=0.10, store=store)
        tracker._spent = 0.09
        assert tracker.remaining == pytest.approx(0.01)
        assert store.get_spent() == pytest.approx(0.09)


# ─────────────────────────────────────────────────────────
# atomic_check_and_reserve — all three backends
# ─────────────────────────────────────────────────────────

class AtomicReserveMixin:
    """
    Run against any BudgetStore implementation.
    Subclasses provide make_store() and optionally a shared-store factory
    for the concurrent-safety test.
    """

    def make_store(self, namespace="default"):
        raise NotImplementedError

    # ── basic success / failure ───────────────────────────────────────────────

    def test_reserve_within_budget_returns_true(self):
        store = self.make_store()
        result = store.atomic_check_and_reserve(total_budget=1.0, amount=0.05)
        assert result is True
        assert store.get_spent() == pytest.approx(0.05)

    def test_reserve_that_would_exceed_returns_false(self):
        store = self.make_store()
        store.add_spent(0.95)
        result = store.atomic_check_and_reserve(total_budget=1.0, amount=0.10)
        assert result is False
        # Store must not have been modified
        assert store.get_spent() == pytest.approx(0.95)

    def test_reserve_exactly_at_budget_returns_true(self):
        store = self.make_store()
        store.add_spent(0.90)
        result = store.atomic_check_and_reserve(total_budget=1.0, amount=0.10)
        assert result is True
        assert store.get_spent() == pytest.approx(1.00)

    def test_reserve_one_cent_over_budget_returns_false(self):
        store = self.make_store()
        store.add_spent(0.90)
        result = store.atomic_check_and_reserve(total_budget=1.0, amount=0.11)
        assert result is False
        assert store.get_spent() == pytest.approx(0.90)

    # ── concurrent safety ─────────────────────────────────────────────────────

    def test_concurrent_reservations_only_one_succeeds(self):
        """
        Two threads race to reserve 0.60 against a budget of 1.00 with 0.50
        already spent.  Only the first reservation (0.50 + 0.60 = 1.10 > 1.00)
        ... wait, that would fail too.  Use 0.40 already spent so that exactly
        one thread can succeed (0.40 + 0.60 = 1.00 <= 1.00) and the second
        (1.00 + 0.60 = 1.60 > 1.00) must fail.
        """
        store = self.make_store(namespace="concurrent_test")
        store.add_spent(0.40)

        results = []

        def _reserve():
            r = store.atomic_check_and_reserve(total_budget=1.0, amount=0.60)
            results.append(r)

        t1 = threading.Thread(target=_reserve)
        t2 = threading.Thread(target=_reserve)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one thread should have succeeded
        assert results.count(True) == 1
        assert results.count(False) == 1
        # Final spend must be exactly 1.00
        assert store.get_spent() == pytest.approx(1.00)


class TestAtomicCheckAndReserveMemory(AtomicReserveMixin):
    def make_store(self, namespace="default"):
        return MemoryBudgetStore()


class TestAtomicCheckAndReserveFile(AtomicReserveMixin):
    @pytest.fixture(autouse=True)
    def _set_tmp(self, tmp_path):
        self._tmp_path = tmp_path

    def make_store(self, namespace="default"):
        path = str(self._tmp_path / "atomic_budgets.json")
        return FileBudgetStore(path, namespace=namespace)


class TestAtomicCheckAndReserveSQLite(AtomicReserveMixin):
    @pytest.fixture(autouse=True)
    def _set_tmp(self, tmp_path):
        self._tmp_path = tmp_path

    def make_store(self, namespace="default"):
        db = str(self._tmp_path / "atomic_budgets.db")
        return SQLiteBudgetStore(db, namespace=namespace)
