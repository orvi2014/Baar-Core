"""Tests for BudgetWindow and WindowedBudgetStore (baar/core/budget.py)."""

import os
import tempfile
from unittest.mock import patch

import pytest

from baar.core.budget import BudgetTracker, BudgetWindow, WindowedBudgetStore
from baar.core.stores import MemoryBudgetStore, SQLiteBudgetStore, FileBudgetStore, BudgetStore


# ── BudgetWindow enum ─────────────────────────────────────────────────────────

class TestBudgetWindow:
    def test_values(self):
        assert BudgetWindow.HOURLY == "hourly"
        assert BudgetWindow.DAILY == "daily"
        assert BudgetWindow.MONTHLY == "monthly"


# ── WindowedBudgetStore ───────────────────────────────────────────────────────

class TestWindowedBudgetStore:
    def test_wraps_memory_store(self):
        base = MemoryBudgetStore()
        ws = WindowedBudgetStore(base, BudgetWindow.DAILY)
        ws.add_spent(0.05)
        assert ws.get_spent() == pytest.approx(0.05)

    def test_same_period_accumulates(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.DAILY)
        ws.add_spent(0.01)
        ws.add_spent(0.02)
        assert ws.get_spent() == pytest.approx(0.03)

    def test_new_period_resets_to_zero(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.DAILY)
        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
            ws.add_spent(0.05)
            assert ws.get_spent() == pytest.approx(0.05)

            # Advance to next day
            mock_dt.now.return_value = datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc)
            assert ws.get_spent() == pytest.approx(0.0)

    def test_hourly_period_key(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.HOURLY)
        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)
            key = ws._period_key()
            assert key == "2026-05-19T14"

    def test_daily_period_key(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.DAILY)
        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
            assert ws._period_key() == "2026-05-19"

    def test_monthly_period_key(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.MONTHLY)
        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
            assert ws._period_key() == "2026-05"

    def test_set_spent(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.DAILY)
        ws.set_spent(0.42)
        assert ws.get_spent() == pytest.approx(0.42)

    def test_reset(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.DAILY)
        ws.add_spent(0.10)
        ws.reset()
        assert ws.get_spent() == pytest.approx(0.0)

    def test_atomic_check_and_reserve_allows(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.DAILY)
        result = ws.atomic_check_and_reserve(1.0, 0.05)
        assert result is True
        assert ws.get_spent() == pytest.approx(0.05)

    def test_atomic_check_and_reserve_blocks(self):
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.DAILY)
        ws.add_spent(0.95)
        result = ws.atomic_check_and_reserve(1.0, 0.10)
        assert result is False
        assert ws.get_spent() == pytest.approx(0.95)

    def test_wraps_sqlite_store(self, tmp_path):
        db = str(tmp_path / "budget.db")
        base = SQLiteBudgetStore(db, namespace="user:alice")
        ws = WindowedBudgetStore(base, BudgetWindow.DAILY)
        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
            ws.add_spent(0.07)
            assert ws.get_spent() == pytest.approx(0.07)

            # New day — fresh zero
            mock_dt.now.return_value = datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc)
            assert ws.get_spent() == pytest.approx(0.0)

    def test_wraps_file_store(self, tmp_path):
        path = str(tmp_path / "budget.json")
        base = FileBudgetStore(path, namespace="user:bob")
        ws = WindowedBudgetStore(base, BudgetWindow.MONTHLY)
        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
            ws.add_spent(0.03)
            assert ws.get_spent() == pytest.approx(0.03)

            mock_dt.now.return_value = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
            assert ws.get_spent() == pytest.approx(0.0)

    def test_sqlite_uses_namespaced_sub_store(self, tmp_path):
        """Verifies that each period creates a distinct namespace in the DB."""
        db = str(tmp_path / "budget.db")
        base = SQLiteBudgetStore(db, namespace="user:alice")
        ws = WindowedBudgetStore(base, BudgetWindow.DAILY)
        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
            ws.add_spent(0.05)

            mock_dt.now.return_value = datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc)
            ws.add_spent(0.10)

            # Each period is independent
            mock_dt.now.return_value = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
            assert ws.get_spent() == pytest.approx(0.05)
            mock_dt.now.return_value = datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc)
            assert ws.get_spent() == pytest.approx(0.10)


# ── BudgetTracker with window ─────────────────────────────────────────────────

class TestBudgetTrackerWithWindow:
    def test_tracker_wraps_store_with_window(self):
        tracker = BudgetTracker(total_budget=0.10, window=BudgetWindow.DAILY)
        assert isinstance(tracker._store, WindowedBudgetStore)

    def test_tracker_no_window_uses_store_directly(self):
        tracker = BudgetTracker(total_budget=0.10)
        assert isinstance(tracker._store, MemoryBudgetStore)

    def test_tracker_with_window_resets_daily(self):
        tracker = BudgetTracker(total_budget=0.10, window=BudgetWindow.DAILY)
        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
            tracker._store.add_spent(0.08)
            assert tracker.spent == pytest.approx(0.08)
            assert tracker.utilization == pytest.approx(0.8)

            mock_dt.now.return_value = datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc)
            assert tracker.spent == pytest.approx(0.0)
            assert tracker.remaining == pytest.approx(0.10)

    def test_tracker_window_with_sqlite(self, tmp_path):
        db = str(tmp_path / "b.db")
        store = SQLiteBudgetStore(db, namespace="t1")
        tracker = BudgetTracker(total_budget=1.0, window=BudgetWindow.MONTHLY, store=store)
        assert isinstance(tracker._store, WindowedBudgetStore)


class TestWindowedBudgetStoreEdgeCases:
    def test_unknown_store_type_raises_type_error(self):
        """Custom BudgetStore subclasses must raise rather than silently fall back."""
        class MyRedisStore(BudgetStore):
            def get_spent(self): return 0.0
            def add_spent(self, a): return 0.0
            def set_spent(self, v): pass
            def reset(self): pass
            def atomic_check_and_reserve(self, b, a): return True

        store = MyRedisStore()
        ws = WindowedBudgetStore(store, BudgetWindow.DAILY)
        with pytest.raises(TypeError, match="MyRedisStore"):
            ws.get_spent()

    def test_memory_store_is_explicitly_handled(self):
        """MemoryBudgetStore must still work (not hit the TypeError path)."""
        from baar.core.stores import MemoryBudgetStore
        ws = WindowedBudgetStore(MemoryBudgetStore(), BudgetWindow.DAILY)
        ws.add_spent(0.05)
        assert ws.get_spent() == pytest.approx(0.05)


class TestAlertWindowRolloverReset:
    def test_once_alert_re_arms_on_window_rollover(self):
        """Alert._fired must reset when the billing window rolls over."""
        from baar.core.budget import Alert
        fired_count = [0]

        def cb(_):
            fired_count[0] += 1

        alert = Alert(threshold=0.5, callback=cb, once=True)
        tracker = BudgetTracker(total_budget=0.10, window=BudgetWindow.DAILY, alerts=[alert])

        with patch("baar.core.budget.datetime") as mock_dt:
            from datetime import datetime, timezone
            with patch("baar.core.budget.cost_per_token", return_value=(0.00003, 0.00003)), \
                 patch("baar.core.budget.completion_cost", return_value=0.00006):

                # Day 1 — alert fires
                mock_dt.now.return_value = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
                tracker._store.add_spent(0.06)
                tracker._fire_alerts()
                assert fired_count[0] == 1
                assert alert._fired is True

                # Day 2 — new period; alert should re-arm and fire again
                mock_dt.now.return_value = datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc)
                tracker._store.add_spent(0.06)
                tracker._fire_alerts()
                assert fired_count[0] == 2

    def test_reset_method_re_arms_alerts(self):
        """BudgetTracker.reset() must re-arm once=True alerts."""
        from baar.core.budget import Alert
        fired = [0]
        alert = Alert(threshold=0.5, callback=lambda _: fired.__setitem__(0, fired[0] + 1), once=True)
        tracker = BudgetTracker(total_budget=1.0, alerts=[alert])
        tracker._store.set_spent(0.6)
        tracker._fire_alerts()
        assert fired[0] == 1
        assert alert._fired is True

        tracker.reset()
        assert alert._fired is False
        assert tracker.spent == pytest.approx(0.0)

        tracker._store.set_spent(0.6)
        tracker._fire_alerts()
        assert fired[0] == 2
