"""Tests for budget threshold alerts (baar/core/budget.py)."""

import pytest
from unittest.mock import MagicMock, patch, call

from baar.core.budget import Alert, BudgetTracker, BudgetWindow


# ── Alert dataclass ───────────────────────────────────────────────────────────

class TestAlertDataclass:
    def test_threshold_clamped_to_0_1(self):
        assert Alert(threshold=1.5, callback=lambda _: None).threshold == 1.0
        assert Alert(threshold=-0.1, callback=lambda _: None).threshold == 0.0

    def test_once_default_true(self):
        a = Alert(threshold=0.8, callback=lambda _: None)
        assert a.once is True

    def test_once_false(self):
        a = Alert(threshold=0.8, callback=lambda _: None, once=False)
        assert a.once is False

    def test_fired_starts_false(self):
        a = Alert(threshold=0.8, callback=lambda _: None)
        assert a._fired is False


# ── BudgetTracker._fire_alerts ────────────────────────────────────────────────

class TestFireAlerts:
    def _tracker_with_spend(self, total, spent, alerts):
        tracker = BudgetTracker(total_budget=total, alerts=alerts)
        tracker._store.set_spent(spent)
        return tracker

    def test_no_alerts_no_error(self):
        tracker = BudgetTracker(total_budget=1.0)
        tracker._fire_alerts()  # should not raise

    def test_fires_when_threshold_crossed(self):
        cb = MagicMock()
        tracker = self._tracker_with_spend(1.0, 0.85, [Alert(threshold=0.8, callback=cb)])
        tracker._fire_alerts()
        cb.assert_called_once()
        info = cb.call_args[0][0]
        assert info["threshold"] == 0.8
        assert info["utilization"] == pytest.approx(0.85)

    def test_does_not_fire_below_threshold(self):
        cb = MagicMock()
        tracker = self._tracker_with_spend(1.0, 0.70, [Alert(threshold=0.8, callback=cb)])
        tracker._fire_alerts()
        cb.assert_not_called()

    def test_once_true_fires_only_once(self):
        cb = MagicMock()
        alert = Alert(threshold=0.8, callback=cb, once=True)
        tracker = self._tracker_with_spend(1.0, 0.85, [alert])
        tracker._fire_alerts()
        tracker._fire_alerts()
        assert cb.call_count == 1
        assert alert._fired is True

    def test_once_false_fires_repeatedly(self):
        cb = MagicMock()
        alert = Alert(threshold=0.8, callback=cb, once=False)
        tracker = self._tracker_with_spend(1.0, 0.90, [alert])
        tracker._fire_alerts()
        tracker._fire_alerts()
        assert cb.call_count == 2

    def test_multiple_alerts_all_fire(self):
        cb1, cb2 = MagicMock(), MagicMock()
        tracker = self._tracker_with_spend(1.0, 0.95, [
            Alert(threshold=0.8, callback=cb1),
            Alert(threshold=0.9, callback=cb2),
        ])
        tracker._fire_alerts()
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_only_crossed_thresholds_fire(self):
        cb1, cb2 = MagicMock(), MagicMock()
        tracker = self._tracker_with_spend(1.0, 0.85, [
            Alert(threshold=0.8, callback=cb1),
            Alert(threshold=0.9, callback=cb2),
        ])
        tracker._fire_alerts()
        cb1.assert_called_once()
        cb2.assert_not_called()

    def test_callback_exception_does_not_propagate(self):
        def bad_cb(_):
            raise RuntimeError("boom")
        tracker = self._tracker_with_spend(1.0, 0.85, [Alert(threshold=0.8, callback=bad_cb)])
        tracker._fire_alerts()  # must not raise

    def test_info_dict_has_all_keys(self):
        received = {}
        def cb(info):
            received.update(info)
        tracker = self._tracker_with_spend(1.0, 0.85, [Alert(threshold=0.8, callback=cb)])
        tracker._fire_alerts()
        assert "threshold" in received
        assert "utilization" in received
        assert "spent" in received
        assert "remaining" in received
        assert "total_budget" in received

    def test_remaining_is_accurate(self):
        received = {}
        def cb(info):
            received.update(info)
        tracker = self._tracker_with_spend(1.0, 0.85, [Alert(threshold=0.8, callback=cb)])
        tracker._fire_alerts()
        assert received["remaining"] == pytest.approx(0.15)
        assert received["spent"] == pytest.approx(0.85)
        assert received["total_budget"] == pytest.approx(1.0)


# ── Integration: record() and record_manual() trigger alerts ──────────────────

class TestAlertTriggeredByRecord:
    def _make_response(self, prompt=50, completion=30):
        resp = MagicMock()
        resp.model = "gpt-4o-mini"
        resp.usage.prompt_tokens = prompt
        resp.usage.completion_tokens = completion
        return resp

    def test_record_fires_alert(self):
        cb = MagicMock()
        alert = Alert(threshold=0.5, callback=cb)
        tracker = BudgetTracker(total_budget=0.001, alerts=[alert])
        with patch("baar.core.budget.completion_cost", return_value=0.0006), \
             patch("baar.core.budget.cost_per_token", return_value=(0.0003, 0.0003)):
            tracker.record(self._make_response(), model="gpt-4o-mini")
        cb.assert_called_once()

    def test_record_manual_fires_alert(self):
        cb = MagicMock()
        alert = Alert(threshold=0.5, callback=cb)
        tracker = BudgetTracker(total_budget=0.001, alerts=[alert])
        with patch("baar.core.budget.cost_per_token", return_value=(0.0003, 0.0003)):
            tracker.record_manual("gpt-4o-mini", prompt_tokens=50, completion_tokens=50)
        cb.assert_called_once()

    def test_alert_not_fired_before_threshold(self):
        cb = MagicMock()
        alert = Alert(threshold=0.9, callback=cb)
        tracker = BudgetTracker(total_budget=1.0, alerts=[alert])
        with patch("baar.core.budget.completion_cost", return_value=0.0001), \
             patch("baar.core.budget.cost_per_token", return_value=(0.00005, 0.00005)):
            tracker.record(self._make_response(), model="gpt-4o-mini")
        cb.assert_not_called()


# ── BAARRouter integration ────────────────────────────────────────────────────

class TestRouterWithAlerts:
    def test_alert_fires_on_router_chat(self):
        cb = MagicMock()
        resp = MagicMock()
        resp.model = "gpt-4o-mini"
        resp.choices[0].message.content = "hi"
        resp.usage.prompt_tokens = 50
        resp.usage.completion_tokens = 30

        with patch("baar.router.litellm.completion", return_value=resp), \
             patch("baar.core.budget.completion_cost", return_value=0.00006), \
             patch("baar.core.budget.cost_per_token", return_value=(0.00003, 0.00003)), \
             patch("baar.router.token_counter", return_value=50):
            from baar import BAARRouter, Alert
            router = BAARRouter(
                budget=0.0001,
                use_llm_router=False,
                alerts=[Alert(threshold=0.5, callback=cb)],
            )
            router.chat("hello")

        cb.assert_called_once()

    def test_record_manual_warns_on_pricing_failure(self):
        """record_manual must emit UserWarning (not silently record $0) on price failure."""
        import warnings as _warnings
        tracker = BudgetTracker(total_budget=1.0)
        with patch("baar.core.budget.cost_per_token", side_effect=Exception("no price")):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                tracker.record_manual("unknown-model", prompt_tokens=50, completion_tokens=30)
        price_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(price_warns) == 1
        assert "record_manual" in str(price_warns[0].message)

    def test_router_window_and_alert_together(self):
        cb = MagicMock()
        resp = MagicMock()
        resp.model = "gpt-4o-mini"
        resp.choices[0].message.content = "ok"
        resp.usage.prompt_tokens = 50
        resp.usage.completion_tokens = 30

        from baar import BAARRouter, Alert, BudgetWindow
        with patch("baar.router.litellm.completion", return_value=resp), \
             patch("baar.core.budget.completion_cost", return_value=0.00006), \
             patch("baar.core.budget.cost_per_token", return_value=(0.00003, 0.00003)), \
             patch("baar.router.token_counter", return_value=50):
            router = BAARRouter(
                budget=0.0001,
                use_llm_router=False,
                window=BudgetWindow.DAILY,
                alerts=[Alert(threshold=0.5, callback=cb)],
            )
            router.chat("hello")

        cb.assert_called_once()
