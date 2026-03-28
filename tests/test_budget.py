"""
tests/test_budget.py — BudgetTracker unit tests

Runs entirely without API keys. Uses mock responses.
Every test is deterministic and fast.
"""

import pytest
from unittest.mock import MagicMock, patch
from baar.core.budget import BudgetTracker, BudgetExceeded


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def make_mock_response(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50):
    """Build a mock LiteLLM response object."""
    response = MagicMock()
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.choices[0].message.content = "mock response"
    return response


# ─────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────

class TestBudgetTrackerInit:
    def test_initial_state(self):
        tracker = BudgetTracker(total_budget=1.00)
        assert tracker.total_budget == 1.00
        assert tracker.spent == 0.0
        assert tracker.remaining == 1.00
        assert tracker.utilization == 0.0
        assert tracker.records == []

    def test_zero_budget_utilization(self):
        tracker = BudgetTracker(total_budget=0.0)
        assert tracker.utilization == 1.0  # already "full"

    def test_negative_remaining_clamped_to_zero(self):
        """Remaining should never go below 0 even if overspent."""
        tracker = BudgetTracker(total_budget=0.01)
        # Manually force overspend
        tracker._spent = 0.05
        assert tracker.remaining == 0.0


# ─────────────────────────────────────────────────────────
# Cost recording
# ─────────────────────────────────────────────────────────

class TestCostRecording:
    @patch("baar.core.budget.completion_cost", return_value=0.000250)
    def test_record_updates_spent(self, mock_cost):
        tracker = BudgetTracker(total_budget=1.00)
        resp = make_mock_response()
        record = tracker.record(resp, "gpt-4o-mini")

        assert tracker.spent == pytest.approx(0.000250)
        assert tracker.remaining == pytest.approx(0.999750)
        assert record.cost == pytest.approx(0.000250)
        assert record.step == 1

    @patch("baar.core.budget.completion_cost", return_value=0.000250)
    def test_record_accumulates_across_steps(self, mock_cost):
        tracker = BudgetTracker(total_budget=1.00)
        resp = make_mock_response()

        tracker.record(resp, "gpt-4o-mini")
        tracker.record(resp, "gpt-4o-mini")
        tracker.record(resp, "gpt-4o-mini")

        assert tracker.spent == pytest.approx(0.000750)
        assert len(tracker.records) == 3
        assert tracker.records[2].cumulative_cost == pytest.approx(0.000750)

    @patch("baar.core.budget.completion_cost", side_effect=Exception("API error"))
    def test_cost_fallback_when_completion_cost_fails(self, mock_cost):
        """Falls back to cost_per_token when completion_cost raises."""
        tracker = BudgetTracker(total_budget=1.00)
        resp = make_mock_response(prompt_tokens=100, completion_tokens=50)

        with patch("baar.core.budget.cost_per_token", return_value=(0.0001, 0.0002)):
            record = tracker.record(resp, "gpt-4o-mini")
            assert record.cost == pytest.approx(0.0003)

    @patch("baar.core.budget.completion_cost", side_effect=Exception("fail"))
    def test_cost_returns_zero_on_double_failure(self, mock_cost):
        """Returns 0.0 cost if both methods fail — never crashes."""
        tracker = BudgetTracker(total_budget=1.00)
        resp = make_mock_response()
        resp.usage = None  # no usage data either

        record = tracker.record(resp, "gpt-4o-mini")
        assert record.cost == 0.0


# ─────────────────────────────────────────────────────────
# Budget enforcement (BCD)
# ─────────────────────────────────────────────────────────

class TestBudgetConstrainedDecoding:
    @patch("baar.core.budget.cost_per_token", return_value=(0.0001, 0.0002))
    def test_check_affordability_passes_when_enough_budget(self, mock_cpt):
        tracker = BudgetTracker(total_budget=1.00)
        # Should not raise
        tracker.check_affordability("gpt-4o", prompt_tokens=1000)

    @patch("baar.core.budget.cost_per_token", return_value=(0.05, 0.10))
    def test_check_affordability_raises_budget_exceeded(self, mock_cpt):
        tracker = BudgetTracker(total_budget=0.10)
        tracker._spent = 0.09  # Only $0.01 remaining

        with pytest.raises(BudgetExceeded) as exc_info:
            tracker.check_affordability("gpt-4o", prompt_tokens=5000)

        assert "gpt-4o" in str(exc_info.value)
        assert exc_info.value.remaining == pytest.approx(0.01)

    def test_budget_exceeded_attributes(self):
        exc = BudgetExceeded(requested=0.05, remaining=0.01, model="gpt-4o")
        assert exc.requested == 0.05
        assert exc.remaining == 0.01
        assert exc.model == "gpt-4o"

    @patch("baar.core.budget.cost_per_token", side_effect=Exception("pricing failure"))
    def test_check_affordability_passes_on_pricing_failure(self, mock_cpt):
        """If we can't estimate cost, we allow the call (fail open, not closed)."""
        tracker = BudgetTracker(total_budget=0.001)
        tracker._spent = 0.0009
        # Should not raise even with near-zero budget when cost estimation fails
        tracker.check_affordability("gpt-4o", prompt_tokens=1000)


# ─────────────────────────────────────────────────────────
# Summary and utilization
# ─────────────────────────────────────────────────────────

class TestSummary:
    @patch("baar.core.budget.completion_cost", return_value=0.001)
    def test_summary_structure(self, mock_cost):
        tracker = BudgetTracker(total_budget=0.10)
        resp = make_mock_response()
        tracker.record(resp, "gpt-4o-mini")

        summary = tracker.summary()
        assert "total_budget" in summary
        assert "spent" in summary
        assert "remaining" in summary
        assert "utilization_pct" in summary
        assert "records" in summary
        assert len(summary["records"]) == 1

    @patch("baar.core.budget.completion_cost", return_value=0.05)
    def test_utilization_at_50_percent(self, mock_cost):
        tracker = BudgetTracker(total_budget=0.10)
        resp = make_mock_response()
        tracker.record(resp, "gpt-4o")

        assert tracker.utilization == pytest.approx(0.5)
        assert tracker.summary()["utilization_pct"] == pytest.approx(50.0)
