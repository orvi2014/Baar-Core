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


# ─────────────────────────────────────────────────────────
# record_manual (streaming cost recording)
# ─────────────────────────────────────────────────────────

class TestRecordManual:
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    def test_record_manual_updates_spent(self, mock_cpt):
        tracker = BudgetTracker(total_budget=0.10)
        rec = tracker.record_manual("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50
        assert rec.cost == pytest.approx(0.000001 + 0.000002)
        assert tracker.spent == pytest.approx(rec.cost)

    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    def test_record_manual_increments_step(self, mock_cpt):
        tracker = BudgetTracker(total_budget=0.10)
        assert tracker._step == 0
        tracker.record_manual("gpt-4o-mini", 50, 25)
        assert tracker._step == 1

    @patch("baar.core.budget.cost_per_token", side_effect=Exception("no pricing"))
    def test_record_manual_falls_back_to_zero_cost_on_error(self, mock_cpt):
        tracker = BudgetTracker(total_budget=0.10)
        rec = tracker.record_manual("unknown-model", 100, 50)
        assert rec.cost == 0.0
        # Still records the tokens
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50

    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    def test_record_manual_appears_in_summary_records(self, mock_cpt):
        tracker = BudgetTracker(total_budget=0.10)
        tracker.record_manual("gpt-4o-mini", 60, 30)
        summary = tracker.summary()
        assert summary["steps"] == 1
        assert len(summary["records"]) == 1
        assert summary["records"][0]["model"] == "gpt-4o-mini"


# ─────────────────────────────────────────────────────────
# check_and_reserve — atomic budget reservation
# ─────────────────────────────────────────────────────────

class TestCheckAndReserve:
    @patch("baar.core.budget.cost_per_token", return_value=(0.01, 0.02))
    def test_passes_when_enough_budget(self, mock_cpt):
        """check_and_reserve should succeed and not raise when budget is sufficient."""
        tracker = BudgetTracker(total_budget=1.0)
        # Should not raise
        tracker.check_and_reserve(0.05)
        # The reservation should be reflected in the spent amount
        assert tracker.spent == pytest.approx(0.05)

    @patch("baar.core.budget.cost_per_token", return_value=(0.01, 0.02))
    def test_raises_budget_exceeded_when_would_exceed(self, mock_cpt):
        """check_and_reserve should raise BudgetExceeded if amount > remaining."""
        tracker = BudgetTracker(total_budget=0.10)
        tracker._spent = 0.09  # Only $0.01 remaining

        with pytest.raises(BudgetExceeded) as exc_info:
            tracker.check_and_reserve(0.05)

        assert exc_info.value.requested == pytest.approx(0.05)
        assert exc_info.value.model == "(reservation)"

    @patch("baar.core.budget.cost_per_token", return_value=(0.01, 0.02))
    def test_store_not_modified_on_failure(self, mock_cpt):
        """When check_and_reserve fails, the store must remain unchanged."""
        tracker = BudgetTracker(total_budget=0.10)
        tracker._spent = 0.09
        spent_before = tracker.spent

        try:
            tracker.check_and_reserve(0.05)
        except BudgetExceeded:
            pass

        assert tracker.spent == pytest.approx(spent_before)

    @patch("baar.core.budget.cost_per_token", return_value=(0.01, 0.02))
    def test_reservation_reflected_in_remaining(self, mock_cpt):
        """After a successful check_and_reserve, remaining decreases by the amount."""
        tracker = BudgetTracker(total_budget=1.0)
        tracker.check_and_reserve(0.30)
        assert tracker.remaining == pytest.approx(0.70)
        assert tracker.spent == pytest.approx(0.30)

    @patch("baar.core.budget.cost_per_token", return_value=(0.01, 0.02))
    def test_sequential_reservations_accumulate(self, mock_cpt):
        """Multiple successful reservations are cumulative."""
        tracker = BudgetTracker(total_budget=1.0)
        tracker.check_and_reserve(0.10)
        tracker.check_and_reserve(0.20)
        assert tracker.spent == pytest.approx(0.30)

    @patch("baar.core.budget.cost_per_token", return_value=(0.01, 0.02))
    def test_exactly_at_budget_succeeds(self, mock_cpt):
        """Reserving exactly the remaining budget should succeed."""
        tracker = BudgetTracker(total_budget=0.10)
        tracker._spent = 0.05
        # Should not raise
        tracker.check_and_reserve(0.05)
        assert tracker.spent == pytest.approx(0.10)
