"""
tests/test_models.py — RoutingLog and StepResult tests.

Tests the data layer: log aggregation, savings calculation,
report printing. All without API calls.
"""

import pytest
from unittest.mock import patch, MagicMock
from baar.core.models import RoutingLog, StepResult
from baar.core.router import RoutingDecision, ModelTier


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def make_step(
    step_num=1,
    task="test task",
    tier=ModelTier.SMALL,
    model="gpt-4o-mini",
    complexity=0.3,
    cost=0.000025,
    cumulative=0.000025,
    forced=False,
):
    decision = RoutingDecision(
        tier=tier,
        model=model,
        confidence=0.8,
        complexity_score=complexity,
        reason="test reason",
        forced_by_budget=forced,
    )
    return StepResult(
        step_num=step_num,
        task=task,
        decision=decision,
        response_text="response",
        cost=cost,
        cumulative_cost=cumulative,
        prompt_tokens=50,
        completion_tokens=30,
        latency_ms=120.0,
    )


# ─────────────────────────────────────────────────────────
# RoutingLog aggregation
# ─────────────────────────────────────────────────────────

class TestRoutingLogAggregation:
    def test_empty_log(self):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        assert log.total_steps == 0
        assert log.total_cost == 0.0
        assert log.big_calls == 0
        assert log.small_calls == 0

    def test_add_steps(self):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(step_num=1, cost=0.001, cumulative=0.001))
        log.add(make_step(step_num=2, cost=0.002, cumulative=0.003))
        assert log.total_steps == 2
        assert log.total_cost == pytest.approx(0.003)

    def test_big_and_small_counts(self):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(tier=ModelTier.SMALL, model="gpt-4o-mini"))
        log.add(make_step(tier=ModelTier.SMALL, model="gpt-4o-mini"))
        log.add(make_step(tier=ModelTier.BIG, model="gpt-4o"))
        assert log.small_calls == 2
        assert log.big_calls == 1

    def test_budget_forced_downgrade_count(self):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(forced=False))
        log.add(make_step(forced=True))
        log.add(make_step(forced=True))
        assert log.budget_forced_downgrades == 2

    def test_pct_routed_to_small(self):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(tier=ModelTier.SMALL))
        log.add(make_step(tier=ModelTier.SMALL))
        log.add(make_step(tier=ModelTier.SMALL))
        log.add(make_step(tier=ModelTier.BIG))
        summary = log.summary()
        assert summary["pct_routed_to_small"] == pytest.approx(75.0)


# ─────────────────────────────────────────────────────────
# Savings calculation
# ─────────────────────────────────────────────────────────

class TestSavingsCalculation:
    def test_all_small_shows_positive_savings(self):
        """If we routed everything to SMALL, savings vs always-big should be positive."""
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(tier=ModelTier.SMALL, cost=0.001, cumulative=0.001))
        log.add(make_step(tier=ModelTier.SMALL, cost=0.001, cumulative=0.002))
        savings = log.savings_vs_always_big()
        assert savings["savings_pct"] > 0
        assert savings["saved_usd"] > 0

    def test_all_big_shows_zero_savings(self):
        """If we used BIG for everything, savings should be ~0%."""
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(tier=ModelTier.BIG, model="gpt-4o", cost=0.005))
        log.add(make_step(tier=ModelTier.BIG, model="gpt-4o", cost=0.005))
        savings = log.savings_vs_always_big()
        # When all calls ARE big, estimated_always_big ≈ actual cost
        assert savings["savings_pct"] == pytest.approx(0.0, abs=1.0)

    def test_savings_never_negative(self):
        """Savings % should never be negative."""
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(tier=ModelTier.BIG, cost=0.01))
        savings = log.savings_vs_always_big()
        assert savings["savings_pct"] >= 0

    def test_empty_log_savings(self):
        """Empty log should not crash."""
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        savings = log.savings_vs_always_big()
        assert savings["savings_pct"] == 0.0


# ─────────────────────────────────────────────────────────
# StepResult
# ─────────────────────────────────────────────────────────

class TestStepResult:
    def test_used_big_property(self):
        big_step = make_step(tier=ModelTier.BIG, model="gpt-4o")
        small_step = make_step(tier=ModelTier.SMALL, model="gpt-4o-mini")
        assert big_step.used_big is True
        assert small_step.used_big is False

    def test_model_used_property(self):
        step = make_step(model="gpt-4o-mini")
        assert step.model_used == "gpt-4o-mini"

    def test_to_dict_has_required_keys(self):
        step = make_step()
        d = step.to_dict()
        required = [
            "step", "task_preview", "model", "tier", "complexity_score",
            "confidence", "routing_reason", "forced_by_budget",
            "prompt_tokens", "completion_tokens", "cost_usd",
            "cumulative_cost_usd", "latency_ms",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_truncates_long_task(self):
        long_task = "x" * 200
        step = make_step(task=long_task)
        d = step.to_dict()
        assert len(d["task_preview"]) <= 83  # 80 + "..."

    def test_to_dict_does_not_truncate_short_task(self):
        short_task = "short task"
        step = make_step(task=short_task)
        d = step.to_dict()
        assert d["task_preview"] == short_task


# ─────────────────────────────────────────────────────────
# Summary structure
# ─────────────────────────────────────────────────────────

class TestSummaryStructure:
    def test_summary_keys(self):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step())
        summary = log.summary()
        required = [
            "budget_usd", "spent_usd", "remaining_usd", "utilization_pct",
            "total_steps", "small_model_calls", "big_model_calls",
            "budget_forced_downgrades", "pct_routed_to_small",
            "savings_vs_always_big", "steps",
        ]
        for k in required:
            assert k in summary

    def test_utilization_at_50_percent(self):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(cost=0.05, cumulative=0.05))
        summary = log.summary()
        assert summary["utilization_pct"] == pytest.approx(50.0)

    def test_remaining_calculation(self):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(cost=0.03, cumulative=0.03))
        summary = log.summary()
        assert summary["remaining_usd"] == pytest.approx(0.07)

    def test_print_report_does_not_crash(self, capsys):
        log = RoutingLog(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        log.add(make_step(tier=ModelTier.SMALL, cost=0.001, cumulative=0.001))
        log.add(make_step(tier=ModelTier.BIG, model="gpt-4o", cost=0.01, cumulative=0.011))
        log.print_report()  # should not raise
        captured = capsys.readouterr()
        assert "BAAR ROUTING REPORT" in captured.out
        assert "Saved" in captured.out
