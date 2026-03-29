from types import SimpleNamespace

import pytest

import benchmarks.standard_eval as se
from benchmarks.data_loader import StandardTask


def test_percentile_helper_basic():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert se._percentile(vals, 0) == 1.0
    assert se._percentile(vals, 100) == 4.0
    assert se._percentile(vals, 50) == pytest.approx(2.5)


def test_derive_alpha_from_data_percentile(monkeypatch):
    tasks = [
        StandardTask(id="t1", dataset="x", task="a", ground_truth=""),
        StandardTask(id="t2", dataset="x", task="bb", ground_truth=""),
        StandardTask(id="t3", dataset="x", task="ccc", ground_truth=""),
    ]

    class FakeRouter:
        def decide(self, task, remaining_budget, budget_utilization):
            return SimpleNamespace(model="small", tier=SimpleNamespace(value="small"))

    class FakeTracker:
        utilization = 0.0

        def estimate_cost(self, model, prompt_tokens):
            return float(prompt_tokens)

    class FakeBAAR:
        def __init__(self, budget, value_fn, small_exploration_rate):
            self._router = FakeRouter()
            self._tracker = FakeTracker()
            self.remaining = budget

    monkeypatch.setattr(se, "BAARRouter", FakeBAAR)
    monkeypatch.setattr(se, "token_counter", lambda text, model: len(text))
    value_fn = lambda task: float(len(task))

    alpha, records, ratios = se.derive_alpha_from_data(
        tasks=tasks,
        budget=1.0,
        value_fn=value_fn,
        reject_rate_target=0.2,
        source="percentile",
        sample_size=10,
        small_exploration_rate=0.0,
    )

    assert len(records) == 3
    assert ratios == [1.0, 1.0, 1.0]
    assert alpha == pytest.approx(1.0)
