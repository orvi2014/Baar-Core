"""
tests/test_calibration.py — "calibrated" output mode and run-level Learn-then-Test.

Covers:
  - exact binomial p-values (vs brute force)
  - fixed-sequence Learn-then-Test selection and its validity guarantee
  - QuantileTable lookups, fallback and JSON round trip
  - caps.plan_output("calibrated") and router wiring

Runs without API keys; pricing is faked as in test_output_overrun.py.
"""

import math
import random

import pytest
from unittest.mock import patch

from baar import (
    BAARConfig,
    BAARRouter,
    BudgetExhausted,
    CalibrationResult,
    QuantileTable,
    calibrate_from_failures,
    ltt_select,
)
from baar.calibration import binom_cdf, ltt_p_value, resolve_beta
from baar.core import caps
from baar.core.budget import BudgetTracker
from baar.core.caps import OutputLimitUnavailable, plan_output

from tests.test_output_overrun import MAX_OUT, completion_echo, priced  # noqa: F401  (fixture)


def make_router(**kwargs):
    kwargs.setdefault("use_llm_router", False)
    return BAARRouter(**kwargs)


def table_for(model="gpt-4o-mini", values=range(1, 101), kind="any", min_samples=30):
    return QuantileTable({(model, kind): list(values)}, min_samples=min_samples)


# ── Binomial p-value ──────────────────────────────────────────────────────────

class TestBinomial:
    @pytest.mark.parametrize("n", [1, 5, 12, 30])
    @pytest.mark.parametrize("p", [0.01, 0.05, 0.3, 0.5, 0.9])
    def test_cdf_matches_brute_force(self, n, p):
        for k in range(-1, n + 2):
            brute = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, min(k, n) + 1))
            assert binom_cdf(k, n, p) == pytest.approx(brute, rel=1e-9, abs=1e-15)

    def test_large_n_is_finite_and_monotone(self):
        values = [binom_cdf(k, 5000, 0.05) for k in range(0, 400, 20)]
        assert all(0.0 <= v <= 1.0 for v in values)
        assert values == sorted(values)

    def test_edge_probabilities(self):
        assert binom_cdf(0, 10, 0.0) == 1.0
        assert binom_cdf(9, 10, 1.0) == 0.0
        assert binom_cdf(10, 10, 1.0) == 1.0

    def test_p_value_rejects_bad_counts(self):
        with pytest.raises(ValueError):
            ltt_p_value(11, 10, 0.05)
        with pytest.raises(ValueError):
            ltt_p_value(-1, 10, 0.05)

    def test_zero_failures_p_value(self):
        # P(Bin(100, 0.05) <= 0) = 0.95^100
        assert ltt_p_value(0, 100, 0.05) == pytest.approx(0.95**100)


# ── Learn-then-Test ───────────────────────────────────────────────────────────

class TestLTT:
    BETAS = [0.01, 0.05, 0.1, 0.2, 0.5]

    def test_stops_at_first_non_rejection(self):
        calls = []
        failures = {0.01: 0, 0.05: 2, 0.1: 60, 0.2: 0, 0.5: 0}

        def run_fn(beta, n):
            calls.append(beta)
            return failures[beta]

        res = ltt_select(self.BETAS, run_fn, n=1000, eps=0.05, delta=0.05)
        assert res.beta == 0.05
        assert calls == [0.01, 0.05, 0.1]  # 0.2 and 0.5 never run, despite 0 failures
        assert res.tested == [0.01, 0.05, 0.1]
        assert res.failures == [0, 2, 60]
        assert res.p_values[-1] > 0.05

    def test_returns_none_when_first_level_fails(self):
        res = ltt_select(self.BETAS, lambda b, n: n // 2, n=200, eps=0.05, delta=0.05)
        assert res.beta is None
        assert res.tested == [0.01]
        assert resolve_beta(res) is None

    def test_all_levels_certified(self):
        res = ltt_select(self.BETAS, lambda b, n: 0, n=1000, eps=0.05, delta=0.05)
        assert res.beta == 0.5
        assert len(res.tested) == len(self.BETAS)

    def test_calibrate_from_failures_matches_ltt(self):
        res = calibrate_from_failures(self.BETAS, [0, 2, 60], n=1000, eps=0.05, delta=0.05)
        assert res.beta == 0.05
        assert res.tested == [0.01, 0.05, 0.1]

    def test_calibrate_from_failures_partial_counts(self):
        res = calibrate_from_failures(self.BETAS, [0, 1], n=1000, eps=0.05, delta=0.05)
        assert res.beta == 0.05

    @pytest.mark.parametrize("kwargs", [
        dict(betas=[], n=10, eps=0.1, delta=0.1),
        dict(betas=[0.1], n=0, eps=0.1, delta=0.1),
        dict(betas=[0.1], n=10, eps=0.0, delta=0.1),
        dict(betas=[0.1], n=10, eps=0.1, delta=1.0),
    ])
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            ltt_select(run_fn=lambda b, n: 0, **kwargs)

    def test_guarantee_holds_over_repeated_calibrations(self):
        """
        Synthetic, non-monotone true risks. The deployed level must have risk
        <= eps in at least 1 - delta of independent calibrations.
        """
        eps, delta, n, reps = 0.1, 0.1, 200, 400
        risk = {0.01: 0.02, 0.05: 0.06, 0.1: 0.095, 0.2: 0.13, 0.3: 0.05, 0.5: 0.3}
        betas = list(risk)
        rng = random.Random(0)

        def run_fn(beta, n_runs):
            return sum(rng.random() < risk[beta] for _ in range(n_runs))

        violations = 0
        for _ in range(reps):
            res = ltt_select(betas, run_fn, n=n, eps=eps, delta=delta)
            deployed = 0.0 if res.beta is None else risk[res.beta]
            violations += deployed > eps
        # Expected violation rate <= delta; allow Monte Carlo slack.
        assert violations / reps <= delta + 0.04

    def test_result_json_round_trip(self, tmp_path):
        res = ltt_select(self.BETAS, lambda b, n: 0, n=50, eps=0.2, delta=0.1)
        path = tmp_path / "cal.json"
        res.save(str(path))
        loaded = CalibrationResult.load(str(path))
        assert loaded == res


# ── QuantileTable ─────────────────────────────────────────────────────────────

class TestQuantileTable:
    def test_quantile_values(self):
        t = table_for(values=range(1, 101))
        assert t.bound("gpt-4o-mini", "any", 0.0) == 100
        assert t.bound("gpt-4o-mini", "any", 0.1) == 90
        assert t.bound("gpt-4o-mini", "any", 0.5) == 50

    def test_beta_none_returns_none(self):
        assert table_for().bound("gpt-4o-mini", "any", None) is None

    def test_too_few_samples_returns_none(self):
        t = table_for(values=range(10), min_samples=30)
        assert t.bound("gpt-4o-mini", "any", 0.1) is None

    def test_unknown_model_returns_none(self):
        assert table_for().bound("gpt-4o", "any", 0.1) is None

    def test_sparse_kind_falls_back_to_pooled_model_samples(self):
        t = QuantileTable({("m", "tool"): range(1, 101), ("m", "final"): [5000]}, min_samples=30)
        assert t.bound("m", "tool", 0.0) == 100
        assert t.bound("m", "final", 0.0) == 5000  # pooled: 101 samples, max is 5000
        assert t.bound("m", "other", 0.5) == 51

    def test_from_records_tuples_and_mappings(self):
        recs = [("m", "tool", i) for i in range(40)] + [
            {"model": "m", "completion_tokens": 7},
            {"model": "m", "kind": "final", "completion_tokens": 9},
        ]
        t = QuantileTable.from_records(recs)
        assert t.count("m", "tool") == 40
        assert t.count("m", "any") == 1
        assert t.count("m", "final") == 1

    def test_rejects_bad_beta_and_tokens(self):
        t = table_for()
        with pytest.raises(ValueError):
            t.bound("gpt-4o-mini", "any", 1.0)
        with pytest.raises(ValueError):
            t.add("m", "any", -1)

    def test_json_round_trip(self, tmp_path):
        t = QuantileTable({("a", "tool"): [3, 1, 2], ("b", "any"): [9]}, min_samples=1)
        path = tmp_path / "q.json"
        t.save(str(path))
        loaded = QuantileTable.load(str(path))
        assert loaded.to_dict() == t.to_dict()
        assert loaded.bound("a", "tool", 0.0) == 3


# ── caps.plan_output("calibrated") ────────────────────────────────────────────

class TestPlanCalibrated:
    def plan(self, **kw):
        kw.setdefault("model", "gpt-4o-mini")
        kw.setdefault("prompt_tokens", 100)
        kw.setdefault("expected_output_tokens", 500)
        kw.setdefault("remaining", 1.0)
        kw.setdefault("tracker", BudgetTracker(total_budget=1.0))
        return plan_output("calibrated", **kw)

    def test_mode_is_registered(self):
        assert "calibrated" in caps.OUTPUT_OVERRUN_MODES
        caps.validate_mode("calibrated")

    def test_reserves_quantile_not_max_and_sends_no_max_tokens(self, priced):
        p = self.plan(quantiles=table_for(values=range(1, 1001)), beta=0.1)
        assert p.completion_tokens == 900
        assert p.completion_tokens < MAX_OUT["gpt-4o-mini"]
        assert p.max_tokens is None
        assert p.budget_limited is False

    def test_user_limit_caps_plan_and_is_forwarded(self, priced):
        p = self.plan(quantiles=table_for(values=range(1, 1001)), beta=0.1, max_output_tokens=300)
        assert p.completion_tokens == 300
        assert p.max_tokens == 300

    def test_no_table_falls_back_to_preflight(self, priced):
        p = self.plan(quantiles=None, beta=0.1)
        assert p.completion_tokens == MAX_OUT["gpt-4o-mini"]

    def test_beta_none_falls_back_to_preflight(self, priced):
        p = self.plan(quantiles=table_for(), beta=None)
        assert p.completion_tokens == MAX_OUT["gpt-4o-mini"]

    def test_fallback_without_known_limit_raises(self, priced):
        with pytest.raises(OutputLimitUnavailable):
            self.plan(model="mystery", quantiles=None, beta=0.1)

    def test_kind_selects_stratum(self, priced):
        t = QuantileTable({("gpt-4o-mini", "tool"): range(1, 101),
                           ("gpt-4o-mini", "final"): range(1001, 1101)})
        assert self.plan(quantiles=t, beta=0.0, kind="tool").completion_tokens == 100
        assert self.plan(quantiles=t, beta=0.0, kind="final").completion_tokens == 1100


# ── Router wiring ─────────────────────────────────────────────────────────────

class TestRouterCalibrated:
    def test_requires_table_and_beta(self):
        with pytest.raises(ValueError, match="calibrated"):
            make_router(budget=1.0, on_output_overrun="calibrated")
        with pytest.raises(ValueError, match="calibrated"):
            make_router(budget=1.0, on_output_overrun="calibrated", output_quantiles=table_for())

    def test_per_call_without_calibration_rejected(self, priced):
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(ValueError, match="calibrated"):
                router.chat("hello", on_output_overrun="calibrated")
            m.assert_not_called()

    def test_invalid_beta_rejected(self):
        with pytest.raises(ValueError):
            make_router(budget=1.0, output_quantiles=table_for(), calibration_beta=1.5)

    def test_never_sends_max_tokens(self, priced):
        router = make_router(budget=1.0, on_output_overrun="calibrated",
                             output_quantiles=table_for(values=range(1, 1001)), calibration_beta=0.1)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert "max_tokens" not in m.call_args.kwargs
        assert router.log.steps[-1].truncated_by_budget is False

    def test_admits_call_that_preflight_blocks(self, priced):
        # small worst case ≈ $0.0098 > budget; quantile plan of 1000 tokens ≈ $0.0006 fits.
        q = table_for(values=range(1, 1001))
        pre = make_router(budget=0.005, on_output_overrun="preflight")
        cal = make_router(budget=0.005, on_output_overrun="calibrated",
                          output_quantiles=q, calibration_beta=0.0)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            with pytest.raises(BudgetExhausted):
                pre.chat("hello")
            assert m.call_count == 0
            cal.chat("hello")
            assert m.call_count == 1

    def test_accepts_calibration_result(self, priced):
        res = CalibrationResult(beta=0.2, eps=0.05, delta=0.05, n=100)
        router = make_router(budget=1.0, on_output_overrun="calibrated",
                             output_quantiles=table_for(), calibration_beta=res)
        assert router._calibration_beta == 0.2

    def test_uncertified_result_behaves_like_preflight(self, priced):
        res = CalibrationResult(beta=None, eps=0.05, delta=0.05, n=100)
        router = make_router(budget=0.005, on_output_overrun="calibrated",
                             output_quantiles=table_for(values=range(1, 1001)), calibration_beta=res)
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(BudgetExhausted):
                router.chat("hello")
            m.assert_not_called()

    def test_from_config_carries_fields(self):
        q = table_for()
        router = BAARRouter.from_config(BAARConfig(
            budget=1.0, use_llm_router=False, on_output_overrun="calibrated",
            output_quantiles=q, calibration_beta=0.1,
        ))
        assert router._output_quantiles is q
        assert router._calibration_beta == 0.1
