"""
Tests for certified early stopping (baar/early_stop.py) and its router hook.
"""

import math
import random

import pytest
from unittest.mock import patch

from baar import (
    BAARRouter,
    BudgetExhausted,
    EarlyStopPolicy,
    RunState,
    RunStopped,
    calibrate_early_stop,
    fit_early_stop,
)
from baar.early_stop import featurize, fit_logistic, state_at, _sigmoid
from tests.test_output_overrun import completion_echo, priced  # noqa: F401  (fixture)


# ── Synthetic logged runs ─────────────────────────────────────────────────────

def make_run(rng, budget=0.30):
    """A run whose length is hidden; long runs show more tool errors early."""
    length = 1 + int(rng.expovariate(1 / 18))
    length = min(length, 90)
    err_rate = 0.05 + 0.6 * min(1.0, length / 60)
    steps, ctx, errs = [], 500, 0
    for t in range(length):
        out = rng.randint(20, 200)
        cost = ctx * 2.5e-6 + out * 1e-5
        errs += 1 if rng.random() < err_rate else 0
        steps.append(dict(prompt_tokens=ctx, completion_tokens=out, cost=cost, counters={"err": errs}))
        ctx += out + rng.randint(50, 400)
    total = sum(s["cost"] for s in steps)
    return dict(steps=steps, finished=rng.random() < 0.9, resolved=rng.random() < 0.5, _cost=total)


def make_runs(n, seed):
    rng = random.Random(seed)
    return [make_run(rng) for _ in range(n)]


BUDGET = 0.30


# ── Features and model ────────────────────────────────────────────────────────

def test_featurize_matches_research_formula():
    s0 = RunState(step=0, first_prompt_tokens=1000)
    assert featurize(s0) == [1.0, math.log1p(1000), 0.0, 0.0, 0.0, 0.0]
    s = RunState(step=3, spent=0.02, first_prompt_tokens=1000, context_tokens=4000,
                 output_tokens=[10, 30, 50], counters={"err": 4})
    expect = [1.0, math.log1p(1000), math.log1p(0.02 * 1e4), math.log1p(4000),
              math.log1p(30.0), math.log1p(50), math.log1p(4)]
    assert featurize(s, ["err"]) == pytest.approx(expect)


def test_logistic_recovers_separable_signal():
    rng = random.Random(0)
    X, y = [], []
    for _ in range(400):
        x = rng.uniform(-3, 3)
        X.append([1.0, x])
        y.append(1.0 if x + rng.gauss(0, 0.3) > 0.5 else 0.0)
    w = fit_logistic(X, y)
    assert w[1] > 2.0                       # strong positive slope
    assert -w[0] / w[1] == pytest.approx(0.5, abs=0.15)  # decision boundary


def test_state_at_reads_logged_run():
    run = make_runs(1, 3)[0]
    st = state_at(run, 2)
    assert st.step == 2
    assert st.spent == pytest.approx(run["steps"][0]["cost"] + run["steps"][1]["cost"])
    assert st.context_tokens == run["steps"][2]["prompt_tokens"] if len(run["steps"]) > 2 else True
    assert st.counters == run["steps"][1]["counters"]


# ── Policy behaviour ──────────────────────────────────────────────────────────

def always(step=0, threshold=0.5):
    return EarlyStopPolicy(checkpoints=[step], weights={step: [10.0, 0, 0, 0, 0, 0]}, threshold=threshold)


def test_should_stop_only_at_checkpoints():
    p = EarlyStopPolicy(checkpoints=[0, 5], weights={0: [10.0] + [0] * 5, 5: [10.0] + [0] * 5}, threshold=0.5)
    assert p.should_stop(RunState(step=0))
    assert not p.should_stop(RunState(step=3))
    assert p.should_stop(RunState(step=5))


def test_checkpoint_without_model_never_stops():
    p = EarlyStopPolicy(checkpoints=[0, 5], weights={0: [10.0] + [0] * 5}, threshold=0.5)
    assert not p.should_stop(RunState(step=5))


def test_threshold_none_never_stops():
    p = always(threshold=None)
    assert not p.should_stop(RunState(step=0))


def test_json_round_trip(tmp_path):
    runs = make_runs(400, 1)
    pol = fit_early_stop(runs, BUDGET, counter_keys=["err"])
    pol = calibrate_early_stop(pol, make_runs(300, 2), eps=0.05, delta=0.05)
    path = tmp_path / "es.json"
    pol.save(str(path))
    back = EarlyStopPolicy.load(str(path))
    assert back.threshold == pol.threshold
    assert back.weights == pol.weights
    assert back.counter_keys == ["err"]
    assert back.meta["calibration"].beta == pol.meta["calibration"].beta
    st = state_at(runs[0], 0)
    assert back.score(st) == pytest.approx(pol.score(st))


# ── Fitting and calibration ───────────────────────────────────────────────────

def test_fit_skips_sparse_checkpoints():
    runs = make_runs(300, 4)
    pol = fit_early_stop(runs, BUDGET, checkpoints=(0, 5, 85))
    assert 0 in pol.weights and 5 in pol.weights
    assert 85 not in pol.weights            # fewer than 50 runs still going
    assert pol.threshold is None


def test_fit_learns_that_long_runs_overrun():
    runs = make_runs(1500, 5)
    pol = fit_early_stop(runs, BUDGET, counter_keys=["err"])
    over = [r for r in runs if r["_cost"] > BUDGET and len(r["steps"]) > 20]
    under = [r for r in runs if r["_cost"] <= BUDGET and len(r["steps"]) > 20]
    s_over = sum(pol.score(state_at(r, 20)) for r in over) / len(over)
    s_under = sum(pol.score(state_at(r, 20)) for r in under) / len(under)
    assert s_over > s_under + 0.2


def test_calibrate_none_when_first_level_fails():
    # A policy that stops every run at step 0: every harmless-to-stop run is harmed,
    # so even the most conservative threshold cannot be certified.
    pol = always(0)
    pol.budget = BUDGET
    runs = make_runs(300, 6)
    out = calibrate_early_stop(pol, runs, eps=0.01, delta=0.05, taus=(0.99, 0.5))
    assert out.threshold is None
    assert out.meta["calibration"].tested == [0.99]


def test_calibrate_picks_expected_threshold():
    # Scores: run i is stopped at step 0 iff tau <= its fixed score; only runs whose
    # score exceeds 0.9 are ever harmed. With harm-free taus above 0.9 the last
    # certified tau is the smallest one above 0.9.
    class Fixed(EarlyStopPolicy):
        def score(self, state):
            return state.counters.get("s", 0.0)

    runs = []
    rng = random.Random(7)
    for i in range(400):
        doomed = i % 2 == 0
        s = rng.uniform(0.95, 1.0) if doomed else rng.uniform(0.0, 0.85)
        cost = 1.0 if doomed else 0.1
        runs.append(dict(steps=[dict(prompt_tokens=10, completion_tokens=1, cost=cost, counters={})],
                         finished=True))
        runs[-1]["steps"][0]["counters"] = {}
        runs[-1]["_s"] = s
    pol = Fixed(checkpoints=[0], weights={0: [0.0]}, threshold=None, budget=0.5)
    with patch("baar.early_stop.state_at", side_effect=lambda r, k: RunState(step=0, counters={"s": r["_s"]})):
        out = calibrate_early_stop(pol, runs, eps=0.05, delta=0.05, taus=(0.99, 0.95, 0.9, 0.5))
    # 0.99, 0.95, 0.9 stop only doomed runs (no harm); 0.5 stops harmless runs with s>=0.5
    assert out.threshold == 0.9


def test_calibration_validity_by_simulation():
    """Over many independent calibrations, the certified threshold's true harm rate
    exceeds eps in at most about delta of them."""
    eps, delta = 0.05, 0.10
    pol = fit_early_stop(make_runs(800, 100), BUDGET, counter_keys=["err"])
    pool = make_runs(4000, 101)
    taus = (0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    from baar.early_stop import _stops
    true_harm = {}
    for t in taus:
        harmed = sum(1 for r in pool if r["finished"] and r["_cost"] <= BUDGET and _stops(pol, r, t))
        true_harm[t] = harmed / len(pool)
    assert any(h > eps for h in true_harm.values())  # the family does contain unsafe levels
    reps, bad = 60, 0
    for rep in range(reps):
        cal = make_runs(250, 1000 + rep)
        out = calibrate_early_stop(pol, cal, eps=eps, delta=delta, taus=taus)
        if out.threshold is not None and true_harm[out.threshold] > eps:
            bad += 1
    assert bad / reps <= delta + 0.10


def test_resolve_harm_uses_resolved_flag():
    pol = always(0)
    pol.budget = 10.0
    runs = [dict(steps=[dict(prompt_tokens=10, completion_tokens=1, cost=0.01)], finished=True, resolved=False)
            for _ in range(300)]
    out = calibrate_early_stop(pol, runs, eps=0.02, delta=0.05, harm="resolve", taus=(0.99,))
    assert out.threshold == 0.99            # stopping unresolved runs harms nothing
    with pytest.raises(ValueError):
        calibrate_early_stop(pol, runs, eps=0.02, delta=0.05, harm="other")


# ── Router integration ────────────────────────────────────────────────────────

def make_router(**kwargs):
    kwargs.setdefault("use_llm_router", False)
    return BAARRouter(**kwargs)


def test_router_stops_before_any_call(priced):
    router = make_router(budget=1.0, early_stop=always(0))
    with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
        with pytest.raises(RunStopped) as exc:
            router.chat("hello")
    assert m.call_count == 0
    assert router.spent == 0.0
    assert isinstance(exc.value, BudgetExhausted)
    assert exc.value.step == 0 and exc.value.threshold == 0.5


def test_router_stops_at_later_checkpoint(priced):
    pol = EarlyStopPolicy(checkpoints=[1], weights={1: [10.0, 0, 0, 0, 0, 0]}, threshold=0.5)
    router = make_router(budget=1.0, early_stop=pol)
    with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
        router.chat("first")
        spent = router.spent
        with pytest.raises(RunStopped):
            router.chat("second")
    assert m.call_count == 1
    assert router.spent == spent             # nothing spent on the stopped call


@pytest.mark.asyncio
async def test_router_stops_async(priced):
    from unittest.mock import AsyncMock
    router = make_router(budget=1.0, early_stop=always(0))
    with patch("baar.router.litellm.acompletion", new=AsyncMock()) as m:
        with pytest.raises(RunStopped):
            await router.achat("hello")
    assert m.call_count == 0


def test_record_event_reaches_policy(priced):
    pol = EarlyStopPolicy(checkpoints=[0], weights={0: [-10.0, 0, 0, 0, 0, 0, 20.0]},
                          threshold=0.5, counter_keys=["err"])
    router = make_router(budget=1.0, early_stop=pol)
    assert not pol.should_stop(router.run_state(100))
    router.record_event("err", 5)
    assert router.events == {"err": 5}
    with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
        with pytest.raises(RunStopped):
            router.chat("hello")
    assert m.call_count == 0


def test_default_router_unaffected(priced):
    router = make_router(budget=1.0)
    with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
        router.chat("a")
        router.chat("b")
    assert m.call_count == 2


def test_sigmoid_is_stable():
    assert _sigmoid(1000) == 1.0
    assert _sigmoid(-1000) == 0.0
