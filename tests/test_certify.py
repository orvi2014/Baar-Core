"""
Tests for run logs (baar/runlog.py), certification from logs (baar/certify.py),
the guarantee monitor (baar/monitor.py), the router's run logging and the
baar-calibrate CLI.
"""

import gzip
import json
import math
import random

import pytest
from unittest.mock import patch

from baar import BAARRouter, EarlyStopPolicy, RunStopped
from baar.calibrate_cli import main as cli_main
from baar.calibration import CalibrationResult, QuantileTable
from baar.certify import (
    Certificate,
    NotEnoughRuns,
    _level_plan,
    certify_calibrated,
    certify_early_stop,
    min_calibration_runs,
    min_total_runs,
    overspends,
    replay_spend,
)
from baar.monitor import GuaranteeAlarm, GuaranteeMonitor
from baar.runlog import load_runs, split_runs
from tests.test_output_overrun import completion_echo, priced  # noqa: F401  (fixture)

P_IN, P_OUT = 1e-6, 4e-6
YMAX = 4000


# ── Synthetic logged runs ─────────────────────────────────────────────────────

def make_run(rng, rid, model="m"):
    """An agent run: growing context, heavy-tailed outputs, priced per token."""
    steps, ctx = [], 600
    for t in range(rng.randint(2, 12)):
        out = min(YMAX, int(rng.lognormvariate(5.0, 0.8)))
        real = ctx + rng.randint(0, 30)
        steps.append({"type": "step", "run": rid, "step": t, "model": model, "est_prompt_tokens": ctx,
                      "prompt_tokens": real, "completion_tokens": out, "cost": real * P_IN + out * P_OUT,
                      "price_in": P_IN, "price_out": P_OUT, "max_output_tokens": YMAX,
                      "counters": {"err": t // 3}})
        ctx += out + rng.randint(100, 500)
    return {"run": rid, "finished": rng.random() < 0.9, "resolved": rng.random() < 0.6,
            "stopped_by_budget": False, "budget": None, "cost": sum(s["cost"] for s in steps), "steps": steps}


def make_runs(n, seed=0):
    rng = random.Random(seed)
    return [make_run(rng, f"r{i}") for i in range(n)]


def write_log(path, runs, gz=False):
    op = gzip.open if gz else open
    with op(path, "wt") as f:
        for r in runs:
            for s in r["steps"]:
                f.write(json.dumps(s) + "\n")
            f.write(json.dumps({"type": "run", "run": r["run"], "finished": r["finished"],
                                "resolved": r["resolved"], "stopped_by_budget": r["stopped_by_budget"],
                                "cost": r["cost"]}) + "\n")


def budget_at(runs, q):
    costs = sorted(r["cost"] for r in runs)
    return costs[int(q * (len(costs) - 1))]


# ── Run logs ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gz", [False, True])
def test_load_runs_round_trip(tmp_path, gz):
    runs = make_runs(5)
    path = tmp_path / ("log.jsonl.gz" if gz else "log.jsonl")
    write_log(str(path), runs, gz=gz)
    out = load_runs(str(path))
    assert [r["run"] for r in out.runs] == [r["run"] for r in runs]
    assert out.runs[0]["steps"] == runs[0]["steps"]
    assert out.runs[0]["cost"] == pytest.approx(runs[0]["cost"])


def test_load_runs_skips_incomplete_and_stopped(tmp_path):
    runs = make_runs(4)
    runs[1]["stopped_by_budget"] = True
    path = str(tmp_path / "log.jsonl")
    write_log(path, runs[:3])
    with open(path, "a") as f:                       # run r3: steps but no end line
        for s in runs[3]["steps"]:
            f.write(json.dumps(s) + "\n")
        f.write(json.dumps({"type": "run", "run": "empty", "finished": True}) + "\n")
    out = load_runs(path)
    assert [r["run"] for r in out.runs] == ["r0", "r2"]
    assert (out.incomplete, out.stopped_by_budget, out.empty) == (1, 1, 1)
    assert [r["run"] for r in load_runs(path, include_stopped=True).runs] == ["r0", "r1", "r2"]


def test_load_runs_sorts_steps_and_reports_bad_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"run": "a", "step": 1, "cost": 2}\n{"run": "a", "step": 0, "cost": 1}\n'
                    '{"type": "run", "run": "a", "finished": true}\n')
    assert [s["step"] for s in load_runs(str(path)).runs[0]["steps"]] == [0, 1]
    path.write_text('{"run": "a"}\nnot json\n')
    with pytest.raises(ValueError, match=":2:"):
        load_runs(str(path))
    path.write_text('{"step": 0}\n')
    with pytest.raises(ValueError, match="no 'run' id"):
        load_runs(str(path))


def test_split_runs_is_a_fixed_partition():
    a = split_runs(100, (0.3, 0.7), seed=3)
    assert a == split_runs(100, (0.3, 0.7), seed=3)
    assert len(a[0]) == 30 and sorted(a[0] + a[1]) == list(range(100))
    assert a != split_runs(100, (0.3, 0.7), seed=4)


# ── Replay ────────────────────────────────────────────────────────────────────

def step(cost, est=100, out=10):
    return {"model": "m", "est_prompt_tokens": est, "prompt_tokens": est, "completion_tokens": out,
            "cost": cost, "price_in": 0.0, "price_out": 0.001, "max_output_tokens": 100}


def test_replay_admits_while_reservation_fits():
    run = {"steps": [step(0.01), step(0.01), step(0.01)]}
    plan = lambda s: 50                                  # reserve 50 * 0.001 = 0.05
    assert replay_spend(run, plan, budget=0.065, margin=0.0) == (pytest.approx(0.02), 2)
    assert replay_spend(run, plan, budget=1.0, margin=0.0) == (pytest.approx(0.03), None)
    assert replay_spend(run, lambda s: None, budget=1.0, margin=0.0) == (0.0, 0)


def test_replay_prices_prompt_with_margin():
    s = dict(step(0.0), price_in=0.001, price_out=0.0)
    run = {"steps": [s]}
    assert replay_spend(run, lambda _: 0, budget=0.1, margin=0.0)[1] is None      # 100 tokens = 0.1
    assert replay_spend(run, lambda _: 0, budget=0.1, margin=0.2)[1] == 0         # 120 tokens > 0.1


def test_overspend_counts_runs_refused_after_crossing_the_budget():
    # The first call costs more than it reserved and crosses the budget; the
    # second is then refused. The run still spent more than its budget.
    run = {"steps": [step(0.2), step(0.01)]}
    spent, refused_at = replay_spend(run, lambda s: 10, budget=0.1, margin=0.0)
    assert refused_at == 1 and spent == pytest.approx(0.2)
    assert overspends(run, lambda s: 10, budget=0.1, margin=0.0)


def test_level_plan_caps_quantile_at_worst_case():
    table = QuantileTable({("m", "any"): [10] * 20 + [5000] * 20}, min_samples=30)
    s = step(0.0)
    assert _level_plan(table, 0.01, None)(s) == 100      # quantile 5000 capped at max_output 100
    assert _level_plan(table, 0.9, None)(s) == 10
    assert _level_plan(table, None, None)(s) == 100      # fallback: worst case
    assert _level_plan(table, None, None)(dict(s, max_output_tokens=None)) is None
    assert _level_plan(table, None, 77)(dict(s, max_output_tokens=None)) == 77
    thin = QuantileTable({("m", "any"): [10] * 5}, min_samples=30)
    assert _level_plan(thin, 0.5, None)(s) == 100        # too few samples: worst case


def test_prices_argument_and_missing_prices():
    s = step(0.01)
    del s["price_in"], s["price_out"]
    with pytest.raises(ValueError, match="no price"):
        replay_spend({"steps": [s]}, lambda _: 1, 1.0, 0.0)
    assert replay_spend({"steps": [s]}, lambda _: 1, 1.0, 0.0, prices={"m": (0, 0.001)})[1] is None


def test_unknown_price_in_log_needs_prices():
    s = dict(step(0.01), price_in=None, price_out=None)
    with pytest.raises(ValueError, match="no price"):
        replay_spend({"steps": [s]}, lambda _: 1, 1.0, 0.0)


def brute_force_overspend(run, u, budget, margin):
    spent = 0.0
    for s in run["steps"]:
        planned = min(u, s["max_output_tokens"])
        if spent + s["price_in"] * math.ceil(s["est_prompt_tokens"] * (1 + margin)) \
                + s["price_out"] * planned > budget:
            break
        spent += s["cost"]
    return spent > budget


def test_replay_matches_brute_force():
    runs = make_runs(300, seed=5)
    budget = budget_at(runs, 0.6)
    for u in [50, 150, 400, 4000]:
        plan = lambda s, u=u: min(u, s["max_output_tokens"])
        assert [overspends(r, plan, budget, 0.2) for r in runs] == \
               [brute_force_overspend(r, u, budget, 0.2) for r in runs]


# ── certify_calibrated ────────────────────────────────────────────────────────

def test_min_runs():
    assert min_calibration_runs(0.05, 0.05) == 59
    assert min_calibration_runs(0.01, 0.05) == 299
    assert min_calibration_runs(0.02, 0.05) == 149
    n = min_total_runs(0.05, 0.05, 0.3)
    assert n - round(0.3 * n) >= 59 and all(t - round(0.3 * t) < 59 for t in range(59, n))


def test_not_enough_runs_says_how_many():
    runs = make_runs(40)
    with pytest.raises(NotEnoughRuns) as exc:
        certify_calibrated(runs, budget=1.0, eps=0.05)
    assert exc.value.have == 40 and exc.value.need == min_total_runs(0.05, 0.05, 0.3)
    assert f"log {exc.value.need - 40} more" in str(exc.value)


def test_certifies_and_round_trips(tmp_path):
    runs = make_runs(600, seed=1)
    budget = budget_at(runs, 0.9)
    cert = certify_calibrated(runs, budget=budget, eps=0.05, delta=0.05)
    assert cert.certified and cert.kind == "calibrated"
    assert cert.n_calibration == 420 and cert.n_runs == 600
    path = str(tmp_path / "cert.json")
    cert.save(path)
    back = Certificate.load(path)
    assert back.level == cert.level and back.payload == cert.payload
    kw = back.router_kwargs()
    assert kw["on_output_overrun"] == "calibrated" and kw["prompt_margin"] == 0.2
    assert isinstance(kw["output_quantiles"], QuantileTable)
    assert isinstance(kw["calibration_beta"], CalibrationResult) and kw["calibration_beta"].beta == cert.level
    assert "Certified" in cert.summary() and str(cert.eps) in cert.summary()


def underestimated_runs(n, seed):
    """Prompt estimates far below the billed prompt, and prompts that dominate the cost."""
    runs = make_runs(n, seed)
    for r in runs:
        for s in r["steps"]:
            s["est_prompt_tokens"], s["price_in"] = 1, 1e-4
            s["cost"] = s["prompt_tokens"] * 1e-4 + s["completion_tokens"] * P_OUT
        r["cost"] = sum(s["cost"] for s in r["steps"])
    return runs


def test_a_tight_budget_refuses_rather_than_overspends():
    # Refusing is not overspending: at a tight budget the aggressive levels still pass.
    runs = make_runs(400, seed=2)
    assert certify_calibrated(runs, budget=budget_at(runs, 0.05), eps=0.05).certified


def test_nothing_certified_when_every_level_overspends():
    runs = underestimated_runs(400, seed=2)
    cert = certify_calibrated(runs, budget=budget_at(runs, 0.5), eps=0.05)
    assert not cert.certified and cert.level is None
    assert "Not certified" in cert.summary()
    assert cert.router_kwargs()["calibration_beta"].beta is None


def test_warns_when_fallback_overspends():
    runs = underestimated_runs(300, seed=3)
    cert = certify_calibrated(runs, budget=budget_at(runs, 0.5), eps=0.05)
    assert cert.diagnostics["fallback_failures"] > 0
    assert any("worst-case reservation" in w for w in cert.diagnostics["warnings"])


def test_validity_on_fresh_runs():
    """Over repeated calibrations, the held-out overspend rate exceeds eps in at most ~delta of them."""
    eps, delta = 0.1, 0.1
    pool = make_runs(6000, seed=11)
    budget = budget_at(pool, 0.8)
    rng = random.Random(0)
    bad = 0
    reps = 30
    for k in range(reps):
        sample = rng.sample(pool, 300)
        cert = certify_calibrated(sample, budget=budget, eps=eps, delta=delta, seed=k)
        table = QuantileTable.from_dict(cert.payload)
        plan = _level_plan(table, cert.level, None)
        risk = sum(overspends(r, plan, budget, 0.2) for r in pool) / len(pool)
        bad += risk > eps
    assert bad <= reps * delta + 3


def test_input_validation():
    runs = make_runs(100)
    with pytest.raises(ValueError):
        certify_calibrated(runs, budget=0, eps=0.05)
    with pytest.raises(ValueError):
        certify_calibrated(runs, budget=1, eps=1.5)
    with pytest.raises(ValueError):
        certify_calibrated(runs, budget=1, eps=0.05, table_share=1.0)
    with pytest.raises(ValueError, match="no steps"):
        certify_calibrated(runs + [{"run": "x", "steps": []}], budget=1, eps=0.05)


# ── certify_early_stop ────────────────────────────────────────────────────────

def make_long_runs(n, seed):
    from tests.test_early_stop import make_runs as es_runs
    runs = es_runs(n, seed)
    for i, r in enumerate(runs):
        r["run"], r["cost"] = f"e{i}", r.pop("_cost")
    return runs


def test_certify_early_stop():
    runs = make_long_runs(1200, seed=4)
    cert = certify_early_stop(runs, budget=0.30, eps=0.05, counter_keys=["err"])
    assert cert.kind == "early_stop" and cert.n_calibration == 600
    pol = cert.router_kwargs()["early_stop"]
    assert isinstance(pol, EarlyStopPolicy) and pol.threshold == cert.level
    assert cert.diagnostics["fitted_checkpoints"]
    assert cert.calibration["n"] == 600


def test_certify_early_stop_checks_inputs():
    runs = make_long_runs(400, seed=4)
    runs[0]["resolved"] = None
    with pytest.raises(ValueError, match="resolved"):
        certify_early_stop(runs, budget=0.30, eps=0.05, harm="resolve")
    with pytest.raises(NotEnoughRuns):
        certify_early_stop(runs[:60], budget=0.30, eps=0.05)
    with pytest.raises(ValueError):
        certify_early_stop(runs, budget=0.30, eps=0.05, harm="x")


def test_certificate_rejects_newer_schema():
    with pytest.raises(ValueError, match="newer"):
        Certificate.from_dict({"schema": 99})
    bad = Certificate(kind="x", budget=1, eps=0.1, delta=0.1, certified=False, level=None, n_runs=0,
                      n_calibration=0, calibration={}, payload={})
    with pytest.raises(ValueError):
        bad.router_kwargs()


# ── GuaranteeMonitor ──────────────────────────────────────────────────────────

def run_monitor(p, n, seed, eps=0.05, alpha=0.05):
    rng = random.Random(seed)
    mon = GuaranteeMonitor(eps, alpha=alpha, on_alarm=lambda m: None)
    for _ in range(n):
        mon.update(rng.random() < p)
    return mon


def test_monitor_false_alarm_rate_is_controlled():
    alarms = sum(run_monitor(0.05, 1500, seed).alarm for seed in range(200))
    assert alarms <= 200 * 0.05 + 6


def test_monitor_detects_drift():
    mon = run_monitor(0.2, 400, seed=1)
    assert mon.alarm and mon.alarm_at < 200
    assert mon.evidence >= 20


def test_monitor_alarm_fires_once_and_warns():
    calls = []
    mon = GuaranteeMonitor(0.05, alpha=0.05, on_alarm=calls.append)
    for _ in range(50):
        mon.update(True)
    assert calls == [mon] and mon.alarm_at is not None
    mon2 = GuaranteeMonitor(0.05)
    with pytest.warns(GuaranteeAlarm, match="no longer holds"):
        for _ in range(20):
            mon2.update(True)


def test_monitor_state_round_trip_and_checks():
    mon = run_monitor(0.1, 100, seed=2)
    back = GuaranteeMonitor.from_dict(json.loads(json.dumps(mon.to_dict())))
    assert (back.n, back.failures, back.alarm) == (mon.n, mon.failures, mon.alarm)
    assert back.evidence == pytest.approx(mon.evidence)
    with pytest.raises(ValueError):
        GuaranteeMonitor(0)
    with pytest.raises(ValueError):
        GuaranteeMonitor(0.1, alpha=1)
    es = Certificate(kind="early_stop", budget=1, eps=0.1, delta=0.1, certified=True, level=0.9,
                     n_runs=1, n_calibration=1, calibration={}, payload={})
    with pytest.raises(ValueError, match="observable"):
        GuaranteeMonitor.for_certificate(es)
    cal = Certificate(**dict(es.to_dict(), kind="calibrated"))
    assert GuaranteeMonitor.for_certificate(cal).eps == 0.1


# ── Router run logging ────────────────────────────────────────────────────────

def make_router(**kwargs):
    kwargs.setdefault("use_llm_router", False)
    return BAARRouter(**kwargs)


def read_lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_router_writes_steps_and_outcome(priced, tmp_path):
    path = str(tmp_path / "runs.jsonl")
    router = make_router(budget=1.0, run_log_path=path, run_id="run-1")
    router.record_event("err", 2)
    with patch("baar.router.litellm.completion", side_effect=completion_echo()):
        router.chat("a")
        router.chat("b")
    router.finish_run(finished=True, resolved=False, task="t1")
    lines = read_lines(path)
    assert [line["type"] for line in lines] == ["step", "step", "run"]
    s0 = lines[0]
    assert s0["run"] == "run-1" and s0["step"] == 0 and lines[1]["step"] == 1
    for key in ("model", "est_prompt_tokens", "prompt_tokens", "completion_tokens", "cost",
                "price_in", "price_out", "max_output_tokens", "finish_reason"):
        assert key in s0
    assert s0["counters"] == {"err": 2}
    assert s0["price_in"] > 0 and s0["price_out"] > 0
    tracker = router._tracker
    assert s0["price_in"] * 100 + s0["price_out"] * 30 == pytest.approx(tracker.estimate_cost("gpt-4o-mini", 100, 30))
    end = lines[2]
    assert end["finished"] is True and end["resolved"] is False and end["steps"] == 2
    assert end["cost"] == pytest.approx(router.spent) and end["budget"] == 1.0 and end["task"] == "t1"
    loaded = load_runs(path).runs
    assert len(loaded) == 1 and len(loaded[0]["steps"]) == 2
    with pytest.raises(RuntimeError, match="already finished"):
        router.finish_run(finished=True)


def test_recording_run_marks_budget_stops(priced, tmp_path):
    path = str(tmp_path / "runs.jsonl")
    always = EarlyStopPolicy(checkpoints=[0], weights={0: [10.0, 0, 0, 0, 0, 0]}, threshold=0.5)
    router = make_router(budget=1.0, run_log_path=path, early_stop=always)
    with pytest.raises(RunStopped):
        with router.recording_run():
            router.chat("x")
    end = read_lines(path)[-1]
    assert end["stopped_by_budget"] is True and end["finished"] is False
    assert end["run"] == router.run_id
    assert load_runs(path).runs == []


def test_recording_run_success_and_other_errors(priced, tmp_path):
    path = str(tmp_path / "runs.jsonl")
    router = make_router(budget=1.0, run_log_path=path)
    with patch("baar.router.litellm.completion", side_effect=completion_echo()):
        with router.recording_run() as run:
            router.chat("x")
            run.finished, run.resolved = True, True
            run.extra["task"] = "t"
    end = read_lines(path)[-1]
    assert end["finished"] and end["resolved"] and end["task"] == "t" and not end["stopped_by_budget"]
    router2 = make_router(budget=1.0, run_log_path=path)
    with pytest.raises(KeyError):
        with router2.recording_run():
            raise KeyError("boom")
    end = read_lines(path)[-1]
    assert end["run"] == router2.run_id and end["finished"] is False and not end["stopped_by_budget"]


def test_streaming_calls_are_logged(priced, tmp_path):
    from tests.test_output_overrun import make_chunks
    path = str(tmp_path / "runs.jsonl")
    router = make_router(budget=1.0, run_log_path=path)
    usage = (100, 7)
    with patch("baar.router.litellm.completion", return_value=iter(make_chunks(["a", "b"], usage=usage))):
        list(router.stream_chat("x"))
    router.finish_run(finished=True)
    lines = read_lines(path)
    assert lines[0]["type"] == "step" and lines[0]["completion_tokens"] == 7


@pytest.mark.asyncio
async def test_async_calls_are_logged(priced, tmp_path):
    from unittest.mock import AsyncMock
    from tests.test_output_overrun import make_response
    path = str(tmp_path / "runs.jsonl")
    router = make_router(budget=1.0, run_log_path=path)
    with patch("baar.router.litellm.acompletion", new=AsyncMock(return_value=make_response())):
        await router.achat("x")
    router.finish_run(finished=True)
    assert [line["type"] for line in read_lines(path)] == ["step", "run"]


def test_router_updates_monitor(priced):
    mon = GuaranteeMonitor(0.05)
    router = make_router(budget=1.0, guarantee_monitor=mon)
    with patch("baar.router.litellm.completion", side_effect=completion_echo()):
        router.chat("x")
    router.finish_run(finished=True)
    assert (mon.n, mon.failures) == (1, 0)


def test_no_run_log_by_default(priced, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    router = make_router(budget=1.0)
    with patch("baar.router.litellm.completion", side_effect=completion_echo()):
        router.chat("x")
    router.finish_run(finished=True)               # allowed without a log or monitor
    assert list(tmp_path.iterdir()) == []


def test_end_to_end_router_log_to_certificate(priced, tmp_path):
    """Runs logged by the router certify, and the certificate configures a working router."""
    path = str(tmp_path / "runs.jsonl")
    rng = random.Random(0)
    with patch("baar.router.litellm.completion", side_effect=completion_echo()):
        for i in range(120):
            router = make_router(budget=100.0, run_log_path=path, run_id=f"r{i}")
            for _ in range(rng.randint(1, 4)):
                router.chat("q")
            router.finish_run(finished=True)
    runs = load_runs(path).runs
    assert len(runs) == 120
    budget = budget_at(runs, 0.99) * 1.5
    cert = certify_calibrated(runs, budget=budget, eps=0.05)
    assert cert.certified
    router = make_router(budget=budget, **cert.router_kwargs())
    with patch("baar.router.litellm.completion", side_effect=completion_echo()):
        router.chat("q")


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_certifies(tmp_path, capsys):
    runs = make_runs(500, seed=6)
    log = str(tmp_path / "runs.jsonl")
    write_log(log, runs)
    out = str(tmp_path / "cert.json")
    code = cli_main([log, "--budget", str(budget_at(runs, 0.9)), "--eps", "0.05", "-o", out])
    text = capsys.readouterr().out
    assert code == 0 and "Certified" in text and "Saved" in text
    assert Certificate.load(out).certified


def test_cli_json_and_not_certified(tmp_path, capsys):
    runs = underestimated_runs(300, seed=6)
    log = str(tmp_path / "runs.jsonl")
    write_log(log, runs)
    code = cli_main([log, "--budget", str(budget_at(runs, 0.5)), "--json", "-o", str(tmp_path / "c.json")])
    data = json.loads(capsys.readouterr().out)
    assert code == 1 and data["certified"] is False and data["kind"] == "calibrated"


def test_cli_too_few_runs_and_bad_file(tmp_path, capsys):
    runs = make_runs(30)
    log = str(tmp_path / "runs.jsonl")
    write_log(log, runs)
    assert cli_main([log, "--budget", "1", "-o", str(tmp_path / "c.json")]) == 2
    assert "more" in capsys.readouterr().err
    assert cli_main([str(tmp_path / "missing.jsonl"), "--budget", "1"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_cli_early_stop(tmp_path, capsys):
    runs = make_long_runs(400, seed=8)
    log = str(tmp_path / "runs.jsonl")
    with open(log, "w") as f:
        for r in runs:
            for t, s in enumerate(r["steps"]):
                f.write(json.dumps(dict(s, run=r["run"], step=t)) + "\n")
            f.write(json.dumps({"type": "run", "run": r["run"], "finished": r["finished"],
                                "resolved": r["resolved"]}) + "\n")
    out = str(tmp_path / "es.json")
    code = cli_main([log, "--budget", "0.30", "--early-stop", "--harm", "resolve", "--counter", "err", "-o", out])
    assert code in (0, 1)
    cert = Certificate.load(out)
    assert cert.kind == "early_stop" and cert.settings["harm"] == "resolve"
    assert cert.settings["counter_keys"] == ["err"]
