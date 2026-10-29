"""
tests/test_loop_eval.py — keeps benchmarks/loop_eval.py honest.

Locks the properties the README claims, so a detector change that starts
stopping healthy runs (or missing exact loops) fails CI.
"""

from benchmarks.loop_eval import evaluate


def _rows():
    return {r["method"]: r for r in evaluate()}


def test_detector_catches_every_exact_loop_and_stops_no_healthy_run_when_polls_are_exempt():
    r = _rows()["detector+exempt"]
    for cat in ["exact_repeat", "error_retry", "ping_pong", "volatile_args"]:
        stopped, total = r["by_category"][cat].split("/")
        assert stopped == total, cat
    assert r["false_stop_pct"] == 0.0
    assert r["median_wasted_calls"] <= 3


def test_rephrased_loops_are_a_known_miss_that_the_budget_catches():
    rows = _rows()
    assert rows["detector+exempt"]["by_category"]["rephrased"] == "0/15"
    assert rows["detector+budget"]["caught_pct"] == 100.0
    assert rows["detector+budget"]["false_stop_pct"] == 0.0


def test_detector_wastes_less_than_the_budget_alone():
    rows = _rows()
    assert rows["detector+budget"]["usd_wasted"] < rows["budget_only"]["usd_wasted"]
    assert rows["detector+budget"]["median_wasted_calls"] < rows["budget_only"]["median_wasted_calls"]


def test_result_hashing_beats_call_hashing_on_false_stops():
    rows = _rows()
    assert rows["detector"]["false_stop_pct"] < rows["call_hash_only"]["false_stop_pct"]
