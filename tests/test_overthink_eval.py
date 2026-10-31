"""
tests/test_overthink_eval.py — the OverThink benchmark's headline claims hold.

Runs benchmarks/overthink_eval.py on a few seeds (deterministic, no network).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

import overthink_eval  # noqa: E402

SEEDS = [1, 2, 3, 7]


@pytest.fixture(scope="module")
def results():
    return {seed: {r["setup"]: r for r in overthink_eval.evaluate(seed)} for seed in SEEDS}


def test_hard_modes_never_overshoot(results):
    for rows in results.values():
        for setup in ("preflight", "preflight+thinking", "truncate"):
            assert rows[setup]["overshoot_usd"] == 0, (setup, rows[setup])


def test_preflight_never_cuts_an_answer_off(results):
    for rows in results.values():
        for setup in ("preflight", "preflight+thinking", "preflight+learned"):
            assert rows[setup]["cut_off"] == 0, (setup, rows[setup])


def test_estimate_and_alerts_overshoot_under_inflated_reasoning(results):
    assert all(rows["alert only"]["overshoot_usd"] > 0 for rows in results.values())
    assert sum(rows["estimate"]["overshoot_usd"] > 0 for rows in results.values()) >= len(SEEDS) - 1


def test_thinking_cap_gets_more_work_done_than_plain_preflight(results):
    for rows in results.values():
        assert rows["preflight+thinking"]["done"] > rows["preflight"]["done"]


def test_learned_lengths_use_more_of_the_budget_than_worst_case(results):
    for rows in results.values():
        assert rows["preflight+learned"]["used_pct"] >= rows["preflight"]["used_pct"]


def test_cli_table_and_json(capsys):
    overthink_eval.main(["--seed", "7"])
    out = capsys.readouterr().out
    assert "| preflight+thinking |" in out
    overthink_eval.main(["--json", "--seed", "7"])
    assert '"setup": "truncate"' in capsys.readouterr().out
