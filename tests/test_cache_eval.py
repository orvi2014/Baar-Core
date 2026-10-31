"""
tests/test_cache_eval.py — the cache benchmark's headline holds on a small slice (≈ a few seconds).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

import cache_eval  # noqa: E402


def test_routing_beats_always_big_even_after_cache_losses():
    convs = cache_eval.make_conversations(7)[:3]
    rows = {r["setup"]: r for r in cache_eval.evaluate(conversations=convs)}
    big, routed = rows["always big"], rows["routing"]
    assert routed["cost_usd"] < big["cost_usd"]
    assert routed["switches"] > 0 and big["switches"] == 0
    assert big["cache_hit_pct"] > routed["cache_hit_pct"] > 50


def test_cli_table(capsys, monkeypatch):
    monkeypatch.setattr(cache_eval, "CONVERSATIONS", 1)
    monkeypatch.setattr(cache_eval, "TURNS", 2)
    cache_eval.main(["--seed", "3"])
    assert "| routing |" in capsys.readouterr().out
    cache_eval.main(["--json", "--seed", "3"])
    assert '"setup": "always big"' in capsys.readouterr().out
