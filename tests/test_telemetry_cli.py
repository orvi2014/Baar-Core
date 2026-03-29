import json
import sys
import pytest

from baar.telemetry_cli import summarize_records, load_jsonl, main


def test_summarize_records_computes_reject_failover_and_spend():
    records = [
        {
            "tier": "small",
            "model": "gpt-4o-mini",
            "cost_usd": 0.001,
            "failover_count": 0,
            "attempted_models": ["gpt-4o-mini"],
        },
        {
            "tier": "big",
            "model": "gpt-4o",
            "cost_usd": 0.01,
            "failover_count": 1,
            "attempted_models": ["gpt-4o", "fallback-big"],
        },
        {
            "tier": "reject",
            "model": "",
            "cost_usd": 0.0,
            "failover_count": 0,
            "attempted_models": [],
        },
    ]
    s = summarize_records(records)
    assert s["total_steps"] == 3
    assert s["reject_steps"] == 1
    assert s["reject_rate_pct"] == 33.3
    assert s["failover_steps"] == 1
    assert s["failover_rate_pct"] == 33.3
    assert s["total_spend_usd"] == 0.011
    assert s["per_model"][0]["model"] == "gpt-4o"
    assert s["per_model"][0]["spend_usd"] == 0.01


def test_load_jsonl_reads_records(tmp_path):
    p = tmp_path / "telemetry.jsonl"
    rows = [{"tier": "small", "model": "gpt-4o-mini", "cost_usd": 0.1}]
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    loaded = load_jsonl(p)
    assert loaded == rows


def test_load_jsonl_raises_on_malformed_line(tmp_path):
    p = tmp_path / "telemetry.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write('{"tier":"small"}\n')
        f.write("not-json\n")
    with pytest.raises(json.JSONDecodeError):
        load_jsonl(p)


def test_cli_main_prints_summary(capsys, tmp_path, monkeypatch):
    p = tmp_path / "telemetry.jsonl"
    rows = [
        {"tier": "small", "model": "gpt-4o-mini", "cost_usd": 0.001},
        {"tier": "reject", "model": "", "cost_usd": 0.0},
    ]
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    monkeypatch.setattr(sys, "argv", ["baar-telemetry", str(p)])
    main()
    out = capsys.readouterr().out
    assert "BAAR Telemetry Summary" in out
    assert "Reject steps" in out
    assert "Failover steps" in out
