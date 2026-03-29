"""
Telemetry summary CLI for BAAR JSONL exports.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = len(records)
    reject_steps = 0
    failover_steps = 0
    total_spend = 0.0
    model_spend: dict[str, float] = defaultdict(float)
    model_calls: dict[str, int] = defaultdict(int)

    for row in records:
        tier = str(row.get("tier", "")).lower()
        if tier == "reject":
            reject_steps += 1

        failover_count = int(row.get("failover_count", 0) or 0)
        attempted_models = row.get("attempted_models", []) or []
        if failover_count > 0 or len(attempted_models) > 1:
            failover_steps += 1

        cost = float(row.get("cost_usd", 0.0) or 0.0)
        total_spend += cost

        model = str(row.get("model", "")).strip()
        if model:
            model_calls[model] += 1
            model_spend[model] += cost

    reject_rate = (reject_steps / total_steps * 100.0) if total_steps else 0.0
    failover_rate = (failover_steps / total_steps * 100.0) if total_steps else 0.0

    per_model = [
        {
            "model": m,
            "calls": model_calls[m],
            "spend_usd": round(model_spend[m], 8),
            "spend_pct": round((model_spend[m] / total_spend * 100.0), 1) if total_spend > 0 else 0.0,
        }
        for m in model_spend
    ]
    per_model.sort(key=lambda x: x["spend_usd"], reverse=True)

    return {
        "total_steps": total_steps,
        "reject_steps": reject_steps,
        "reject_rate_pct": round(reject_rate, 1),
        "failover_steps": failover_steps,
        "failover_rate_pct": round(failover_rate, 1),
        "total_spend_usd": round(total_spend, 8),
        "per_model": per_model,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_summary(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("BAAR Telemetry Summary")
    lines.append("=" * 60)
    lines.append(f"Total steps:    {summary['total_steps']}")
    lines.append(f"Reject steps:   {summary['reject_steps']} ({summary['reject_rate_pct']}%)")
    lines.append(f"Failover steps: {summary['failover_steps']} ({summary['failover_rate_pct']}%)")
    lines.append(f"Total spend:    ${summary['total_spend_usd']:.6f}")
    lines.append("-" * 60)
    lines.append(f"{'Model':<35} {'Calls':>6} {'Spend (USD)':>12} {'Spend %':>8}")
    lines.append("-" * 60)
    for row in summary["per_model"]:
        lines.append(
            f"{row['model']:<35} {row['calls']:>6} {row['spend_usd']:>12.6f} {row['spend_pct']:>7.1f}%"
        )
    if not summary["per_model"]:
        lines.append("(no model spend rows found)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BAAR telemetry JSONL.")
    parser.add_argument("path", type=str, help="Path to telemetry JSONL file.")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    rows = load_jsonl(path)
    summary = summarize_records(rows)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
