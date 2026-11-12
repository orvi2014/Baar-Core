"""
baar/runlog.py — run-level logs for certification.

The certified features (``on_output_overrun="calibrated"`` and
``EarlyStopPolicy``) are calibrated on whole agent runs. This module writes
those runs in one JSONL file and reads them back.

Each line is one JSON object:

* a step (one LLM call)::

    {"type": "step", "run": "r1", "step": 0, "model": "gpt-4o-mini",
     "est_prompt_tokens": 812, "prompt_tokens": 820, "completion_tokens": 95,
     "cost": 0.000180, "price_in": 1.5e-07, "price_out": 6e-07,
     "max_output_tokens": 16384, "finish_reason": "stop", "counters": {...}}

  ``est_prompt_tokens`` is what the router counted *before* the call (what the
  controller knows); ``prompt_tokens`` and ``cost`` are what the provider
  billed. ``price_in``/``price_out`` are USD per token and
  ``max_output_tokens`` is the worst-case output the router would reserve, so
  logs can be replayed without a pricing table.

* the end of a run::

    {"type": "run", "run": "r1", "finished": true, "resolved": true,
     "stopped_by_budget": false, "cost": 0.0123, "budget": 10.0, "steps": 14}

Runs without an end line (crashed, still going) are skipped when loading.

Calibration needs runs that the budget did not cut short: log with a budget
that does not bind (for example ten times the budget you will certify), so each
run shows what it would have cost. Runs with ``stopped_by_budget`` are
excluded by ``load_runs`` unless you ask for them.

This module has no third-party dependencies.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCHEMA = 1

_file_locks: Dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    key = os.path.abspath(path)
    with _file_locks_guard:
        return _file_locks.setdefault(key, threading.Lock())


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


class RunLogWriter:
    """Appends step and run records to a JSONL file (thread-safe within a process)."""

    def __init__(self, path: str):
        self.path = str(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=True) + "\n"
        with _lock_for(self.path):
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)


@dataclass
class RunOutcome:
    """Filled in by the caller inside ``router.recording_run()``."""

    run_id: str
    finished: bool = False
    resolved: Optional[bool] = None
    extra: dict = field(default_factory=dict)


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


@dataclass
class LoadedRuns:
    runs: List[dict]
    incomplete: int = 0          # runs with steps but no end line
    stopped_by_budget: int = 0   # runs excluded because the budget cut them short
    empty: int = 0               # runs that ended without any call


def load_runs(path: str, include_stopped: bool = False) -> LoadedRuns:
    """
    Read a run log into run dicts::

        {"run": id, "finished": bool, "resolved": bool | None, "cost": float,
         "stopped_by_budget": bool, "budget": float | None, "steps": [step, ...]}

    Steps are sorted by their index. The run dicts are what ``fit_early_stop``,
    ``calibrate_early_stop`` and ``baar.certify`` take.
    """
    steps: Dict[str, List[dict]] = {}
    ends: Dict[str, dict] = {}
    order: List[str] = []
    with _open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not valid JSON ({e.msg})") from None
            rid = str(rec.get("run", ""))
            if not rid:
                raise ValueError(f"{path}:{n}: record has no 'run' id")
            if rid not in steps and rid not in ends:
                order.append(rid)
            if rec.get("type") == "run":
                ends[rid] = rec
            else:
                steps.setdefault(rid, []).append(rec)
    out = LoadedRuns(runs=[])
    for rid in order:
        st = sorted(steps.get(rid, []), key=lambda s: int(s.get("step", 0)))
        end = ends.get(rid)
        if end is None:
            out.incomplete += 1
            continue
        if not st:
            out.empty += 1
            continue
        stopped = bool(end.get("stopped_by_budget", False))
        if stopped and not include_stopped:
            out.stopped_by_budget += 1
            continue
        resolved = end.get("resolved")
        out.runs.append({
            "run": rid,
            "finished": bool(end.get("finished", False)),
            "resolved": None if resolved is None else bool(resolved),
            "stopped_by_budget": stopped,
            "budget": end.get("budget"),
            "cost": sum(float(s["cost"]) for s in st),
            "steps": st,
        })
    return out


def split_runs(n: int, fractions: Tuple[float, ...], seed: int = 0) -> List[List[int]]:
    """Shuffle range(n) with a fixed seed and cut it into consecutive parts of the given shares."""
    import random

    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    parts, start = [], 0
    for i, frac in enumerate(fractions):
        end = n if i == len(fractions) - 1 else start + int(round(frac * n))
        parts.append(idx[start:end])
        start = end
    return parts
