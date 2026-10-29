#!/usr/bin/env python3
"""
benchmarks/loop_eval.py — How well does LoopDetector catch agent loops?

Replays a labeled, deterministic set of agent traces (no API keys, no network)
through LoopDetector and three baselines, and reports per method:

  caught      share of looping traces stopped before they ran out
  false stop  share of healthy traces stopped (the number that sets defaults)
  wasted      median calls made after the loop began, before the stop
  $ wasted    total spend after loop onset, across all looping traces
  µs/step     detector overhead per observed step

Looping traces (the loop runs to the end of the trace unless stopped):
  exact_repeat   same call, same result
  error_retry    same call, same error
  ping_pong      two calls alternating, same results
  volatile_args  same call, but a request_id changes every time
  rephrased      same intent, new wording each time, same empty result
                 (a semantic loop; exact matching is expected to miss it)

Healthy traces that look like loops:
  poll_progress  a job polled until done, at most 3 identical "running"
  long_poll      a job polled with 6 identical "running" before done
  pagination     one tool, new cursor each time
  retry_429      one or two rate-limit errors, then success
  flaky_api      the same call failing on and off (never 3 times in a row)
                 while it keeps succeeding in between
  reread         the same file read again after it changed
  research       many different searches

Methods:
  detector             LoopDetector defaults, ignore_args=["request_id"]
  detector+exempt      the same, with the job-status tool in exempt_tools
  detector+budget      detector+exempt plus the $0.40 run budget (how you'd
                       ship it: the budget catches what matching can't)
  call_hash_only       stop on the 5th identical call, results ignored
  max_calls            baar_guard(max_calls=8) on every tool
  budget_only          a $0.40 run budget (above any healthy run) and nothing else

Run:
    python benchmarks/loop_eval.py            # table to stdout
    python benchmarks/loop_eval.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baar.core.loop import LoopDetected, LoopDetector  # noqa: E402

SEED = 7
# A loop doesn't end by itself: looping traces run until something stops them.
LOOP_LEN = 100
TRACES_PER_CATEGORY = 15
COST_PER_STEP = 0.01
# Sized above the longest healthy trace (32 calls), as you would size a real
# run budget: it should only ever stop runs that went wrong.
RUN_BUDGET = 0.40
MAX_CALLS = 8
CALL_HASH_LIMIT = 4

LOOP_CATEGORIES = ["exact_repeat", "error_retry", "ping_pong", "volatile_args", "rephrased"]
HEALTHY_CATEGORIES = [
    "poll_progress", "long_poll", "pagination", "retry_429", "flaky_api", "reread", "research",
]


@dataclass
class Step:
    tool: str
    args: dict
    result: object = None
    error: Optional[str] = None


@dataclass
class Trace:
    category: str
    loop: bool
    steps: list[Step]
    onset: int = -1  # first step of the loop (looping traces only)


# ── trace generation ──────────────────────────────────────────────────────────

TOPICS = ["pricing", "churn", "latency", "refunds", "onboarding", "retention", "billing", "search"]


def _prefix(rng: random.Random) -> list[Step]:
    """A few distinct, productive steps before anything goes wrong."""
    steps = []
    for i in range(rng.randint(2, 6)):
        topic = rng.choice(TOPICS)
        steps.append(Step("search_web", {"query": f"{topic} report {i}"}, result=f"{topic} results #{i}"))
    return steps


def _fill(steps: list[Step], step_fn: Callable[[int], Step], start: int) -> list[Step]:
    i = 0
    while len(steps) < LOOP_LEN:
        steps.append(step_fn(start + i))
        i += 1
    return steps


def make_loop(category: str, rng: random.Random) -> Trace:
    steps = _prefix(rng)
    onset = len(steps)
    topic = rng.choice(TOPICS)
    if category == "exact_repeat":
        call = Step("search_web", {"query": f"latest {topic} numbers"}, result="no results")
        _fill(steps, lambda _: call, 0)
    elif category == "error_retry":
        _fill(steps, lambda _: Step("deploy", {"env": "prod"}, error="ValueError: missing config key"), 0)
    elif category == "ping_pong":
        a = Step("read_file", {"path": f"{topic}.py"}, result="def f(): pass")
        b = Step("run_tests", {"path": f"{topic}.py"}, result="1 failed")
        _fill(steps, lambda i: a if i % 2 == 0 else b, 0)
    elif category == "volatile_args":
        _fill(steps, lambda i: Step(
            "fetch_page", {"url": f"https://example.com/{topic}", "request_id": f"req-{rng.random():.8f}"},
            result="403 Forbidden"), 0)
    elif category == "rephrased":
        phrasings = ["what is", "tell me", "find", "look up", "search for", "show"]
        _fill(steps, lambda i: Step(
            "search_web", {"query": f"{phrasings[i % len(phrasings)]} {topic} for q{i}"}, result="no results"), 0)
    else:
        raise ValueError(category)
    # The first repetition is the second step of the loop (a ping-pong needs a full cycle).
    first_repeat = onset + (2 if category == "ping_pong" else 1)
    return Trace(category, True, steps, onset=first_repeat)


def make_healthy(category: str, rng: random.Random) -> Trace:
    steps = _prefix(rng)
    if category in ("poll_progress", "long_poll"):
        running = rng.randint(1, 3) if category == "poll_progress" else 6
        job = {"job_id": f"job-{rng.randint(100, 999)}"}
        steps.append(Step("job_status", job, result="queued"))
        steps += [Step("job_status", job, result="running") for _ in range(running)]
        steps.append(Step("job_status", job, result="done"))
        steps.append(Step("fetch_result", job, result="report.csv"))
    elif category == "pagination":
        for page in range(rng.randint(10, 20)):
            steps.append(Step("list_orders", {"cursor": page * 50}, result=f"orders {page * 50}..{page * 50 + 49}"))
    elif category == "retry_429":
        for _ in range(rng.randint(1, 2)):
            steps.append(Step("crm_lookup", {"email": "a@b.co"}, error="RateLimitError: 429"))
        steps.append(Step("crm_lookup", {"email": "a@b.co"}, result="contact #42"))
    elif category == "flaky_api":
        # Intermittent failures: at most 2 in a row, with successes between.
        failures_in_a_row = 0
        for i in range(rng.randint(8, 14)):
            if failures_in_a_row < 2 and rng.random() < 0.5:
                steps.append(Step("inventory", {"sku": "A-1"}, error="ConnectionError: reset by peer"))
                failures_in_a_row += 1
            else:
                steps.append(Step("inventory", {"sku": "A-1"}, result=f"stock {100 - i}"))
                failures_in_a_row = 0
    elif category == "reread":
        for v in range(rng.randint(3, 6)):
            steps.append(Step("read_file", {"path": "app.py"}, result=f"version {v}"))
            steps.append(Step("edit_file", {"path": "app.py", "patch": f"fix {v}"}, result="ok"))
    elif category == "research":
        for i in range(rng.randint(15, 25)):
            topic = rng.choice(TOPICS)
            steps.append(Step("search_web", {"query": f"{topic} deep dive {i}"}, result=f"{topic} findings {i}"))
    else:
        raise ValueError(category)
    steps.append(Step("finish", {"answer": "done"}, result="ok"))
    return Trace(category, False, steps)


def build_traces(seed: int = SEED) -> list[Trace]:
    rng = random.Random(seed)
    traces = []
    for _ in range(TRACES_PER_CATEGORY):
        traces += [make_loop(c, rng) for c in LOOP_CATEGORIES]
        traces += [make_healthy(c, rng) for c in HEALTHY_CATEGORIES]
    return traces


# ── methods: each returns the index of the refused step, or None ─────────────

def run_detector(trace: Trace, **opts) -> tuple[Optional[int], float]:
    d = LoopDetector(ignore_args=["request_id"], **opts)
    t0 = time.perf_counter()
    stopped = None
    for i, s in enumerate(trace.steps):
        try:
            d.before(s.tool, s.args)
        except LoopDetected:
            stopped = i
            break
        err = RuntimeError(s.error) if s.error else None
        d.after(s.tool, s.args, result=s.result, error=err)
    elapsed = time.perf_counter() - t0
    observed = (stopped if stopped is not None else len(trace.steps)) or 1
    return stopped, elapsed / observed


def run_call_hash_only(trace: Trace) -> Optional[int]:
    seen: Counter = Counter()
    for i, s in enumerate(trace.steps):
        key = (s.tool, json.dumps({k: v for k, v in s.args.items() if k != "request_id"}, sort_keys=True))
        if seen[key] >= CALL_HASH_LIMIT:
            return i
        seen[key] += 1
    return None


def run_max_calls(trace: Trace) -> Optional[int]:
    per_tool: Counter = Counter()
    for i, s in enumerate(trace.steps):
        if per_tool[s.tool] >= MAX_CALLS:
            return i
        per_tool[s.tool] += 1
    return None


def run_budget_only(trace: Trace) -> Optional[int]:
    for i, _ in enumerate(trace.steps):
        if (i + 1) * COST_PER_STEP > RUN_BUDGET + 1e-9:
            return i
    return None


def run_detector_with_budget(trace: Trace, **opts) -> tuple[Optional[int], float]:
    stopped, per_step = run_detector(trace, **opts)
    budget = run_budget_only(trace)
    if budget is not None and (stopped is None or budget < stopped):
        stopped = budget
    return stopped, per_step


# ── scoring ───────────────────────────────────────────────────────────────────

@dataclass
class Score:
    method: str
    caught: int = 0
    loops: int = 0
    false_stops: int = 0
    healthy: int = 0
    wasted: list[int] = field(default_factory=list)
    by_category: dict[str, list[int]] = field(default_factory=dict)  # [stopped, total]
    us_per_step: list[float] = field(default_factory=list)

    def add(self, trace: Trace, stopped: Optional[int]) -> None:
        cat = self.by_category.setdefault(trace.category, [0, 0])
        cat[1] += 1
        if stopped is not None:
            cat[0] += 1
        if trace.loop:
            self.loops += 1
            end = stopped if stopped is not None else len(trace.steps)
            self.wasted.append(max(0, end - trace.onset))
            if stopped is not None:
                self.caught += 1
        else:
            self.healthy += 1
            if stopped is not None:
                self.false_stops += 1

    def row(self) -> dict:
        return {
            "method": self.method,
            "caught_pct": round(100 * self.caught / self.loops, 1),
            "false_stop_pct": round(100 * self.false_stops / self.healthy, 1),
            "median_wasted_calls": statistics.median(self.wasted),
            "usd_wasted": round(sum(self.wasted) * COST_PER_STEP, 2),
            "us_per_step": round(statistics.mean(self.us_per_step), 1) if self.us_per_step else None,
            "by_category": {k: f"{v[0]}/{v[1]}" for k, v in self.by_category.items()},
        }


def evaluate(seed: int = SEED) -> list[dict]:
    traces = build_traces(seed)
    methods: dict[str, Callable[[Trace], tuple[Optional[int], Optional[float]]]] = {
        "detector": lambda t: run_detector(t),
        "detector+exempt": lambda t: run_detector(t, exempt_tools=["job_status"]),
        "detector+budget": lambda t: run_detector_with_budget(t, exempt_tools=["job_status"]),
        "call_hash_only": lambda t: (run_call_hash_only(t), None),
        "max_calls": lambda t: (run_max_calls(t), None),
        "budget_only": lambda t: (run_budget_only(t), None),
    }
    rows = []
    for name, fn in methods.items():
        score = Score(name)
        for t in traces:
            stopped, per_step = fn(t)
            score.add(t, stopped)
            if per_step is not None:
                score.us_per_step.append(per_step * 1e6)
        rows.append(score.row())
    return rows


def render(rows: list[dict]) -> str:
    n_loops = TRACES_PER_CATEGORY * len(LOOP_CATEGORIES)
    n_healthy = TRACES_PER_CATEGORY * len(HEALTHY_CATEGORIES)
    out = [
        f"Loop eval — {n_loops} looping + {n_healthy} healthy traces (seed {SEED}), "
        f"${COST_PER_STEP:.2f}/call\n",
        "| Method | Caught | False stops | Median wasted calls | $ wasted | µs/step |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        us = f"{r['us_per_step']}" if r["us_per_step"] is not None else "—"
        out.append(
            f"| {r['method']} | {r['caught_pct']}% | {r['false_stop_pct']}% | "
            f"{r['median_wasted_calls']} | ${r['usd_wasted']:.2f} | {us} |"
        )
    cats = LOOP_CATEGORIES + HEALTHY_CATEGORIES
    out += ["", "Stopped / traces, by category (looping first, then healthy):", "",
            "| Method | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for r in rows:
        out.append(f"| {r['method']} | " + " | ".join(r["by_category"][c] for c in cats) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rows = evaluate(args.seed)
    print(json.dumps(rows, indent=2) if args.json else render(rows))


if __name__ == "__main__":
    main()
