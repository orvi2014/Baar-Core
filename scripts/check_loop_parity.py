#!/usr/bin/env python3
"""
scripts/check_loop_parity.py — do baar-core and noburn detect loops the same way?

noburn's SDK loop detection started as a copy of baar-core's LoopDetector. To
keep them from drifting, both must give the same verdicts on the shared
scenarios in tests/fixtures/loop_parity.json: which steps warn, and whether the
next call is refused (and as which pattern).

Run:
    python scripts/check_loop_parity.py                         # baar-core vs the expected verdicts
    python scripts/check_loop_parity.py --other path/to/loop.py # also noburn's loop.py

Exits 1 on any mismatch. tests/test_loop_parity.py runs the baar-core half in CI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = os.path.join(ROOT, "tests", "fixtures", "loop_parity.json")

# Placeholders that would be unwieldy inline in JSON.
_SPECIAL = {"LONG_RUNNING_PAGE": "Results: " + "running shoes, " * 30}


def load_scenarios(path: str = SCENARIOS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["scenarios"]


def _value(v: Any) -> Any:
    return _SPECIAL.get(v, v) if isinstance(v, str) else v


def verdict(module: Any, scenario: dict) -> dict:
    """Replay one scenario through ``module.LoopDetector``; return its verdict."""
    detector = module.LoopDetector(**scenario.get("options", {}))
    warn_steps = []
    for i, step in enumerate(scenario["steps"]):
        detector.before(step["tool"], step["args"])
        if "error" in step:
            event = detector.after(step["tool"], step["args"], error=RuntimeError(step["error"]))
        else:
            event = detector.after(step["tool"], step["args"], result=_value(step["result"]))
        if event is not None:
            warn_steps.append(i)
    nxt = scenario["next"]
    stop = detector.peek(nxt["tool"], nxt["args"])
    return {"warn_steps": warn_steps, "stop": stop.pattern if stop is not None else None}


def load_module(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(implementations: dict[str, Any], scenarios: list[dict]) -> list[str]:
    problems = []
    for scenario in scenarios:
        expected = scenario["expect"]
        for label, module in implementations.items():
            try:
                got = verdict(module, scenario)
            except Exception as e:  # e.g. an option the other implementation doesn't support
                got = f"{type(e).__name__}: {e}"
            if got != expected:
                problems.append(f"{label}: '{scenario['name']}' expected {expected}, got {got}")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--other", help="path to another loop.py to compare (e.g. noburn's sdk/python/src/noburn/loop.py)")
    args = parser.parse_args(argv)
    sys.path.insert(0, ROOT)
    from baar.core import loop as baar_loop

    implementations = {"baar-core": baar_loop}
    if args.other:
        implementations["other"] = load_module(args.other, "_other_loop")
    scenarios = load_scenarios()
    problems = check(implementations, scenarios)
    for p in problems:
        print("MISMATCH", p)
    print(f"{len(scenarios)} scenarios × {len(implementations)} implementation(s): "
          f"{'all match' if not problems else f'{len(problems)} mismatch(es)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
