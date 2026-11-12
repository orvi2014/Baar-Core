"""
baar-calibrate: certify a router setting from a run log.

    baar-calibrate runs.jsonl --budget 0.50 --eps 0.05
    baar-calibrate runs.jsonl --budget 0.50 --eps 0.05 --early-stop --harm resolve

Writes a certificate (default ``baar_cert.json``) that the router loads with
``BAARRouter(budget=0.50, **Certificate.load("baar_cert.json").router_kwargs())``.
Exit status: 0 certified, 1 nothing certified (safe fallback), 2 error or too few runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from baar.certify import NotEnoughRuns, certify_calibrated, certify_early_stop, min_total_runs
from baar.runlog import load_runs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="baar-calibrate",
        description="Certify a run-level budget setting from logged agent runs (see baar.runlog).")
    p.add_argument("log", help="run log written with BAARRouter(run_log_path=...) (.jsonl or .jsonl.gz)")
    p.add_argument("--budget", type=float, required=True, help="budget per run in USD (fixed before calibrating)")
    p.add_argument("--eps", type=float, default=0.05, help="allowed risk per run (default 0.05)")
    p.add_argument("--delta", type=float, default=0.05, help="1 - confidence (default 0.05)")
    p.add_argument("-o", "--out", default="baar_cert.json", help="certificate path (default baar_cert.json)")
    p.add_argument("--early-stop", action="store_true",
                   help="certify an early-stopping policy instead of the calibrated reservation level")
    p.add_argument("--harm", choices=["finish", "resolve"], default="finish",
                   help="early stopping: harm is stopping a run that would have finished / solved within budget")
    p.add_argument("--counter", action="append", default=[], metavar="NAME",
                   help="early stopping: a record_event() counter to use as a feature (repeatable)")
    p.add_argument("--margin", type=float, default=0.2, help="prompt margin the router will use (default 0.2)")
    p.add_argument("--seed", type=int, default=0, help="seed of the run split (default 0)")
    p.add_argument("--json", action="store_true", help="print the certificate summary as JSON")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_runs(args.log)
    except (OSError, ValueError) as e:
        print(f"baar-calibrate: cannot read {args.log}: {e}", file=sys.stderr)
        return 2
    notes = []
    if loaded.incomplete:
        notes.append(f"{loaded.incomplete} runs have no end record (finish_run was not called) and were skipped")
    if loaded.stopped_by_budget:
        notes.append(f"{loaded.stopped_by_budget} runs were cut short by the budget and were skipped; "
                     "log calibration runs with a budget that does not bind")
    runs = loaded.runs
    try:
        if args.early_stop:
            cert = certify_early_stop(runs, args.budget, args.eps, args.delta, harm=args.harm,
                                      seed=args.seed, counter_keys=args.counter)
        else:
            cert = certify_calibrated(runs, args.budget, args.eps, args.delta, margin=args.margin,
                                      seed=args.seed)
    except NotEnoughRuns as e:
        for n in notes:
            print(f"Note: {n}", file=sys.stderr)
        print(f"baar-calibrate: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"baar-calibrate: {e}", file=sys.stderr)
        return 2
    cert.diagnostics.setdefault("warnings", [])
    cert.diagnostics["notes"] = notes
    cert.save(args.out)
    if args.json:
        print(json.dumps({"certified": cert.certified, "kind": cert.kind, "level": cert.level,
                          "budget": cert.budget, "eps": cert.eps, "delta": cert.delta,
                          "n_runs": cert.n_runs, "n_calibration": cert.n_calibration,
                          "out": args.out, "warnings": cert.diagnostics["warnings"], "notes": notes}))
    else:
        for n in notes:
            print(f"Note: {n}")
        print(cert.summary())
        if not cert.certified:
            share = 0.5 if args.early_stop else 0.3
            print(f"Tip: with more runs, fewer of them need to stay within budget to certify; "
                  f"{min_total_runs(args.eps, args.delta, share)} is the bare minimum at this eps and delta.")
        print(f"Saved: {args.out}")
    return 0 if cert.certified else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
