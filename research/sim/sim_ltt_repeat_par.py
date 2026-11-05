"""Parallel repeat of LTT calibration: fraction of calibrations whose true overrun
rate exceeds eps must be <= delta. Writes per-rep results to JSONL for later analysis."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

import sim_ltt as L

BUDGET, EPS, DELTA, N_CAL, N_TEST = 0.02, 0.05, 0.05, 1000, 4000


def one(args):
    scenario, r = args
    beta, quant = L.ltt(BUDGET, EPS, DELTA, scenario, "A", n_cal=N_CAL, seed=1000 + r)
    if beta is None:
        return dict(scenario=scenario, rep=r, beta=None, overrun=0.0)
    res = L.runs(np.random.default_rng(5000 + r), BUDGET, quant, beta, EPS, scenario, "A", N_TEST)
    return dict(scenario=scenario, rep=r, beta=beta, overrun=float(res[:, 0].mean()),
                big=float(res[:, 5].sum() / res[:, 4].sum()), complete=float(res[:, 2].mean()))


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    out = sys.argv[2] if len(sys.argv) > 2 else "ltt_repeat.jsonl"
    jobs = [(s, r) for s in ["valid", "broken"] for r in range(reps)]
    with Pool(os.cpu_count()) as pool, open(out, "w") as f:
        for i, res in enumerate(pool.imap_unordered(one, jobs), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} done", flush=True)
    rows = [json.loads(l) for l in open(out)]
    for s in ["valid", "broken"]:
        rates = np.array([r["overrun"] for r in rows if r["scenario"] == s])
        k = int(np.sum(rates > EPS))
        print(f"[{s}] reps={len(rates)} mean_overrun={rates.mean():.4f} max={rates.max():.4f} "
              f"violating={k}/{len(rates)} = {k/len(rates):.3f} (delta={DELTA})")
