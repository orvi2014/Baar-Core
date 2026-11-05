"""
Simulation v2: a large task pool, every policy run on every task.

For each (scenario, mode, budget) we draw N tasks and run each one under every
policy with common random numbers (the same task seed for every policy):
preflight, estimate, the calibrated policy at every beta in a fine grid, and the
per-step (union-bound / risk-spending) baseline at several eps. Results are
saved per configuration; analysis_v2.py then draws calibration sets from one
half of the pool and measures the true risk of whatever was selected on the
other half (100k tasks, standard error <= 0.07 percentage points).

Scenarios
  valid  — output lengths in the tables come from the deployment distribution.
  shift  — tables (and estimate-mode means) are fitted on easy tasks, but in
           deployment every task has a hidden difficulty d ~ N(0, 0.6^2) that
           scales all its output lengths by e^d: the tables are wrong in mean
           and tail, and steps within a run are correlated.

Usage: python sim_v2.py OUT_DIR [N]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

import sim_thm1 as S

BETAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16,
         0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
POLICIES = ["preflight", "estimate"] + [f"b{b}" for b in BETAS]
PERSTEP_EPS = [0.01, 0.05, 0.10]
CONFIGS = ([(s, "A", b) for s in ["valid", "shift"] for b in [0.01, 0.02, 0.05, 0.10]]
           + [(s, "B", 0.02) for s in ["valid", "shift"]])
N_TAB = 2000
CHUNK = 5000


def table_samples(seed=11):
    """Logged output lengths used to build the policy family (no difficulty)."""
    rng = np.random.default_rng(seed)
    return {k: np.sort([S.draw_y(rng, k[0], k[1]) for _ in range(N_TAB)]) for k in S.STRATA}


SAMPLES = table_samples()
PLANS = {"preflight": {k: S.YMAX[k[0]] for k in S.STRATA},
         "estimate": {k: int(SAMPLES[k].mean()) for k in S.STRATA}}
for _b in BETAS:
    PLANS[f"b{_b}"] = {k: int(np.quantile(SAMPLES[k], 1 - _b)) for k in S.STRATA}


def run_policy(seed, policy, budget, scenario, mode):
    rng = np.random.default_rng(seed)
    run_scen = "broken" if scenario == "shift" else "valid"
    if policy == "preflight":
        return S.run_once(rng, budget, "preflight", None, None, 1.0, run_scen, mode)
    if policy == "estimate":
        return S.run_once(rng, budget, "estimate", None, PLANS["estimate"], 1.0, run_scen, mode)
    plan = PLANS[policy]
    return S.run_once(rng, budget, "fixed", lambda k, _e: plan[k], None, 1.0, run_scen, mode)


def chunk(args):
    ci, (scenario, mode, budget), start, n = args
    out = np.zeros((n, len(POLICIES), 6), dtype=np.float32)
    for i in range(n):
        seed = [ci, start + i]
        for p, pol in enumerate(POLICIES):
            out[i, p] = run_policy(seed, pol, budget, scenario, mode)
    return start, out


def perstep(args):
    ci, (scenario, mode, budget), eps, n = args
    q, means = S.calibrate(np.random.default_rng(12), N_TAB, 0.05, "valid")
    run_scen = "broken" if scenario == "shift" else "valid"
    res = np.array([S.run_once(np.random.default_rng([ci, 10 ** 7 + i]), budget, "calibrated",
                               q, means, eps, run_scen, mode) for i in range(n)], dtype=np.float32)
    return eps, res


if __name__ == "__main__":
    out_dir = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    os.makedirs(out_dir, exist_ok=True)
    with Pool() as pool:
        for ci, cfg in enumerate(CONFIGS):
            name = "{}_{}_{}".format(*cfg)
            res = np.zeros((N, len(POLICIES), 6), dtype=np.float32)
            jobs = [(ci, cfg, s, min(CHUNK, N - s)) for s in range(0, N, CHUNK)]
            for start, part in pool.imap_unordered(chunk, jobs):
                res[start:start + len(part)] = part
            ps = dict(pool.map(perstep, [(ci, cfg, e, 20_000) for e in PERSTEP_EPS]))
            np.savez_compressed(os.path.join(out_dir, name + ".npz"), res=res,
                                policies=np.array(POLICIES), betas=np.array(BETAS),
                                **{f"perstep_{e}": ps[e] for e in PERSTEP_EPS})
            print(name, "done", flush=True)
