"""Produce every number used in the paper's simulation tables, as JSON lines.
Baselines (estimate, preflight, per-step calibrated) and LTT, for both scenarios,
modes A/B and eps in {0.01, 0.05, 0.10}. Budget $0.02, delta 0.05."""
import json
import sys
from multiprocessing import Pool

import numpy as np

import sim_ltt as L
import sim_thm1 as S

BUDGET, DELTA, N_CAL, N_TEST = 0.02, 0.05, 1000, 20000


def summarize(res):
    return dict(overrun=float(res[:, 0].mean()), trunc=float(res[:, 1].mean()),
                complete=float(res[:, 2].mean()), util=float(res[:, 3].mean() / BUDGET),
                steps=float(res[:, 4].mean()),
                big=float(res[:, 5].sum() / max(res[:, 4].sum(), 1)))


def job(spec):
    scenario, mode, eps, method = spec
    row = dict(scenario=scenario, mode=mode, eps=eps, method=method)
    rng = np.random.default_rng(123)
    if method == "ltt":
        beta, quant = L.ltt(BUDGET, eps, DELTA, scenario, mode, N_CAL, seed=3)
        row["beta"] = beta
        if beta is None:
            return row
        row.update(summarize(L.runs(rng, BUDGET, quant, beta, eps, scenario, mode, N_TEST)))
        return row
    q, means = S.calibrate(np.random.default_rng(8), 2000, DELTA, scenario)
    res = np.array([S.run_once(rng, BUDGET, method, q, means, eps, scenario, mode)
                    for _ in range(N_TEST)], dtype=float)
    row.update(summarize(res))
    return row


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "tables.jsonl"
    specs = [(s, m, e, meth) for s in ["valid", "broken"] for m in ["A", "B"]
             for e in [0.01, 0.05, 0.10] for meth in ["estimate", "preflight", "calibrated", "ltt"]]
    with Pool() as pool, open(out, "w") as f:
        for row in pool.imap_unordered(job, specs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row, flush=True)
