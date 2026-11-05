"""
Analysis of the simulation v2 pools (sim_v2.py output).

Half of each pool (100k tasks) is the calibration source, the other half is
the evaluation set, on which every policy's true risk is known to within
about +-0.07 percentage points. For each eps and calibration size n we draw
2,000 calibration sets of n tasks (without replacement) from the calibration
half, run fixed-sequence Learn-then-Test over the beta grid (the same tasks
for every beta), and score the selected policy on the evaluation half.

Ground truth: if TRUTH.json (sim_truth.py) is given, every policy's risk and
outcomes come from a fresh pool of 400k-1M tasks with disjoint seeds, and a
selected policy counts as a violation if its true risk exceeds eps; it is
"ambiguous" if the true risk is within two standard errors of eps.

Usage: python analysis_v2.py POOL_DIR OUT.json [TRUTH.json]
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binom

EPSILONS = [0.01, 0.02, 0.05, 0.10]
NS = [100, 300, 1000, 3000]
DELTA = 0.05
REPS = 2000


def metrics(block, budget, loss_col):
    """block: [tasks, 6] = overrun, trunc, completed, spent, steps, big."""
    return dict(risk=float(block[:, loss_col].mean()), done=float(block[:, 2].mean()),
                big=float(block[:, 5].sum() / max(block[:, 4].sum(), 1)),
                util=float(block[:, 3].mean() / budget))


def main(pool_dir, out_path, truth_path=None):
    rng = np.random.default_rng(0)
    truth_all = json.load(open(truth_path)) if truth_path else None
    out = []
    for path in sorted(glob.glob(os.path.join(pool_dir, "*.npz"))):
        z = np.load(path)
        scenario, mode, budget = os.path.basename(path)[:-4].split("_")
        budget = float(budget)
        loss_col = 0 if mode == "A" else 1
        res, policies = z["res"], list(z["policies"])
        betas = [float(b) for b in z["betas"]]
        half = len(res) // 2
        cal, ev = res[:half], res[half:]
        ev_m = {p: metrics(ev[:, i], budget, loss_col) for i, p in enumerate(policies)}
        risk = {p: ev_m[p]["risk"] for p in policies}
        # risk in the population calibration sets are drawn from (what Theorem 1 is about)
        risk_cal = {p: float(cal[:, i, loss_col].mean()) for i, p in enumerate(policies)}
        se_ev = {p: float(np.sqrt(max(risk[p] * (1 - risk[p]), 1e-12) / len(ev))) for p in policies}
        name = os.path.basename(path)[:-4]
        if truth_all is not None:
            T = truth_all[name]
            N_true = T["_n"]
            key = "overrun" if loss_col == 0 else "trunc"
            ev_m = {p: dict(risk=T[p][key], done=T[p]["done"], big=T[p]["big_steps"] / max(T[p]["steps"], 1e-12),
                            util=T[p]["spent"] / budget) for p in policies}
            risk = {p: ev_m[p]["risk"] for p in policies}
            se_ev = {p: float(np.sqrt(max(risk[p] * (1 - risk[p]), 1e-12) / N_true)) for p in policies}
        cal_loss = cal[:, :, loss_col]
        bidx = [policies.index(f"b{b}") for b in betas]
        eidx = policies.index("estimate")
        cfg = dict(scenario=scenario, mode=mode, budget=budget, pool=len(res),
                   truth_n=(truth_all[name]["_n"] if truth_all is not None else None),
                   risk_curve={str(b): risk[f"b{b}"] for b in betas},
                   risk_curve_cal={str(b): risk_cal[f"b{b}"] for b in betas},
                   preflight=ev_m["preflight"], estimate=ev_m["estimate"], runs={})
        for eps in EPSILONS:
            ps = z[f"perstep_{eps}"] if f"perstep_{eps}" in z.files else None
            # oracle: the most capable member of the family whose true risk is <= eps
            good = [b for b in betas if risk[f"b{b}"] <= eps]
            ob = max(good, key=lambda b: ev_m[f"b{b}"]["big"]) if good else None
            oracle = ev_m[f"b{ob}"] if ob is not None else ev_m["preflight"]
            entry = dict(oracle=dict(oracle, beta=ob),
                         perstep=metrics(ps, budget, loss_col) if ps is not None else None)
            for n in NS:
                sel_ltt, sel_ce = [], []
                for _ in range(REPS):
                    rows = rng.choice(half, n, replace=False)
                    fails = cal_loss[rows][:, bidx].sum(0)
                    chosen = "preflight"
                    for j, f in enumerate(fails):        # fixed sequence, conservative first
                        if binom.cdf(f, n, eps) > DELTA:
                            break
                        chosen = f"b{betas[j]}"
                    sel_ltt.append(chosen)
                    fe = cal_loss[rows, eidx].sum()
                    sel_ce.append("estimate" if binom.cdf(fe, n, eps) <= DELTA else "preflight")

                def agg(sel):
                    r = np.array([risk[s] for s in sel])
                    return dict(violation_calpop=float(np.mean([risk_cal[s] > eps for s in sel])),
                                violation=float(np.mean([risk[s] > eps for s in sel])),
                                violation_clear=float(np.mean([risk[s] > eps + 2 * se_ev[s] for s in sel])),
                                ambiguous=float(np.mean([abs(risk[s] - eps) <= 2 * se_ev[s] for s in sel])),
                                violation_heldout_2se=float(np.mean([risk[s] > eps + 2 * se_ev[s] for s in sel])),
                                violation_heldout_raw=float(np.mean(r > eps)),
                                risk_mean=float(r.mean()), risk_max=float(r.max()),
                                done=float(np.mean([ev_m[s]["done"] for s in sel])),
                                big=float(np.mean([ev_m[s]["big"] for s in sel])),
                                util=float(np.mean([ev_m[s]["util"] for s in sel])),
                                fallback=float(np.mean([s == "preflight" for s in sel])))
                entry[f"n{n}"] = dict(ltt=agg(sel_ltt), certified_estimate=agg(sel_ce))
            cfg["runs"][str(eps)] = entry
        out.append(cfg)
        print(os.path.basename(path), "done", flush=True)
    json.dump(dict(eps=EPSILONS, ns=NS, delta=DELTA, reps=REPS, configs=out), open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
