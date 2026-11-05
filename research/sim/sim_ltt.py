"""
Approach (b): run-level calibration via Learn-then-Test (fixed-sequence testing).

Policy family: per-step bound = empirical (1 - beta) quantile of the stratum's output
length. Smaller beta = more conservative. LTT walks the grid from conservative to
aggressive, testing H0: overrun_rate(beta) > eps with an exact binomial p-value on
n_cal fresh calibration runs, and stops at the first non-rejection.
Guarantee: P(overrun_rate(beta_hat) <= eps) >= 1 - delta (no stratum assumption needed;
requires runs i.i.d. from the task distribution).
"""
import sys

import numpy as np
from scipy.stats import binom

import sim_thm1 as S

BETAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def make_quantiles(rng, n=2000, scenario="valid"):
    tab = {}
    for k in S.STRATA:
        d = rng.normal(0, 0.6, n) if scenario == "broken" else np.zeros(n)
        tab[k] = np.sort([S.draw_y(rng, k[0], k[1], di) for di in d])
    return lambda k, beta: int(np.quantile(tab[k], 1 - beta)) if beta > 0 else S.YMAX[k[0]]


def runs(rng, budget, quant, beta, eps_unused, scenario, mode, n):
    q = lambda k, _e: quant(k, beta)
    # 'calibrated' branch with wealth large makes fits() use q at a fixed level
    out = [S.run_once(rng, budget, "fixed", q, None, 1.0, scenario, mode) for _ in range(n)]
    return np.array(out, dtype=float)


def ltt(budget, eps, delta, scenario, mode, n_cal, seed):
    rng = np.random.default_rng(seed)
    quant = make_quantiles(np.random.default_rng(seed + 99), scenario=scenario)
    chosen = None
    for beta in BETAS:                      # fixed sequence: conservative -> aggressive
        r = runs(rng, budget, quant, beta, eps, scenario, mode, n_cal)
        fails = r[:, 0].sum() if mode == "A" else r[:, 1].sum()
        p = binom.cdf(fails, n_cal, eps)    # small p => evidence that rate <= eps
        if p > delta:
            break
        chosen = beta
    return chosen, quant


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    n_cal, n_test = 1000, 10000
    for scenario in ["valid", "broken"]:
        for mode in ["A", "B"]:
            for eps in [0.01, 0.05, 0.10]:
                beta, quant = ltt(budget, eps, 0.05, scenario, mode, n_cal, seed=3)
                if beta is None:
                    print(f"[{scenario} {mode} eps={eps}] LTT: no beta certified (fallback = preflight)")
                    continue
                r = runs(np.random.default_rng(123), budget, quant, beta, eps, scenario, mode, n_test)
                big = r[:, 5].sum() / r[:, 4].sum()
                print(f"[{scenario} {mode} eps={eps}] beta={beta} overrun={r[:,0].mean():.4f} "
                      f"trunc={r[:,1].mean():.4f} complete={r[:,2].mean():.3f} "
                      f"util={r[:,3].mean()/budget:.3f} big={big:.3f}")
