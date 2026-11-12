"""
Simulator for Theorem 1 (run-level budget guarantee).

Synthetic agent: steps with growing context, adaptive model choice, heavy-tailed
output lengths per stratum (model x step type). Compares output-reservation
policies: estimate (mean), preflight (worst case), calibrated (ours).

Scenarios:
  valid   — stratum-Markov assumption holds (Y_t | history ~ F_k).
  broken  — output length depends on a hidden per-task difficulty (assumption fails).
"""
import math
import sys

import numpy as np

PRICE = {  # $ per token (in, out)
    "small": (0.15e-6, 0.60e-6),
    "big": (2.50e-6, 10.0e-6),
}
YMAX = {"small": 4096, "big": 8192}           # provider max output
LOGN = {                                       # lognormal (mu, sigma) of output tokens
    ("small", "tool"): (5.0, 0.9),
    ("small", "final"): (5.8, 0.8),
    ("big", "tool"): (5.4, 1.0),
    ("big", "final"): (6.2, 0.9),
}
OBS_MAX = 800                                  # tool observation cap (tool-side)
N0 = 600                                       # initial prompt tokens
RHO = 0.5                                      # fraction of risk wealth spent per tight step
STRATA = list(LOGN)


def draw_y(rng, model, kind, difficulty=0.0):
    mu, s = LOGN[(model, kind)]
    y = int(math.exp(mu + difficulty + s * rng.standard_normal()))
    return max(1, min(y, YMAX[model]))


GRID = np.geomspace(1e-4, 0.5, 40)


def calibrate(rng, n_per_stratum, delta, scenario):
    """Logged samples per stratum (from an arbitrary logging policy).
    Returns q(k, eps): an upper bound with P(Y > q) <= eps for every stratum k and
    grid level <= eps, simultaneously with prob >= 1 - delta over calibration
    (exact binomial order-statistic bound, Bonferroni over strata x grid)."""
    from scipy.stats import binom
    d_each = delta / (len(STRATA) * len(GRID))
    tables = {}
    for k in STRATA:
        d = rng.normal(0, 0.6, n_per_stratum) if scenario == "broken" else np.zeros(n_per_stratum)
        ys = np.sort([draw_y(rng, k[0], k[1], di) for di in d])
        n = len(ys)
        js = np.arange(1, n + 1)
        qs = {}
        for g in GRID:
            # q = y_(j): P(F(y_(j)) < 1-g) = P(Binom(n, 1-g) >= j); need <= d_each
            ok = js[binom.sf(js - 1, n, 1 - g) <= d_each]
            qs[g] = int(ys[ok[0] - 1]) if len(ok) else YMAX[k[0]]
        tables[k] = qs

    def q(k, eps):
        usable = GRID[GRID <= eps]
        if len(usable) == 0:
            return YMAX[k[0]]
        return tables[k][usable[-1]]

    means = {k: float(np.mean([draw_y(rng, k[0], k[1]) for _ in range(n_per_stratum)])) for k in STRATA}
    return q, means


def run_once(rng, budget, policy, q, means, eps, scenario, mode):
    """Returns (overrun, truncated, completed, spent, steps)."""
    want = min(30, 1 + rng.geometric(1 / 6))
    difficulty = rng.normal(0, 0.6) if scenario == "broken" else 0.0
    spent, n, t, steps = 0.0, N0, 0, 0
    truncated = False
    wealth = eps
    big_steps = 0

    def bound(model, kind, eps_t):
        if policy == "estimate":
            return int(means[(model, kind)])
        if policy == "preflight":
            return YMAX[model]
        return q((model, kind), eps_t)

    def cost(model, n_in, y):
        pi, po = PRICE[model]
        return pi * n_in + po * y

    def fits(m, kind, eps_t):
        b = bound(m, kind, eps_t)
        need = cost(m, n, b)
        if kind == "tool":
            need += cost("small", n + b + OBS_MAX, bound("small", "final", eps_t))
        return spent + need <= budget

    while True:
        t += 1
        final = steps >= want - 1
        kinds = ["final"] if final else ["tool"]
        prefer = ["big", "small"] if (budget - spent) > 0.5 * budget else ["small"]
        admitted = None
        # try normal step (with reserve for wrap-up), downgrading if needed
        spend = 0.0
        candidates = [(m, k) for k in kinds for m in prefer]
        if not final:
            candidates.append(("small", "final"))  # early wrap-up
        for m, kind in candidates:
            if fits(m, kind, 0.0):                  # worst case fits: zero risk
                admitted, spend = (m, kind), 0.0
                break
            if policy != "calibrated":
                if fits(m, kind, eps):              # eps ignored by non-calibrated bounds
                    admitted = (m, kind)
                    break
                continue
            e = RHO * wealth                        # risk split across the step and its reserve
            if fits(m, kind, e / 2):
                admitted, spend = (m, kind), e
                break
        if admitted is not None and policy == "calibrated":
            wealth -= spend
        if admitted is None:
            # refused, run incomplete; it has overspent if earlier calls cost more than reserved
            return spent > budget + 1e-12, truncated, False, spent, steps, big_steps
        m, kind = admitted
        big_steps += m == "big"
        y = draw_y(rng, m, kind, difficulty)
        if mode == "B":
            cap = int((budget - spent - PRICE[m][0] * n) / PRICE[m][1])
            if y > cap:
                y, truncated = cap, True
        spent += cost(m, n, y)
        steps += 1
        if kind == "final":
            return spent > budget + 1e-12, truncated, True, spent, steps, big_steps
        n += y + int(rng.integers(50, OBS_MAX + 1))


def experiment(scenario, mode, eps, budget, runs, seed, delta=0.01, n_cal=2000):
    rng = np.random.default_rng(seed)
    out = {}
    for policy in ["estimate", "preflight", "calibrated"]:
        q, means = calibrate(np.random.default_rng(seed + 1), n_cal, delta, scenario)
        res = np.array([run_once(rng, budget, policy, q, means, eps, scenario, mode) for _ in range(runs)], dtype=float)
        out[policy] = dict(
            overrun=res[:, 0].mean(), trunc=res[:, 1].mean(), complete=res[:, 2].mean(),
            util=res[:, 3].mean() / budget, steps=res[:, 4].mean(), big=res[:, 5].sum() / res[:, 4].sum(),
        )
    return out


def calibration_repeat(scenario, eps, budget, reps, runs, delta=0.01):
    """Violation rate across independent calibration draws (tests the +delta term)."""
    rates = []
    for r in range(reps):
        q, means = calibrate(np.random.default_rng(10_000 + r), 2000, delta, scenario)
        rng = np.random.default_rng(20_000 + r)
        v = np.mean([run_once(rng, budget, "calibrated", q, means, eps, scenario, "A")[0] for _ in range(runs)])
        rates.append(v)
    return np.array(rates)


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    for scenario in ["valid", "broken"]:
        for mode in ["A", "B"]:
            for eps in [0.01, 0.05, 0.10]:
                r = experiment(scenario, mode, eps, budget, runs, seed=7)
                print(f"\n[{scenario} mode={mode} eps={eps}]")
                for p, m in r.items():
                    print(f"  {p:10s} overrun={m['overrun']:.4f} trunc={m['trunc']:.4f} "
                          f"complete={m['complete']:.3f} util={m['util']:.3f} big={m['big']:.3f}")
    rates = calibration_repeat("valid", 0.05, budget, reps=50, runs=2000)
    print(f"\n[calibration repeats valid eps=0.05] mean={rates.mean():.4f} "
          f"max={rates.max():.4f} frac>eps={np.mean(rates > 0.05):.2f}")
