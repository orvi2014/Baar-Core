"""
Real-trace analysis of the HotpotQA pilot (pilot_log.jsonl from pilot.py).

1. Descriptive statistics: per-stratum output-length tails, run-cost spread,
   prompt-token estimation error.
2. Exact replay of admit/refuse budget controllers on the logged runs. A
   controller that only admits or refuses steps never changes the agent's
   trajectory before it refuses, so replaying logged runs gives the exact outcome
   each policy would have had on those tasks.
3. Learn-then-Test on real runs: tables fitted on a held-out split; calibration
   sets drawn i.i.d. from the remaining runs (the "population"), whose true risk
   is computed exactly. Repeated many times to check P(R > eps) <= delta.

Usage: python real_replay.py pilot_log.jsonl [out.json]
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

PRICE = {"gpt-4o-mini": (0.15e-6, 0.60e-6), "gpt-4o": (2.50e-6, 10.0e-6)}
YMAX = 16384
BETAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
EPS, DELTA = 0.05, 0.05


def load(path):
    steps, runs = defaultdict(list), {}
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "run":
            runs[r["run"]] = r
        else:
            steps[r["run"]].append(r)
    out = []
    for rid, r in runs.items():
        st = sorted(steps[rid], key=lambda s: s["step"])
        out.append(dict(model=r["model"], f1=r["f1"], finished=r["finished"], steps=st,
                        cost=sum(s["cost"] for s in st)))
    return out


def pct(a, q):
    return float(np.percentile(a, q))


def describe(runs):
    d = {}
    for m in PRICE:
        rs = [r for r in runs if r["model"] == m]
        if not rs:
            continue
        costs = np.array([r["cost"] for r in rs])
        strata = {}
        for kind in ["tool", "final"]:
            y = np.array([s["completion_tokens"] for r in rs for s in r["steps"] if s["kind"] == kind])
            if len(y):
                strata[kind] = dict(n=int(len(y)), mean=float(y.mean()), p50=pct(y, 50), p90=pct(y, 90),
                                    p99=pct(y, 99), max=int(y.max()), ymax_over_p99=YMAX / max(pct(y, 99), 1))
        est = np.array([s["est_prompt_tokens"] for r in rs for s in r["steps"]], float)
        act = np.array([s["prompt_tokens"] for r in rs for s in r["steps"]], float)
        d[m] = dict(runs=len(rs), steps_mean=float(np.mean([len(r["steps"]) for r in rs])),
                    f1=float(np.mean([r["f1"] for r in rs])),
                    finished=float(np.mean([r["finished"] for r in rs])),
                    cost_mean=float(costs.mean()), cost_cv=float(costs.std() / costs.mean()),
                    cost_p50=pct(costs, 50), cost_p90=pct(costs, 90), cost_p99=pct(costs, 99),
                    cost_max=float(costs.max()), max_over_median=float(costs.max() / np.median(costs)),
                    prompt_est_rel_err_median=float(np.median(np.abs(est - act) / act)),
                    prompt_est_rel_err_p99=pct(np.abs(est - act) / act, 99),
                    length_finishes=int(sum(s["finish"] == "length" for r in rs for s in r["steps"])),
                    strata=strata)
    return d


class Table:
    def __init__(self, runs):
        self.y = {k: np.sort([s["completion_tokens"] for r in runs for s in r["steps"] if s["kind"] == k])
                  for k in ["tool", "final"]}

    def plan(self, policy, kind, beta=None):
        y = self.y[kind]
        if policy == "preflight" or len(y) == 0:
            return YMAX
        if policy == "estimate":
            return float(y.mean())
        return float(np.quantile(y, 1 - beta))


def replay(run, plan, budget, model):
    """Mode A replay: returns (overrun, completed, f1_if_completed)."""
    pin, pout = PRICE[model]
    spent = 0.0
    for s in run["steps"]:
        # reserve with the controller's own pre-call prompt estimate, as Baar does
        if spent + pin * s["est_prompt_tokens"] + pout * plan[s["kind"]] > budget:
            return 0, 0, 0.0                      # refused: run ends without an answer
        spent += s["cost"]
    return int(spent > budget + 1e-12), int(run["finished"]), run["f1"]


def evaluate(runs, plan, budget, model):
    res = np.array([replay(r, plan, budget, model) for r in runs], float)
    return dict(overrun=float(res[:, 0].mean()), done=float(res[:, 1].mean()), f1=float(res[:, 2].mean()))


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(int(k) + 1))


def ltt(losses_by_beta, n):
    """losses_by_beta: list of failure counts in BETAS order. Returns index of chosen beta or None."""
    chosen = None
    for j, f in enumerate(losses_by_beta):
        if binom_cdf(f, n, EPS) > DELTA:
            break
        chosen = j
    return chosen


def ltt_study(runs, model, budget_q, reps=500, n_cal=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(runs))
    n_tab = len(runs) // 4
    table = Table([runs[i] for i in idx[:n_tab]])
    pop = [runs[i] for i in idx[n_tab:]]
    budget = pct([r["cost"] for r in runs], budget_q)
    plans = {b: {k: table.plan("calibrated", k, b) for k in ["tool", "final"]} for b in BETAS}
    # per-run loss and outcome for each beta (deterministic replay)
    out = {b: np.array([replay(r, plans[b], budget, model) for r in pop], float) for b in BETAS}
    true_risk = {b: float(out[b][:, 0].mean()) for b in BETAS}
    base = {p: evaluate(pop, {k: table.plan(p, k) for k in ["tool", "final"]}, budget, model)
            for p in ["estimate", "preflight"]}
    n = n_cal or len(pop) // 2
    chosen, viol = [], 0
    for _ in range(reps):
        sample = rng.integers(0, len(pop), n)
        fails = [int(out[b][sample, 0].sum()) for b in BETAS]
        j = ltt(fails, n)
        b = BETAS[j] if j is not None else None
        chosen.append(b)
        viol += (true_risk[b] if b is not None else 0.0) > EPS
    from collections import Counter
    mode_b = Counter(chosen).most_common(1)[0][0]
    deployed = (dict(overrun=true_risk[mode_b], done=float(out[mode_b][:, 1].mean()),
                     f1=float(out[mode_b][:, 2].mean())) if mode_b is not None else base["preflight"])
    return dict(model=model, budget=budget, budget_quantile=budget_q, n_table=n_tab, n_pop=len(pop),
                n_cal=n, reps=reps, violations=viol, beta_counts={str(k): v for k, v in Counter(chosen).items()},
                true_risk_by_beta={str(b): r for b, r in true_risk.items()},
                estimate=base["estimate"], preflight=base["preflight"], ltt_typical=dict(beta=mode_b, **deployed),
                unconstrained=dict(done=float(np.mean([r["finished"] for r in pop])),
                                   f1=float(np.mean([r["f1"] for r in pop]))))


if __name__ == "__main__":
    runs = load(sys.argv[1])
    result = dict(describe=describe(runs), ltt=[])
    for m in PRICE:
        rs = [r for r in runs if r["model"] == m]
        if len(rs) < 100:
            continue
        for q in [50, 75, 90]:
            result["ltt"].append(ltt_study(rs, m, q))
    print(json.dumps(result, indent=1))
    if len(sys.argv) > 2:
        json.dump(result, open(sys.argv[2], "w"), indent=1)
