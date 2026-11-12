"""
Real-trace analysis v2: held-out evaluation of the shipped controller.

Replays logged agent runs under each reservation policy exactly as Baar-Core
0.11.0 plans them:
  - the plan is per model, pooling step types (the router does not know in
    advance whether a call will be a tool call or a final answer);
  - preflight and calibrated price the prompt with prompt_margin = 0.2
    (estimate does not), using the pre-call prompt estimate that was logged;
  - quantiles come from baar.calibration.QuantileTable and the level is chosen
    by baar.calibration.calibrate_from_failures (fixed-sequence Learn-then-Test).
A step is admitted iff spent + planned cost <= B; otherwise the run is refused
(ends without an answer). Refusing never changes the steps before it, so the
replay is exact for this admit/refuse controller.

For each model and budget, 200 random splits: 25% of runs build the quantile
table, 37.5% calibrate (the same runs are reused for every beta, which keeps
the guarantee valid), 37.5% are held out for evaluation.

Usage: python real_v2.py LOG.jsonl[.gz] OUT.json [--margin 0.2] [--true-prompt]
"""
import argparse
import gzip
import json
import math
from collections import defaultdict

import numpy as np

from baar.calibration import QuantileTable, binom_cdf, calibrate_from_failures, ltt_p_value

PRICE = {"gpt-4o-mini": (0.15e-6, 0.60e-6), "gpt-4o": (2.50e-6, 10.0e-6),
         "gpt-4o-priced": (2.50e-6, 10.0e-6)}
YMAX = 16384
BETAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16,
         0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
EPSILONS = [0.01, 0.02, 0.05]
DELTA = 0.05
BUDGET_Q = [50, 75, 90, 95, 99]
SPLITS = 200


def load(path, prompt_add=0):
    op = gzip.open if path.endswith(".gz") else open
    steps, runs = defaultdict(list), {}
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "run":
                runs[r["run"]] = r
            else:
                steps[r["run"]].append(r)
    out = []
    for rid, r in runs.items():
        st = sorted(steps[rid], key=lambda s: s["step"])
        if not st:
            continue
        if prompt_add:
            pin = PRICE[r["model"]][0]
            st = [dict(s, prompt_tokens=s["prompt_tokens"] + prompt_add,
                       est_prompt_tokens=s["est_prompt_tokens"] + prompt_add,
                       cost=s["cost"] + pin * prompt_add) for s in st]
        out.append(dict(run=rid, model=r["model"], f1=float(r.get("f1", 0.0)), finished=bool(r["finished"]),
                        cost=sum(s["cost"] for s in st), steps=st))
    return out


class Replayer:
    """
    Vectorized exact replay. Before step t the controller admits the call iff
    spent_{t-1} + p_in * ceil(prompt_t * (1 + margin)) + p_out * u <= B, i.e. iff
    u <= slack_t. A run is refused at the first step whose running minimum of
    slack falls below u; otherwise every step runs and the run overruns iff its
    total cost exceeds B.
    """

    def __init__(self, runs, margin, true_prompt):
        T = max(len(r["steps"]) for r in runs)
        n = len(runs)
        # need_t = spent_{t-1} + p_in * ceil(prompt_t * (1 + margin)); admitted iff need_t + p_out * u <= B
        self.maxneed = np.full((n, T), -np.inf)
        self.spent_before = np.zeros((n, T))
        self.pout = np.zeros(n)
        self.total = np.array([r["cost"] for r in runs])
        self.finished = np.array([float(r["finished"]) for r in runs])
        self.quality = np.array([r["f1"] for r in runs])
        for i, r in enumerate(runs):
            pin, pout = PRICE[r["model"]]
            self.pout[i] = pout
            spent, need, before = 0.0, [], []
            for s in r["steps"]:
                p = s["prompt_tokens"] if true_prompt else s["est_prompt_tokens"]
                before.append(spent)
                need.append(spent + pin * math.ceil(p * (1 + margin)))
                spent += s["cost"]
            k = len(need)
            self.maxneed[i, :k] = np.maximum.accumulate(need)
            self.maxneed[i, k:] = self.maxneed[i, k - 1]
            self.spent_before[i, :k] = before
            self.spent_before[i, k:] = before[-1]
            self.total[i] = spent

    def outcomes(self, u, idx, budget):
        """[len(idx), 4] = overrun, done, quality, spent for planned output u and budget B."""
        over = self.maxneed[idx] + (self.pout[idx] * u)[:, None] > budget
        refused = over[:, -1]
        first = np.argmax(over, axis=1)
        spent = np.where(refused, self.spent_before[idx, first], self.total[idx])
        admitted = ~refused
        # overspent: what the run was admitted to spend exceeds B (a refused run can have
        # crossed B before the refusal, when an admitted call cost more than reserved)
        return np.stack([spent > budget + 1e-12,
                         admitted * self.finished[idx], admitted * self.quality[idx], spent], axis=1).astype(float)


def summarize(o, budget):
    return dict(overrun=float(o[:, 0].mean()), done=float(o[:, 1].mean()),
                quality=float(o[:, 2].mean()), util=float(o[:, 3].mean() / budget))


def study(runs, model, q, args, rng):
    """Budgets are set per split from the table split only, so they are fixed before calibration."""
    res = defaultdict(list)
    curve = defaultdict(list)
    budgets = []
    N = len(runs)
    costs = np.array([r["cost"] for r in runs])
    rep = Replayer(runs, args.margin, args.true_prompt)
    for _ in range(SPLITS):
        idx = rng.permutation(N)
        a, b = N // 4, N // 4 + (3 * N) // 8
        tab, cal, test = idx[:a], idx[a:b], idx[b:]
        budget = float(np.percentile(costs[tab], q))
        budgets.append(budget)
        table = QuantileTable.from_records(
            {"model": model, "completion_tokens": s["completion_tokens"]} for i in tab for s in runs[i]["steps"])
        mean_u = float(np.mean([s["completion_tokens"] for i in tab for s in runs[i]["steps"]]))
        plans = {"preflight": YMAX, "estimate": mean_u}
        plans.update({f"b{bt}": table.bound(model, "any", bt) or YMAX for bt in BETAS})
        # Family: every calibrated level plus estimate mode, ordered from the largest
        # reservation to the smallest. The order depends only on the table split,
        # so it is fixed before calibration.
        family = sorted([f"b{bt}" for bt in BETAS] + ["estimate"], key=lambda k: -plans[k])
        cal_o = {k: rep.outcomes(u, cal, budget) for k, u in plans.items()}
        test_o = {k: rep.outcomes(u, test, budget) for k, u in plans.items()}
        for bt in BETAS:
            curve[bt].append(float(test_o[f"b{bt}"][:, 0].mean()))
        n = len(cal)
        for k in ["preflight", "estimate"]:
            res[k].append(summarize(test_o[k], budget))
        res["unconstrained"].append(dict(done=float(np.mean([runs[i]["finished"] for i in test])),
                                         quality=float(np.mean([runs[i]["f1"] for i in test]))))
        for eps in EPSILONS:
            fails = [int(cal_o[k][:, 0].sum()) for k in family]
            cr = calibrate_from_failures(list(range(len(family))), fails, n, eps, DELTA)
            key = family[int(cr.beta)] if cr.beta is not None else "preflight"
            f_test = int(test_o[key][:, 0].sum())
            # one-sided test of H: held-out risk <= eps; small p = significant violation
            p_viol = 1.0 - binom_cdf(f_test - 1, len(test), eps) if f_test > 0 else 1.0
            res[f"ltt_{eps}"].append(dict(summarize(test_o[key], budget), beta=key,
                                          sig_violation=float(p_viol < 0.05)))
            ok = ltt_p_value(int(cal_o["estimate"][:, 0].sum()), n, eps) <= DELTA
            res[f"certest_{eps}"].append(summarize(test_o["estimate" if ok else "preflight"], budget))
            # oracle: the member of the family that finishes most held-out runs with held-out risk <= eps
            good = [k for k in family if test_o[k][:, 0].mean() <= eps] or ["preflight"]
            best = max(good, key=lambda k: test_o[k][:, 1].mean())
            res[f"oracle_{eps}"].append(summarize(test_o[best], budget))
    agg = {}
    for k, rows in res.items():
        agg[k] = {m: float(np.mean([r[m] for r in rows]))
                  for m in rows[0] if isinstance(rows[0][m], (int, float))}
        if k.startswith("ltt_"):
            eps = float(k.split("_")[1])
            ov = np.array([r["overrun"] for r in rows])
            agg[k].update(overrun_p95=float(np.percentile(ov, 95)),
                          frac_test_over_eps=float(np.mean(ov > eps)),
                          frac_sig_violation=float(np.mean([r["sig_violation"] for r in rows])),
                          beta_counts={b: sum(r["beta"] == b for r in rows)
                                       for b in sorted({r["beta"] for r in rows})})
            agg[k].pop("beta", None)
            agg[k].pop("sig_violation", None)
    return dict(model=model, budget_quantile=q, budget=float(np.mean(budgets)),
                budget_range=[float(min(budgets)), float(max(budgets))], n_runs=len(runs), splits=SPLITS,
                risk_by_beta={str(bt): float(np.mean(v)) for bt, v in curve.items()},
                n_cal=len(runs) // 4 + (3 * len(runs)) // 8 - len(runs) // 4, results=agg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--true-prompt", action="store_true")
    ap.add_argument("--prompt-add", type=int, default=0,
                    help="tokens added to every prompt (e.g. tool schemas missing from a log)")
    args = ap.parse_args()
    runs = load(args.log, args.prompt_add)
    rng = np.random.default_rng(0)
    out = []
    for model in sorted({r["model"] for r in runs}):
        rs = [r for r in runs if r["model"] == model]
        for q in BUDGET_Q:
            out.append(study(rs, model, q, args, rng))
            print(model, q, "done", flush=True)
    json.dump(dict(margin=args.margin, true_prompt=args.true_prompt, prompt_add=args.prompt_add,
                   family="calibrated levels then estimate", eps=EPSILONS, delta=DELTA,
                   betas=BETAS, studies=out), open(args.out, "w"), indent=1)
