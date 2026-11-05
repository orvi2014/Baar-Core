"""
Certified early stopping (and start-of-run admission) for long-horizon agents.

At checkpoints k in K (k = 0 is the start of the run), a predictor fitted on a
held-out split scores how likely the run is to cost more than the budget B.
Policy tau stops the run at the first checkpoint whose score is >= tau
(tau = +inf: never stop). Stopping never changes the steps before it, so logged
runs replay exactly.

Harm we certify against (loss = 1):
  "finish"  — the policy stops a run that would have finished within B;
  "resolve" — the policy stops a run that would have resolved the task within B.
Fixed-sequence Learn-then-Test over tau, from never-stop downward, gives
P(risk > eps) <= delta; the fallback "never stop" has zero risk.

Benefit: spend saved on runs that cannot finish within B anyway.

Usage: python admission.py LOG.jsonl.gz OUT.json [--prompt-add 3121]
"""
import argparse
import gzip
import json

import numpy as np

import real_v2 as R
from baar.calibration import binom_cdf, calibrate_from_failures

CHECKPOINTS = [0, 5, 10, 15, 20, 30, 40, 50, 60, 80]
TAUS = [0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
EPSILONS = [0.01, 0.02, 0.05]
DELTA = 0.05
SPLITS = 200
BUDGET_Q = [50, 75, 90]


def features(run, k, behavior=None):
    """
    What the controller knows before step k+1 (k = 0: before the first call).
    base: prompt size, spend, context, output lengths so far. behavior (optional):
    counts of the agent's actions and tool errors after k turns (log1p).
    """
    st = run["steps"]
    first = st[0]["prompt_tokens"]
    if k == 0:
        base = [1.0, np.log1p(first), 0.0, 0.0, 0.0, 0.0]
    else:
        done = st[:k]
        spent = sum(s["cost"] for s in done)
        ctx = st[k]["prompt_tokens"] if k < len(st) else done[-1]["prompt_tokens"]
        outs = np.array([s["completion_tokens"] for s in done], float)
        base = [1.0, np.log1p(first), np.log1p(spent * 1e4), np.log1p(ctx),
                np.log1p(outs.mean()), np.log1p(outs.max())]
    if behavior is None:
        return np.array(base)
    return np.array(base + list(np.log1p(np.array(behavior[k], float))))


def fit_logistic(X, y, l2=1e-3, iters=50):
    """Ridge-regularised logistic regression by Newton's method (no dependencies)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--prompt-add", type=int, default=0)
    ap.add_argument("--behavior", help="behavior.json.gz from swe/scripts/behavior.py")
    ap.add_argument("--features", choices=["base", "behavior", "full"], default="base",
                    help="full = behavior + a repository cost prior fitted on the train split")
    args = ap.parse_args()
    runs = R.load(args.log, args.prompt_add)
    beh = None
    if args.features != "base":
        bj = json.load(gzip.open(args.behavior, "rt"))["runs"]
        ids = [r["run"] for r in runs]
        beh = [bj[i]["rows"] for i in ids]
        repos = [bj[i]["repo"] for i in ids]
    N = len(runs)
    cost = np.array([r["cost"] for r in runs])
    finished = np.array([r["finished"] for r in runs], bool)
    resolved = np.array([r["f1"] > 0 for r in runs], bool)
    nsteps = np.array([len(r["steps"]) for r in runs])
    # spend before step k (k = 0 .. max): cum[i, k]
    T = nsteps.max()
    cum = np.zeros((N, T + 1))
    for i, r in enumerate(runs):
        c = np.cumsum([0.0] + [s["cost"] for s in r["steps"]])
        cum[i, :len(c)] = c
        cum[i, len(c):] = c[-1]
    # features per checkpoint (only for runs still going at k)
    dim = len(features(runs[0], 0, beh[0] if beh else None))
    F = {k: np.array([features(r, k, beh[i] if beh else None) if k < len(r["steps"]) else np.full(dim, np.nan)
                      for i, r in enumerate(runs)]) for k in CHECKPOINTS}
    rng = np.random.default_rng(0)
    out = []
    for q in BUDGET_Q:
        agg = {f"{loss}_{eps}": [] for loss in ["finish", "resolve"] for eps in EPSILONS}
        agg["none"] = []
        agg["naive_0.5"] = []
        for _ in range(SPLITS):
            idx = rng.permutation(N)
            a, b = N // 4, N // 4 + (3 * N) // 8
            tr, cal, te = idx[:a], idx[a:b], idx[b:]
            B = float(np.percentile(cost[tr], q))
            # budget-only controller: a run is cut when its next call is unaffordable; replay
            # with the estimate-mode admission rule is within 3 points of "cost > B" (Sec. 8),
            # so we use cost > B as the budget-only refusal and spend min(cost, B).
            over = cost > B
            fits = finished & ~over                     # would finish within B
            fits_res = resolved & ~over                 # would resolve within B
            stop_at_budget = np.where(over, np.minimum(cost, B), cost)
            # scores per checkpoint from a predictor of "cost > B" fitted on the train split
            S = np.full((N, len(CHECKPOINTS)), -np.inf)
            prior = None
            if args.features == "full":
                # repository prior: shrunken mean log cost of train runs from the same repo
                lc = np.log(cost)
                g = lc[tr].mean()
                sums, cnts = {}, {}
                for i in tr:
                    sums[repos[i]] = sums.get(repos[i], 0.0) + lc[i]
                    cnts[repos[i]] = cnts.get(repos[i], 0) + 1
                prior = np.array([(sums.get(repos[i], 0.0) + 2 * g) / (cnts.get(repos[i], 0) + 2) - g
                                  for i in range(N)])
            for j, k in enumerate(CHECKPOINTS):
                alive = nsteps > k
                trk = tr[alive[tr]]
                if len(trk) < 50 or over[trk].all() or (~over[trk]).all():
                    continue
                X = F[k] if prior is None else np.column_stack([F[k], prior])
                w = fit_logistic(X[trk], over[trk].astype(float))
                al = np.where(alive)[0]
                S[al, j] = 1 / (1 + np.exp(-X[al] @ w))

            def apply(tau, ids):
                """(stopped, spend) for each run in ids under threshold tau."""
                hit = S[ids] >= tau
                first = np.where(hit.any(1), hit.argmax(1), -1)
                stopped = first >= 0
                kstop = np.array([CHECKPOINTS[f] if f >= 0 else 0 for f in first])
                spend = np.where(stopped, np.minimum(cum[ids, np.minimum(kstop, nsteps[ids])], B),
                                 stop_at_budget[ids])
                return stopped, spend

            def evaluate(tau, ids):
                if tau is None:
                    stopped, spend = np.zeros(len(ids), bool), stop_at_budget[ids]
                else:
                    stopped, spend = apply(tau, ids)
                base = stop_at_budget[ids].sum()
                doomed = over[ids]
                return dict(false_kill=float(np.mean(stopped & fits[ids])),
                            lost_resolve=float(np.mean(stopped & fits_res[ids])),
                            saved=float(1 - spend.sum() / base),
                            doomed_stopped=float(stopped[doomed].mean()) if doomed.any() else 0.0,
                            finished=float(np.mean(~stopped & fits[ids])),
                            resolved=float(np.mean(~stopped & fits_res[ids])))

            agg["none"].append(evaluate(None, te))
            agg["naive_0.5"].append(evaluate(0.5, te))
            for loss, harm in [("finish", fits), ("resolve", fits_res)]:
                fails = [int(np.sum(apply(t, cal)[0] & harm[cal])) for t in TAUS]
                for eps in EPSILONS:
                    cr = calibrate_from_failures(list(range(len(TAUS))), fails, len(cal), eps, DELTA)
                    tau = TAUS[int(cr.beta)] if cr.beta is not None else None
                    e = evaluate(tau, te)
                    ft = int(round(e["false_kill" if loss == "finish" else "lost_resolve"] * len(te)))
                    p_viol = 1.0 - binom_cdf(ft - 1, len(te), eps) if ft > 0 else 1.0
                    agg[f"{loss}_{eps}"].append(dict(e, tau=tau if tau is not None else 1.0,
                                                     sig_violation=float(p_viol < 0.05)))
        res = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in agg.items()}
        for k, v in agg.items():
            if k.startswith(("finish", "resolve")):
                key = "false_kill" if k.startswith("finish") else "lost_resolve"
                res[k]["harm_p95"] = float(np.percentile([r[key] for r in v], 95))
        out.append(dict(budget_quantile=q, results=res))
        print("p%d" % q, json.dumps({k: {m: round(x, 4) for m, x in v.items()} for k, v in res.items()}), flush=True)
    json.dump(dict(checkpoints=CHECKPOINTS, taus=TAUS, eps=EPSILONS, delta=DELTA, splits=SPLITS,
                   prompt_add=args.prompt_add, features=args.features, studies=out), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
