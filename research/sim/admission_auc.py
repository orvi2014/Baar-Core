"""How predictable is "this run will cost more than B" at each checkpoint, and how much
could an oracle that stops every such run at that checkpoint save? (Section 8.)
Usage: python admission_auc.py LOG.jsonl.gz BEHAVIOR.json.gz OUT.json [--prompt-add 3121]"""
import gzip
import json
import sys

import numpy as np

import admission as A
import real_v2 as R

add = int(sys.argv[sys.argv.index("--prompt-add") + 1]) if "--prompt-add" in sys.argv else 0
runs = R.load(sys.argv[1], add)
bj = json.load(gzip.open(sys.argv[2], "rt"))["runs"]
beh = [bj[r["run"]]["rows"] for r in runs]
repos = [bj[r["run"]]["repo"] for r in runs]
N = len(runs)
cost = np.array([r["cost"] for r in runs])
ns = np.array([len(r["steps"]) for r in runs])
cum = [np.cumsum([0.0] + [s["cost"] for s in r["steps"]]) for r in runs]
KS = [0, 5, 10, 15, 20, 30, 40]


def auc(s, y):
    o = np.argsort(s)
    rk = np.empty(len(s))
    rk[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


out = {}
for q in [50, 75]:
    res = {(k, f): [] for k in KS for f in ["base", "behavior", "full"]}
    oracle = {k: [] for k in KS}
    for seed in range(20):
        idx = np.random.default_rng(seed).permutation(N)
        tr0, te0 = idx[: N // 2], idx[N // 2:]
        B = np.percentile(cost[tr0], q)
        over = cost > B
        lc = np.log(cost)
        g = lc[tr0].mean()
        sums, cnts = {}, {}
        for i in tr0:
            sums[repos[i]] = sums.get(repos[i], 0) + lc[i]
            cnts[repos[i]] = cnts.get(repos[i], 0) + 1
        prior = np.array([(sums.get(repos[i], 0) + 2 * g) / (cnts.get(repos[i], 0) + 2) - g for i in range(N)])
        base = np.minimum(cost[te0], B).sum()
        for k in KS:
            tr, te = tr0[ns[tr0] > k], te0[ns[te0] > k]
            for f in ["base", "behavior", "full"]:
                feat = lambda i: A.features(runs[i], k, None if f == "base" else beh[i])
                Xtr = np.array([feat(i) for i in tr])
                Xte = np.array([feat(i) for i in te])
                if f == "full":
                    Xtr, Xte = np.column_stack([Xtr, prior[tr]]), np.column_stack([Xte, prior[te]])
                w = A.fit_logistic(Xtr, over[tr].astype(float))
                res[(k, f)].append(auc(Xte @ w, over[te]))
            doomed = [i for i in te0 if over[i] and ns[i] > k]
            oracle[k].append(float(sum(min(cost[i], B) - min(cum[i][k], B) for i in doomed) / base))
    out[f"p{q}"] = dict(checkpoints=KS,
                        auc={f: [float(np.mean(res[(k, f)])) for k in KS] for f in ["base", "behavior", "full"]},
                        oracle_saving=[float(np.mean(oracle[k])) for k in KS])
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps(out, indent=1))
