import json, gzip, sys, numpy as np, collections
p = sys.argv[1]; op = gzip.open if p.endswith(".gz") else open
runs, steps = [], collections.defaultdict(list)
for l in op(p, "rt"):
    d = json.loads(l)
    if d.get("type") == "run": runs.append(d)
    else: steps[d["kind"]].append(d["completion_tokens"])
q = lambda a, ps: {f"p{x}": round(float(np.percentile(a, x)), 4) for x in ps}
S = np.array([r["steps"] for r in runs]); C = np.array([r["cost"] for r in runs])
print("runs", len(runs))
print("steps/run mean", round(S.mean(), 2), q(S, [50, 90]), "max", S.max())
print("cost/run mean", round(C.mean(), 4), q(C, [50, 90, 99]), "max", round(C.max(), 4), "max/median", round(C.max() / np.median(C), 2))
for k in ("tool", "final"):
    a = np.array(steps[k]); print("completion", k, "n", len(a), q(a, [50, 90, 99]), "max", a.max())
print("success rate", round(np.mean([r["f1"] for r in runs]), 4), "finished rate", round(np.mean([r["finished"] for r in runs]), 4),
      "last_turn_finish_call", round(np.mean([r["last_turn_is_finish_call"] for r in runs]), 4))
