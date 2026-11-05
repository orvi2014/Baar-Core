"""Mid-run refusals on the coding-agent runs (Section 8): share of runs cut off,
share of all spend that went into them, and the share of runs whose unconstrained
cost exceeds B (which no reservation policy can finish).
Usage: python swe_waste.py ../swe/swe_log.jsonl.gz OUT.json [--prompt-add 3121]"""
import json
import sys

import numpy as np

import real_v2 as R

add = int(sys.argv[sys.argv.index("--prompt-add") + 1]) if "--prompt-add" in sys.argv else 0
runs = R.load(sys.argv[1], add)
costs = np.array([r["cost"] for r in runs])
idx = np.arange(len(runs))
rep = R.Replayer(runs, 0.2, False)
mean_u = float(np.mean([s["completion_tokens"] for r in runs for s in r["steps"]]))
out = {"prompt_add": add, "input_share": float(sum(s["prompt_tokens"] * R.PRICE[r["model"]][0]
                                                    for r in runs for s in r["steps"]) / costs.sum())}
for q in [50, 75, 90]:
    B = float(np.percentile(costs, q))
    row = {"budget": B, "kappa": 16384 * 10e-6 / B, "cannot_finish": float(np.mean(costs > B))}
    for name, u in [("preflight", R.YMAX), ("estimate", mean_u)]:
        o = rep.outcomes(u, idx, B)
        cut = o[:, 3] < costs - 1e-12
        row[name] = dict(cut=float(cut.mean()), spend_on_cut=float(o[cut, 3].sum() / o[:, 3].sum()),
                         overrun=float(o[:, 0].mean()))
    out[f"p{q}"] = row
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
