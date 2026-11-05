"""Early-stopping tables (Section 9) from admission_*.json and admission_auc.json.
Usage: python make_tex_es.py admission_base.json admission_behavior.json admission_full.json admission_auc.json OUT_DIR"""
import json
import os
import sys

base, beh, full, aucj = (json.load(open(p)) for p in sys.argv[1:5])
out = sys.argv[5]
pc = lambda x: f"{100 * x:.1f}"

# Table: certified early stopping with behaviour features
rows = []
for s in beh["studies"]:
    R = s["results"]
    first = True
    for loss, hk in [("resolve", "lost_resolve"), ("finish", "false_kill")]:
        for e in beh["eps"]:
            r = R[f"{loss}_{e}"]
            lead = (f"\\multirow{{6}}{{*}}{{p{s['budget_quantile']}}}" if first else "")
            first = False
            rows.append(f"{lead} & {'solving' if loss == 'resolve' else 'finishing'} & {e:g} & "
                        f"{pc(r[hk])} & {pc(r['harm_p95'])} & {pc(r['sig_violation'])} & "
                        f"{pc(r['saved'])} & {pc(r['doomed_stopped'])} \\\\")
    n = R["naive_0.5"]
    rows.append(f" & \\multicolumn{{2}}{{l}}{{uncertified $\\tau = 0.5$}} & {pc(n['false_kill'])} / {pc(n['lost_resolve'])} & & & "
                f"{pc(n['saved'])} & {pc(n['doomed_stopped'])} \\\\")
    rows.append("\\addlinespace")
open(os.path.join(out, "early_stop.tex"), "w").write("\n".join(rows[:-1]))

# Table: predictability and the oracle ceiling, by checkpoint (p75 budget), plus feature sets' savings
v = aucj["p75"]
rows = []
for i, k in enumerate(v["checkpoints"]):
    rows.append(f"{k} & {v['auc']['base'][i]:.2f} & {v['auc']['behavior'][i]:.2f} & {v['auc']['full'][i]:.2f} & "
                f"{pc(v['oracle_saving'][i])} \\\\")
open(os.path.join(out, "early_auc.tex"), "w").write("\n".join(rows))

rows = []
for q_i, q in enumerate([50, 75, 90]):
    cells = []
    for d in (base, beh, full):
        r = d["studies"][q_i]["results"]["resolve_0.05"]
        cells.append(f"{pc(r['saved'])}")
    rows.append(f"p{q} & " + " & ".join(cells) + " \\\\")
open(os.path.join(out, "early_features.tex"), "w").write("\n".join(rows))
print("ok")
