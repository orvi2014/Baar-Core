"""
Write every results table of the preprint as LaTeX, straight from the JSON
outputs, so no number in the paper is copied by hand.

Usage: python make_tex.py analysis_v2.json real_v2.json swe_v2.json OUT_DIR
"""
import json
import os
import sys

sim, hot, swe = (json.load(open(p)) for p in sys.argv[1:4])
out = sys.argv[4]
os.makedirs(out, exist_ok=True)
pc = lambda x: "--" if x is None else f"{100 * x:.1f}"
YMAX_BIG_COST = 8192 * 10e-6  # simulator: one maximum-length capable-model output


def write(name, body):
    open(os.path.join(out, name), "w").write(body)


# ── Table: simulation across budgets (eps = 0.05, n = 1000) ──────────────────
def sim_budget_table(scenario):
    rows = []
    for c in sorted([c for c in sim["configs"] if c["scenario"] == scenario and c["mode"] == "A"],
                    key=lambda c: c["budget"]):
        r = c["runs"]["0.05"]
        L, C, O, P = r["n1000"]["ltt"], r["n1000"]["certified_estimate"], r["oracle"], r["perstep"]
        rows.append(
            f"\\${c['budget']:.2f} & {YMAX_BIG_COST / c['budget']:.1f} & {pc(c['preflight']['big'])} & "
            f"{pc(P['big'])} & {pc(c['estimate']['risk'])} / {pc(c['estimate']['big'])} & "
            f"{pc(C['big'])} & {pc(L['risk_mean'])} / {pc(L['big'])} & {pc(L['violation'])} & {pc(O['big'])} \\\\")
    return "\n".join(rows)


for scen in ["valid", "shift"]:
    write(f"sim_budget_{scen}.tex", sim_budget_table(scen))

# ── Table: validity over every setting, by calibration size ──────────────────
REPS = sim["reps"]
rows = []
for n in sim["ns"]:
    L = [r[f"n{n}"]["ltt"] for c in sim["configs"] for r in c["runs"].values()]
    pooled = sum(x["violation"] for x in L) / len(L)
    worst = max(L, key=lambda x: x["violation"])
    rows.append(f"{n:,} & {len(L)} & {pc(pooled)} & {pc(worst['violation'])} & "
                f"{pc(max(x['violation_clear'] for x in L))} & {pc(max(x['violation_calpop'] for x in L))} & "
                f"{pc(sum(x['fallback'] for x in L) / len(L))} \\\\".replace(",", "{,}"))
write("sim_validity.tex", "\n".join(rows))

# ── Table: HotpotQA held-out ─────────────────────────────────────────────────
rows = []
for s in hot["studies"]:
    if s["budget_quantile"] not in (50, 75, 90):
        continue
    R = s["results"]
    E = R["estimate"]
    first = True
    for eps in hot["eps"]:
        L, C, O = R[f"ltt_{eps}"], R[f"certest_{eps}"], R[f"oracle_{eps}"]
        lead = (f"\\multirow{{3}}{{*}}{{{s['model'].replace('gpt-', '')}}} & \\multirow{{3}}{{*}}{{p{s['budget_quantile']}}} & "
                f"\\multirow{{3}}{{*}}{{{pc(E['overrun'])} / {pc(E['done'])}}}") if first else " & & "
        first = False
        rows.append(f"{lead} & {eps:g} & {pc(C['done'])} & {pc(L['overrun'])} & {pc(L['overrun_p95'])} & "
                    f"{pc(L['done'])} & {pc(L['frac_sig_violation'])} & {pc(O['done'])} \\\\")
    rows.append("\\addlinespace")
write("hotpot.tex", "\n".join(rows[:-1]))

# ── Table: SWE (long-horizon) ────────────────────────────────────────────────
rows = []
for s in swe["studies"]:
    R = s["results"]
    L = R["ltt_0.05"]
    rows.append(f"p{s['budget_quantile']} & \\${s['budget']:.2f} & {16384 * 10e-6 / s['budget']:.3f} & "
                f"{pc(R['preflight']['overrun'])} / {pc(R['preflight']['done'])} & "
                f"{pc(R['estimate']['overrun'])} / {pc(R['estimate']['done'])} & "
                f"{pc(L['overrun'])} / {pc(L['done'])} & {pc(R['unconstrained']['done'])} \\\\")
write("swe.tex", "\n".join(rows))
print("tables written to", out)
