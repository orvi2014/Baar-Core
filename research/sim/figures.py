"""Figures for the preprint. Usage: python figures.py analysis_v2.json OUT_DIR"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: run-level risk vs per-call level beta
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, scen in zip(axes, ["valid", "shift"]):
    for c in d["configs"]:
        if c["scenario"] != scen or c["mode"] != "A":
            continue
        xs = [float(b) for b in c["risk_curve"]]
        ys = [100 * v for v in c["risk_curve"].values()]
        ax.plot(xs, ys, marker="o", ms=2.5, lw=1.2, label=f"B = ${c['budget']:.2f}")
    ax.plot([1e-3, 0.12], [0.1, 12], ls=":", c="grey", lw=0.8, label="run risk = per-call level")
    ax.set_xscale("log")
    ax.set_ylim(0, 12)
    ax.set_xlabel(r"per-call exceedance level $\beta$")
    ax.set_title("tables match deployment" if scen == "valid" else "tables fitted on easier tasks")
axes[0].set_ylabel("run overrun probability (%)")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(out, "risk_curve.pdf"))

# Fig 2: capability bought at eps = 0.05 across budgets
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, scen in zip(axes, ["valid", "shift"]):
    cs = sorted([c for c in d["configs"] if c["scenario"] == scen and c["mode"] == "A"], key=lambda c: c["budget"])
    B = [c["budget"] for c in cs]
    r = [c["runs"]["0.05"] for c in cs]
    series = {
        "preflight": [100 * c["preflight"]["big"] for c in cs],
        "per-step (union bound)": [100 * x["perstep"]["big"] for x in r],
        "certified estimate": [100 * x["n1000"]["certified_estimate"]["big"] for x in r],
        "run-level LTT (ours)": [100 * x["n1000"]["ltt"]["big"] for x in r],
        "oracle in family": [100 * x["oracle"]["big"] for x in r],
    }
    for name, ys in series.items():
        ax.plot(B, ys, marker="o", ms=3, lw=1.2, ls="--" if name.startswith("oracle") else "-", label=name)
    ax.set_xscale("log")
    ax.set_xticks(B)
    ax.set_xticklabels([f"${b:.2f}" for b in B])
    ax.set_xlabel("run budget B")
    ax.set_title("tables match deployment" if scen == "valid" else "tables fitted on easier tasks")
axes[0].set_ylabel("steps on capable model (%)")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(out, "capability.pdf"))
print("ok")
