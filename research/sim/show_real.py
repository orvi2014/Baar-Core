import json, sys
d = json.load(open(sys.argv[1]))
f = lambda x: "  -  " if x is None else f"{100*x:5.1f}"
for s in d["studies"]:
    R = s["results"]
    print(f"\n{s['model']} p{s['budget_quantile']} B=${s['budget']:.5f} n_cal={s['n_cal']}  "
          f"uncon done {f(R['unconstrained']['done'])} q {R['unconstrained']['quality']:.3f}")
    for k in ["preflight", "estimate"]:
        r = R[k]; print(f"  {k:12s} over {f(r['overrun'])} done {f(r['done'])} q {r['quality']:.3f} util {f(r['util'])}")
    for e in d["eps"]:
        for k in [f"certest_{e}", f"ltt_{e}", f"oracle_{e}"]:
            r = R[k]; extra = ""
            if k.startswith("ltt"):
                extra = f" p95 {f(r['overrun_p95'])} test>eps {f(r['frac_test_over_eps'])} sig {f(r['frac_sig_violation'])} betas {r['beta_counts']}"
            print(f"  {k:12s} over {f(r['overrun'])} done {f(r['done'])} q {r['quality']:.3f} util {f(r['util'])}{extra}")
