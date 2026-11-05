"""
Ground truth for the simulation study: every policy's risk and outcomes on a
FRESH pool of N tasks per configuration (seeds disjoint from sim_v2.py's
calibration/evaluation pools). With N = 1,000,000 the standard error of a risk
near 10% is 0.03 percentage points.

Writes sums per policy (loss counts, completions, steps, capable-model steps,
spend) so means can be formed exactly.

Usage: python sim_truth.py OUT.json [N]
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

import sim_v2 as V

CHUNK = 20_000
SEED_OFFSET = 1000  # sim_v2 uses config index ci as the first seed word; stay disjoint


def chunk(args):
    ci, cfg, start, n = args
    scenario, mode, budget = cfg
    sums = np.zeros((len(V.POLICIES), 6))
    for i in range(n):
        seed = [SEED_OFFSET + ci, start + i]
        for p, pol in enumerate(V.POLICIES):
            sums[p] += V.run_policy(seed, pol, budget, scenario, mode)
    return sums


if __name__ == "__main__":
    out = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    import os
    result = json.load(open(out)) if os.path.exists(out) else {}  # resume: keep finished configs
    with Pool() as pool:
        for ci, cfg in enumerate(V.CONFIGS):
            if "{}_{}_{}".format(*cfg) in result:
                continue
            jobs = [(ci, cfg, s, min(CHUNK, N - s)) for s in range(0, N, CHUNK)]
            sums = sum(pool.imap_unordered(chunk, jobs))
            name = "{}_{}_{}".format(*cfg)
            result[name] = {pol: dict(overrun=sums[p, 0] / N, trunc=sums[p, 1] / N, done=sums[p, 2] / N,
                                      spent=sums[p, 3] / N, steps=sums[p, 4] / N, big_steps=sums[p, 5] / N)
                            for p, pol in enumerate(V.POLICIES)}
            result[name]["_n"] = N
            json.dump(result, open(out, "w"), indent=1)
            print(name, "done", flush=True)
