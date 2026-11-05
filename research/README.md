# Run-level budget guarantees for LLM agents

Code, data and paper for *Bounding the Probability That an LLM Agent Exceeds Its
Budget: Run-Level Risk Control with Learn-then-Test* (Robat Das).

| Path | What it is |
|---|---|
| `preprint/main.tex`, `refs.bib`, `main.pdf` | The paper |
| `sim/sim_thm1.py` | Synthetic agent simulator; estimate / preflight / per-step baselines |
| `sim/sim_ltt.py` | Learn-then-Test calibration over the policy family |
| `sim/make_tables.py` → `sim/tables.jsonl` | Tables 1–2 (simulation) |
| `sim/sim_ltt_repeat_par.py` → `sim/ltt_repeat.jsonl` | 500-calibration validity check |
| `sim/pilot.py` → `sim/pilot_log.jsonl` | Real-model study: ReAct agent on HotpotQA, gpt-4o-mini / gpt-4o |
| `sim/real_replay.py` → `sim/real_results.json` | Exact replay + Learn-then-Test on real runs (Table 3) |

Reproduce (Python 3.12, numpy, scipy):

```bash
cd research/sim
python make_tables.py tables.jsonl            # simulation tables
python sim_ltt_repeat_par.py 500              # validity check
python real_replay.py pilot_log.jsonl real_results.json   # real-trace analysis (no API calls)
```

Re-running `pilot.py` needs an OpenAI key and the HotpotQA distractor dev set
(`hotpot_dev_distractor_v1.json`); the logged runs cost $1.85.
