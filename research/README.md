# Run-level budget risk for LLM agents

Code, data and paper for *How Often Will My Agent Overspend? Certifying
Run-Level Budget Risk for LLM Agents* (Robat Das). The method is the
`calibrated` output mode of Baar-Core (v0.11.0+).

Every table in the paper is generated from the analysis outputs by
`sim/make_tex.py`; no number is copied by hand.

## Layout

| Path | What it is |
|---|---|
| `preprint/main.tex`, `refs.bib`, `main.pdf` | The paper |
| `preprint/tables/*.tex`, `preprint/figs/*.pdf` | Generated tables and figures |
| `sim/sim_thm1.py` | Simulator (two models, growing context, stylised controller) and the per-step baseline |
| `sim/sim_v2.py` | 200,000-task pools per setting, every policy run on every task |
| `sim/sim_truth.py` → `sim/truth.json` | Ground truth: every policy's risk on 400k–1M fresh tasks per setting |
| `sim/analysis_v2.py` → `sim/analysis_v3.json` | 2,000 calibrations per setting, scored against the ground truth (Tables 1–2, Figures 1–2) |
| `sim/pilot.py` → `sim/pilot_log.jsonl` | Real HotpotQA runs, gpt-4o-mini and gpt-4o ($1.85) |
| `sim/real_v2.py` → `sim/real_v3.json` | Exact replay of Baar-Core's admission rule, 200 held-out splits, budgets from the table split, same prompt margin for every policy (Table 3) |
| `swe/` | 3,000 OpenHands trajectories on SWE-rebench, converted to the same log format (see `swe/README.md`) |
| `sim/real_v2.py --prompt-add 3121` → `sim/swe_v3.json` | The same analysis on the coding-agent runs, with tool-schema tokens added (Table 4) |
| `sim/swe_waste.py` → `sim/swe_waste.json` | Runs stopped mid-task and the spend they absorb (Section 8) |
| `swe/scripts/behavior.py` → `swe/behavior.json.gz` | Per-step agent behaviour (commands, edits, tests, tool errors, repeats) for the coding-agent runs |
| `sim/admission.py` → `sim/admission_{base,behavior,full}.json` | Certified early stopping, 200 held-out splits, three feature sets (Tables 5–6) |
| `sim/admission_auc.py` → `sim/admission_auc.json` | Predictability by checkpoint and the oracle saving ceiling (Table 6) |
| `sim/analysis_v2.json`, `sim/real_v2.json`, `sim/swe_v2.json` | Earlier outputs superseded after the second review, kept for the record |
| `sim/sim_ltt.py`, `sim_ltt_repeat_par.py`, `make_tables.py`, `ltt_repeat.jsonl`, `tables.jsonl`, `real_replay.py`, `real_results.json` | First-version analyses, kept for the record |

## Reproduce

Python 3.12 with numpy, scipy, matplotlib and `baar-core>=0.11.0`.

```bash
cd research/sim
python sim_v2.py pool 200000                       # ~45M simulated runs
python sim_truth.py truth.json 400000              # ground truth (one setting in the paper used 1,000,000)
python analysis_v2.py pool analysis_v3.json truth.json   # 320,000 calibrations
python real_v2.py pilot_log.jsonl real_v3.json
python real_v2.py ../swe/swe_log.jsonl.gz swe_v3.json --prompt-add 3121
python swe_waste.py ../swe/swe_log.jsonl.gz swe_waste.json --prompt-add 3121
python figures.py analysis_v3.json ../preprint/figs
python make_tex.py analysis_v3.json real_v3.json swe_v3.json ../preprint/tables
python -I ../swe/scripts/behavior.py ../swe/raw/sample_3000_seed0.parquet ../swe/behavior.json.gz
for f in base behavior full; do
  python admission.py ../swe/swe_log.jsonl.gz admission_$f.json --prompt-add 3121 --behavior ../swe/behavior.json.gz --features $f
done
python admission_auc.py ../swe/swe_log.jsonl.gz ../swe/behavior.json.gz admission_auc.json --prompt-add 3121
python make_tex_es.py admission_base.json admission_behavior.json admission_full.json admission_auc.json ../preprint/tables
```

All seeds are fixed. The simulation pools (`pool/*.npz`, about 70 MB) and the
sampled raw trajectories (`swe/raw/sample_3000_seed0.parquet`, 103 MB) are not
committed; `sim_v2.py` and `swe/scripts/fetch.py` rebuild them exactly.
Re-running `pilot.py` needs an OpenAI key and the HotpotQA distractor dev set.
