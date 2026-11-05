# Per-step behavioural features of the sampled OpenHands trajectories, aligned with
# swe_log.jsonl.gz (runs numbered swe0.. in trajectory_id order, one step per assistant turn).
# Row k of a run = what the controller has seen after k assistant turns and their tool results
# (row 0 = before the first call). Run with: python -I behavior.py SAMPLE.parquet OUT.json.gz
import gzip
import json
import re
import sys

import pyarrow.parquet as pq

FEATS = ["bash", "edit", "view", "think", "test", "err", "repeat", "edited", "tested_after_edit", "err_last5"]
ERR = re.compile(r"Traceback|\bError\b|error:|exit code [1-9]|FAILED|command not found|No such file", re.I)
TEST = re.compile(r"pytest|unittest|\btox\b|python -m test|run_tests|\btest_\w+\.py")


def args_of(tc):
    try:
        return json.loads(((tc or {}).get("function") or {}).get("arguments") or "{}")
    except Exception:
        return {}


rows = pq.read_table(sys.argv[1]).to_pylist()
rows.sort(key=lambda r: r["trajectory_id"])
out = {}
for k, r in enumerate(rows):
    msgs = r["trajectory"]
    c = dict.fromkeys(FEATS, 0)
    seen_cmds, recent_err = set(), []
    table = [[0] * len(FEATS)]
    pending = False
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            if pending:
                table.append([c[f] for f in FEATS])
            pending = True
            for tc in (m.get("tool_calls") or []):
                name = ((tc or {}).get("function") or {}).get("name")
                a = args_of(tc)
                if name == "execute_bash":
                    c["bash"] += 1
                    cmd = str(a.get("command", ""))
                    if TEST.search(cmd):
                        c["test"] += 1
                        if c["edited"]:
                            c["tested_after_edit"] = 1
                    if cmd in seen_cmds:
                        c["repeat"] += 1
                    seen_cmds.add(cmd)
                elif name == "str_replace_editor":
                    sub = a.get("command")
                    if sub in ("create", "str_replace", "insert", "undo_edit"):
                        c["edit"] += 1
                        c["edited"] = 1
                    else:
                        c["view"] += 1
                elif name == "think":
                    c["think"] += 1
        elif m["role"] == "tool":
            e = 1 if ERR.search(str(m.get("content") or "")) else 0
            c["err"] += e
            recent_err = (recent_err + [e])[-5:]
            c["err_last5"] = sum(recent_err)
    table.append([c[f] for f in FEATS])  # after the last assistant turn
    out[f"swe{k}"] = dict(repo=r["repo"], rows=table)
with gzip.open(sys.argv[2], "wt") as f:
    json.dump(dict(features=FEATS, runs=out), f)
print("runs", len(out))
