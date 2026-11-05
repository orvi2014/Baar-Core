# Convert sampled OpenHands trajectories to the pilot_log JSONL format (gpt-4o re-pricing).
import pyarrow.parquet as pq, tiktoken, json, gzip, sys, os
SRC, OUT = sys.argv[1], sys.argv[2]
SRC_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
IN_P, OUT_P, OVH = 2.50e-6, 10.00e-6, 3
enc = tiktoken.get_encoding("o200k_base")
def ntok(s): return len(enc.encode_ordinary(s)) if s else 0
def msg_tokens(m):
    t = ntok(m.get("content"))
    for tc in (m.get("tool_calls") or []):
        t += ntok(((tc or {}).get("function") or {}).get("arguments"))
    return t
def is_finish(m):
    return any(((tc or {}).get("function") or {}).get("name") == "finish" for tc in (m.get("tool_calls") or []))
rows = pq.read_table(SRC).to_pylist()
rows.sort(key=lambda r: r["trajectory_id"])
lines = []; anomalies = {"no_assistant": 0, "finish_not_last": 0}
for k, r in enumerate(rows):
    msgs = r["trajectory"]; run = f"swe{k}"
    a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if not a_idx: anomalies["no_assistant"] += 1; continue
    last = a_idx[-1]; ctx = 0; step = 0; total = 0.0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            step += 1
            p = ctx; c = msg_tokens(m)
            fin = (i == last) or is_finish(m)
            if is_finish(m) and i != last: anomalies["finish_not_last"] += 1
            cost = p * IN_P + c * OUT_P; total += cost
            lines.append({"run": run, "model": "gpt-4o-priced", "src_model": SRC_MODEL, "step": step,
                          "kind": "final" if fin else "tool", "prompt_tokens": p, "est_prompt_tokens": p,
                          "completion_tokens": c, "cost": cost,
                          "finish": "tool_calls" if m.get("tool_calls") else "stop"})
        ctx += msg_tokens(m) + OVH
    lines.append({"run": run, "model": "gpt-4o-priced", "src_model": SRC_MODEL, "id": r["trajectory_id"],
                  "instance_id": r["instance_id"], "repo": r["repo"], "steps": step, "cost": total,
                  "finished": (r["exit_status"] == "submit"), "exit_status_is_submit": r["exit_status"] == "submit",
                  "last_turn_is_finish_call": is_finish(msgs[last]),
                  "f1": 1.0 if r["resolved"] == 1 else 0.0, "type": "run"})
data = "".join(json.dumps(l) + "\n" for l in lines).encode()
if len(data) > 20 * 2**20:
    with gzip.open(OUT + ".gz", "wb") as g: g.write(data); print("wrote", OUT + ".gz")
else:
    open(OUT, "wb").write(data); print("wrote", OUT)
print("bytes", len(data), "anomalies", anomalies)
