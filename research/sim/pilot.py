"""
Real-model pilot: log per-step output lengths and per-run costs of a small
ReAct-style agent on HotpotQA (distractor setting), for two OpenAI models.

The experiment's total spend is hard-capped with Baar's own BudgetTracker in
worst-case (preflight-style) reservation mode, persisted to disk so a restart
cannot exceed the cap.

Usage: python pilot.py --small gpt-4o-mini --big gpt-4o --n-small 150 --n-big 60 --cap 4.50
"""
import argparse
import json
import os
import random
import re
import string
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import litellm

from baar.core.budget import BudgetExceeded, BudgetTracker
from baar.core.caps import model_max_output_tokens
from baar.core.stores import FileBudgetStore

MAX_STEPS = 8
SYSTEM = (
    "Answer the question. You may call the lookup tool with an exact title from the list "
    "to read that article's paragraph. Look up only what you need. When you know the "
    "answer, reply with 'FINAL ANSWER: <short answer>' and nothing after it.\n\nTitles:\n{titles}"
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Return the paragraph for an article title from the list.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
    },
}]
_write_lock = threading.Lock()


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def call(tracker, model, messages):
    """One LLM call under the global hard cap: reserve worst case, call, reconcile."""
    prompt_tokens = litellm.token_counter(model=model, messages=messages)
    worst = tracker.estimate_cost(model, prompt_tokens, model_max_output_tokens(model) or 16384)
    tracker.check_and_reserve(worst)          # raises BudgetExceeded: no request is sent
    try:
        resp = litellm.completion(model=model, messages=messages, tools=TOOLS, temperature=0.7)
    except Exception:
        tracker.cancel_reservation(worst)
        raise
    rec = tracker.record(resp, model)
    tracker.cancel_reservation(worst)
    return resp, rec, prompt_tokens


def run_task(tracker, model, task, run_id, log):
    paras = {title: " ".join(sents) for title, sents in task["context"]}
    messages = [
        {"role": "system", "content": SYSTEM.format(titles="\n".join(paras))},
        {"role": "user", "content": task["question"]},
    ]
    steps, answer, total = [], None, 0.0
    for step in range(1, MAX_STEPS + 1):
        resp, rec, est_prompt = call(tracker, model, messages)
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        kind = "tool" if tool_calls else "final"
        total += rec.cost
        steps.append(dict(run=run_id, model=model, step=step, kind=kind,
                          prompt_tokens=rec.prompt_tokens, est_prompt_tokens=est_prompt,
                          completion_tokens=rec.completion_tokens, cost=rec.cost,
                          finish=resp.choices[0].finish_reason))
        if not tool_calls:
            m = re.search(r"FINAL ANSWER:\s*(.*)", msg.content or "", re.S)
            answer = (m.group(1) if m else (msg.content or "")).strip()
            break
        messages.append(msg.model_dump(exclude_none=True))
        for tc in tool_calls:
            try:
                title = json.loads(tc.function.arguments).get("title", "")
            except Exception:
                title = ""
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": paras.get(title, f"No article titled '{title}'.")})
    result = dict(run=run_id, model=model, id=task["_id"], steps=len(steps), cost=total,
                  finished=answer is not None, f1=f1(answer, task["answer"]) if answer else 0.0)
    with _write_lock:
        for s in steps:
            log.write(json.dumps(s) + "\n")
        log.write(json.dumps(dict(result, type="run")) + "\n")
        log.flush()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="gpt-4o-mini")
    ap.add_argument("--big", default="gpt-4o")
    ap.add_argument("--n-small", type=int, default=150)
    ap.add_argument("--n-big", type=int, default=60)
    ap.add_argument("--cap", type=float, default=4.50)
    ap.add_argument("--data", default="hotpot_dev_distractor_v1.json")
    ap.add_argument("--out", default="pilot_log.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tracker = BudgetTracker(total_budget=args.cap,
                            store=FileBudgetStore(os.path.abspath("pilot_budget.json"), namespace="pilot"))
    tasks = json.load(open(args.data))
    random.Random(0).shuffle(tasks)
    jobs = [(args.small, t, f"s{i}") for i, t in enumerate(tasks[:args.n_small])]
    jobs += [(args.big, t, f"b{i}") for i, t in enumerate(tasks[:args.n_big])]  # same tasks: paired

    stopped = threading.Event()

    def work(job):
        if stopped.is_set():
            return None
        try:
            return run_task(tracker, *job, log)
        except BudgetExceeded:
            stopped.set()
            return None
        except Exception as e:  # keep going on transient API errors
            print(f"error {job[2]}: {type(e).__name__}: {e}", flush=True)
            return None

    with open(args.out, "a") as log, ThreadPoolExecutor(args.workers) as pool:
        results = [r for r in pool.map(work, jobs) if r]
    print(f"runs={len(results)} spent=${tracker.spent:.4f} of cap ${args.cap} "
          f"stopped_by_cap={stopped.is_set()}")
    for m in (args.small, args.big):
        rs = [r for r in results if r["model"] == m]
        if rs:
            print(f"{m}: runs={len(rs)} mean_cost=${sum(r['cost'] for r in rs)/len(rs):.5f} "
                  f"F1={sum(r['f1'] for r in rs)/len(rs):.3f} finished={sum(r['finished'] for r in rs)/len(rs):.2f}")


if __name__ == "__main__":
    main()
