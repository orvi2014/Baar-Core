#!/usr/bin/env python3
"""
benchmarks/overthink_eval.py — Does the budget hold when reasoning is inflated?

OverThink (arXiv 2502.02542) hides decoy puzzles in content a reasoning model
reads, multiplying its hidden reasoning tokens (up to 46× in the paper) while
the visible answer stays the same. Reasoning tokens are billed as output, so
every cost estimate made before the call is wrong by that factor.

This replays a deterministic agent run (no API keys, no network) against a
simulated Claude-class reasoning model and reports, per setup:

  done        tasks answered before the run stopped
  cut off     answers stopped by a limit below the model's own max output
              (Baar's budget-derived max_tokens, or your max_output_tokens)
  model max   answers that ran into the model's own max output (64k) — a
              poisoned task can want more than the model can produce
  overshoot   spend above the budget, in USD (what a hard cap must keep at 0)
  used        share of the budget actually spent

Workload: 150 tasks, 30% poisoned (reasoning × 10–46), $5.00 budget. The model
is registered with LiteLLM, so Baar's real pricing, limits and thinking
detection run unmodified; only the provider call is simulated. The simulated
provider honours `max_tokens` (output stops there) and a thinking budget
(reasoning stops there), as real APIs do.

Setups:
  alert only          no pre-call check; a spend alert stops the run after the
                      call that crosses the budget (post-hoc tools)
  estimate            Baar default: reserve the expected output
  preflight           reserve the model's full max output (64k tokens)
  preflight+thinking  preflight with thinking_budget=2048, max_output_tokens=2000
  preflight+learned   preflight with overrun_risk=0.05 (learned output lengths)
  truncate            max_tokens from the remaining budget (opt-in cut-offs)

Run:
    python benchmarks/overthink_eval.py            # table
    python benchmarks/overthink_eval.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import litellm  # noqa: E402

from baar import BAARRouter, BudgetExhausted  # noqa: E402

SEED = 7
N_TASKS = 150
POISON_RATE = 0.3
BUDGET = 5.00
MODEL = "sim-reasoner"
IN_PRICE, OUT_PRICE = 3e-6, 1.5e-5  # Claude Sonnet-class, per token
MAX_OUTPUT, CONTEXT = 64_000, 200_000

litellm.register_model({MODEL: {
    "input_cost_per_token": IN_PRICE,
    "output_cost_per_token": OUT_PRICE,
    "max_output_tokens": MAX_OUTPUT,
    "max_input_tokens": CONTEXT,
    "max_tokens": MAX_OUTPUT,
    "litellm_provider": "anthropic",
    "mode": "chat",
    "supports_reasoning": True,
}})


@dataclass(frozen=True)
class Task:
    index: int
    reasoning: int  # hidden reasoning tokens the model wants to spend
    answer: int     # visible answer tokens
    poisoned: bool


def make_tasks(seed: int = SEED) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(N_TASKS):
        reasoning = int(rng.lognormvariate(math.log(800), 0.5))
        answer = int(rng.lognormvariate(math.log(300), 0.4))
        poisoned = rng.random() < POISON_RATE
        if poisoned:
            reasoning = int(reasoning * rng.uniform(10, 46))
        tasks.append(Task(i, reasoning, answer, poisoned))
    return tasks


PROMPT = "task {index}: summarize the attached document"


def prompt_tokens(index: int) -> int:
    """What the provider bills for the prompt (same tokenizer Baar estimates with)."""
    return litellm.token_counter(model=MODEL, messages=[{"role": "user", "content": PROMPT.format(index=index)}])


class SimProvider:
    """A provider that honours max_tokens and a thinking budget."""

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = tasks
        self.cut_off = 0
        self.model_max = 0

    def completion(self, **kwargs):
        index = int(kwargs["messages"][-1]["content"].split()[1].rstrip(":"))
        task = self.tasks[index]
        reasoning = task.reasoning
        thinking = kwargs.get("thinking")
        if thinking:
            reasoning = min(reasoning, thinking["budget_tokens"])
        total = reasoning + task.answer
        finish = "stop"
        limit = min(kwargs.get("max_tokens") or MAX_OUTPUT, MAX_OUTPUT)
        if total > limit:
            total, finish = limit, "length"
            if limit < MAX_OUTPUT:
                self.cut_off += 1
            else:
                self.model_max += 1
        return litellm.ModelResponse(
            model=MODEL,
            choices=[{"index": 0, "finish_reason": finish,
                      "message": {"role": "assistant", "content": "answer"}}],
            usage=litellm.Usage(prompt_tokens=prompt_tokens(index), completion_tokens=total,
                                total_tokens=prompt_tokens(index) + total),
        )


def call_cost(task: Task) -> float:
    return prompt_tokens(task.index) * IN_PRICE + min(task.reasoning + task.answer, MAX_OUTPUT) * OUT_PRICE


def run_alert_only(tasks: list[Task]) -> dict:
    spent, done, model_max = 0.0, 0, 0
    for task in tasks:
        spent += call_cost(task)
        model_max += task.reasoning + task.answer > MAX_OUTPUT
        done += 1
        if spent >= BUDGET:
            break
    return {"done": done, "cut_off": 0, "model_max": model_max, "spent": spent}


def run_baar(tasks: list[Task], **router_kwargs) -> dict:
    provider = SimProvider(tasks)
    router = BAARRouter(budget=BUDGET, small_model=MODEL, big_model=MODEL,
                        use_llm_router=False, min_cost_threshold=0.0, **router_kwargs)
    done = 0
    with patch("baar.router.litellm.completion", side_effect=provider.completion), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for task in tasks:
            try:
                router.chat(PROMPT.format(index=task.index))
            except BudgetExhausted:
                break
            done += 1
    return {"done": done, "cut_off": provider.cut_off, "model_max": provider.model_max, "spent": router.spent}


SETUPS = {
    "alert only": None,
    "estimate": {},
    "preflight": {"on_output_overrun": "preflight"},
    "preflight+thinking": {"on_output_overrun": "preflight", "thinking_budget": 2048, "max_output_tokens": 2000},
    "preflight+learned": {"on_output_overrun": "preflight", "overrun_risk": 0.05},
    "truncate": {"on_output_overrun": "truncate"},
}


def evaluate(seed: int = SEED) -> list[dict]:
    tasks = make_tasks(seed)
    rows = []
    for name, kwargs in SETUPS.items():
        result = run_alert_only(tasks) if kwargs is None else run_baar(tasks, **kwargs)
        rows.append({
            "setup": name,
            "done": result["done"],
            "cut_off": result["cut_off"],
            "model_max": result["model_max"],
            "overshoot_usd": round(max(0.0, result["spent"] - BUDGET), 4),
            "used_pct": round(min(result["spent"], BUDGET) / BUDGET * 100, 1),
            "spent_usd": round(result["spent"], 4),
        })
    return rows


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    rows = evaluate(args.seed)
    if args.json:
        print(json.dumps(rows))
        return
    tasks = make_tasks(args.seed)
    poisoned = sum(t.poisoned for t in tasks)
    print(f"OverThink eval — {N_TASKS} tasks ({poisoned} poisoned), ${BUDGET:.2f} budget, seed {args.seed}\n")
    print("| Setup | Done | Cut off | Model max | Overshoot | Budget used |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['setup']} | {r['done']} | {r['cut_off']} | {r['model_max']} | "
              f"${r['overshoot_usd']:.4f} | {r['used_pct']}% |")


if __name__ == "__main__":
    main()
