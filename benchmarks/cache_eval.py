#!/usr/bin/env python3
"""
benchmarks/cache_eval.py — What does switching models cost an agent with a long context?

A prompt cache is per model: a conversation that moves from the big model to the
small one pays full price for its whole history again ("Don't Break the Cache",
arXiv 2601.06007). This replays multi-turn agent conversations (no API keys, no
network) with real gpt-5 / gpt-5-mini pricing from LiteLLM, and a simulated
provider that caches like OpenAI: the prompt prefix matching the previous request
to the same model is billed at the cache rate (once it reaches 1024 tokens, in
128-token blocks).

Workload: 24 conversations × 12 turns. Each has a system prompt + tool context of
2k, 20k or 80k tokens, and the history grows every turn. 30% of turns are hard
(routed to the big model), the rest easy.

Setups:
  always big          every turn on gpt-5
  routing             Baar routing: easy turns go to gpt-5-mini

Reports total cost, share of prompt tokens served from cache and model switches.

Why there's no "sticky" routing: staying on gpt-5 for an easy turn, because its
cached prefix makes that turn cheaper than an uncached switch, was tried and
cost *more* overall here ($2.34 vs $2.20 at seed 7, even with a 30% margin).
Both models keep their own cache and gpt-5-mini is ~5x cheaper on every line
(input, cached input, output), so rebuilding its cache once pays for itself
within a turn or two; a per-turn rule never sees that.

Run:
    python benchmarks/cache_eval.py            # table
    python benchmarks/cache_eval.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import litellm  # noqa: E402

from baar import BAARRouter, Policy, Rule  # noqa: E402

SEED = 7
CONVERSATIONS = 24
TURNS = 12
HARD_RATE = 0.3
CONTEXT_SIZES = (2_000, 20_000, 80_000)  # tokens of system prompt + tool context
BIG, SMALL = "gpt-5", "gpt-5-mini"
ANSWER_TOKENS = 300
WORDS = ("budget agent model token cache prompt route call tool step plan answer "
         "search file read write test check value cost limit window history").split()

HARD = Policy(rules=[Rule(when={"turn": "hard"}, then="force_big")])


def _text(rng: random.Random, tokens: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(tokens))


def _count(messages: list) -> int:
    return litellm.token_counter(model=BIG, messages=messages)


def make_conversations(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    convs = []
    for c in range(CONVERSATIONS):
        size = CONTEXT_SIZES[c % len(CONTEXT_SIZES)]
        turns = [{"text": f"conv {c} turn {t}: " + _text(rng, 30), "hard": rng.random() < HARD_RATE}
                 for t in range(TURNS)]
        convs.append({"system": _text(rng, size), "turns": turns, "context": size})
    return convs


class CachingProvider:
    """OpenAI-style automatic prefix caching, per model."""

    def __init__(self) -> None:
        self.last: dict[str, list] = {}
        self.prompt_tokens = 0
        self.cached_tokens = 0

    def completion(self, **kwargs):
        model, msgs = kwargs["model"], kwargs["messages"]
        prev = self.last.get(model, [])
        k = 0
        while k < min(len(prev), len(msgs)) and prev[k] == msgs[k]:
            k += 1
        prefix = _count(msgs[:k]) if k else 0
        cached = (prefix // 128) * 128 if prefix >= 1024 else 0
        prompt = _count(msgs)
        cached = min(cached, prompt)
        self.last[model] = list(msgs)
        self.prompt_tokens += prompt
        self.cached_tokens += cached
        return litellm.ModelResponse(
            model=model,
            choices=[{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "ok"}}],
            usage=litellm.Usage(prompt_tokens=prompt, completion_tokens=ANSWER_TOKENS,
                                total_tokens=prompt + ANSWER_TOKENS,
                                prompt_tokens_details={"cached_tokens": cached}),
        )


def run(convs: list[dict], setup: str) -> dict:
    provider = CachingProvider()
    small = BIG if setup == "always big" else SMALL
    router = BAARRouter(budget=1_000.0, small_model=small, big_model=BIG, use_llm_router=False,
                        policy=HARD)
    switches, prev_model = 0, None
    with patch("baar.router.litellm.completion", side_effect=provider.completion), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for conv in convs:
            history = [{"role": "system", "content": conv["system"]}]
            prev_model = None
            for turn in conv["turns"]:
                history.append({"role": "user", "content": turn["text"]})
                router.chat(turn["text"], messages=history,
                            context={"turn": "hard" if turn["hard"] else "easy"})
                model = router.log.steps[-1].model_used
                switches += prev_model is not None and model != prev_model
                prev_model = model
                history.append({"role": "assistant", "content": "answer " * ANSWER_TOKENS})
    steps = list(router.log.steps)
    return {
        "setup": setup,
        "cost_usd": round(router.spent, 4),
        "cache_hit_pct": round(provider.cached_tokens / provider.prompt_tokens * 100, 1),
        "switches": switches,
        "big_turns": sum(1 for s in steps if s.model_used == BIG),
    }


SETUPS = ("always big", "routing")


def evaluate(seed: int = SEED, conversations: Optional[list] = None) -> list[dict]:
    convs = conversations if conversations is not None else make_conversations(seed)
    return [run(convs, setup) for setup in SETUPS]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    rows = evaluate(args.seed)
    if args.json:
        print(json.dumps(rows))
        return
    print(f"Cache eval — {CONVERSATIONS} conversations × {TURNS} turns, {BIG} / {SMALL}, seed {args.seed}\n")
    print("| Setup | Cost | Prompt tokens from cache | Model switches | Turns on big |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['setup']} | ${r['cost_usd']:.4f} | {r['cache_hit_pct']}% | {r['switches']} | "
              f"{r['big_turns']} |")


if __name__ == "__main__":
    main()
