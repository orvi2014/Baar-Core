"""
tests/test_cache_aware.py — prompt-cache-aware recording and reporting.

  - cached prompt tokens are read from usage (OpenAI and Anthropic shapes),
    kept on each step, and priced at the cache rate when a stream is recorded
  - reservations still use the uncached price
  - RoutingLog reports cache hit rate and cached tokens
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from baar import BAARRouter, Policy, Rule
from baar.core.cache import cached_tokens

# gpt-5 / gpt-5-mini-like pricing: the big model's cache read is cheaper than the
# small model's uncached input.
PRICE = {
    "big": {"in": 1.25e-6, "cache": 1.25e-7, "out": 1e-5},
    "small": {"in": 2.5e-7, "cache": 2.5e-8, "out": 2e-6},
}
MAX_OUT = {"big": 128000, "small": 128000}
CONTEXT = {"big": 400000, "small": 400000}


def _cost_per_token(model, prompt_tokens=0, completion_tokens=0, cache_read_input_tokens=0, **_):
    p = PRICE[model]
    cached = cache_read_input_tokens or 0
    return (prompt_tokens - cached) * p["in"] + cached * p["cache"], completion_tokens * p["out"]


def cost(model, prompt, completion, cached=0):
    i, o = _cost_per_token(model, prompt, completion, cached)
    return i + o


def _completion_cost(completion_response=None, **_):
    u = completion_response.usage
    return cost(completion_response.model, u.prompt_tokens, u.completion_tokens, cached_tokens(u))


def _token_counter(text="", model="", messages=None):
    """1 character = 1 token, so prompts can be sized exactly."""
    if messages is None:
        return len(text)
    return sum(len(m["content"]) if isinstance(m.get("content"), str) else 0 for m in messages)


@pytest.fixture
def priced():
    with patch("baar.core.budget.cost_per_token", side_effect=_cost_per_token), \
         patch("baar.core.budget.completion_cost", side_effect=_completion_cost), \
         patch("baar.router.token_counter", side_effect=_token_counter), \
         patch("baar.core.caps.model_max_output_tokens", side_effect=MAX_OUT.get), \
         patch("baar.core.caps.model_context_window", side_effect=CONTEXT.get), \
         patch("baar.core.caps.supports_stream_usage", return_value=True):
        yield


class CachingProvider:
    """
    Automatic prefix caching, OpenAI-style: the part of the prompt that matches
    the previous request to the same model is billed as cached.
    """

    def __init__(self, completion_tokens=500):
        self.completion_tokens = completion_tokens
        self.last = {}
        self.calls = []

    def _usage(self, kw):
        msgs = kw["messages"]
        prev = self.last.get(kw["model"], [])
        k = 0
        while k < min(len(prev), len(msgs)) and prev[k] == msgs[k]:
            k += 1
        cached = _token_counter(messages=msgs[:k])
        self.last[kw["model"]] = list(msgs)
        prompt = _token_counter(messages=msgs)
        return prompt, cached

    def completion(self, **kw):
        self.calls.append(kw["model"])
        prompt, cached = self._usage(kw)
        if kw.get("stream"):
            return iter([SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=self.completion_tokens,
                                      prompt_tokens_details=SimpleNamespace(cached_tokens=cached)))])
        r = MagicMock()
        r.model = kw["model"]
        r.choices[0].message.content = "ok"
        r.choices[0].finish_reason = "stop"
        r.usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=self.completion_tokens,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
        return r


HARD = Policy(rules=[Rule(when={"turn": "hard"}, then="force_big")])


def router(**kw):
    kw.setdefault("use_llm_router", False)
    kw.setdefault("small_model", "small")
    kw.setdefault("big_model", "big")
    kw.setdefault("policy", HARD)
    return BAARRouter(**kw)


def conversation(system_chars, turns):
    """A system prompt plus alternating user/assistant history."""
    msgs = [{"role": "system", "content": "s" * system_chars}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


def next_turn(history, text):
    return history + [{"role": "user", "content": text}]


# ── usage parsing ─────────────────────────────────────────────────────────────

class TestCachedTokens:
    def test_openai_shape(self):
        assert cached_tokens(SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=1500))) == 1500

    def test_dict_details(self):
        assert cached_tokens(SimpleNamespace(prompt_tokens_details={"cached_tokens": 7})) == 7

    def test_anthropic_shape(self):
        assert cached_tokens(SimpleNamespace(prompt_tokens_details=None, cache_read_input_tokens=900)) == 900

    def test_missing_or_mock(self):
        assert cached_tokens(None) == 0
        assert cached_tokens(SimpleNamespace()) == 0
        assert cached_tokens(MagicMock()) == 0


# ── recording ─────────────────────────────────────────────────────────────────

class TestRecording:
    def test_step_keeps_cached_tokens(self, priced):
        p = CachingProvider()
        r = router(budget=10.0)
        history = conversation(5000, 1)
        with patch("baar.router.litellm.completion", side_effect=p.completion):
            r.chat("q1", messages=next_turn(history, "q1"))
            r.chat("q2", messages=next_turn(history, "q2"))
        assert r.log.steps[0].cached_prompt_tokens == 0
        assert r.log.steps[1].cached_prompt_tokens == _token_counter(messages=history)
        assert r.log.steps[1].to_dict()["cached_prompt_tokens"] > 0

    def test_stream_is_recorded_at_the_cache_rate(self, priced):
        p = CachingProvider()
        r = router(budget=10.0)
        history = conversation(5000, 1)
        with patch("baar.router.litellm.completion", side_effect=p.completion):
            list(r.stream_chat("q1", messages=next_turn(history, "q1")))
            before = r.spent
            list(r.stream_chat("q2", messages=next_turn(history, "q2")))
        step = r.log.steps[-1]
        assert step.cached_prompt_tokens > 0
        assert r.spent - before == pytest.approx(
            cost("small", step.prompt_tokens, step.completion_tokens, step.cached_prompt_tokens))


# ── metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_summary_and_report(self, priced, capsys):
        p = CachingProvider()
        r = router(budget=10.0)
        history = conversation(5000, 1)
        with patch("baar.router.litellm.completion", side_effect=p.completion):
            r.chat("q1", messages=next_turn(history, "q1"))
            r.chat("q2", messages=next_turn(history, "q2"))
        s = r.log.summary()
        assert s["cached_prompt_tokens"] == sum(st.cached_prompt_tokens for st in r.log.steps) > 0
        assert 0 < s["cache_hit_rate_pct"] < 100
        r.log.print_report()
        assert "prompt cache" in capsys.readouterr().out

    def test_empty_log(self):
        s = router().log.summary()
        assert (s["cached_prompt_tokens"], s["cache_hit_rate_pct"]) == (0, 0.0)


def test_reservation_uses_the_uncached_price(priced):
    p = CachingProvider()
    r = router(budget=10.0, prompt_margin=0.0)
    held = []

    def _call(**kw):
        held.append(r.spent)
        return p.completion(**kw)

    history = conversation(5000, 1)
    second = next_turn(history, "q2")
    with patch("baar.router.litellm.completion", side_effect=_call):
        r.chat("q1", messages=next_turn(history, "q1"))
        before = r.spent
        r.chat("q2", messages=second)
    assert r.log.steps[-1].cached_prompt_tokens > 0
    eot = r._router.decide(task="q2", budget_utilization=0.0).estimated_output_tokens
    assert held[1] - before == pytest.approx(cost("small", _token_counter(messages=second), eot))
