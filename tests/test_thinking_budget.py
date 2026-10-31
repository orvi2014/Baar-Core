"""
tests/test_thinking_budget.py — thinking_budget: cap hidden reasoning, never the answer.

Providers expose reasoning control two ways (via LiteLLM):
  - a token budget (Anthropic, Gemini, DeepSeek): thinking={"type": "enabled",
    "budget_tokens": N}. max_tokens covers thinking + answer, so Baar sends
    answer limit + N and the visible answer keeps its full limit.
  - an effort level (OpenAI o-series, GPT-5): reasoning_effort low/medium/high.
    Advisory only — the output limit still counts reasoning — so the
    worst-case reservation is unchanged.
Runs without API keys.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baar import BAARRouter, BudgetExceeded, BudgetExhausted, OutputLimitUnavailable
from baar.core import caps
from baar.core.budget import BudgetTracker
from baar.core.caps import plan_output

IN_PRICE = {"gpt-4o-mini": 1.5e-7, "claude-think": 3e-6, "o-reasoner": 2e-6}
OUT_PRICE = {"gpt-4o-mini": 6e-7, "claude-think": 1.5e-5, "o-reasoner": 8e-6}
MAX_OUT = {"gpt-4o-mini": 16384, "claude-think": 64000, "o-reasoner": 100000}
CONTEXT = {"gpt-4o-mini": 128000, "claude-think": 200000, "o-reasoner": 200000}
STYLE = {"claude-think": "budget", "o-reasoner": "effort"}
PROMPT = 100


def _cost_per_token(model, prompt_tokens=0, completion_tokens=0, **_):
    return prompt_tokens * IN_PRICE[model], completion_tokens * OUT_PRICE[model]


def cost(model, completion_tokens, prompt_tokens=PROMPT):
    i, o = _cost_per_token(model, prompt_tokens, completion_tokens)
    return i + o


def _completion_cost(completion_response=None, **_):
    u = completion_response.usage
    return cost(completion_response.model, u.completion_tokens, u.prompt_tokens)


@pytest.fixture
def priced():
    with patch("baar.core.budget.cost_per_token", side_effect=_cost_per_token), \
         patch("baar.core.budget.completion_cost", side_effect=_completion_cost), \
         patch("baar.router.token_counter", return_value=PROMPT), \
         patch("baar.core.caps.model_max_output_tokens", side_effect=MAX_OUT.get), \
         patch("baar.core.caps.model_context_window", side_effect=CONTEXT.get), \
         patch("baar.core.caps.supports_stream_usage", return_value=True), \
         patch("baar.core.caps.thinking_style", side_effect=STYLE.get):
        yield


def response(model, completion_tokens=30):
    r = MagicMock()
    r.model = model
    r.choices[0].message.content = "ok"
    r.choices[0].finish_reason = "stop"
    r.usage.prompt_tokens = PROMPT
    r.usage.completion_tokens = completion_tokens
    return r


def echo(**kwargs):
    return response(kwargs["model"])


def plan(mode, model, **kw):
    kw.setdefault("remaining", 100.0)
    return plan_output(mode, model=model, prompt_tokens=PROMPT, expected_output_tokens=500,
                       tracker=BudgetTracker(total_budget=1000.0), **kw)


def router(**kw):
    kw.setdefault("use_llm_router", False)
    return BAARRouter(**kw)


# ── which control a model has ─────────────────────────────────────────────────

class TestThinkingStyle:
    def _style(self, params, reasoning=True):
        with patch("baar.core.caps.litellm.get_supported_openai_params", return_value=params), \
             patch("baar.core.caps.litellm.supports_reasoning", return_value=reasoning):
            return caps.thinking_style("m")

    def test_token_budget_when_thinking_is_supported(self):
        assert self._style(["max_tokens", "thinking", "reasoning_effort"]) == "budget"

    def test_effort_when_only_reasoning_effort(self):
        assert self._style(["max_tokens", "reasoning_effort"]) == "effort"

    def test_none_for_non_reasoning_models(self):
        assert self._style(["max_tokens"], reasoning=False) is None
        assert self._style(["max_tokens", "reasoning_effort"], reasoning=False) is None

    def test_none_when_litellm_errors(self):
        with patch("baar.core.caps.litellm.get_supported_openai_params", side_effect=Exception("x")):
            assert caps.thinking_style("mystery") is None

    @pytest.mark.parametrize("budget,effort", [(1024, "low"), (4095, "low"), (4096, "medium"),
                                               (16383, "medium"), (16384, "high"), (60000, "high")])
    def test_effort_mapping(self, budget, effort):
        assert caps.reasoning_effort_for(budget) == effort


# ── planning ──────────────────────────────────────────────────────────────────

class TestPlan:
    def test_budget_style_adds_thinking_to_the_answer_limit(self, priced):
        p = plan("preflight", "claude-think", max_output_tokens=2000, thinking_budget=4000)
        assert p.max_tokens == 6000
        assert p.completion_tokens == 6000
        assert p.params == {"thinking": {"type": "enabled", "budget_tokens": 4000}}

    def test_budget_style_without_user_limit_fills_model_max(self, priced):
        p = plan("preflight", "claude-think", thinking_budget=4000)
        assert p.max_tokens == MAX_OUT["claude-think"]

    def test_budget_larger_than_model_max_raises(self, priced):
        with pytest.raises(OutputLimitUnavailable, match="thinking_budget"):
            plan("preflight", "claude-think", thinking_budget=64000)

    def test_answer_limit_still_clamped_to_model_max(self, priced):
        p = plan("preflight", "claude-think", max_output_tokens=1_000_000, thinking_budget=4000)
        assert p.max_tokens == MAX_OUT["claude-think"]

    def test_effort_style_sends_effort_and_keeps_worst_case(self, priced):
        p = plan("preflight", "o-reasoner", thinking_budget=2000)
        assert p.params == {"reasoning_effort": "low"}
        assert p.max_tokens == MAX_OUT["o-reasoner"]
        assert p.completion_tokens == MAX_OUT["o-reasoner"]

    def test_non_reasoning_model_unchanged(self, priced):
        with_budget = plan("preflight", "gpt-4o-mini", thinking_budget=4000)
        without = plan("preflight", "gpt-4o-mini")
        assert with_budget == without
        assert with_budget.params == {}

    def test_no_budget_means_no_params(self, priced):
        assert plan("preflight", "claude-think").params == {}

    def test_estimate_mode_budget_style(self, priced):
        p = plan("estimate", "claude-think", thinking_budget=4000)
        # Anthropic needs max_tokens > budget_tokens, so it is always sent.
        assert p.max_tokens == MAX_OUT["claude-think"]
        assert p.completion_tokens == 500 + 4000
        assert p.params["thinking"]["budget_tokens"] == 4000

    def test_estimate_mode_budget_style_with_user_limit(self, priced):
        p = plan("estimate", "claude-think", max_output_tokens=300, thinking_budget=4000)
        assert p.max_tokens == 4300
        assert p.completion_tokens == 300 + 4000

    def test_truncate_keeps_thinking_and_shortens_the_answer(self, priced):
        remaining = cost("claude-think", 4000 + 500) + OUT_PRICE["claude-think"] / 2
        p = plan("truncate", "claude-think", remaining=remaining, thinking_budget=4000)
        assert p.max_tokens == 4500
        assert p.budget_limited is True
        assert cost("claude-think", p.max_tokens) <= remaining

    def test_truncate_without_room_for_a_useful_answer_raises(self, priced):
        remaining = cost("claude-think", 4000 + 10)
        with pytest.raises(BudgetExceeded):
            plan("truncate", "claude-think", remaining=remaining, thinking_budget=4000,
                 min_useful_tokens=64)


# ── router ────────────────────────────────────────────────────────────────────

class TestRouter:
    def test_default_is_off(self):
        assert router().config.thinking_budget is None

    def test_budget_below_provider_minimum_rejected(self):
        with pytest.raises(ValueError, match="thinking_budget"):
            router(thinking_budget=500)

    def test_budget_model_gets_thinking_param(self, priced):
        r = router(budget=10.0, small_model="claude-think", big_model="claude-think",
                   thinking_budget=4000, max_output_tokens=2000)
        with patch("baar.router.litellm.completion", side_effect=echo) as m:
            r.chat("hello")
        assert m.call_args.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4000}
        assert m.call_args.kwargs["max_tokens"] == 6000

    def test_effort_model_gets_reasoning_effort(self, priced):
        r = router(budget=10.0, small_model="o-reasoner", big_model="o-reasoner", thinking_budget=20000)
        with patch("baar.router.litellm.completion", side_effect=echo) as m:
            r.chat("hello")
        assert m.call_args.kwargs["reasoning_effort"] == "high"
        assert "thinking" not in m.call_args.kwargs

    def test_non_reasoning_model_gets_nothing(self, priced):
        r = router(budget=10.0, thinking_budget=4000)
        with patch("baar.router.litellm.completion", side_effect=echo) as m:
            r.chat("hello")
        assert "thinking" not in m.call_args.kwargs
        assert "reasoning_effort" not in m.call_args.kwargs

    def test_without_budget_nothing_is_sent(self, priced):
        r = router(budget=10.0, small_model="claude-think", big_model="claude-think")
        with patch("baar.router.litellm.completion", side_effect=echo) as m:
            r.chat("hello")
        assert "thinking" not in m.call_args.kwargs

    def test_per_call_override(self, priced):
        r = router(budget=10.0, small_model="claude-think", big_model="claude-think")
        with patch("baar.router.litellm.completion", side_effect=echo) as m:
            r.chat("hello", thinking_budget=2048, max_output_tokens=1000)
        assert m.call_args.kwargs["thinking"]["budget_tokens"] == 2048
        assert m.call_args.kwargs["max_tokens"] == 3048

    def test_per_call_budget_below_minimum_rejected(self, priced):
        r = router(budget=10.0)
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(ValueError, match="thinking_budget"):
                r.chat("hello", thinking_budget=10)
            m.assert_not_called()

    def test_preflight_becomes_affordable_on_a_reasoning_model(self, priced):
        # Uncapped worst case: 64000 output tokens ≈ $0.96; capped: 6000 ≈ $0.09.
        budget = 0.2
        uncapped = router(budget=budget, small_model="claude-think", big_model="claude-think",
                          on_output_overrun="preflight")
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(BudgetExhausted):
                uncapped.chat("hello")
            m.assert_not_called()
        capped = router(budget=budget, small_model="claude-think", big_model="claude-think",
                        on_output_overrun="preflight", thinking_budget=4000, max_output_tokens=2000)
        with patch("baar.router.litellm.completion", side_effect=echo) as m:
            capped.chat("hello")
        assert m.call_count == 1

    def test_preflight_worst_case_never_overshoots(self, priced):
        r = router(budget=0.2, small_model="claude-think", big_model="claude-think",
                   on_output_overrun="preflight", thinking_budget=4000, max_output_tokens=2000,
                   prompt_margin=0.0)
        with patch("baar.router.litellm.completion",
                   side_effect=lambda **kw: response(kw["model"], completion_tokens=kw["max_tokens"])):
            r.chat("hello")
        assert r.spent <= 0.2

    def test_stream_gets_thinking_param(self, priced):
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"), finish_reason="stop")],
                                usage=SimpleNamespace(prompt_tokens=PROMPT, completion_tokens=30))
        r = router(budget=10.0, small_model="claude-think", big_model="claude-think", thinking_budget=4000)
        with patch("baar.router.litellm.completion", return_value=iter([chunk])) as m:
            assert "".join(r.stream_chat("hello")) == "ok"
        assert m.call_args.kwargs["thinking"]["budget_tokens"] == 4000

    def test_async_gets_thinking_param(self, priced):
        r = router(budget=10.0, small_model="claude-think", big_model="claude-think", thinking_budget=4000)
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock,
                   return_value=response("claude-think")) as m:
            asyncio.run(r.achat("hello"))
        assert m.call_args.kwargs["thinking"]["budget_tokens"] == 4000


# ── edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_reasoning_model_without_either_control(self):
        with patch("baar.core.caps.litellm.get_supported_openai_params", return_value=["max_tokens"]), \
             patch("baar.core.caps.litellm.supports_reasoning", return_value=True):
            assert caps.thinking_style("m") is None

    def test_budget_style_with_unknown_model_max_raises(self, priced):
        with patch("baar.core.caps.thinking_style", return_value="budget"):
            with pytest.raises(OutputLimitUnavailable, match="max output"):
                plan("preflight", "mystery", max_output_tokens=1000, thinking_budget=2000)

    def test_truncate_with_plenty_of_budget_is_not_shortened(self, priced):
        p = plan("truncate", "claude-think", remaining=100.0, max_output_tokens=2000, thinking_budget=4000)
        assert p.max_tokens == 6000
        assert p.budget_limited is False

    def test_truncate_on_unpriced_model_keeps_the_full_limit(self, priced):
        with patch.dict(MAX_OUT, {"claude-free": 64000}), patch.dict(CONTEXT, {"claude-free": 200000}), \
             patch.dict(STYLE, {"claude-free": "budget"}):
            p = plan("truncate", "claude-free", remaining=0.001, max_output_tokens=2000, thinking_budget=4000)
        assert p.max_tokens == 6000
        assert p.budget_limited is False

    def test_truncate_big_model_must_fit_answer_plus_thinking(self, priced):
        from baar import Policy, Rule
        force_big = Policy(rules=[Rule(when={}, then="force_big")])
        # Fits: answer 2000 + thinking 4000 on the big model.
        r = router(budget=1.0, small_model="gpt-4o-mini", big_model="claude-think", policy=force_big,
                   on_output_overrun="truncate", thinking_budget=4000, max_output_tokens=2000)
        with patch("baar.router.litellm.completion", side_effect=echo) as m:
            r.chat("hello")
        assert m.call_args.kwargs["model"] == "claude-think"
        assert m.call_args.kwargs["max_tokens"] == 6000
        # Doesn't fit uncut: downgrade instead of cutting the big model off.
        r = router(budget=0.05, small_model="gpt-4o-mini", big_model="claude-think", policy=force_big,
                   on_output_overrun="truncate", thinking_budget=4000, max_output_tokens=2000)
        with patch("baar.router.litellm.completion", side_effect=echo) as m, \
             pytest.warns(UserWarning, match="force_big"):
            r.chat("hello")
        assert m.call_args.kwargs["model"] == "gpt-4o-mini"
