"""
tests/test_budget_awareness.py — the agent can see its budget, and wraps up instead of crashing.

  - router.budget_status() → BudgetStatus (spent, remaining, calls left at the current rate)
  - budget_awareness=True adds a short status line to the end of the last user message
    (not the system prompt, so a cached prompt prefix stays intact)
  - wind_down=WindDown(at=0.9, reserve=0.05): the next call after the threshold is the
    final one, with a wrap-up instruction; later calls raise BudgetExhausted carrying
    the final answer in .partial_result
  - Policy rule then="wrap_up" makes a call the final one
(BATS, arXiv 2511.17006: a budget status block each step makes agents spend better.)
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from baar import BAARRouter, BudgetExhausted, BudgetStatus, Policy, Rule, WindDown

IN_PRICE = {"gpt-4o-mini": 1.5e-7, "gpt-4o": 2.5e-6}
OUT_PRICE = {"gpt-4o-mini": 6e-7, "gpt-4o": 1e-5}
MAX_OUT = {"gpt-4o-mini": 16384, "gpt-4o": 16384}
CONTEXT = {"gpt-4o-mini": 128000, "gpt-4o": 128000}
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
         patch("baar.core.caps.supports_stream_usage", return_value=True):
        yield


class Model:
    """Answers each call with `tokens` completion tokens; records what it was sent."""

    def __init__(self, tokens=1000):
        self.tokens = tokens
        self.calls = []

    def completion(self, **kw):
        self.calls.append(kw)
        text = f"answer {len(self.calls)}"
        if kw.get("stream"):
            return iter([SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=PROMPT, completion_tokens=self.tokens))])
        r = MagicMock()
        r.model = kw["model"]
        r.choices[0].message.content = text
        r.choices[0].finish_reason = "stop"
        r.usage.prompt_tokens = PROMPT
        r.usage.completion_tokens = self.tokens
        return r

    async def acompletion(self, **kw):
        r = self.completion(**kw)
        if kw.get("stream"):
            chunks = list(r)

            async def _agen():
                for c in chunks:
                    yield c
            return _agen()
        return r

    def last_user_text(self, i=-1):
        msgs = self.calls[i]["messages"]
        return [m for m in msgs if m["role"] == "user"][-1]["content"]


def patched(model):
    return (patch("baar.router.litellm.completion", side_effect=model.completion),
            patch("baar.router.litellm.acompletion", side_effect=model.acompletion))


def router(**kw):
    kw.setdefault("use_llm_router", False)
    return BAARRouter(**kw)


CALL = cost("gpt-4o-mini", 1000)


# ── BudgetStatus ──────────────────────────────────────────────────────────────

class TestBudgetStatus:
    def test_fresh_router(self):
        s = router(budget=1.0).budget_status()
        assert isinstance(s, BudgetStatus)
        assert (s.total, s.spent, s.remaining, s.utilization) == (1.0, 0.0, 1.0, 0.0)
        assert s.calls == 0 and s.avg_call_cost is None and s.calls_left is None
        assert s.winding_down is False

    def test_after_calls(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0)
        with p1, p2:
            r.chat("a")
            r.chat("b")
        s = r.budget_status()
        assert s.calls == 2
        assert s.avg_call_cost == pytest.approx(CALL)
        assert s.calls_left == int((1.0 - 2 * CALL) // CALL)

    def test_note_text(self):
        s = BudgetStatus(total=1.0, spent=0.32, remaining=0.68, utilization=0.32, calls=8,
                         avg_call_cost=0.04, calls_left=17, winding_down=False)
        assert s.note() == "[budget] $0.3200 of $1.00 used (32%) · ~17 calls left at the current rate"

    def test_note_without_history(self):
        s = BudgetStatus(total=1.0, spent=0.0, remaining=1.0, utilization=0.0, calls=0,
                         avg_call_cost=None, calls_left=None, winding_down=False)
        assert s.note() == "[budget] $0.0000 of $1.00 used (0%)"

    def test_calls_count_survives_log_eviction(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0, max_log_steps=1)
        with p1, p2:
            for _ in range(3):
                r.chat("a")
        assert r.budget_status().calls == 3


# ── budget_awareness ──────────────────────────────────────────────────────────

class TestAwareness:
    def test_off_by_default(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0)
        with p1, p2:
            r.chat("hello")
        assert m.last_user_text() == "hello"

    def test_status_line_is_appended_to_the_last_user_message(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0, budget_awareness=True, system_prompt="be brief")
        with p1, p2:
            r.chat("first")
            r.chat("second")
        sent = m.calls[1]["messages"]
        assert sent[0] == {"role": "system", "content": "be brief"}  # prefix untouched
        assert sent[-1]["content"].startswith("second\n\n[budget] $")
        assert "calls left" in sent[-1]["content"]

    def test_history_messages_are_not_mutated(self, priced):
        m = Model()
        p1, p2 = patched(m)
        history = [{"role": "user", "content": "q"}]
        r = router(budget=1.0, budget_awareness=True)
        with p1, p2:
            r.chat("q", messages=history)
        assert history == [{"role": "user", "content": "q"}]
        assert "[budget]" in m.last_user_text()

    def test_multipart_content_gets_a_text_part(self, priced):
        m = Model()
        p1, p2 = patched(m)
        history = [{"role": "user", "content": [{"type": "text", "text": "look"}]}]
        r = router(budget=1.0, budget_awareness=True)
        with p1, p2:
            r.chat("look", messages=history)
        parts = m.last_user_text()
        assert parts[0] == {"type": "text", "text": "look"}
        assert parts[-1]["type"] == "text" and parts[-1]["text"].startswith("[budget]")

    def test_custom_formatter(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0, budget_awareness=lambda s: f"LEFT={s.remaining:.2f}")
        with p1, p2:
            r.chat("hi")
        assert m.last_user_text() == "hi\n\nLEFT=1.00"

    def test_history_without_a_user_message_gets_one(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0, budget_awareness=True)
        with p1, p2:
            r.chat("go", messages=[{"role": "assistant", "content": "earlier"}])
        assert m.calls[0]["messages"][-1]["role"] == "user"
        assert m.calls[0]["messages"][-1]["content"].startswith("[budget]")

    @pytest.mark.parametrize("path", ["stream", "achat", "astream"])
    def test_other_paths(self, priced, path):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0, budget_awareness=True)

        async def _astream():
            return "".join([t async for t in r.astream_chat("hi")])

        with p1, p2:
            if path == "stream":
                list(r.stream_chat("hi"))
            elif path == "achat":
                asyncio.run(r.achat("hi"))
            else:
                asyncio.run(_astream())
        assert "[budget]" in m.last_user_text()


# ── wind_down ─────────────────────────────────────────────────────────────────

class TestWindDown:
    def test_validation(self):
        for bad in ({"at": 0}, {"at": 1.5}, {"reserve": -0.1}, {"reserve": 1.0}):
            with pytest.raises(ValueError):
                WindDown(**bad)

    def test_final_call_then_partial_result(self, priced):
        m = Model()
        p1, p2 = patched(m)
        # Each call uses 27.8% of the budget: the 4th starts at 83% (past 75%) → final.
        # (The final call must still afford the router's estimate, ~0.5 of a call.)
        r = router(budget=CALL * 3.6, wind_down=WindDown(at=0.75, reserve=0.0))
        answers = []
        with p1, p2:
            with pytest.raises(BudgetExhausted) as exc:
                for i in range(10):
                    answers.append(r.chat(f"step {i}"))
        final = answers[-1]
        assert "wrap up" in m.last_user_text().lower() or "final" in m.last_user_text().lower()
        assert exc.value.partial_result == final
        assert len(m.calls) == len(answers)  # the refused call made no network request
        assert r.budget_status().winding_down is True

    def test_reserve_triggers_wind_down(self, priced):
        m = Model()
        p1, p2 = patched(m)
        budget = CALL * 3.5
        r = router(budget=budget, wind_down=WindDown(at=0.99, reserve=CALL * 1.6 / budget))
        with p1, p2:
            r.chat("1")
            r.chat("2")
            assert "[budget]" not in m.last_user_text()
            r.chat("3")  # 1.5 calls left, below the 1.6-call reserve → final
            with pytest.raises(BudgetExhausted) as exc:
                r.chat("4")
        assert exc.value.partial_result == "answer 3"
        assert "final" in m.last_user_text().lower()

    def test_custom_instruction(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=CALL * 1.8, wind_down=WindDown(at=0.5, reserve=0.0, instruction="FINISH NOW"))
        with p1, p2:
            r.chat("1")
            r.chat("2")
        assert m.last_user_text().endswith("FINISH NOW")

    def test_run_stops_cleanly_with_the_final_answer_logged(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=CALL * 2.7, wind_down=WindDown(at=0.5, reserve=0.0))
        with p1, p2:
            log = r.run([f"t{i}" for i in range(6)])
        assert log.total_steps == 3
        assert log.steps[-1].response_text == "answer 3"

    def test_rolls_back_when_budget_frees_up(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=CALL * 1.8, wind_down=WindDown(at=0.5, reserve=0.0))
        with p1, p2:
            r.chat("1")
            r.chat("2")  # final
            with pytest.raises(BudgetExhausted):
                r.chat("3")
            r._tracker.reset()  # e.g. a new budget window
            assert r.budget_status().winding_down is False
            r.chat("4")
        assert "final" not in m.last_user_text().lower()

    @pytest.mark.parametrize("path", ["stream", "achat", "astream"])
    def test_other_paths(self, priced, path):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=CALL * 1.8, wind_down=WindDown(at=0.5, reserve=0.0))

        async def _astream(task):
            return "".join([t async for t in r.astream_chat(task)])

        def call(task):
            if path == "stream":
                return "".join(r.stream_chat(task))
            if path == "achat":
                return asyncio.run(r.achat(task))
            return asyncio.run(_astream(task))

        with p1, p2:
            call("1")
            final = call("2")
            with pytest.raises(BudgetExhausted) as exc:
                call("3")
        assert exc.value.partial_result == final


# ── policy wrap_up ────────────────────────────────────────────────────────────

class TestPolicyWrapUp:
    def test_rule_makes_the_call_final(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0, policy=Policy(rules=[Rule(when={"phase": "last"}, then="wrap_up")]))
        with p1, p2:
            r.chat("1", context={"phase": "work"})
            assert "final" not in m.last_user_text().lower()
            answer = r.chat("2", context={"phase": "last"})
            assert "final" in m.last_user_text().lower()
            with pytest.raises(BudgetExhausted) as exc:
                r.chat("3", context={"phase": "work"})
        assert exc.value.partial_result == answer

    def test_wrap_up_is_a_valid_action(self):
        Rule(when={}, then="wrap_up")

    def test_reason_is_marked(self, priced):
        m = Model()
        p1, p2 = patched(m)
        r = router(budget=1.0, policy=Policy(rules=[Rule(when={}, then="wrap_up")]))
        with p1, p2:
            r.chat("1")
        assert "[POLICY WRAP_UP]" in r.log.steps[-1].decision.reason


# ── integrations ──────────────────────────────────────────────────────────────

class TestIntegrations:
    def test_langchain(self):
        pytest.importorskip("langchain_core")
        from baar.integrations.langchain import BaarChatModel
        r = router(budget=1.0)
        assert BaarChatModel(router=r).budget_status == r.budget_status()

    def test_langgraph(self):
        from baar.integrations.langgraph import BaarMiddleware
        r = router(budget=1.0)
        assert BaarMiddleware(MagicMock(), router=r).budget_status == r.budget_status()

    def test_crewai(self):
        import sys
        if "crewai" not in sys.modules:
            class _LLM:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)
            sys.modules["crewai"] = MagicMock(LLM=_LLM)
        from baar.integrations.crewai import BaarCrewLLM
        r = router(budget=1.0)
        assert BaarCrewLLM(router=r).budget_status == r.budget_status()

    def test_exhausted_without_partial_result_by_default(self):
        assert BudgetExhausted("x").partial_result is None


def test_wrap_up_on_the_first_call_is_cleared_by_a_reset(priced):
    m = Model()
    p1, p2 = patched(m)
    r = router(budget=1.0, policy=Policy(rules=[Rule(when={"go": "last"}, then="wrap_up")]))
    with p1, p2:
        r.chat("1", context={"go": "last"})
        with pytest.raises(BudgetExhausted):
            r.chat("2")
        r._tracker.reset()
        assert r.chat("3") == "answer 2"


@pytest.mark.parametrize("path", ["chat", "stream", "achat", "astream"])
def test_policy_wrap_up_on_every_path(priced, path):
    m = Model()
    p1, p2 = patched(m)
    r = router(budget=1.0, policy=Policy(rules=[Rule(when={"go": "last"}, then="wrap_up")]))

    async def _astream(task, **kw):
        return "".join([t async for t in r.astream_chat(task, **kw)])

    def call(task, **kw):
        if path == "chat":
            return r.chat(task, **kw)
        if path == "stream":
            return "".join(r.stream_chat(task, **kw))
        if path == "achat":
            return asyncio.run(r.achat(task, **kw))
        return asyncio.run(_astream(task, **kw))

    with p1, p2:
        final = call("1", context={"go": "last"})
        assert "final call" in m.last_user_text().lower()
        with pytest.raises(BudgetExhausted) as exc:
            call("2")
    assert exc.value.partial_result == final
