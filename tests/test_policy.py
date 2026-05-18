"""Tests for the policy engine (baar/core/policy.py)."""

import pytest
from unittest.mock import MagicMock, patch

from baar.core.policy import Policy, PolicyAction, Rule, _match_value


# ── _match_value ──────────────────────────────────────────────────────────────

class TestMatchValue:
    def test_exact_string_match(self):
        assert _match_value("gpt-4o", "gpt-4o") is True

    def test_exact_string_no_match(self):
        assert _match_value("gpt-4o", "gpt-4o-mini") is False

    def test_comparison_gte_true(self):
        assert _match_value(">= 0.8", 0.9) is True

    def test_comparison_gte_false(self):
        assert _match_value(">= 0.8", 0.7) is False

    def test_comparison_lte(self):
        assert _match_value("<= 0.5", 0.5) is True
        assert _match_value("<= 0.5", 0.6) is False

    def test_comparison_gt(self):
        assert _match_value("> 0.5", 0.51) is True
        assert _match_value("> 0.5", 0.5) is False

    def test_comparison_lt(self):
        assert _match_value("< 0.5", 0.49) is True
        assert _match_value("< 0.5", 0.5) is False

    def test_comparison_eq(self):
        assert _match_value("== 1.0", 1.0) is True
        assert _match_value("== 1.0", 0.9) is False

    def test_comparison_ne(self):
        assert _match_value("!= 1.0", 0.9) is True
        assert _match_value("!= 1.0", 1.0) is False

    def test_comparison_bad_rhs_returns_false(self):
        assert _match_value(">= abc", 0.5) is False

    def test_comparison_non_numeric_actual_returns_false(self):
        assert _match_value(">= 0.5", "not-a-number") is False

    def test_non_string_condition_equality(self):
        assert _match_value(42, 42) is True
        assert _match_value(42, 43) is False

    def test_none_actual_string_condition(self):
        assert _match_value("gpt-4o", None) is False

    def test_none_actual_comparison(self):
        assert _match_value(">= 0.8", None) is False


# ── Rule ──────────────────────────────────────────────────────────────────────

class TestRule:
    def test_valid_rule(self):
        r = Rule(when={"model": "gpt-4o"}, then="block")
        assert r.action == PolicyAction.BLOCK

    def test_invalid_then_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            Rule(when={"model": "x"}, then="invalid_action")

    def test_matches_single_condition(self):
        r = Rule(when={"model": "gpt-4o"}, then="block")
        assert r.matches({"model": "gpt-4o"}) is True
        assert r.matches({"model": "gpt-4o-mini"}) is False

    def test_matches_multiple_conditions_all_must_match(self):
        r = Rule(when={"plan": "free", "model": "gpt-4o"}, then="block")
        assert r.matches({"plan": "free", "model": "gpt-4o"}) is True
        assert r.matches({"plan": "pro", "model": "gpt-4o"}) is False
        assert r.matches({"plan": "free", "model": "gpt-4o-mini"}) is False

    def test_matches_missing_key_returns_false(self):
        r = Rule(when={"plan": "free"}, then="block")
        assert r.matches({}) is False

    def test_matches_comparison(self):
        r = Rule(when={"utilization": ">= 0.8"}, then="force_small")
        assert r.matches({"utilization": 0.9}) is True
        assert r.matches({"utilization": 0.7}) is False

    def test_all_actions(self):
        for action in ["block", "force_small", "force_big", "allow"]:
            r = Rule(when={}, then=action)
            assert r.action == PolicyAction(action)


# ── Policy ────────────────────────────────────────────────────────────────────

class TestPolicy:
    def test_no_rules_returns_none(self):
        p = Policy()
        assert p.evaluate({"model": "gpt-4o"}) is None

    def test_first_match_wins(self):
        p = Policy(rules=[
            Rule(when={"plan": "free"}, then="block"),
            Rule(when={"plan": "free"}, then="force_small"),
        ])
        assert p.evaluate({"plan": "free"}) == PolicyAction.BLOCK

    def test_no_match_returns_none(self):
        p = Policy(rules=[
            Rule(when={"plan": "free"}, then="block"),
        ])
        assert p.evaluate({"plan": "pro"}) is None

    def test_second_rule_matches_when_first_doesnt(self):
        p = Policy(rules=[
            Rule(when={"plan": "free", "model": "gpt-4o"}, then="block"),
            Rule(when={"utilization": ">= 0.8"},           then="force_small"),
        ])
        result = p.evaluate({"plan": "pro", "model": "gpt-4o", "utilization": 0.9})
        assert result == PolicyAction.FORCE_SMALL

    def test_allow_action(self):
        p = Policy(rules=[
            Rule(when={"domain": "code"}, then="allow"),
        ])
        assert p.evaluate({"domain": "code"}) == PolicyAction.ALLOW

    def test_context_keys_are_matched(self):
        p = Policy(rules=[
            Rule(when={"tier": "enterprise"}, then="force_big"),
        ])
        assert p.evaluate({"tier": "enterprise"}) == PolicyAction.FORCE_BIG
        assert p.evaluate({"tier": "starter"}) is None


# ── Router integration ────────────────────────────────────────────────────────

@pytest.fixture
def mock_response():
    resp = MagicMock()
    resp.model = "gpt-4o-mini"
    resp.choices[0].message.content = "answer"
    resp.usage.prompt_tokens = 50
    resp.usage.completion_tokens = 30
    return resp


@pytest.fixture
def patches(mock_response):
    with patch("baar.router.litellm.completion", return_value=mock_response), \
         patch("baar.core.budget.completion_cost", return_value=0.000025), \
         patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002)), \
         patch("baar.router.token_counter", return_value=50):
        yield


class TestRouterPolicyIntegration:
    def test_block_raises_policy_violation(self, patches):
        from baar import BAARRouter, PolicyViolation
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"model": "gpt-4o-mini"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        with pytest.raises(PolicyViolation):
            router.chat("hello")

    def test_force_small_overrides_big_decision(self, patches):
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"complexity": ">= 0.0"}, then="force_small")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        result = router.chat("Write a complex algorithm with recursion and dynamic programming")
        assert result == "answer"
        # Policy-forced downgrade must NOT set forced_by_budget — it's a governance
        # decision, not a budget constraint.
        assert router.log.steps[-1].decision.forced_by_budget is False
        assert "[POLICY FORCE_SMALL]" in router.log.steps[-1].decision.reason

    def test_force_big_overrides_small_decision(self, patches):
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"domain": "general"}, then="force_big")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        result = router.chat("hi")
        assert result == "answer"
        assert router.log.steps[-1].decision.model == "gpt-4o"

    def test_allow_is_passthrough(self, patches):
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"domain": "general"}, then="allow")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        result = router.chat("hi")
        assert result == "answer"

    def test_context_dict_used_in_policy(self, patches):
        from baar import BAARRouter, PolicyViolation
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"plan": "free"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        with pytest.raises(PolicyViolation):
            router.chat("hello", context={"plan": "free"})
        # No violation without the matching context
        result = router.chat("hello", context={"plan": "pro"})
        assert result == "answer"

    def test_no_policy_no_impact(self, patches):
        from baar import BAARRouter
        router = BAARRouter(budget=1.0, use_llm_router=False)
        result = router.chat("hello")
        assert result == "answer"

    def test_policy_violation_has_facts(self, patches):
        from baar import BAARRouter, PolicyViolation
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"plan": "free"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        with pytest.raises(PolicyViolation) as exc_info:
            router.chat("hello", context={"plan": "free"})
        assert exc_info.value.facts.get("plan") == "free"

    @pytest.mark.asyncio
    async def test_achat_block_raises_policy_violation(self, patches):
        from baar import BAARRouter, PolicyViolation
        from baar.core.policy import Policy, Rule
        with patch("baar.router.litellm.acompletion") as mock_ac:
            mock_ac.return_value = MagicMock(
                model="gpt-4o-mini",
                choices=[MagicMock(message=MagicMock(content="answer"))],
                usage=MagicMock(prompt_tokens=50, completion_tokens=30),
            )
            policy = Policy(rules=[Rule(when={"plan": "blocked"}, then="block")])
            router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
            with pytest.raises(PolicyViolation):
                await router.achat("hello", context={"plan": "blocked"})

    def test_stream_chat_block_raises_policy_violation(self, patches):
        from baar import BAARRouter, PolicyViolation
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"plan": "free"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        with pytest.raises(PolicyViolation):
            list(router.stream_chat("hello", context={"plan": "free"}))

    def test_stream_chat_force_small(self, patches):
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule
        chunks = []
        with patch("baar.router.litellm.completion") as mock_comp:
            stream_chunk = MagicMock()
            stream_chunk.choices[0].delta.content = "chunk"
            mock_comp.return_value = iter([stream_chunk])
            policy = Policy(rules=[Rule(when={"complexity": ">= 0.0"}, then="force_small")])
            router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
            chunks = list(router.stream_chat("write a recursive algorithm", context={}))
        assert "chunk" in chunks

    def test_run_context_forwarded_to_policy(self, patches):
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"plan": "free"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        log = router.run(["hello"], context={"plan": "free"})
        assert len(log.errors) == 0
        assert len(log.steps) == 0

    def test_run_no_context_no_policy_impact(self, patches):
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"plan": "free"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        log = router.run(["hello"])  # no context → plan key absent → no match
        assert len(log.steps) == 1

    def test_run_policy_block_is_not_counted_as_error(self, patches):
        """PolicyViolation must be treated like TaskRejected — continue, not error."""
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"plan": "free"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy, max_consecutive_errors=1)
        # 2 tasks, both blocked — run must not abort due to consecutive_errors
        log = router.run(["hello", "world"], context={"plan": "free"})
        assert len(log.errors) == 0
        assert len(log.steps) == 0  # no successful steps

    def test_context_keys_do_not_override_system_facts(self, patches):
        """User context cannot spoof the model seen by policy."""
        from baar import BAARRouter, PolicyViolation
        from baar.core.policy import Policy, Rule
        # Rule blocks if model == "gpt-4o-mini" (the actual model)
        # Caller tries to spoof model as "gpt-4o" in context — should still be blocked
        policy = Policy(rules=[Rule(when={"model": "gpt-4o-mini"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        with pytest.raises(PolicyViolation):
            # context tries to override model — system fact must win
            router.chat("hello", context={"model": "gpt-4o"})

    def test_force_big_voided_by_budget_emits_warning(self):
        """When force_big is overridden by budget constraint, a UserWarning is emitted."""
        import warnings as _warnings
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule

        def mock_cost_per_token(model, prompt_tokens, completion_tokens):
            if "mini" in model:
                return (0.000001, 0.000002)
            return (1.0, 1.0)

        mock_resp = MagicMock()
        mock_resp.model = "gpt-4o-mini"
        mock_resp.choices[0].message.content = "answer"
        mock_resp.usage.prompt_tokens = 50
        mock_resp.usage.completion_tokens = 30

        policy = Policy(rules=[Rule(when={"complexity": "<= 1.0"}, then="force_big")])
        with patch("baar.router.litellm.completion", return_value=mock_resp), \
             patch("baar.core.budget.completion_cost", return_value=0.000003), \
             patch("baar.core.budget.cost_per_token", side_effect=mock_cost_per_token), \
             patch("baar.router.token_counter", return_value=50):
            router = BAARRouter(budget=0.01, use_llm_router=False, policy=policy)
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                try:
                    router.chat("hello")
                except Exception:
                    pass
        force_big_warns = [w for w in caught if "force_big" in str(w.message).lower()]
        assert len(force_big_warns) >= 1

    @pytest.mark.asyncio
    async def test_arun_policy_block_is_not_counted_as_error(self, patches):
        """PolicyViolation in arun must not count as a consecutive error."""
        from baar import BAARRouter
        from baar.core.policy import Policy, Rule
        with patch("baar.router.litellm.acompletion") as mock_ac:
            mock_ac.return_value = MagicMock(
                model="gpt-4o-mini",
                choices=[MagicMock(message=MagicMock(content="answer"))],
                usage=MagicMock(prompt_tokens=50, completion_tokens=30),
            )
            policy = Policy(rules=[Rule(when={"plan": "free"}, then="block")])
            router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy, max_consecutive_errors=1)
            log = await router.arun(["hello", "world"], context={"plan": "free"})
            assert len(log.errors) == 0

    @pytest.mark.asyncio
    async def test_astream_chat_policy_block(self, patches):
        """astream_chat must raise PolicyViolation on block action."""
        from baar import BAARRouter, PolicyViolation
        from baar.core.policy import Policy, Rule
        policy = Policy(rules=[Rule(when={"plan": "free"}, then="block")])
        router = BAARRouter(budget=1.0, use_llm_router=False, policy=policy)
        with pytest.raises(PolicyViolation):
            async for _ in router.astream_chat("hello", context={"plan": "free"}):
                pass
