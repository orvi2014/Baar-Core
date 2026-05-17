"""
tests/test_router.py — Router unit tests

Tests heuristic scoring, LLM routing decisions,
budget pressure effects, and forced downgrades.
All tests run without API keys.
"""

import pytest
from unittest.mock import MagicMock, patch
from baar.core.router import Router, ModelTier, RoutingDecision
from baar.router import BAARConfig, BAARRouter


# ─────────────────────────────────────────────────────────
# Heuristic scoring (no API calls)
# ─────────────────────────────────────────────────────────

class TestHeuristicScoring:
    """Tests heuristic fallback — no LLM calls."""

    def setup_method(self):
        # use_llm_router=False → always uses heuristic
        self.router = Router(use_llm_router=False)

    def test_simple_greeting_routes_to_small(self):
        decision = self.router.decide("hi", budget_utilization=0.0)
        assert decision.tier == ModelTier.SMALL

    def test_factual_short_routes_to_small(self):
        decision = self.router.decide("What is 2+2?", budget_utilization=0.0)
        assert decision.tier == ModelTier.SMALL

    def test_code_task_routes_to_big(self):
        task = "def parse_json(): # Debug this function and add error handling"
        decision = self.router.decide(task, budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_analysis_task_routes_to_big(self):
        task = "Analyze and compare the trade-offs between microservices and monolithic architecture"
        decision = self.router.decide(task, budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_structured_math_prompt_routes_to_big(self):
        task = (
            "A store sells pencils in packs of 3. If Ana buys 7 packs, how many pencils does she get?\n\n"
            "Show your work and end with 'The answer is #### <number>'."
        )
        decision = self.router.decide(task, budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_multi_choice_prompt_routes_to_big(self):
        task = (
            "What is the primary function of mitochondria?\n\n"
            "Choices:\n"
            "A. Protein synthesis\n"
            "B. Cellular respiration\n"
            "C. DNA replication\n"
            "D. Signal transduction\n\n"
            "Respond with ONLY the letter of the correct choice."
        )
        decision = self.router.decide(task, budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_easy_pattern_capital_routes_to_small(self):
        decision = self.router.decide(
            "What is the capital of Japan?",
            budget_utilization=0.0)
        assert decision.tier == ModelTier.SMALL
        assert "easy-pattern" in decision.reason

    def test_long_task_increases_complexity(self):
        short_task = "What time is it"
        long_task = " ".join(["word"] * 60)  # 60 words
        router = Router(use_llm_router=False)
        short_score, _ = router._heuristic_score(short_task)
        long_score, _ = router._heuristic_score(long_task)
        assert long_score > short_score

    def test_complexity_score_in_range(self):
        for task in ["hello", "write complex code with error handling and unit tests", "x" * 1000]:
            score, _ = self.router._heuristic_score(task)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for task: {task[:50]}"

    def test_decision_contains_reason(self):
        decision = self.router.decide("hello", budget_utilization=0.0)
        assert len(decision.reason) > 0

    def test_decision_model_matches_tier(self):
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o", use_llm_router=False)
        for task in ["hi", "complex analysis with multiple steps and reasoning"]:
            decision = router.decide(task, budget_utilization=0.0)
            if decision.tier == ModelTier.SMALL:
                assert decision.model == "gpt-4o-mini"
            else:
                assert decision.model == "gpt-4o"


# ─────────────────────────────────────────────────────────
# Budget pressure effects
# ─────────────────────────────────────────────────────────

class TestBudgetPressure:
    def setup_method(self):
        self.router = Router(
            complexity_threshold=0.65,
            use_llm_router=False)

    def test_high_utilization_raises_threshold(self):
        """At 85% budget used, even moderately complex tasks go to SMALL."""
        # A task that would normally go to BIG at low utilization
        task = "Explain the concept of recursion with examples"

        decision_low_util = self.router.decide(task, budget_utilization=0.1)
        decision_high_util = self.router.decide(task, budget_utilization=0.85)

        # High utilization should either keep same or downgrade — never upgrade
        if decision_low_util.tier == ModelTier.SMALL:
            assert decision_high_util.tier == ModelTier.SMALL
        # (if originally BIG, high utilization may downgrade it)

    def test_near_zero_budget_always_routes_small(self):
        """When almost out of budget, always use SMALL regardless of complexity."""
        complex_task = "Write a comprehensive analysis with code generation and debugging"
        decision = self.router.decide(
            complex_task,
            budget_utilization=0.99)
        assert decision.tier == ModelTier.SMALL

    def test_full_budget_allows_big_for_complex(self):
        """With full budget, complex tasks should route to BIG."""
        complex_task = "debug this complex recursive algorithm and explain each step"
        decision = self.router.decide(
            complex_task,
            budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG


# ─────────────────────────────────────────────────────────
# Force downgrade
# ─────────────────────────────────────────────────────────

class TestForceSmall:
    def test_force_small_changes_tier_and_model(self):
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o", use_llm_router=False)
        original = RoutingDecision(
            tier=ModelTier.BIG,
            model="gpt-4o",
            confidence=0.9,
            complexity_score=0.8,
            reason="complex task",
            forced_by_budget=False)
        forced = router.force_small(original, reason="insufficient funds")

        assert forced.tier == ModelTier.SMALL
        assert forced.model == "gpt-4o-mini"
        assert forced.forced_by_budget is True
        assert "BUDGET FORCED" in forced.reason
        assert forced.complexity_score == original.complexity_score

    def test_force_small_preserves_complexity_score(self):
        router = Router(use_llm_router=False)
        original = RoutingDecision(
            tier=ModelTier.BIG,
            model="gpt-4o",
            confidence=0.85,
            complexity_score=0.75,
            reason="analysis needed",
            forced_by_budget=False)
        forced = router.force_small(original, "budget low")
        assert forced.complexity_score == 0.75

    def test_force_small_preserves_routing_cache_hit(self):
        router = Router(use_llm_router=False)
        original = RoutingDecision(
            tier=ModelTier.BIG,
            model="gpt-4o",
            confidence=0.85,
            complexity_score=0.75,
            reason="analysis needed",
            forced_by_budget=False,
            routing_cache_hit=True)
        forced = router.force_small(original, "budget low")
        assert forced.routing_cache_hit is True


# ─────────────────────────────────────────────────────────
# LLM router (mocked)
# ─────────────────────────────────────────────────────────

class TestLLMRouter:
    def test_llm_router_parses_valid_json(self):
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.8, "reason": "requires deep reasoning"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            score, reason, domain, eot = router._llm_score("complex task")
            assert 0.0 <= score <= 1.0
            assert score == pytest.approx(0.8)  # no length penalty — trust the LLM's score
            assert "reasoning" in reason

    def test_llm_router_handles_markdown_code_block(self):
        """Some models wrap JSON in ```json ... ```"""
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '```json\n{"complexity": 0.6, "reason": "medium"}\n```'

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            score, reason, domain, eot = router._llm_score("medium task")
            assert 0.0 <= score <= 1.0
            assert score == pytest.approx(0.6)  # no length penalty — trust the LLM's score

    def test_llm_router_clamps_score_to_valid_range(self):
        """Score from LLM must be clamped to [0, 1]."""
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 1.5, "reason": "extreme"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            score, *_ = router._llm_score("extreme task")
            assert 0.95 <= score <= 1.0

    def test_llm_router_length_does_not_reduce_score(self):
        """The LLM's score is trusted directly — long tasks are not penalised downward.
        Penalising long tasks was wrong: long prompts tend to be more complex, not less."""
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.9, "reason": "hard"}'
        short_task = "hard question"
        long_task = "x" * 1500

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            short_score, *_ = router._llm_score(short_task)
            long_score, *_ = router._llm_score(long_task)
            assert long_score == pytest.approx(short_score)

    def test_llm_router_falls_back_on_api_error(self):
        """If LLM call fails, falls back to heuristic — never crashes."""
        router = Router(use_llm_router=True, complexity_threshold=0.65)

        with patch("baar.core.router.litellm.completion", side_effect=Exception("API down")):
            # Should use heuristic fallback
            decision = router.decide(
                "hello world",
                budget_utilization=0.0)
            assert decision.tier in (ModelTier.SMALL, ModelTier.BIG)  # got a valid answer

    def test_llm_router_falls_back_on_json_parse_error(self):
        """If LLM returns invalid JSON, falls back to heuristic."""
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I think this is complex."

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            decision = router.decide("some task", budget_utilization=0.0)
            assert decision.tier in (ModelTier.SMALL, ModelTier.BIG)

    def test_router_task_view_preserves_head_middle_and_tail(self):
        router = Router(use_llm_router=True, routing_task_char_limit=99)
        task = ("A" * 120) + ("M" * 120) + ("Z" * 120)
        view = router._router_task_view(task)

        assert len(view) > 99  # includes truncation markers
        assert view.startswith("A" * 33)
        assert ("M" * 33) in view
        assert view.endswith("Z" * 33)
        assert view.count("[TRUNCATED SEGMENT]") == 2

    def test_llm_score_uses_router_task_view_in_prompt(self):
        router = Router(use_llm_router=True, routing_task_char_limit=90)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.5, "reason": "mid"}'
        task = ("A" * 100) + ("M" * 100) + ("Z" * 100)

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            router._llm_score(task)

        # System message is messages[0]; user message (with truncated task) is messages[1]
        user_msg_sent = m.call_args.kwargs["messages"][1]["content"]
        assert user_msg_sent.count("[TRUNCATED SEGMENT]") == 2
        assert ("A" * 30) in user_msg_sent
        assert ("M" * 30) in user_msg_sent
        assert ("Z" * 30) in user_msg_sent


# ─────────────────────────────────────────────────────────
# Complexity LRU (routing) cache
# ─────────────────────────────────────────────────────────

class TestRoutingCache:
    def test_llm_second_identical_task_one_completion_call(self):
        router = Router(use_llm_router=True, routing_cache_size=256)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.5, "reason": "mid"}'
        # Use a task that is not in the easy_small_reason fast-path so it
        # actually reaches the LLM router and we can verify caching behaviour.
        task = "Summarize the key differences between REST and GraphQL APIs"

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            d1 = router.decide(task, budget_utilization=0.0)
            d2 = router.decide(task, budget_utilization=0.0)

        assert m.call_count == 1
        assert d1.routing_cache_hit is False
        assert d2.routing_cache_hit is True
        assert d1.complexity_score == d2.complexity_score

    def test_whitespace_normalized_same_cache_key(self):
        router = Router(use_llm_router=True, routing_cache_size=256)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.3, "reason": "x"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            router.decide("  hello   world  ", 0.0)
            router.decide("hello world", 0.0)

        assert m.call_count == 1

    def test_cache_disabled_invokes_scorer_each_time(self):
        router = Router(
            use_llm_router=True,
            routing_cache_enabled=False)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.4, "reason": "x"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            router.decide("same", 0.0)
            router.decide("same", 0.0)

        assert m.call_count == 2

    def test_cache_size_zero_disables_cache(self):
        router = Router(use_llm_router=True, routing_cache_size=0)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.4, "reason": "x"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            router.decide("same", 0.0)
            router.decide("same", 0.0)

        assert m.call_count == 2

    def test_tier_recomputed_on_cache_hit_when_utilization_changes(self):
        router = Router(use_llm_router=False, complexity_threshold=0.65)
        # Heuristic ~0.7: BIG at low utilization, SMALL once cost pressure raises the bar
        task = "Explain the concept of recursion with examples"
        d_low = router.decide(task, 0.0)
        d_high = router.decide(task, 0.85)
        assert d_low.tier == ModelTier.BIG
        assert d_high.tier == ModelTier.SMALL
        assert d_high.routing_cache_hit is True

    def test_heuristic_scorer_also_cached(self):
        router = Router(use_llm_router=False, routing_cache_size=256)
        d1 = router.decide("hello there", 0.0)
        d2 = router.decide("hello there", 0.0)
        assert d1.routing_cache_hit is False
        assert d2.routing_cache_hit is True

    def test_cache_key_changes_when_routing_window_changes(self):
        task = ("A" * 100) + ("Z" * 100)
        r_small = Router(use_llm_router=True, routing_task_char_limit=80)
        r_large = Router(use_llm_router=True, routing_task_char_limit=160)

        k_small = r_small._routing_cache_key(task)
        k_large = r_large._routing_cache_key(task)
        assert k_small != k_large


class TestExploration:
    def test_exploration_can_downshift_big_to_small(self):
        router = Router(use_llm_router=False, small_exploration_rate=0.1)
        task = "def parse_json(): # Debug this function and add error handling"
        with patch("baar.core.router.random.random", return_value=0.01):
            decision = router.decide(task, 0.0)
        assert decision.tier == ModelTier.SMALL
        assert "EXPLORE SMALL" in decision.reason


# ─────────────────────────────────────────────────────────
# Custom thresholds
# ─────────────────────────────────────────────────────────

class TestCustomThresholds:
    def test_very_low_threshold_routes_almost_everything_to_big(self):
        router = Router(complexity_threshold=0.1, use_llm_router=False)
        decision = router.decide("explain recursion", budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_very_high_threshold_routes_almost_everything_to_small(self):
        router = Router(complexity_threshold=0.99, use_llm_router=False)
        decision = router.decide("write complex recursive fibonacci", budget_utilization=0.0)
        assert decision.tier == ModelTier.SMALL


# ─────────────────────────────────────────────────────────
# Structured routing signal (domain + estimated_output_tokens)
# ─────────────────────────────────────────────────────────

class TestStructuredRoutingSignal:
    def test_llm_score_returns_domain_and_eot(self):
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"complexity": 0.9, "reason": "complex code", '
            '"domain": "code", "estimated_output_tokens": 800}'
        )
        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            score, reason, domain, eot = router._llm_score("write a parser")
        assert domain == "code"
        assert eot == 800

    def test_llm_score_falls_back_to_defaults_for_missing_fields(self):
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"complexity": 0.5, "reason": "mid"}'  # no domain or eot
        )
        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            score, reason, domain, eot = router._llm_score("summarize this")
        assert domain == "general"
        assert eot == 500

    def test_llm_score_clamps_eot_to_valid_range(self):
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"complexity": 0.7, "reason": "x", "domain": "math", "estimated_output_tokens": 99999}'
        )
        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            _, _, _, eot = router._llm_score("task")
        assert eot == 4000  # clamped to max

    def test_decision_carries_domain_and_eot(self):
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"complexity": 0.5, "reason": "x", "domain": "analysis", "estimated_output_tokens": 600}'
        )
        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            decision = router.decide("analyze this", budget_utilization=0.0)
        assert decision.domain == "analysis"
        assert decision.estimated_output_tokens == 600

    def test_heuristic_fallback_defaults_domain_and_eot(self):
        router = Router(use_llm_router=False)
        decision = router.decide("explain something", budget_utilization=0.0)
        assert decision.domain == "general"
        assert decision.estimated_output_tokens == 500

    def test_easy_decision_has_domain_and_eot(self):
        router = Router(use_llm_router=True)
        decision = router.decide("hi", budget_utilization=0.0)
        assert decision.domain == "general"
        assert decision.estimated_output_tokens == 100  # easy tasks are brief

    def test_force_small_preserves_domain_and_eot(self):
        router = Router(use_llm_router=False)
        original = RoutingDecision(
            tier=ModelTier.BIG,
            model="gpt-4o",
            confidence=0.9,
            complexity_score=0.8,
            reason="complex",
            forced_by_budget=False,
            domain="code",
            estimated_output_tokens=900)
        forced = router.force_small(original, "budget low")
        assert forced.domain == "code"
        assert forced.estimated_output_tokens == 900


# ─────────────────────────────────────────────────────────
# Injectable shared routing cache
# ─────────────────────────────────────────────────────────

class TestInjectableSharedCache:
    def test_external_cache_is_populated_on_first_call(self):
        shared = {}
        router = Router(use_llm_router=False, routing_cache=shared)
        router.decide("explain recursion", 0.0)
        assert len(shared) == 1

    def test_two_routers_share_cache_hit(self):
        shared = {}
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.5, "reason": "mid"}'
        task = "Summarize the differences between REST and GraphQL"

        r1 = Router(use_llm_router=True, routing_cache=shared)
        r2 = Router(use_llm_router=True, routing_cache=shared)

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            d1 = r1.decide(task, 0.0)
            d2 = r2.decide(task, 0.0)

        # Only one LLM call made across both router instances
        assert m.call_count == 1
        assert d1.routing_cache_hit is False
        assert d2.routing_cache_hit is True

    def test_external_cache_skips_lru_eviction(self):
        shared = {}
        router = Router(
            use_llm_router=False,
            routing_cache=shared,
            routing_cache_size=1,  # tiny internal LRU — should be ignored
        )
        # Fill with multiple tasks; external cache should keep all
        router.decide("task alpha", 0.0)
        router.decide("task beta", 0.0)
        router.decide("task gamma", 0.0)
        assert len(shared) == 3  # external dict not evicted

    def test_internal_cache_used_when_no_external_cache(self):
        router = Router(use_llm_router=False, routing_cache_size=256)
        d1 = router.decide("hello there", 0.0)
        d2 = router.decide("hello there", 0.0)
        assert d1.routing_cache_hit is False
        assert d2.routing_cache_hit is True


# ─────────────────────────────────────────────────────────
# BAARConfig / BAARRouter presets
# ─────────────────────────────────────────────────────────

class TestPresets:
    """Model-pair presets on BAARConfig and BAARRouter."""

    def test_anthropic_config_sets_claude_models(self):
        cfg = BAARConfig.anthropic(budget=0.05)
        assert cfg.small_model == "claude-haiku-4-5-20251001"
        assert cfg.big_model == "claude-sonnet-4-6"
        assert cfg.budget == 0.05

    def test_openai_config_sets_gpt_models(self):
        cfg = BAARConfig.openai(budget=0.20)
        assert cfg.small_model == "gpt-4o-mini"
        assert cfg.big_model == "gpt-4o"
        assert cfg.budget == 0.20

    def test_anthropic_config_allows_model_override(self):
        cfg = BAARConfig.anthropic(big_model="claude-opus-4-7")
        assert cfg.small_model == "claude-haiku-4-5-20251001"
        assert cfg.big_model == "claude-opus-4-7"

    def test_openai_config_allows_model_override(self):
        cfg = BAARConfig.openai(small_model="gpt-4o-mini", big_model="o1")
        assert cfg.big_model == "o1"

    def test_anthropic_router_shortcut(self):
        router = BAARRouter.anthropic(budget=0.10, use_llm_router=False)
        assert router.small_model == "claude-haiku-4-5-20251001"
        assert router.big_model == "claude-sonnet-4-6"
        assert router.budget == 0.10

    def test_openai_router_shortcut(self):
        router = BAARRouter.openai(budget=0.50, use_llm_router=False)
        assert router.small_model == "gpt-4o-mini"
        assert router.big_model == "gpt-4o"
        assert router.budget == 0.50

    def test_presets_preserve_other_defaults(self):
        cfg = BAARConfig.anthropic()
        assert cfg.complexity_threshold == 0.80
        assert cfg.routing_cache_enabled is True


class TestCoverageGaps:
    """Covers lines in baar/core/router.py not yet exercised elsewhere."""

    def setup_method(self):
        self.router = Router(use_llm_router=False)

    def test_unit_conversion_fast_path(self):
        decision = self.router.decide("32°F to Celsius", 0.0)
        assert decision.tier == ModelTier.SMALL
        assert "unit conversion" in decision.reason

    def test_heuristic_medium_task_21_to_50_words(self):
        # 27 words, no code/reasoning/multi-choice triggers
        task = (
            "What are the most popular tourist destinations in Europe "
            "I want to visit several countries next year and would like "
            "some recommendations for interesting places to see"
        )
        decision = self.router.decide(task, 0.0)
        assert "medium task" in decision.reason

    @pytest.mark.asyncio
    async def test_async_routing_cache_hit_returns_true(self):
        """Second identical async decision hits the cache (lines 409-410)."""
        task = "What is the boiling point of water?"
        await self.router.adecide(task, 0.0)
        d2 = await self.router.adecide(task, 0.0)
        assert d2.routing_cache_hit is True

    def test_sync_routing_timeout_forwarded_to_llm(self):
        """routing_timeout is added to kwargs in _compute_complexity_uncached (line 450)."""
        with patch("baar.core.router.litellm.completion") as mock_comp:
            mock_comp.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content='{"complexity":0.8,"reason":"hard","domain":"general","estimated_output_tokens":500}'
                ))]
            )
            r = Router(use_llm_router=True, routing_timeout=7.0)
            r._compute_complexity_uncached("Write a complex distributed system")
        _, kwargs = mock_comp.call_args
        assert kwargs.get("timeout") == 7.0

    @pytest.mark.asyncio
    async def test_async_routing_timeout_forwarded_to_llm(self):
        """routing_timeout is added to kwargs in _acompute_complexity_uncached (line 473)."""
        with patch("baar.core.router.litellm.acompletion") as mock_acomp:
            mock_acomp.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content='{"complexity":0.8,"reason":"hard","domain":"general","estimated_output_tokens":500}'
                ))]
            )
            r = Router(use_llm_router=True, routing_timeout=7.0)
            await r._acompute_complexity_uncached("Write a complex distributed system")
        _, kwargs = mock_acomp.call_args
        assert kwargs.get("timeout") == 7.0
