"""
tests/test_router.py — Router unit tests

Tests heuristic scoring, LLM routing decisions,
budget pressure effects, and forced downgrades.
All tests run without API keys.
"""

import pytest
from unittest.mock import MagicMock, patch
from baar.core.router import Router, ModelTier, RoutingDecision


# ─────────────────────────────────────────────────────────
# Heuristic scoring (no API calls)
# ─────────────────────────────────────────────────────────

class TestHeuristicScoring:
    """Tests heuristic fallback — no LLM calls."""

    def setup_method(self):
        # use_llm_router=False → always uses heuristic
        self.router = Router(use_llm_router=False)

    def test_simple_greeting_routes_to_small(self):
        decision = self.router.decide("hi", remaining_budget=1.0, budget_utilization=0.0)
        assert decision.tier == ModelTier.SMALL

    def test_factual_short_routes_to_small(self):
        decision = self.router.decide("What is 2+2?", remaining_budget=1.0, budget_utilization=0.0)
        assert decision.tier == ModelTier.SMALL

    def test_code_task_routes_to_big(self):
        task = "def parse_json(): # Debug this function and add error handling"
        decision = self.router.decide(task, remaining_budget=1.0, budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_analysis_task_routes_to_big(self):
        task = "Analyze and compare the trade-offs between microservices and monolithic architecture"
        decision = self.router.decide(task, remaining_budget=1.0, budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_structured_math_prompt_routes_to_big(self):
        task = (
            "A store sells pencils in packs of 3. If Ana buys 7 packs, how many pencils does she get?\n\n"
            "Show your work and end with 'The answer is #### <number>'."
        )
        decision = self.router.decide(task, remaining_budget=1.0, budget_utilization=0.0)
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
        decision = self.router.decide(task, remaining_budget=1.0, budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_easy_pattern_capital_routes_to_small(self):
        decision = self.router.decide(
            "What is the capital of Japan?",
            remaining_budget=1.0,
            budget_utilization=0.0,
        )
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
        decision = self.router.decide("hello", remaining_budget=1.0, budget_utilization=0.0)
        assert len(decision.reason) > 0

    def test_decision_model_matches_tier(self):
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o", use_llm_router=False)
        for task in ["hi", "complex analysis with multiple steps and reasoning"]:
            decision = router.decide(task, remaining_budget=1.0, budget_utilization=0.0)
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
            use_llm_router=False,
        )

    def test_high_utilization_raises_threshold(self):
        """At 85% budget used, even moderately complex tasks go to SMALL."""
        # A task that would normally go to BIG at low utilization
        task = "Explain the concept of recursion with examples"

        decision_low_util = self.router.decide(task, remaining_budget=1.0, budget_utilization=0.1)
        decision_high_util = self.router.decide(task, remaining_budget=0.15, budget_utilization=0.85)

        # High utilization should either keep same or downgrade — never upgrade
        if decision_low_util.tier == ModelTier.SMALL:
            assert decision_high_util.tier == ModelTier.SMALL
        # (if originally BIG, high utilization may downgrade it)

    def test_near_zero_budget_always_routes_small(self):
        """When almost out of budget, always use SMALL regardless of complexity."""
        complex_task = "Write a comprehensive analysis with code generation and debugging"
        decision = self.router.decide(
            complex_task,
            remaining_budget=0.001,
            budget_utilization=0.99,
        )
        assert decision.tier == ModelTier.SMALL

    def test_full_budget_allows_big_for_complex(self):
        """With full budget, complex tasks should route to BIG."""
        complex_task = "debug this complex recursive algorithm and explain each step"
        decision = self.router.decide(
            complex_task,
            remaining_budget=10.0,
            budget_utilization=0.0,
        )
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
            forced_by_budget=False,
        )
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
            forced_by_budget=False,
        )
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
            routing_cache_hit=True,
        )
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
            score, reason = router._llm_score("complex task")
            assert 0.0 <= score <= 1.0
            assert score < 0.8  # normalized (squared + small length correction)
            assert "reasoning" in reason

    def test_llm_router_handles_markdown_code_block(self):
        """Some models wrap JSON in ```json ... ```"""
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '```json\n{"complexity": 0.6, "reason": "medium"}\n```'

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            score, reason = router._llm_score("medium task")
            assert 0.0 <= score <= 1.0
            assert score < 0.6

    def test_llm_router_clamps_score_to_valid_range(self):
        """Score from LLM must be clamped to [0, 1]."""
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 1.5, "reason": "extreme"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            score, _ = router._llm_score("extreme task")
            assert 0.95 <= score <= 1.0

    def test_llm_router_applies_length_penalty(self):
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.9, "reason": "hard"}'
        short_task = "hard question"
        long_task = "x" * 1500

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            short_score, _ = router._llm_score(short_task)
            long_score, _ = router._llm_score(long_task)
            assert long_score < short_score

    def test_llm_router_falls_back_on_api_error(self):
        """If LLM call fails, falls back to heuristic — never crashes."""
        router = Router(use_llm_router=True, complexity_threshold=0.65)

        with patch("baar.core.router.litellm.completion", side_effect=Exception("API down")):
            # Should use heuristic fallback
            decision = router.decide(
                "hello world",
                remaining_budget=1.0,
                budget_utilization=0.0,
            )
            assert decision.tier in (ModelTier.SMALL, ModelTier.BIG)  # got a valid answer

    def test_llm_router_falls_back_on_json_parse_error(self):
        """If LLM returns invalid JSON, falls back to heuristic."""
        router = Router(use_llm_router=True)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I think this is complex."

        with patch("baar.core.router.litellm.completion", return_value=mock_response):
            decision = router.decide("some task", remaining_budget=1.0, budget_utilization=0.0)
            assert decision.tier in (ModelTier.SMALL, ModelTier.BIG)


# ─────────────────────────────────────────────────────────
# Complexity LRU (routing) cache
# ─────────────────────────────────────────────────────────

class TestRoutingCache:
    def test_llm_second_identical_task_one_completion_call(self):
        router = Router(use_llm_router=True, routing_cache_size=256)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.5, "reason": "mid"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            d1 = router.decide("hello", remaining_budget=1.0, budget_utilization=0.0)
            d2 = router.decide("hello", remaining_budget=1.0, budget_utilization=0.0)

        assert m.call_count == 1
        assert d1.routing_cache_hit is False
        assert d2.routing_cache_hit is True
        assert d1.complexity_score == d2.complexity_score

    def test_whitespace_normalized_same_cache_key(self):
        router = Router(use_llm_router=True, routing_cache_size=256)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.3, "reason": "x"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            router.decide("  hello   world  ", 1.0, 0.0)
            router.decide("hello world", 1.0, 0.0)

        assert m.call_count == 1

    def test_cache_disabled_invokes_scorer_each_time(self):
        router = Router(
            use_llm_router=True,
            routing_cache_enabled=False,
        )
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.4, "reason": "x"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            router.decide("same", 1.0, 0.0)
            router.decide("same", 1.0, 0.0)

        assert m.call_count == 2

    def test_cache_size_zero_disables_cache(self):
        router = Router(use_llm_router=True, routing_cache_size=0)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"complexity": 0.4, "reason": "x"}'

        with patch("baar.core.router.litellm.completion", return_value=mock_response) as m:
            router.decide("same", 1.0, 0.0)
            router.decide("same", 1.0, 0.0)

        assert m.call_count == 2

    def test_tier_recomputed_on_cache_hit_when_utilization_changes(self):
        router = Router(use_llm_router=False, complexity_threshold=0.65)
        # Heuristic ~0.7: BIG at low utilization, SMALL once cost pressure raises the bar
        task = "Explain the concept of recursion with examples"
        d_low = router.decide(task, 1.0, 0.0)
        d_high = router.decide(task, 1.0, 0.85)
        assert d_low.tier == ModelTier.BIG
        assert d_high.tier == ModelTier.SMALL
        assert d_high.routing_cache_hit is True

    def test_heuristic_scorer_also_cached(self):
        router = Router(use_llm_router=False, routing_cache_size=256)
        d1 = router.decide("hello there", 1.0, 0.0)
        d2 = router.decide("hello there", 1.0, 0.0)
        assert d1.routing_cache_hit is False
        assert d2.routing_cache_hit is True


class TestExploration:
    def test_exploration_can_downshift_big_to_small(self):
        router = Router(use_llm_router=False, small_exploration_rate=0.1)
        task = "def parse_json(): # Debug this function and add error handling"
        with patch("baar.core.router.random.random", return_value=0.01):
            decision = router.decide(task, 1.0, 0.0)
        assert decision.tier == ModelTier.SMALL
        assert "EXPLORE SMALL" in decision.reason


# ─────────────────────────────────────────────────────────
# Custom thresholds
# ─────────────────────────────────────────────────────────

class TestCustomThresholds:
    def test_very_low_threshold_routes_almost_everything_to_big(self):
        router = Router(complexity_threshold=0.1, use_llm_router=False)
        decision = router.decide("explain recursion", remaining_budget=1.0, budget_utilization=0.0)
        assert decision.tier == ModelTier.BIG

    def test_very_high_threshold_routes_almost_everything_to_small(self):
        router = Router(complexity_threshold=0.99, use_llm_router=False)
        decision = router.decide("write complex recursive fibonacci", remaining_budget=1.0, budget_utilization=0.0)
        assert decision.tier == ModelTier.SMALL
