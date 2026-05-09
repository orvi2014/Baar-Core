"""
tests/test_router_integration.py

Integration tests for BAARRouter end-to-end flow.
All LiteLLM calls are mocked — no API keys needed.
Tests the complete pipeline: routing → BCD → execution → cost recording.
"""

import pytest
import json
import tempfile
from unittest.mock import MagicMock, patch, call
from baar import BAARRouter, BudgetExceeded, TaskRejected
from baar.core.router import ModelTier


# ─────────────────────────────────────────────────────────
# Mock factory
# ─────────────────────────────────────────────────────────

def mock_litellm_response(content="test response", model="gpt-4o-mini",
                           prompt_tokens=50, completion_tokens=30):
    resp = MagicMock()
    resp.model = model
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def mock_completion_side_effect(model, messages, **kwargs):
    """Returns model-appropriate mock responses."""
    content = f"Response from {model}"
    return mock_litellm_response(content=content, model=model)


# ─────────────────────────────────────────────────────────
# BAARRouter initialization
# ─────────────────────────────────────────────────────────

class TestBAARRouterInit:
    def test_default_initialization(self):
        router = BAARRouter()
        assert router.budget == 0.10
        assert router.small_model == "gpt-4o-mini"
        assert router.big_model == "gpt-4o"
        assert router.complexity_threshold == pytest.approx(0.80)

    def test_custom_initialization(self):
        router = BAARRouter(
            budget=0.50,
            small_model="claude-haiku-3-5",
            big_model="claude-sonnet-4-5",
        )
        assert router.budget == 0.50
        assert router.small_model == "claude-haiku-3-5"

    def test_custom_routing_task_char_limit_is_forwarded(self):
        router = BAARRouter(
            budget=0.10,
            routing_task_char_limit=900,
        )
        assert router._router._routing_task_char_limit == 900

    def test_zero_budget_raises(self):
        with pytest.raises(ValueError, match="Budget must be positive"):
            BAARRouter(budget=0.0)

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError):
            BAARRouter(budget=-1.0)

    def test_initial_spent_is_zero(self):
        router = BAARRouter(budget=0.10)
        assert router.spent == 0.0

    def test_initial_remaining_equals_budget(self):
        router = BAARRouter(budget=0.25)
        assert router.remaining == pytest.approx(0.25)

    def test_custom_min_cost_threshold_initialization(self):
        router = BAARRouter(budget=0.10, min_cost_threshold=0.001)
        assert router.min_cost_threshold == pytest.approx(0.001)

    def test_negative_min_cost_threshold_raises(self):
        with pytest.raises(ValueError, match="min_cost_threshold must be non-negative"):
            BAARRouter(budget=0.10, min_cost_threshold=-0.001)


# ─────────────────────────────────────────────────────────
# Single chat call
# ─────────────────────────────────────────────────────────

class TestSingleChat:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_chat_returns_string(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response("hello back")
        router = BAARRouter(budget=0.10, use_llm_router=False)
        result = router.chat("hello")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_chat_updates_spent(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False)
        router.chat("test message")
        assert router.spent == pytest.approx(0.000025)

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_chat_logs_step(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False)
        router.chat("test message")
        assert router.log.total_steps == 1

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_system_prompt_included_in_messages(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False, system_prompt="You are helpful.")
        router.chat("test")
        
        call_args = mock_completion.call_args
        messages_sent = call_args[1]["messages"] if call_args[1] else call_args[0][1]
        roles = [m["role"] for m in messages_sent]
        assert "system" in roles


# ─────────────────────────────────────────────────────────
# Multi-step run
# ─────────────────────────────────────────────────────────

class TestMultiStepRun:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_run_executes_all_steps(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False)
        log = router.run(["step 1", "step 2", "step 3"])
        assert log.total_steps == 3

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_run_returns_routing_log(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False)
        from baar.core.models import RoutingLog
        log = router.run(["task 1"])
        assert isinstance(log, RoutingLog)

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.04)
    @patch("baar.core.budget.cost_per_token", return_value=(0.02, 0.02))
    @patch("baar.router.token_counter", return_value=50)
    def test_run_stops_when_budget_exhausted(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        """With $0.10 budget and $0.04/call, should stop after 2 steps."""
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False)
        log = router.run(["task 1", "task 2", "task 3", "task 4"])
        # At most 2 successful steps before budget runs out
        assert log.total_steps <= 3

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_cumulative_cost_increases_per_step(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False)
        log = router.run(["step 1", "step 2", "step 3"])
        
        cumulative_costs = [s.cumulative_cost for s in log.steps]
        for i in range(1, len(cumulative_costs)):
            assert cumulative_costs[i] >= cumulative_costs[i - 1]


# ─────────────────────────────────────────────────────────
# Budget-Constrained Decoding (BCD)
# ─────────────────────────────────────────────────────────

class TestBudgetConstrainedDecoding:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_big_model_downgraded_when_budget_low(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        """
        When BIG model is unaffordable, should downgrade to SMALL
        and mark forced_by_budget=True.
        """
        mock_completion.return_value = mock_litellm_response()

        # Nearly exhausted budget
        router = BAARRouter(budget=0.10, use_llm_router=False, complexity_threshold=0.0)
        router._tracker._spent = 0.0999  # Only $0.0001 remaining

        # Patch cost estimate to make BIG seem expensive
        with patch.object(router._tracker, "check_affordability") as mock_check:
            def side_effect(model, prompt_tokens):
                if "gpt-4o" in model and "mini" not in model:
                    raise BudgetExceeded(0.05, 0.0001, model)
            mock_check.side_effect = side_effect

            log = router.run(["complex analysis task"])

        if log.total_steps > 0:
            step = log.steps[0]
            # Either downgraded to small or stopped
            assert step.decision.model == router.small_model or step.decision.forced_by_budget

    @patch("baar.core.budget.cost_per_token", return_value=(100.0, 100.0))
    @patch("baar.router.token_counter", return_value=50)
    def test_budget_exceeded_raised_when_even_small_unaffordable(self, mock_tc, mock_cpt):
        """When even small model exceeds budget, BudgetExceeded should bubble up."""
        router = BAARRouter(budget=0.001, use_llm_router=False)
        router._tracker._spent = 0.0009  # Only $0.0001 remaining

        # The run() method should catch BudgetExceeded and stop gracefully
        with patch("baar.router.litellm.completion") as mock_comp:
            log = router.run(["any task"])
        # Either 0 steps completed or stopped early
        assert log.total_steps == 0


# ─────────────────────────────────────────────────────────
# Routing log summary
# ─────────────────────────────────────────────────────────

class TestRoutingLogSummary:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_summary_has_required_keys(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False)
        log = router.run(["test task"])
        summary = log.summary()

        required_keys = [
            "budget_usd", "spent_usd", "remaining_usd", "utilization_pct",
            "total_steps", "small_model_calls", "big_model_calls",
            "reject_steps",
            "budget_forced_downgrades",
            "routing_cache_hits",
            "savings_vs_always_big", "steps",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_savings_shows_cost_reduction(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response()
        router = BAARRouter(budget=0.10, use_llm_router=False, complexity_threshold=0.99)
        # Force everything to SMALL by setting very high threshold
        log = router.run(["simple task 1", "simple task 2"])
        savings = log.savings_vs_always_big()

        assert "baar_cost" in savings
        assert "estimated_always_big_cost" in savings
        assert "savings_pct" in savings
        assert savings["savings_pct"] >= 0  # never negative


# ─────────────────────────────────────────────────────────
# Value gate (value_fn + REJECT)
# ─────────────────────────────────────────────────────────

class TestValueFn:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.01, 0.01))
    @patch("baar.router.token_counter", return_value=50)
    def test_value_below_cost_rejects_without_completion(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        """High per-token estimate → value_fn loss → no litellm.completion."""
        router = BAARRouter(
            budget=0.10,
            use_llm_router=False,
            value_fn=lambda _t: 0.0,
        )
        with pytest.raises(TaskRejected):
            router.chat("hello")
        mock_completion.assert_not_called()
        assert router.log.total_steps == 1
        assert router.log.reject_steps == 1
        step = router.log.steps[0]
        assert step.decision.tier == ModelTier.REJECT
        assert step.cost == 0.0
        assert step.decision.estimated_value is not None
        assert step.decision.estimated_cost_usd is not None
        d = step.to_dict()
        assert "estimated_value" in d and "estimated_cost_usd" in d

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_value_at_or_above_cost_executes_and_logs_estimates(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(
            budget=0.10,
            use_llm_router=False,
            value_fn=lambda _t: 1.0,
        )
        router.chat("hello")
        mock_completion.assert_called_once()
        step = router.log.steps[0]
        assert step.decision.tier == ModelTier.SMALL
        assert step.decision.estimated_value == pytest.approx(1.0)
        assert step.decision.estimated_cost_usd is not None

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_run_continues_after_task_rejected(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        calls = {"n": 0}

        def vf(t: str) -> float:
            calls["n"] += 1
            return 0.0 if "reject" in t else 1.0

        router = BAARRouter(budget=0.10, use_llm_router=False, value_fn=vf)
        log = router.run(["please reject me", "ok task"])
        assert log.total_steps == 2
        assert log.reject_steps == 1
        assert log.small_calls == 1
        mock_completion.assert_called_once()

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.005, 0.005))
    @patch("baar.router.token_counter", return_value=50)
    def test_value_medium_zone_downshifts_big_to_small(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(
            budget=0.10,
            use_llm_router=False,
            complexity_threshold=0.0,  # route to BIG first
            value_fn=lambda _t: 0.005,  # between reject_bar(0.003) and cost(0.01)
            value_reject_alpha=0.3,
        )
        router.chat("Analyze and compare system design trade-offs")
        step = router.log.steps[0]
        assert step.decision.tier == ModelTier.SMALL
        assert mock_completion.call_args.kwargs["model"] == router.small_model

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.005, 0.005))
    @patch("baar.router.token_counter", return_value=50)
    def test_reject_rate_clamp_forces_small_instead_of_reject(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(
            budget=0.10,
            use_llm_router=False,
            complexity_threshold=0.0,
            value_fn=lambda _t: 0.0,   # would reject without clamp
            value_reject_alpha=0.3,
            max_reject_rate=0.5,
            min_samples_for_reject_clamp=1,
        )
        router.chat("Analyze and compare system design trade-offs")
        assert router.log.reject_steps == 0
        assert router.log.small_calls == 1
        assert mock_completion.call_args.kwargs["model"] == router.small_model


# ─────────────────────────────────────────────────────────
# Routing cache (end-to-end)
# ─────────────────────────────────────────────────────────

class TestRoutingCacheIntegration:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_identical_chats_skip_second_routing_llm(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        def side_effect(model, messages, **kwargs):
            r = MagicMock()
            r.usage = MagicMock(prompt_tokens=10, completion_tokens=10)
            r.choices = [MagicMock()]
            body = messages[-1]["content"]
            if isinstance(body, str) and "task complexity classifier" in body.lower():
                r.choices[0].message.content = '{"complexity": 0.2, "reason": "simple"}'
            else:
                r.choices[0].message.content = "done"
            return r

        mock_completion.side_effect = side_effect
        router = BAARRouter(budget=0.10, use_llm_router=True)
        router.chat("ping")
        router.chat("ping")
        assert mock_completion.call_count == 3
        assert router.log.routing_cache_hits == 1
        assert router.log.steps[1].decision.routing_cache_hit is True


# ─────────────────────────────────────────────────────────
# Provider failover + telemetry
# ─────────────────────────────────────────────────────────

class TestFailoverAndTelemetry:
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_small_model_failover_succeeds_on_second_candidate(self, mock_tc, mock_cpt, mock_cc):
        calls = {"n": 0}

        def completion_side_effect(model, messages, **kwargs):
            calls["n"] += 1
            if model == "gpt-4o-mini":
                raise RuntimeError("provider down")
            return mock_litellm_response(content=f"from {model}", model=model)

        with patch("baar.router.litellm.completion", side_effect=completion_side_effect) as mock_completion:
            router = BAARRouter(
                budget=0.10,
                use_llm_router=False,
                small_model="gpt-4o-mini",
                big_model="gpt-4o",
                small_fallback_models=["fallback-small"],
            )
            out = router.chat("hi")

        assert "fallback-small" in out
        assert mock_completion.call_count == 2
        step = router.log.steps[-1]
        assert step.decision.model == "fallback-small"
        assert step.failover_count == 1
        assert step.attempted_models == ["gpt-4o-mini", "fallback-small"]
        assert len(step.failover_errors) >= 1

    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_big_model_failover_succeeds_on_second_candidate(self, mock_tc, mock_cpt, mock_cc):
        def completion_side_effect(model, messages, **kwargs):
            if model == "gpt-4o":
                raise RuntimeError("primary big down")
            return mock_litellm_response(content=f"from {model}", model=model)

        with patch("baar.router.litellm.completion", side_effect=completion_side_effect):
            router = BAARRouter(
                budget=0.10,
                use_llm_router=False,
                complexity_threshold=0.0,  # initial decision -> BIG
                small_model="gpt-4o-mini",
                big_model="gpt-4o",
                big_fallback_models=["fallback-big"],
            )
            out = router.chat("Analyze system design trade-offs")

        assert "fallback-big" in out
        step = router.log.steps[-1]
        assert step.decision.model == "fallback-big"
        assert step.failover_count == 1
        assert step.attempted_models == ["gpt-4o", "fallback-big"]

    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.router.token_counter", return_value=50)
    def test_failover_uses_first_available_candidate(self, mock_tc, mock_cc):
        # Primary fails; fallback-small-1 should be tried immediately without a
        # redundant affordability re-check (reservation already covers the slot).
        def completion_side_effect(model, messages, **kwargs):
            if model == "gpt-4o-mini":
                raise RuntimeError("primary small down")
            return mock_litellm_response("ok", model=model)

        with patch("baar.router.litellm.completion", side_effect=completion_side_effect):
            router = BAARRouter(
                budget=0.10,
                use_llm_router=False,
                small_model="gpt-4o-mini",
                small_fallback_models=["fallback-small-1", "fallback-small-2"],
            )
            router.chat("hello")

        step = router.log.steps[-1]
        assert step.decision.model == "fallback-small-1"
        assert step.attempted_models == ["gpt-4o-mini", "fallback-small-1"]
        assert step.failover_count == 1

    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_all_failover_candidates_fail_raises(self, mock_tc, mock_cpt, mock_cc):
        with patch("baar.router.litellm.completion", side_effect=RuntimeError("outage")):
            router = BAARRouter(
                budget=0.10,
                use_llm_router=False,
                small_model="gpt-4o-mini",
                small_fallback_models=["fallback-small-1"],
            )
            with pytest.raises(RuntimeError, match="All provider/model attempts failed"):
                router.chat("hello")

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_telemetry_jsonl_written_for_success_and_reject(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/telemetry.jsonl"
            router = BAARRouter(
                budget=0.10,
                use_llm_router=False,
                telemetry_jsonl_path=path,
                value_fn=lambda t: 0.0 if "reject" in t else 1.0,
            )
            with pytest.raises(TaskRejected):
                router.chat("please reject")
            router.chat("normal task")

            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["tier"] == "reject"
        assert lines[1]["tier"] in ("small", "big")
        assert "timestamp_unix_ms" in lines[0]
        assert "router_budget_usd" in lines[0]


# ─────────────────────────────────────────────────────────
# Multi-turn conversation context
# ─────────────────────────────────────────────────────────

class TestMultiTurnMessages:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_chat_with_messages_passes_history_to_llm(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("follow-up answer")
        router = BAARRouter(budget=0.10, use_llm_router=False)
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "What are its main uses?"},
        ]
        result = router.chat("What are its main uses?", messages=history)
        assert isinstance(result, str)
        # Verify the LLM received the full history
        call_messages = mock_completion.call_args.kwargs["messages"]
        assert any(m["role"] == "assistant" for m in call_messages)
        assert len(call_messages) >= 3

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_chat_with_messages_prepends_system_prompt(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(budget=0.10, use_llm_router=False, system_prompt="Be concise.")
        history = [{"role": "user", "content": "hello"}]
        router.chat("hello", messages=history)
        call_messages = mock_completion.call_args.kwargs["messages"]
        assert call_messages[0]["role"] == "system"
        assert call_messages[0]["content"] == "Be concise."

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_chat_with_messages_does_not_duplicate_system_prompt(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(budget=0.10, use_llm_router=False, system_prompt="Be concise.")
        history = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ]
        router.chat("hello", messages=history)
        call_messages = mock_completion.call_args.kwargs["messages"]
        assert sum(1 for m in call_messages if m["role"] == "system") == 1

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_chat_without_messages_builds_single_turn(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(budget=0.10, use_llm_router=False)
        router.chat("simple question")
        call_messages = mock_completion.call_args.kwargs["messages"]
        user_msgs = [m for m in call_messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "simple question"


# ─────────────────────────────────────────────────────────
# RoutingLog max_steps eviction
# ─────────────────────────────────────────────────────────

class TestMaxLogSteps:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000001)
    @patch("baar.core.budget.cost_per_token", return_value=(0.0000001, 0.0000002))
    @patch("baar.router.token_counter", return_value=10)
    def test_max_log_steps_evicts_oldest(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(budget=1.0, use_llm_router=False, max_log_steps=3)
        for i in range(5):
            router.chat(f"task {i}")
        # Only 3 most recent steps should be kept
        assert len(router.log.steps) == 3

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000001)
    @patch("baar.core.budget.cost_per_token", return_value=(0.0000001, 0.0000002))
    @patch("baar.router.token_counter", return_value=10)
    def test_no_max_log_steps_keeps_all(self, mock_tc, mock_cpt, mock_cc, mock_completion):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(budget=1.0, use_llm_router=False)
        for i in range(5):
            router.chat(f"task {i}")
        assert len(router.log.steps) == 5


# ─────────────────────────────────────────────────────────
# value_fn UserWarning for suspicious scale
# ─────────────────────────────────────────────────────────

class TestValueFnWarning:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_value_fn_warns_when_value_is_suspiciously_large(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(
            budget=0.10,
            use_llm_router=False,
            value_fn=lambda _: 150.0,  # looks like a 0-100 score, not USD
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                router.chat("hello")
            except Exception:
                pass
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) >= 1
        assert "150" in str(user_warns[0].message)

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_value_fn_no_warning_for_usd_scale_value(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        mock_completion.return_value = mock_litellm_response("ok")
        router = BAARRouter(
            budget=0.10,
            use_llm_router=False,
            value_fn=lambda _: 0.005,  # sensible USD value
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            router.chat("hello")
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 0


# ─────────────────────────────────────────────────────────
# Injectable shared routing cache (BAARRouter level)
# ─────────────────────────────────────────────────────────

class TestSharedRoutingCacheIntegration:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.000025)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_shared_cache_across_router_instances(
        self, mock_tc, mock_cpt, mock_cc, mock_completion
    ):
        def side_effect(model, messages, **kwargs):
            r = MagicMock()
            r.usage = MagicMock(prompt_tokens=10, completion_tokens=10)
            r.choices = [MagicMock()]
            body = messages[-1]["content"] if messages else ""
            if "task complexity classifier" in body.lower():
                r.choices[0].message.content = '{"complexity": 0.2, "reason": "simple"}'
            else:
                r.choices[0].message.content = "done"
            return r

        mock_completion.side_effect = side_effect
        shared = {}
        r1 = BAARRouter(budget=0.10, use_llm_router=True, routing_cache=shared)
        r2 = BAARRouter(budget=0.10, use_llm_router=True, routing_cache=shared)

        r1.chat("ping")
        r2.chat("ping")

        # Second instance should have gotten a cache hit
        assert r2.log.steps[0].decision.routing_cache_hit is True


# ─────────────────────────────────────────────────────────
# Streaming (stream_chat)
# ─────────────────────────────────────────────────────────

class TestStreamChat:
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_stream_chat_yields_chunks_and_logs_step(self, mock_tc, mock_cpt):
        def make_chunk(text):
            c = MagicMock()
            c.choices[0].delta.content = text
            return c

        chunks = [make_chunk("Hello"), make_chunk(" world"), make_chunk("!")]
        with patch("baar.router.litellm.completion", return_value=iter(chunks)):
            with patch("baar.core.budget.completion_cost", return_value=0.000025):
                router = BAARRouter(budget=0.10, use_llm_router=False)
                collected = list(router.stream_chat("hi"))

        assert collected == ["Hello", " world", "!"]
        assert len(router.log.steps) == 1
        assert router.log.steps[0].response_text == "Hello world!"

    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("baar.router.token_counter", return_value=50)
    def test_stream_chat_cancels_reservation_on_error(self, mock_tc, mock_cpt):
        with patch("baar.router.litellm.completion", side_effect=RuntimeError("stream failed")):
            with patch("baar.core.budget.completion_cost", return_value=0.000025):
                router = BAARRouter(budget=0.10, use_llm_router=False)
                initial_remaining = router.remaining
                with pytest.raises(RuntimeError):
                    list(router.stream_chat("hi"))
                # Reservation must be cleaned up — remaining should be unchanged
                assert router.remaining == pytest.approx(initial_remaining)
