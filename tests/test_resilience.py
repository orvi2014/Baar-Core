"""
tests/test_resilience.py — The "Zero-Call" Proof.
Ensures that zero network calls are made if the budget is insufficient.
"""
import pytest
from unittest.mock import patch
from baar import BAARRouter, BudgetExceeded

def test_zero_budget_blocks_all_calls():
    """
    Prove that if budget is 0, no network calls are initiated.
    This is the core 'Bullet-Proof' safety guarantee.
    """
    router = BAARRouter(budget=1.0) # Start with something
    router._tracker.total_budget = 0.0 # Force zero budget
    
    with patch("litellm.completion") as mock_completion:
        with pytest.raises((BudgetExceeded, RuntimeError)):
            router.chat("This should be blocked before calling the API.")
        
        # PROOF: confirm litellm.completion was NEVER called
        mock_completion.assert_not_called()

def test_insufficient_budget_blocks_big_model():
    """
    Prove that if budget is too low for the BIG model, it is blocked 
    OR downgraded before the call.
    """
    # Set a tiny budget that can only afford a small model
    # (assuming BIG costs 0.01 per call, SMALL costs 0.00001)
    router = BAARRouter(budget=0.001) 
    
    with patch("litellm.completion") as mock_completion:
        # Mocking small cost to be very low so it passes BCD if possible
        with patch("baar.core.budget.completion_cost", return_value=0.00001):
            router.chat("Simple task")
            
            # The BIG model should NEVER have been called
            for call in mock_completion.call_args_list:
                model_used = call.kwargs.get("model") or call.args[0]
                assert model_used == "gpt-4o-mini"
                assert model_used != "gpt-4o"

def test_extreme_token_inflation_blocks_preflight():
    """
    Prove that a massive prompt is blocked before the call even if budget exists.
    (e.g., 1,000,000 tokens for a $0.05 budget)
    """
    router = BAARRouter(budget=0.05)
    
    # Create a 1M token-equivalent prompt (roughly)
    huge_task = "A " * 500000 
    
    with patch("litellm.completion") as mock_completion:
        with pytest.raises((BudgetExceeded, RuntimeError)):
            router.chat(huge_task)
            
        # PROOF: The cost prediction (BCD) stopped it before the API call
        mock_completion.assert_not_called()


def test_killswitch_below_threshold_has_clear_local_rejection_message():
    """If remaining budget is below threshold, fail locally with clear details."""
    router = BAARRouter(budget=1.0)
    router._tracker._spent = 0.99995  # remaining = 0.00005 (< 0.0001)

    with patch("litellm.completion") as mock_completion:
        with pytest.raises(RuntimeError) as exc:
            router.chat("hello")

        msg = str(exc.value)
        assert "Kill-switch activated" in msg
        assert "$0.000050" in msg
        assert "zero network calls" in msg
        mock_completion.assert_not_called()


def test_affordability_failure_at_threshold_is_wrapped_as_local_rejection():
    """At the exact threshold, cheapest-call affordability failure is wrapped clearly."""
    router = BAARRouter(budget=1.0)
    router._tracker._spent = 0.999899  # remaining ~= 0.000101 (just above threshold)

    with patch("litellm.completion") as mock_completion, \
         patch("baar.router.token_counter", return_value=50), \
         patch.object(router._tracker, "estimate_cost", return_value=0.0), \
         patch.object(
             router._tracker,
             "check_affordability",
             side_effect=BudgetExceeded(
                 requested=0.001,
                 remaining=router.remaining,
                 model=router.small_model,
             ),
         ):
        with pytest.raises(RuntimeError) as exc:
            router.chat("hello")

        msg = str(exc.value)
        assert "cheapest safe call" in msg
        assert router.small_model in msg
        assert "zero network calls" in msg
        mock_completion.assert_not_called()


def test_configurable_threshold_blocks_locally_even_when_default_would_not():
    """A higher configured floor should hard-stop locally before routing."""
    router = BAARRouter(budget=0.001, min_cost_threshold=0.001)
    router._tracker._spent = 0.0005  # remaining = 0.0005 (above old default, below configured floor)

    with patch("litellm.completion") as mock_completion:
        with pytest.raises(RuntimeError) as exc:
            router.chat("hello")

        msg = str(exc.value)
        assert "configured floor $0.001000" in msg
        assert "zero network calls" in msg
        mock_completion.assert_not_called()


def test_dynamic_floor_guard_blocks_when_estimated_small_cost_is_higher():
    """Future guard: dynamic floor uses estimated cheapest-call cost when higher than config."""
    router = BAARRouter(budget=1.0, min_cost_threshold=0.0)
    router._tracker._spent = 0.9998  # remaining = 0.0002

    with patch("litellm.completion") as mock_completion, \
         patch("baar.router.token_counter", return_value=100), \
         patch.object(router._tracker, "estimate_cost", return_value=0.0003), \
         patch.object(router._tracker, "check_affordability") as mock_affordability:
        with pytest.raises(RuntimeError) as exc:
            router.chat("hello")

        msg = str(exc.value)
        assert "Effective preflight floor is $0.000300" in msg
        assert "configured floor $0.000000" in msg
        # This path should fail before affordability check is reached.
        mock_affordability.assert_not_called()
        mock_completion.assert_not_called()
