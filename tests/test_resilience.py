"""
tests/test_resilience.py — The "Zero-Call" Proof.
Ensures that zero network calls are made if the budget is insufficient.
"""
import pytest
from unittest.mock import patch, MagicMock
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
        with pytest.raises(BudgetExceeded):
            router.chat(huge_task)
            
        # PROOF: The cost prediction (BCD) stopped it before the API call
        mock_completion.assert_not_called()
