"""
Budget tracker — real token-based cost using LiteLLM pricing.
This is the financial source of truth for the entire system.
"""

from dataclasses import dataclass, field
from typing import Optional
from litellm import completion_cost, cost_per_token


class BudgetExceeded(Exception):
    """Raised when a model call would exceed the remaining budget."""

    def __init__(self, requested: float, remaining: float, model: str):
        self.requested = requested
        self.remaining = remaining
        self.model = model
        super().__init__(
            f"Budget exceeded: model '{model}' would cost ~${requested:.6f} "
            f"but only ${remaining:.6f} remains."
        )


@dataclass
class SpendRecord:
    step: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    cumulative_cost: float


@dataclass
class BudgetTracker:
    """
    Tracks real spend using LiteLLM's live pricing data.
    Never uses hardcoded costs — always derives from actual token counts.
    """

    total_budget: float
    _spent: float = field(default=0.0, init=False)
    _records: list = field(default_factory=list, init=False)
    _step: int = field(default=0, init=False)

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self._spent)

    @property
    def utilization(self) -> float:
        """0.0 → 1.0 showing how much budget has been consumed."""
        if self.total_budget <= 0:
            return 1.0
        return min(1.0, self._spent / self.total_budget)

    def cost_from_response(self, response) -> float:
        """
        Extract real cost from a LiteLLM completion response.
        Uses completion_cost() which reads live pricing from LiteLLM's
        model_prices_and_context_window.json — no hardcoding.
        """
        try:
            cost = completion_cost(completion_response=response)
            return float(cost)
        except Exception:
            # Fallback: manual calc from usage if completion_cost fails
            usage = getattr(response, "usage", None)
            if usage:
                model = response.model or "gpt-4o-mini"
                try:
                    in_cost, out_cost = cost_per_token(
                        model=model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                    )
                    return float(in_cost + out_cost)
                except Exception:
                    pass
            return 0.0

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int = 200) -> float:
        """
        Pre-flight cost estimate before making a call.
        Uses cost_per_token with estimated output token count.
        """
        try:
            in_cost, out_cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return float(in_cost + out_cost)
        except Exception:
            return 0.0

    def check_affordability(self, model: str, prompt_tokens: int) -> None:
        """
        Hard budget constraint (BCD — Budget-Constrained Decoding).
        Raises BudgetExceeded before the call is ever made.
        """
        estimated = self.estimate_cost(model, prompt_tokens)
        if estimated > self.remaining:
            raise BudgetExceeded(
                requested=estimated,
                remaining=self.remaining,
                model=model,
            )

    def record(self, response, model: str) -> SpendRecord:
        """Record actual spend after a successful call."""
        self._step += 1
        cost = self.cost_from_response(response)
        self._spent += cost

        usage = getattr(response, "usage", None)
        record = SpendRecord(
            step=self._step,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cost=cost,
            cumulative_cost=self._spent,
        )
        self._records.append(record)
        return record

    @property
    def records(self) -> list:
        return list(self._records)

    def summary(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "spent": round(self._spent, 8),
            "remaining": round(self.remaining, 8),
            "utilization_pct": round(self.utilization * 100, 2),
            "steps": self._step,
            "records": [
                {
                    "step": r.step,
                    "model": r.model,
                    "tokens": r.prompt_tokens + r.completion_tokens,
                    "cost": round(r.cost, 8),
                    "cumulative": round(r.cumulative_cost, 8),
                }
                for r in self._records
            ],
        }
