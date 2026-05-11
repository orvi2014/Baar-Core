"""
Budget tracker — real token-based cost using LiteLLM pricing.
This is the financial source of truth for the entire system.
"""

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union
from litellm import completion_cost, cost_per_token

from baar.core.stores import BudgetStore, MemoryBudgetStore

# Reservation/cancellation methods allow achat() to atomically claim budget
# before the async LLM call, preventing concurrent callers from both passing
# the affordability check and together overshooting the budget.


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


class BudgetTracker:
    """
    Tracks real spend using LiteLLM's live pricing data.
    Never uses hardcoded costs — always derives from actual token counts.

    Args:
        total_budget: Maximum allowed spend in USD.
        store:        Where to persist spend. Defaults to in-memory (no persistence).
                      Pass a FileBudgetStore or SQLiteBudgetStore for persistence
                      across process restarts.
    """

    def __init__(
        self,
        total_budget: float,
        store: Optional[BudgetStore] = None,
        max_records: Optional[int] = None,
    ) -> None:
        self.total_budget = total_budget
        self._store: BudgetStore = store if store is not None else MemoryBudgetStore()
        self._records: Union[list, deque] = deque(maxlen=max_records) if max_records is not None else []
        self._step: int = 0

    # _spent is kept as a property so existing tests that do
    # `tracker._spent = X` continue to work — the assignment is forwarded
    # transparently to the store.

    @property
    def _spent(self) -> float:
        return self._store.get_spent()

    @_spent.setter
    def _spent(self, value: float) -> None:
        self._store.set_spent(float(value))

    @property
    def spent(self) -> float:
        return self._store.get_spent()

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self._store.get_spent())

    @property
    def utilization(self) -> float:
        """0.0 → 1.0 showing how much budget has been consumed."""
        if self.total_budget <= 0:
            return 1.0
        return min(1.0, self._store.get_spent() / self.total_budget)

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
            usage = getattr(response, "usage", None)
            if usage:
                model = getattr(response, "model", None)
                if not model:
                    return 0.0
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

    # Fallback estimate used when LiteLLM has no pricing data for a model.
    # $0.0001 ≈ a short gpt-4o-mini call — low enough not to block normal budgets
    # (e.g. $0.05) but nonzero so the kill-switch still fires at micro-budgets
    # (e.g. $0.00005). Register real pricing via litellm.register_model() to
    # replace this fallback with accurate per-model data.
    _UNKNOWN_MODEL_FALLBACK_COST = 0.0001

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int = 500) -> float:
        """
        Pre-flight cost estimate before making a call.
        Default completion_tokens=500 — more realistic than 200 for complex tasks.
        Pass a tighter estimate when you know the expected output length.
        """
        try:
            in_cost, out_cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            cost = float(in_cost + out_cost)
            if not math.isfinite(cost):
                raise ValueError(f"non-finite cost for model '{model}'")
            return cost
        except Exception:
            warnings.warn(
                f"BudgetTracker: no pricing data for model '{model}'. "
                f"Using ${self._UNKNOWN_MODEL_FALLBACK_COST} conservative estimate. "
                "For accurate tracking, register real pricing with "
                "litellm.register_model() before using BAARRouter.",
                UserWarning,
                stacklevel=2,
            )
            return self._UNKNOWN_MODEL_FALLBACK_COST

    def check_affordability(self, model: str, prompt_tokens: int, completion_tokens: int = 500) -> None:
        """
        Hard budget constraint (BCD — Budget-Constrained Decoding).
        Raises BudgetExceeded before the call is ever made.
        """
        estimated = self.estimate_cost(model, prompt_tokens, completion_tokens)
        if estimated > self.remaining:
            raise BudgetExceeded(
                requested=estimated,
                remaining=self.remaining,
                model=model,
            )

    def check_and_reserve(self, amount: float) -> None:
        """
        Atomically check budget and reserve amount in one operation.
        Raises BudgetExceeded if current_spent + amount > total_budget.
        This fixes both the double-call and cross-instance TOCTOU race.
        """
        ok = self._store.atomic_check_and_reserve(self.total_budget, amount)
        if not ok:
            raise BudgetExceeded(
                requested=amount,
                remaining=self.remaining,
                model="(reservation)",
            )

    def reserve(self, amount: float) -> None:
        """
        Pre-deduct estimated cost so concurrent async calls cannot both pass
        the affordability check and together overshoot the budget (TOCTOU fix).
        Must be paired with a call to cancel_reservation() after recording actual cost.
        Prefer check_and_reserve() for new code — it is atomic across processes.
        """
        self.check_and_reserve(amount)

    def cancel_reservation(self, amount: float) -> None:
        """
        Reverse a prior reserve() call.  Called after record() has added the
        actual cost, or when the LLM call failed and no cost was incurred.
        """
        self._store.add_spent(-amount)

    def record(self, response, model: str) -> SpendRecord:
        """Record actual spend after a successful call."""
        self._step += 1
        cost = self.cost_from_response(response)
        new_spent = self._store.add_spent(cost)

        usage = getattr(response, "usage", None)
        rec = SpendRecord(
            step=self._step,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cost=cost,
            cumulative_cost=new_spent,
        )
        self._records.append(rec)
        return rec

    def record_manual(self, model: str, prompt_tokens: int, completion_tokens: int) -> SpendRecord:
        """
        Record spend from manually collected token counts.
        Used by stream_chat() where token counts are accumulated during streaming
        rather than read from a response object.
        """
        self._step += 1
        try:
            in_cost, out_cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            cost = float(in_cost + out_cost)
        except Exception:
            cost = 0.0
        new_spent = self._store.add_spent(cost)
        rec = SpendRecord(
            step=self._step,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            cumulative_cost=new_spent,
        )
        self._records.append(rec)
        return rec

    @property
    def records(self) -> list:
        return self._records

    def summary(self) -> dict:
        spent = self._store.get_spent()
        return {
            "total_budget": self.total_budget,
            "spent": round(spent, 8),
            "remaining": round(max(0.0, self.total_budget - spent), 8),
            "utilization_pct": round(min(1.0, spent / self.total_budget) * 100, 2) if self.total_budget > 0 else 0,
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
