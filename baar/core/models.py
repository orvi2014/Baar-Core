"""
Data models for step results and routing logs.
Every decision is recorded — this is what devs show in benchmarks.
"""

import functools
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List
import litellm
from baar.core.router import RoutingDecision, ModelTier


@functools.lru_cache(maxsize=512)
def _cached_big_model_cost(big_model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """
    Memoized pricing lookup for savings calculation.
    savings_vs_always_big() calls this once per unique (model, token_pair) tuple
    instead of making one live lookup per step — O(unique token pairs) not O(steps).
    """
    try:
        in_c, out_c = litellm.cost_per_token(
            model=big_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(in_c + out_c)
    except Exception:
        return None


@dataclass
class StepResult:
    """Result of a single routed LLM call."""
    step_num: int
    task: str
    decision: RoutingDecision
    response_text: str
    cost: float
    cumulative_cost: float
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    attempted_models: List[str] = field(default_factory=list)
    failover_count: int = 0
    failover_errors: List[str] = field(default_factory=list)

    @property
    def model_used(self) -> str:
        return self.decision.model

    @property
    def used_big(self) -> bool:
        return self.decision.tier == ModelTier.BIG

    def to_dict(self) -> dict:
        d = {
            "step": self.step_num,
            "task_preview": self.task[:80] + ("..." if len(self.task) > 80 else ""),
            "model": self.model_used,
            "tier": self.decision.tier.value,
            "complexity_score": self.decision.complexity_score,
            "confidence": self.decision.confidence,
            "routing_reason": self.decision.reason,
            "forced_by_budget": self.decision.forced_by_budget,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost, 8),
            "cumulative_cost_usd": round(self.cumulative_cost, 8),
            "latency_ms": round(self.latency_ms, 1),
            "attempted_models": list(self.attempted_models),
            "failover_count": self.failover_count,
        }
        if self.failover_errors:
            d["failover_errors"] = list(self.failover_errors)
        if self.decision.estimated_value is not None:
            d["estimated_value"] = self.decision.estimated_value
        if self.decision.estimated_cost_usd is not None:
            d["estimated_cost_usd"] = self.decision.estimated_cost_usd
        if self.decision.routing_cache_hit:
            d["routing_cache_hit"] = True
        return d


@dataclass
class RoutingLog:
    """
    Full audit trail of a BAAR session.
    This is what you show developers in the benchmark report.
    """
    budget: float
    small_model: str
    big_model: str
    steps: List[StepResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    max_steps: Optional[int] = None  # cap in-memory log size; oldest steps are evicted

    def __post_init__(self) -> None:
        if self.max_steps is not None:
            self.steps = deque(self.steps, maxlen=self.max_steps)
        # Accumulate total cost independently of the deque so that evicted steps
        # are not lost from the running total when max_steps is set.
        self._total_cost_all: float = sum(s.cost for s in self.steps)

    def add(self, step: StepResult) -> None:
        self._total_cost_all += step.cost
        self.steps.append(step)
        # deque with maxlen auto-evicts oldest; list is unbounded

    @property
    def total_cost(self) -> float:
        return self._total_cost_all

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def big_calls(self) -> int:
        return sum(1 for s in self.steps if s.decision.tier == ModelTier.BIG)

    @property
    def small_calls(self) -> int:
        return sum(1 for s in self.steps if s.decision.tier == ModelTier.SMALL)

    @property
    def reject_steps(self) -> int:
        return sum(1 for s in self.steps if s.decision.tier == ModelTier.REJECT)

    @property
    def budget_forced_downgrades(self) -> int:
        return sum(1 for s in self.steps if s.decision.forced_by_budget)

    @property
    def routing_cache_hits(self) -> int:
        return sum(1 for s in self.steps if s.decision.routing_cache_hit)

    @property
    def always_big_cost(self) -> float:
        """What this session would have cost using big model for every step."""
        total = 0.0
        for s in self.steps:
            if s.decision.tier == ModelTier.BIG:
                total += s.cost
            elif s.decision.tier == ModelTier.SMALL:
                big_cost = _cached_big_model_cost(
                    self.big_model, s.prompt_tokens, s.completion_tokens
                )
                total += big_cost if big_cost is not None else s.cost
        return total

    def savings_vs_always_big(self) -> dict:
        """
        Calculate savings vs naive always-big strategy using real LiteLLM
        pricing — no hardcoded multipliers.  For each SMALL call, we look up
        what the same prompt+completion tokens would have cost on the big model.
        """
        always_big_cost = 0.0
        for s in self.steps:
            if s.decision.tier == ModelTier.BIG:
                always_big_cost += s.cost
            elif s.decision.tier == ModelTier.SMALL:
                cost = _cached_big_model_cost(
                    self.big_model, s.prompt_tokens, s.completion_tokens
                )
                # Pricing unavailable — fall back to actual cost rather than fabricate.
                always_big_cost += cost if cost is not None else s.cost
            # REJECT steps contributed $0 to both sides — intentionally skipped.

        saved = always_big_cost - self.total_cost
        pct = (saved / always_big_cost * 100) if always_big_cost > 0 else 0.0

        return {
            "baar_cost": round(self.total_cost, 6),
            "estimated_always_big_cost": round(always_big_cost, 6),
            "saved_usd": round(max(0.0, saved), 6),
            "savings_pct": round(max(0.0, pct), 1),
        }

    def summary(self) -> dict:
        savings = self.savings_vs_always_big()
        completion_steps = self.small_calls + self.big_calls
        return {
            "budget_usd": self.budget,
            "spent_usd": round(self.total_cost, 8),
            "remaining_usd": round(self.budget - self.total_cost, 8),
            "utilization_pct": round(self.total_cost / self.budget * 100, 2) if self.budget > 0 else 0,
            "total_steps": self.total_steps,
            "small_model_calls": self.small_calls,
            "big_model_calls": self.big_calls,
            "reject_steps": self.reject_steps,
            "budget_forced_downgrades": self.budget_forced_downgrades,
            "routing_cache_hits": self.routing_cache_hits,
            "pct_routed_to_small": round(
                self.small_calls / max(1, completion_steps) * 100, 1
            ),
            "savings_vs_always_big": savings,
            "steps": [s.to_dict() for s in self.steps],
            "errors": list(self.errors),
        }

    def print_report(self) -> None:
        """Human-readable report — what devs share as screenshots."""
        s = self.summary()
        sav = s["savings_vs_always_big"]

        print("\n" + "═" * 60)
        print("  BAAR ROUTING REPORT")
        print("═" * 60)
        print(f"  Budget:        ${s['budget_usd']:.4f}")
        print(f"  Spent:         ${s['spent_usd']:.6f}  ({s['utilization_pct']}% used)")
        print(f"  Remaining:     ${s['remaining_usd']:.6f}")
        print("─" * 60)
        print(f"  Total steps:   {s['total_steps']}")
        print(f"  → small model: {s['small_model_calls']} calls  ({s['pct_routed_to_small']}%)")
        print(f"  → big model:   {s['big_model_calls']} calls")
        print(f"  → value rejects: {s['reject_steps']}")
        print(f"  → routing cache hits: {s['routing_cache_hits']}")
        print(f"  → budget forced downgrades: {s['budget_forced_downgrades']}")
        print("─" * 60)
        print(f"  BAAR cost:              ${sav['baar_cost']:.6f}")
        print(f"  Always-big estimate:    ${sav['estimated_always_big_cost']:.6f}")
        print(f"  Saved:                  ${sav['saved_usd']:.6f}  ({sav['savings_pct']}% cheaper)")
        print("─" * 60)
        print(f"  {'Step':<5} {'Model':<15} {'Complexity':<11} {'Cost':>10}  Reason")
        print(f"  {'─'*4} {'─'*14} {'─'*10} {'─'*10}  {'─'*20}")
        for step in s["steps"]:
            forced = " [BUDGET]" if step["forced_by_budget"] else ""
            cached = " [ROUTE-CACHE]" if step.get("routing_cache_hit") else ""
            print(
                f"  {step['step']:<5} "
                f"{step['model']:<15} "
                f"{step['complexity_score']:<11.3f} "
                f"${step['cost_usd']:>9.6f}  "
                f"{step['routing_reason'][:30]}{forced}{cached}"
            )
        if self.errors:
            print("─" * 60)
            print(f"  Errors ({len(self.errors)}):")
            for err in self.errors:
                print(f"    {err}")
        print("═" * 60 + "\n")
