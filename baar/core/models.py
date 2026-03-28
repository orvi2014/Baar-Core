"""
Data models for step results and routing logs.
Every decision is recorded — this is what devs show in benchmarks.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from baar.core.router import RoutingDecision, ModelTier


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

    @property
    def model_used(self) -> str:
        return self.decision.model

    @property
    def used_big(self) -> bool:
        return self.decision.tier == ModelTier.BIG

    def to_dict(self) -> dict:
        return {
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
        }


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

    def add(self, step: StepResult) -> None:
        self.steps.append(step)

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.steps)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def big_calls(self) -> int:
        return sum(1 for s in self.steps if s.used_big)

    @property
    def small_calls(self) -> int:
        return sum(1 for s in self.steps if not s.used_big)

    @property
    def budget_forced_downgrades(self) -> int:
        return sum(1 for s in self.steps if s.decision.forced_by_budget)

    @property
    def always_big_cost(self) -> float:
        """What this would have cost if we used big model for every step."""
        return sum(
            s.cost * (s.decision.complexity_score / max(0.01, s.decision.complexity_score))
            if not s.used_big
            else s.cost
            for s in self.steps
        )

    def savings_vs_always_big(self) -> dict:
        """
        Calculate savings vs naive always-big strategy.
        This is the benchmark number that matters.
        """
        # Estimate always-big cost: use ratio of big/small pricing
        # gpt-4o is ~15x more expensive than gpt-4o-mini per token
        estimated_always_big = sum(
            s.cost * 15 if not s.used_big else s.cost
            for s in self.steps
        )
        saved = estimated_always_big - self.total_cost
        pct = (saved / estimated_always_big * 100) if estimated_always_big > 0 else 0

        return {
            "baar_cost": round(self.total_cost, 6),
            "estimated_always_big_cost": round(estimated_always_big, 6),
            "saved_usd": round(saved, 6),
            "savings_pct": round(pct, 1),
        }

    def summary(self) -> dict:
        savings = self.savings_vs_always_big()
        return {
            "budget_usd": self.budget,
            "spent_usd": round(self.total_cost, 8),
            "remaining_usd": round(self.budget - self.total_cost, 8),
            "utilization_pct": round(self.total_cost / self.budget * 100, 2) if self.budget > 0 else 0,
            "total_steps": self.total_steps,
            "small_model_calls": self.small_calls,
            "big_model_calls": self.big_calls,
            "budget_forced_downgrades": self.budget_forced_downgrades,
            "pct_routed_to_small": round(self.small_calls / max(1, self.total_steps) * 100, 1),
            "savings_vs_always_big": savings,
            "steps": [s.to_dict() for s in self.steps],
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
            print(
                f"  {step['step']:<5} "
                f"{step['model']:<15} "
                f"{step['complexity_score']:<11.3f} "
                f"${step['cost_usd']:>9.6f}  "
                f"{step['routing_reason'][:30]}{forced}"
            )
        print("═" * 60 + "\n")
