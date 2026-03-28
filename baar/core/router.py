"""
Router — decides SMALL vs BIG at each step.

Week 1-2: LLM-as-router (cheap model scores complexity).
This is fundamentally different from keyword matching —
it understands semantic complexity, not surface patterns.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import litellm


class ModelTier(str, Enum):
    SMALL = "small"
    BIG = "big"


@dataclass
class RoutingDecision:
    tier: ModelTier
    model: str
    confidence: float          # 0.0–1.0
    complexity_score: float    # 0.0–1.0 from router
    reason: str
    forced_by_budget: bool = False


ROUTER_PROMPT = """You are a task complexity classifier for an AI routing system.

Analyze the following task/message and rate its complexity on a scale of 0.0 to 1.0:
- 0.0–0.3: Simple (factual lookup, greeting, basic formatting, yes/no, simple math)
- 0.4–0.6: Medium (summarization, basic reasoning, short code, explanation)
- 0.7–1.0: Complex (multi-step reasoning, code generation, analysis, planning, debugging)

Respond with ONLY a JSON object in this exact format:
{{"complexity": 0.7, "reason": "requires multi-step reasoning"}}

Task to classify: {task}"""


class Router:
    """
    LLM-based router that semantically scores task complexity.

    Uses a cheap model (gpt-4o-mini) to decide whether a task
    needs gpt-4o — costs ~$0.000015 per routing decision.

    Falls back to a rule-based scorer if routing call fails.
    """

    def __init__(
        self,
        small_model: str = "gpt-4o-mini",
        big_model: str = "gpt-4o",
        complexity_threshold: float = 0.65,
        use_llm_router: bool = True,
    ):
        self.small_model = small_model
        self.big_model = big_model
        self.complexity_threshold = complexity_threshold
        self.use_llm_router = use_llm_router

    def decide(
        self,
        task: str,
        remaining_budget: float,
        budget_utilization: float,
    ) -> RoutingDecision:
        """
        Core routing logic:
        1. Score complexity (LLM-based or rule-based fallback)
        2. Apply budget pressure (high utilization → prefer SMALL)
        3. Return decision with full audit trail
        """
        complexity, reason = self._score_complexity(task)

        # Budget pressure: become more conservative as budget depletes
        # At 80%+ utilization, raise the threshold significantly
        effective_threshold = self.complexity_threshold
        if budget_utilization >= 0.8:
            effective_threshold = min(0.9, self.complexity_threshold + 0.2)
        elif budget_utilization >= 0.6:
            effective_threshold = min(0.85, self.complexity_threshold + 0.1)

        # Route to BIG only if complexity exceeds threshold
        if complexity >= effective_threshold:
            tier = ModelTier.BIG
            model = self.big_model
            confidence = complexity
        else:
            tier = ModelTier.SMALL
            model = self.small_model
            confidence = 1.0 - complexity

        return RoutingDecision(
            tier=tier,
            model=model,
            confidence=round(confidence, 3),
            complexity_score=round(complexity, 3),
            reason=reason,
            forced_by_budget=False,
        )

    def force_small(self, original_decision: RoutingDecision, reason: str) -> RoutingDecision:
        """Override to SMALL when budget cannot afford BIG."""
        return RoutingDecision(
            tier=ModelTier.SMALL,
            model=self.small_model,
            confidence=original_decision.confidence,
            complexity_score=original_decision.complexity_score,
            reason=f"[BUDGET FORCED] {reason}",
            forced_by_budget=True,
        )

    def _score_complexity(self, task: str) -> tuple[float, str]:
        """
        Returns (complexity_score, reason_string).
        Tries LLM-based scoring first, falls back to heuristic.
        """
        if self.use_llm_router:
            try:
                return self._llm_score(task)
            except Exception:
                pass  # Fallback to heuristic
        return self._heuristic_score(task)

    def _llm_score(self, task: str) -> tuple[float, str]:
        """Use cheap model to semantically score complexity."""
        import json

        response = litellm.completion(
            model=self.small_model,
            messages=[
                {
                    "role": "user",
                    "content": ROUTER_PROMPT.format(task=task[:500]),  # cap at 500 chars
                }
            ],
            max_tokens=60,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()

        # Handle markdown code blocks if model wraps response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        complexity = float(parsed.get("complexity", 0.5))
        reason = str(parsed.get("reason", "llm scored"))
        complexity = max(0.0, min(1.0, complexity))
        return complexity, f"llm: {reason}"

    def _heuristic_score(self, task: str) -> tuple[float, str]:
        """
        Rule-based fallback. Better than keyword matching —
        uses multiple signals: length, question words, code markers.
        """
        score = 0.0
        reasons = []

        words = task.lower().split()
        word_count = len(words)

        # Length signal
        if word_count > 50:
            score += 0.3
            reasons.append("long task")
        elif word_count > 20:
            score += 0.15
            reasons.append("medium length")

        # Code signals
        code_markers = ["def ", "class ", "import ", "```", "function", "debug", "error", "traceback"]
        if any(m in task for m in code_markers):
            score += 0.7
            reasons.append("code content")

        # Reasoning signals
        reasoning_words = ["analyze", "explain", "compare", "design", "plan", "evaluate",
                           "why", "how does", "what would happen", "trade-off", "architecture"]
        if any(w in task.lower() for w in reasoning_words):
            score += 0.7
            reasons.append("reasoning required")

        # Multi-step signals
        multi_step = ["step by step", "first", "then", "finally", "multiple", "each"]
        if sum(1 for w in multi_step if w in task.lower()) >= 2:
            score += 0.2
            reasons.append("multi-step")

        score = min(1.0, score)
        reason = ", ".join(reasons) if reasons else "simple task"
        return score, f"heuristic: {reason}"
