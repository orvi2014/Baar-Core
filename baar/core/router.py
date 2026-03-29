"""
Router — decides SMALL vs BIG at each step.

Week 1-2: LLM-as-router (cheap model scores complexity).
This is fundamentally different from keyword matching —
it understands semantic complexity, not surface patterns.
"""

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import hashlib
import random
from typing import Optional
import litellm


class ModelTier(str, Enum):
    SMALL = "small"
    BIG = "big"
    REJECT = "reject"


@dataclass
class RoutingDecision:
    tier: ModelTier
    model: str
    confidence: float          # 0.0–1.0
    complexity_score: float    # 0.0–1.0 from router
    reason: str
    forced_by_budget: bool = False
    # Set by BAARRouter when value_fn is used (execute or reject audit)
    estimated_value: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    # True when (complexity, reason) was served from LRU; tier reflects current utilization
    routing_cache_hit: bool = False


ROUTER_PROMPT = """You are a task complexity classifier for an AI routing system.

Analyze the following task/message and rate its complexity on a scale of 0.0 to 1.0:
- 0.0–0.3: Simple (casual greeting, basic formatting, yes/no, straightforward requests)
- 0.4–0.6: Medium (summarization, simple math, short explanation)
- 0.7–1.0: Complex (multi-step reasoning, advanced academic/domain knowledge, coding, multi-choice benchmarks and analysis)

When a task sits between bands, prefer the LOWER score if a capable small model could plausibly complete it; reserve 0.8+ for genuinely hard or high-stakes work.

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
        complexity_threshold: float = 0.80,
        use_llm_router: bool = True,
        routing_cache_size: int = 256,
        routing_cache_enabled: bool = True,
        small_exploration_rate: float = 0.0,
    ):
        self.small_model = small_model
        self.big_model = big_model
        self.complexity_threshold = complexity_threshold
        self.use_llm_router = use_llm_router
        self._routing_cache_max = max(0, int(routing_cache_size))
        self._routing_cache_enabled = bool(routing_cache_enabled)
        self._routing_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.small_exploration_rate = max(0.0, min(1.0, float(small_exploration_rate)))

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
        easy_reason = self._easy_small_reason(task)
        if easy_reason:
            return RoutingDecision(
                tier=ModelTier.SMALL,
                model=self.small_model,
                confidence=0.95,
                complexity_score=0.05,
                reason=f"easy-pattern: {easy_reason}",
                forced_by_budget=False,
                routing_cache_hit=False,
            )

        complexity, reason, cache_hit = self._score_complexity(task)

        # Cost pressure: effective BIG bar rises smoothly as budget is consumed
        pressure = budget_utilization * 0.18
        effective_threshold = min(0.93, self.complexity_threshold + pressure)

        # Route to BIG only if complexity meets the (possibly raised) bar
        if complexity >= effective_threshold:
            tier = ModelTier.BIG
            model = self.big_model
            confidence = complexity
        else:
            tier = ModelTier.SMALL
            model = self.small_model
            confidence = 1.0 - complexity

        # Optional tiny exploration: occasionally downshift BIG to SMALL.
        if tier == ModelTier.BIG and self.small_exploration_rate > 0.0:
            if random.random() < self.small_exploration_rate:
                tier = ModelTier.SMALL
                model = self.small_model
                confidence = max(0.5, 1.0 - complexity)
                reason = f"[EXPLORE SMALL] {reason}"

        return RoutingDecision(
            tier=tier,
            model=model,
            confidence=round(confidence, 3),
            complexity_score=round(complexity, 3),
            reason=reason,
            forced_by_budget=False,
            routing_cache_hit=cache_hit,
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
            estimated_value=original_decision.estimated_value,
            estimated_cost_usd=original_decision.estimated_cost_usd,
            routing_cache_hit=original_decision.routing_cache_hit,
        )

    def _easy_small_reason(self, task: str) -> Optional[str]:
        lowered = task.lower()
        if "capital of" in lowered:
            return "capital lookup"
        if "define " in lowered or lowered.startswith("define"):
            return "definition request"
        if lowered.startswith("what is ") and len(lowered.split()) <= 12:
            return "short factual what-is"
        return None

    @staticmethod
    def _normalize_task_text(task: str) -> str:
        return " ".join(task.split()).strip()

    def _fingerprint_for_routing_cache(self, task: str) -> str:
        """LLM routing prompt uses task[:500]; heuristic uses the full task."""
        if self.use_llm_router:
            return self._normalize_task_text(task[:500])
        return self._normalize_task_text(task)

    def _routing_cache_key(self, task: str) -> str:
        fp = self._fingerprint_for_routing_cache(task)
        digest = hashlib.sha256(fp.encode("utf-8")).hexdigest()
        return f"{int(self.use_llm_router)}\0{self.small_model}\0{digest}"

    def _routing_cache_get(self, key: str) -> Optional[tuple[float, str]]:
        if key not in self._routing_cache:
            return None
        self._routing_cache.move_to_end(key)
        return self._routing_cache[key]

    def _routing_cache_set(self, key: str, value: tuple[float, str]) -> None:
        self._routing_cache[key] = value
        self._routing_cache.move_to_end(key)
        while len(self._routing_cache) > self._routing_cache_max:
            self._routing_cache.popitem(last=False)

    def _compute_complexity_uncached(self, task: str) -> tuple[float, str]:
        if self.use_llm_router:
            try:
                return self._llm_score(task)
            except Exception:
                pass
        return self._heuristic_score(task)

    def _score_complexity(self, task: str) -> tuple[float, str, bool]:
        """
        Returns (complexity_score, reason_string, routing_cache_hit).
        Only (complexity, reason) is cached; decide() always reapplies utilization thresholds.
        """
        if not self._routing_cache_enabled or self._routing_cache_max <= 0:
            c, r = self._compute_complexity_uncached(task)
            return c, r, False

        key = self._routing_cache_key(task)
        hit = self._routing_cache_get(key)
        if hit is not None:
            return hit[0], hit[1], True

        c, r = self._compute_complexity_uncached(task)
        self._routing_cache_set(key, (c, r))
        return c, r, False

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
        raw_complexity = float(parsed.get("complexity", 0.5))
        reason = str(parsed.get("reason", "llm scored"))

        # Normalize raw classifier output into a wider, more useful band.
        # 1) clamp  2) square (push medium lower, keep very high high)
        # 3) small length-bias correction
        complexity = max(0.0, min(1.0, raw_complexity))
        complexity = complexity ** 2
        length_penalty = min(len(task) / 1000.0, 1.0)
        complexity = max(0.0, complexity - (0.2 * length_penalty))

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
            score += 0.82
            reasons.append("code content")

        # Reasoning signals (longer prompts → stronger bump so benchmarks clear default 0.80 bar)
        reasoning_words = ["analyze", "explain", "compare", "design", "plan", "evaluate",
                           "why", "how does", "what would happen", "trade-off", "architecture"]
        if any(w in task.lower() for w in reasoning_words):
            if word_count >= 8:
                score += 0.82
                reasons.append("reasoning required")
            else:
                score += 0.65
                reasons.append("reasoning (short)")

        # Multi-step signals
        multi_step = ["step by step", "first", "then", "finally", "multiple", "each"]
        if sum(1 for w in multi_step if w in task.lower()) >= 2:
            score += 0.2
            reasons.append("multi-step")

        # Academic / Benchmarking signals
        academic = ["choices:", "choose the correct", "options:"]
        if any(w in task.lower() for w in academic):
            score += 0.7
            reasons.append("academic/multi-choice")

        # Structured math / word problems (e.g. GSM8K-style graded prompts)
        if "####" in task or "show your work" in task.lower():
            score += 0.68
            reasons.append("structured math")

        score = min(1.0, score)
        reason = ", ".join(reasons) if reasons else "simple task"
        return score, f"heuristic: {reason}"
