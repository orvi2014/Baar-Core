"""
Router — decides SMALL vs BIG at each step.

LLM-as-router: a cheap model scores complexity semantically.
Falls back to a rule-based heuristic if the routing call fails.
"""

import json
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
import random
from typing import Optional
import litellm


class ModelTier(str, Enum):
    SMALL = "small"
    BIG = "big"
    REJECT = "reject"


@dataclass(frozen=True)
class RoutingDecision:
    tier: ModelTier
    model: str
    confidence: float          # 0.0–1.0
    complexity_score: float    # 0.0–1.0 from router
    reason: str
    forced_by_budget: bool = False
    estimated_value: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    routing_cache_hit: bool = False
    domain: str = "general"           # task domain inferred by routing LLM
    estimated_output_tokens: int = 500  # router's estimate for affordability checks


_ROUTER_SYSTEM = (
    "You are a task complexity classifier for an AI routing system. "
    "Respond ONLY with a single JSON object — no preamble, no markdown fences.\n\n"
    "JSON schema: "
    '{"complexity": <float 0.0-1.0>, "reason": "<string>", '
    '"domain": "<code|math|analysis|academic|creative|general>", '
    '"estimated_output_tokens": <int 50-4000>}\n\n'
    "Complexity bands:\n"
    "  0.0-0.3  Simple  - greeting, yes/no, basic formatting\n"
    "  0.4-0.6  Medium  - summarization, simple math, short explanation\n"
    "  0.7-1.0  Complex - multi-step reasoning, coding, research, analysis\n\n"
    "When a task sits between bands prefer the LOWER score if a capable small model "
    "could plausibly handle it; reserve 0.8+ for genuinely hard work."
)

_ROUTER_USER_TMPL = "Classify this task:\n\n{task}"

# Deprecated alias — kept so external code that imports ROUTER_PROMPT doesn't crash.
ROUTER_PROMPT = _ROUTER_SYSTEM


# Module-level constants — defined once, not rebuilt on every heuristic call.
# "select " and "where " removed: both are common English words that cause false
# positives on natural language ("select a gift for my mom", "where should I travel?")
# which were being routed to BIG at 0.92 complexity.
_CODE_MARKERS = [
    "def ", "class ", "async def", "import ", "from import",
    "```", "function(", "=>", "traceback", "stacktrace",
    "null pointer", "segfault", "await ",
]

# Precompiled alternation — O(len(task)) instead of O(n_markers × len(task)).
# "async def" precedes "def " so the longer alternative is tried first.
_CODE_RE = re.compile(
    "|".join(re.escape(m) for m in [
        "async def", "def ", "class ", "import ", "from import",
        "```", "function(", "=>", "traceback", "stacktrace",
        "null pointer", "segfault", "await ",
    ])
)

_REASONING = [
    "analyze", "analyse", "compare", "contrast", "evaluate", "assess",
    "critique", "justify",
    "design", "architect", "architecture", "trade-off", "tradeoff",
    "pros and cons",
    "explain", "elaborate", "deep dive", "comprehensive",
    "in depth", "step by step", "in detail",
    "why does", "how does", "what would happen", "difference between",
    "when should", "when to use",
    "debug", "debugging", "optimize", "optimise", "refactor",
    "recursive", "recursion", "algorithm", "implement", "implementation",
    "complexity", "performance", "scalability", "bottleneck",
    "concurrency", "distributed", "microservice",
]


class Router:
    """
    LLM-based router that semantically scores task complexity.

    Uses a cheap model (gpt-4o-mini) to decide whether a task
    needs gpt-4o — costs ~$0.000015 per routing decision.

    Falls back to a rule-based scorer if routing call fails.
    Provides both sync (decide) and async (adecide) entry points.

    Args:
        routing_cache: Optional external dict-like object for cross-instance
                       cache sharing (e.g. for SaaS multi-tenant deployments).
                       When set, the internal LRU cache is bypassed — eviction
                       is the caller's responsibility.
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
        routing_task_char_limit: int = 500,
        routing_cache: Optional[dict] = None,
        routing_timeout: Optional[float] = None,
    ):
        self.small_model = small_model
        self.big_model = big_model
        self.complexity_threshold = complexity_threshold
        self.use_llm_router = use_llm_router
        self._routing_cache_max = max(0, int(routing_cache_size))
        self._routing_cache_enabled = bool(routing_cache_enabled)
        self._routing_cache: OrderedDict[str, tuple] = OrderedDict()
        self.small_exploration_rate = max(0.0, min(1.0, float(small_exploration_rate)))
        self._routing_task_char_limit = max(64, int(routing_task_char_limit))
        # External shared cache: caller manages eviction; internal LRU is skipped when set.
        self._external_routing_cache: Optional[dict] = routing_cache
        self._routing_timeout: Optional[float] = routing_timeout
        self._routing_cache_lock = threading.Lock()
        self._routing_failure_count: int = 0
        self._routing_failure_lock = threading.Lock()

    # ── Public sync/async entry points ────────────────────────────────────────

    def decide(
        self,
        task: str,
        budget_utilization: float,
    ) -> RoutingDecision:
        """
        Core routing logic (sync):
        1. Score complexity (LLM-based or rule-based fallback)
        2. Apply budget pressure (high utilization → prefer SMALL)
        3. Return decision with full audit trail
        """
        easy_reason = self._easy_small_reason(task)
        if easy_reason:
            return self._easy_decision(easy_reason)

        complexity, reason, domain, eot, cache_hit = self._score_complexity(task)
        return self._build_decision(complexity, reason, domain, eot, cache_hit, budget_utilization)

    async def adecide(
        self,
        task: str,
        budget_utilization: float,
    ) -> RoutingDecision:
        """
        Async routing entry point — uses litellm.acompletion for the routing LLM
        call so the event loop is never blocked during routing.
        """
        easy_reason = self._easy_small_reason(task)
        if easy_reason:
            return self._easy_decision(easy_reason)

        complexity, reason, domain, eot, cache_hit = await self._ascore_complexity(task)
        return self._build_decision(complexity, reason, domain, eot, cache_hit, budget_utilization)

    def force_small(self, original: RoutingDecision, reason: str) -> RoutingDecision:
        """Override to SMALL when budget cannot afford BIG."""
        return RoutingDecision(
            tier=ModelTier.SMALL,
            model=self.small_model,
            confidence=original.confidence,
            complexity_score=original.complexity_score,
            reason=f"[BUDGET FORCED] {reason}",
            forced_by_budget=True,
            estimated_value=original.estimated_value,
            estimated_cost_usd=original.estimated_cost_usd,
            routing_cache_hit=original.routing_cache_hit,
            domain=original.domain,
            estimated_output_tokens=original.estimated_output_tokens,
        )

    # ── Shared decision builder ────────────────────────────────────────────────

    def _easy_decision(self, easy_reason: str) -> RoutingDecision:
        return RoutingDecision(
            tier=ModelTier.SMALL,
            model=self.small_model,
            confidence=0.95,
            complexity_score=0.05,
            reason=f"easy-pattern: {easy_reason}",
            forced_by_budget=False,
            routing_cache_hit=False,
            domain="general",
            estimated_output_tokens=100,
        )

    def _build_decision(
        self,
        complexity: float,
        reason: str,
        domain: str,
        eot: int,
        cache_hit: bool,
        budget_utilization: float,
    ) -> RoutingDecision:
        pressure = budget_utilization * 0.18
        effective_threshold = min(0.93, self.complexity_threshold + pressure)

        if complexity >= effective_threshold:
            tier = ModelTier.BIG
            model = self.big_model
            confidence = complexity
        else:
            tier = ModelTier.SMALL
            model = self.small_model
            confidence = 1.0 - complexity

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
            domain=domain,
            estimated_output_tokens=eot,
        )

    # ── Fast-path heuristics ───────────────────────────────────────────────────

    def _easy_small_reason(self, task: str) -> Optional[str]:
        """
        Fast-path for unambiguously trivial tasks only.
        Conservative by design: false positives (complex→small) hurt accuracy
        more than false negatives (simple→big) hurt cost.
        """
        lowered = task.lower().strip()
        words = lowered.split()
        word_count = len(words)

        if word_count <= 4 and lowered in {
            "hi", "hello", "hey", "good morning", "good afternoon",
            "good evening", "thanks", "thank you", "ok", "okay",
            "yes", "no", "sure", "got it",
        }:
            return "greeting"

        if (
            word_count <= 8
            and "capital of" in lowered
            and not any(w in lowered for w in [
                "and", "why", "how", "history", "government", "explain",
                "compare", "between", "culture", "population",
            ])
        ):
            return "capital lookup"

        if (
            word_count <= 10
            and any(p in lowered for p in [
                "°f to", "°c to", " to celsius", " to fahrenheit",
                " to km", " to miles", " to kg", " to pounds",
            ])
            and not any(w in lowered for w in ["explain", "why", "formula", "how"])
        ):
            return "unit conversion"

        if (
            word_count <= 8
            and re.search(r"\d+\s*[\+\-\*\/]\s*\d+", task)
            and not any(w in lowered for w in [
                "explain", "why", "show", "proof", "derive", "step",
            ])
        ):
            return "arithmetic"

        return None

    # ── Routing cache helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_task_text(task: str) -> str:
        return " ".join(task.split()).strip()

    def _fingerprint_for_routing_cache(self, task: str) -> str:
        if self.use_llm_router:
            return self._normalize_task_text(self._router_task_view(task))
        return self._normalize_task_text(task)

    def _router_task_view(self, task: str) -> str:
        """
        Build the task slice sent to the routing LLM.
        For long prompts, preserve head + middle + tail.
        """
        limit = self._routing_task_char_limit
        if len(task) <= limit:
            return task
        base = limit // 3
        rem = limit - (base * 3)
        head_len = base + (1 if rem > 0 else 0)
        mid_len = base + (1 if rem > 1 else 0)
        tail_len = base
        mid_start = max(0, (len(task) - mid_len) // 2)
        mid_end = mid_start + mid_len
        return (
            f"{task[:head_len]}"
            "\n...[TRUNCATED SEGMENT]...\n"
            f"{task[mid_start:mid_end]}"
            "\n...[TRUNCATED SEGMENT]...\n"
            f"{task[-tail_len:]}"
        )

    def _routing_cache_key(self, task: str) -> str:
        fp = self._fingerprint_for_routing_cache(task)
        # MD5 is sufficient for a non-security cache key and is ~4× faster than SHA-256
        digest = hashlib.md5(fp.encode("utf-8")).hexdigest()
        return f"{int(self.use_llm_router)}\0{self.small_model}\0{digest}"

    def _routing_cache_get(self, key: str) -> Optional[tuple]:
        if self._external_routing_cache is not None:
            return self._external_routing_cache.get(key)
        with self._routing_cache_lock:
            if key not in self._routing_cache:
                return None
            self._routing_cache.move_to_end(key)
            return self._routing_cache[key]

    def _routing_cache_set(self, key: str, value: tuple) -> None:
        if self._external_routing_cache is not None:
            self._external_routing_cache[key] = value
            return
        with self._routing_cache_lock:
            self._routing_cache[key] = value
            self._routing_cache.move_to_end(key)
            while len(self._routing_cache) > self._routing_cache_max:
                self._routing_cache.popitem(last=False)

    # ── Complexity scoring (sync) ──────────────────────────────────────────────

    def _score_complexity(self, task: str) -> tuple[float, str, str, int, bool]:
        """Returns (complexity_score, reason_string, domain, estimated_output_tokens, cache_hit)."""
        if not self._routing_cache_enabled or self._routing_cache_max <= 0:
            c, r, d, eot = self._compute_complexity_uncached(task)
            return c, r, d, eot, False

        key = self._routing_cache_key(task)
        hit = self._routing_cache_get(key)
        if hit is not None:
            c, r, d, eot = hit
            return c, r, d, eot, True

        c, r, d, eot = self._compute_complexity_uncached(task)
        self._routing_cache_set(key, (c, r, d, eot))
        return c, r, d, eot, False

    def _compute_complexity_uncached(self, task: str) -> tuple[float, str, str, int]:
        if self.use_llm_router:
            try:
                return self._llm_score(task)
            except Exception as _routing_exc:
                with self._routing_failure_lock:
                    self._routing_failure_count += 1
                    _fc = self._routing_failure_count
                if _fc == 1 or _fc % 10 == 0:
                    warnings.warn(
                        f"Routing LLM call failed ({type(_routing_exc).__name__}: {_routing_exc}). "
                        f"Falling back to heuristic scoring (failure #{_fc}). "
                        "Check your API key and model availability.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        c, r = self._heuristic_score(task)
        return c, r, "general", 500

    # ── Complexity scoring (async) ─────────────────────────────────────────────

    async def _ascore_complexity(self, task: str) -> tuple[float, str, str, int, bool]:
        """Async version of _score_complexity."""
        if not self._routing_cache_enabled or self._routing_cache_max <= 0:
            c, r, d, eot = await self._acompute_complexity_uncached(task)
            return c, r, d, eot, False

        key = self._routing_cache_key(task)
        hit = self._routing_cache_get(key)
        if hit is not None:
            c, r, d, eot = hit
            return c, r, d, eot, True

        c, r, d, eot = await self._acompute_complexity_uncached(task)
        self._routing_cache_set(key, (c, r, d, eot))
        return c, r, d, eot, False

    async def _acompute_complexity_uncached(self, task: str) -> tuple[float, str, str, int]:
        if self.use_llm_router:
            try:
                return await self._allm_score(task)
            except Exception as _routing_exc:
                with self._routing_failure_lock:
                    self._routing_failure_count += 1
                    _fc = self._routing_failure_count
                if _fc == 1 or _fc % 10 == 0:
                    warnings.warn(
                        f"Routing LLM call failed ({type(_routing_exc).__name__}: {_routing_exc}). "
                        f"Falling back to heuristic scoring (failure #{_fc}). "
                        "Check your API key and model availability.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        c, r = self._heuristic_score(task)
        return c, r, "general", 500

    # ── LLM scoring calls ──────────────────────────────────────────────────────

    def _llm_score(self, task: str) -> tuple[float, str, str, int]:
        """Sync LLM complexity scoring."""
        messages = [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": _ROUTER_USER_TMPL.format(task=self._router_task_view(task))},
        ]
        kwargs: dict = dict(
            model=self.small_model,
            messages=messages,
            max_tokens=80,
            temperature=0.0,
        )
        if self._routing_timeout is not None:
            kwargs["timeout"] = self._routing_timeout
        # Prefer structured output (guaranteed JSON) when provider supports it.
        # Only retry without response_format on BadRequestError (HTTP 400 = unsupported
        # parameter). Other errors (auth, timeout, rate limit) propagate immediately.
        try:
            response = litellm.completion(**kwargs, response_format={"type": "json_object"})
        except litellm.exceptions.BadRequestError:
            response = litellm.completion(**kwargs)
        return self._parse_llm_score_response(response)

    async def _allm_score(self, task: str) -> tuple[float, str, str, int]:
        """Async LLM complexity scoring — uses acompletion, never blocks the event loop."""
        messages = [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": _ROUTER_USER_TMPL.format(task=self._router_task_view(task))},
        ]
        kwargs: dict = dict(
            model=self.small_model,
            messages=messages,
            max_tokens=80,
            temperature=0.0,
        )
        if self._routing_timeout is not None:
            kwargs["timeout"] = self._routing_timeout
        try:
            response = await litellm.acompletion(**kwargs, response_format={"type": "json_object"})
        except litellm.exceptions.BadRequestError:
            response = await litellm.acompletion(**kwargs)
        return self._parse_llm_score_response(response)

    @staticmethod
    def _parse_llm_score_response(response) -> tuple[float, str, str, int]:
        """Parse the routing LLM's JSON response into (complexity, reason, domain, estimated_output_tokens)."""
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        _c = parsed.get("complexity")
        complexity = 0.5 if _c is None else max(0.0, min(1.0, float(_c)))
        reason = str(parsed.get("reason", "llm scored"))
        domain = str(parsed.get("domain", "general"))
        _eot = parsed.get("estimated_output_tokens")
        eot = 500 if _eot is None else max(50, min(4000, int(float(_eot))))
        # Length penalty removed: it subtracted from complexity for longer tasks,
        # which is backwards — longer tasks are generally more complex, not less.
        # Trust the LLM's score directly.
        return complexity, f"llm: {reason}", domain, eot

    # ── Heuristic fallback ─────────────────────────────────────────────────────

    def _heuristic_score(self, task: str) -> tuple[float, str]:
        """
        Priority-based complexity scorer — evaluated top to bottom, first match wins.
        Uses module-level constants to avoid rebuilding lists on every call.

        Design principle: when ambiguous, score HIGHER (prefer big model).
        Routing a hard task to small loses accuracy; routing an easy task to big
        only wastes a little money. Accuracy per dollar is maximised by being
        conservative about what counts as "easy".
        """
        lowered = task.lower()
        words = lowered.split()
        word_count = len(words)

        if _CODE_RE.search(lowered):
            return 0.92, "heuristic: code content"

        if (
            "choices:" in lowered
            or "options:" in lowered
            or "choose the correct" in lowered
            or sum(1 for c in "abcd" if f"{c}." in lowered or f"{c})" in lowered) >= 3
        ):
            return 0.88, "heuristic: multi-choice academic"

        if "####" in task or "show your work" in lowered:
            return 0.86, "heuristic: structured math problem"

        has_reasoning = any(w in lowered for w in _REASONING)

        if has_reasoning and word_count >= 10:
            return 0.85, "heuristic: reasoning task"
        if has_reasoning and word_count >= 5:
            return 0.70, "heuristic: reasoning task (moderate)"

        if word_count > 50:
            return 0.65, "heuristic: long task"
        if word_count > 20:
            return 0.40, "heuristic: medium task"

        return 0.15, "heuristic: short task"
