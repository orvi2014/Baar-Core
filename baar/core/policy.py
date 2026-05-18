"""
baar/core/policy.py — Declarative policy engine for LLM call governance.

Rules are evaluated top-to-bottom; first match wins.
"""

import operator
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PolicyAction(str, Enum):
    BLOCK = "block"
    FORCE_SMALL = "force_small"
    FORCE_BIG = "force_big"
    ALLOW = "allow"


_OP_MAP = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}
_OP_RE = re.compile(r"^(>=|<=|>|<|==|!=)\s*(.+)$")


def _match_value(condition_value: Any, actual_value: Any) -> bool:
    """
    Match a condition value against an actual runtime value.
    Supports comparison strings (">= 0.8", "< 0.5") for numeric facts,
    and exact equality for strings and other types.
    """
    if isinstance(condition_value, str):
        m = _OP_RE.match(condition_value.strip())
        if m:
            op_fn = _OP_MAP[m.group(1)]
            try:
                rhs = float(m.group(2))
                return op_fn(float(actual_value), rhs)
            except (ValueError, TypeError):
                return False
    return condition_value == actual_value


@dataclass
class Rule:
    """
    A single policy rule. All keys in `when` must match (AND semantics).

    Supported `when` keys:
      - "model"       — model name chosen by the router
      - "utilization" — budget utilization 0.0–1.0 (supports ">= 0.8" etc.)
      - "domain"      — routing domain: code, math, analysis, creative, general
      - "complexity"  — complexity score 0.0–1.0
      - any key from the per-call ``context`` dict passed to chat() / achat()

    Supported `then` values:
      "block" | "force_small" | "force_big" | "allow"

    Example::

        Rule(when={"plan": "free", "model": "gpt-4o"}, then="block")
        Rule(when={"utilization": ">= 0.8"},           then="force_small")
    """
    when: Dict[str, Any]
    then: str

    def __post_init__(self) -> None:
        valid = [a.value for a in PolicyAction]
        if self.then not in valid:
            raise ValueError(f"Rule 'then' must be one of {valid}, got '{self.then}'")

    def matches(self, facts: Dict[str, Any]) -> bool:
        return all(_match_value(v, facts.get(k)) for k, v in self.when.items())

    @property
    def action(self) -> PolicyAction:
        return PolicyAction(self.then)


@dataclass
class Policy:
    """
    Ordered list of Rules. Evaluated top-to-bottom; first match wins.

    Usage::

        policy = Policy(rules=[
            Rule(when={"plan": "free", "model": "gpt-4o"}, then="block"),
            Rule(when={"utilization": ">= 0.8"},           then="force_small"),
        ])
        router = BAARRouter(budget=0.10, policy=policy)
        router.chat("Summarise this article", context={"plan": "free"})
    """
    rules: List[Rule] = field(default_factory=list)

    def evaluate(self, facts: Dict[str, Any]) -> Optional[PolicyAction]:
        """Return the first matching action, or None if no rule matches."""
        for rule in self.rules:
            if rule.matches(facts):
                return rule.action
        return None
