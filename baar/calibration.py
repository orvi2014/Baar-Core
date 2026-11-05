"""
baar/calibration.py — run-level budget calibration for ``on_output_overrun="calibrated"``.

The calibrated mode plans each call for the empirical (1 - beta) quantile of
logged output lengths for its stratum (model, step kind) instead of the
expected output ("estimate") or the worst case ("preflight"). A single scalar
``beta`` sets how aggressive the plan is.

``beta`` is chosen at the level of whole agent runs with fixed-sequence
Learn-then-Test (Angelopoulos et al., 2021): candidate levels are tested in a
fixed order, from conservative to aggressive, each on ``n`` fresh runs, with an
exact binomial p-value. If the runs are i.i.d. draws from the deployment
distribution, the selected level satisfies

    P( P(run fails) <= eps ) >= 1 - delta

where "fails" is whatever loss you count (e.g. the run spent more than its
budget). No assumption on output-length distributions is needed. If no level
can be certified, the result is ``beta=None``, meaning: use "preflight".

This module has no third-party dependencies.
"""

import bisect
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

ANY_KIND = "any"


# ── Exact binomial p-value ────────────────────────────────────────────────────

def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), computed exactly in log space (no scipy)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(k + 1)
    ]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def ltt_p_value(failures: int, n: int, eps: float) -> float:
    """
    p-value for H0: risk > eps, given ``failures`` among ``n`` i.i.d. runs.
    Small values are evidence that the risk is at most ``eps``.
    """
    if not 0 <= failures <= n:
        raise ValueError("failures must be between 0 and n")
    return binom_cdf(failures, n, eps)


# ── Learn-then-Test selection ─────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """
    Outcome of a Learn-then-Test calibration.

    beta:     the certified level, or None when no level was certified
              (then use "preflight").
    eps:      target failure rate.
    delta:    allowed probability that calibration certifies a bad level.
    n:        runs per tested level.
    tested:   levels actually tested, in order.
    p_values: p-value of each tested level.
    failures: failures observed at each tested level.
    """
    beta: Optional[float]
    eps: float
    delta: float
    n: int
    tested: List[float] = field(default_factory=list)
    p_values: List[float] = field(default_factory=list)
    failures: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "CalibrationResult":
        return cls(
            beta=data["beta"],
            eps=float(data["eps"]),
            delta=float(data["delta"]),
            n=int(data["n"]),
            tested=[float(b) for b in data.get("tested", [])],
            p_values=[float(p) for p in data.get("p_values", [])],
            failures=[int(f) for f in data.get("failures", [])],
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationResult":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _check_ltt_args(betas: Sequence[float], n: int, eps: float, delta: float) -> None:
    if not betas:
        raise ValueError("betas must not be empty")
    if n < 1:
        raise ValueError("n must be a positive integer")
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")


def ltt_select(
    betas: Sequence[float],
    run_fn: Callable[[float, int], int],
    n: int,
    eps: float,
    delta: float,
) -> CalibrationResult:
    """
    Fixed-sequence Learn-then-Test.

    Args:
        betas:  candidate levels, ordered from conservative to aggressive.
                The order must be fixed before seeing any calibration data.
        run_fn: ``run_fn(beta, n)`` executes ``n`` fresh i.i.d. runs under level
                ``beta`` and returns how many failed.
        n:      runs per level.
        eps:    target failure rate.
        delta:  allowed probability of certifying a level whose risk exceeds eps.

    Testing stops at the first level whose p-value exceeds ``delta``; later
    levels are never run. Returns the last certified level (``beta=None`` if
    the first level already fails).
    """
    _check_ltt_args(betas, n, eps, delta)
    result = CalibrationResult(beta=None, eps=eps, delta=delta, n=n)
    for beta in betas:
        failures = int(run_fn(beta, n))
        p = ltt_p_value(failures, n, eps)
        result.tested.append(float(beta))
        result.p_values.append(p)
        result.failures.append(failures)
        if p > delta:
            break
        result.beta = float(beta)
    return result


def calibrate_from_failures(
    betas: Sequence[float],
    failures: Sequence[int],
    n: int,
    eps: float,
    delta: float,
) -> CalibrationResult:
    """
    Learn-then-Test on failure counts you already have, one per level in
    ``betas`` (same order). Each count must come from its own ``n`` runs at
    that level. Counts after the first non-rejection are ignored.
    """
    if len(failures) < 1 or len(failures) > len(betas):
        raise ValueError("failures must have between 1 and len(betas) entries")
    counts = iter(failures)
    return ltt_select(list(betas)[: len(failures)], lambda _b, _n: next(counts), n, eps, delta)


# ── Quantile table ────────────────────────────────────────────────────────────

Record = Union[Mapping, Tuple[str, str, int]]


class QuantileTable:
    """
    Logged output lengths per stratum (model, kind), with quantile lookups.

    ``kind`` labels the step type (for example "tool" or "final"). Records
    without a kind are stored under ``"any"``. A lookup for a stratum with too
    few samples falls back to the model's pooled samples, and then to None
    (the caller should use the worst case).
    """

    def __init__(self, samples: Optional[Mapping[Tuple[str, str], Iterable[int]]] = None,
                 min_samples: int = 30):
        if min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        self.min_samples = int(min_samples)
        self._samples: Dict[Tuple[str, str], List[int]] = {}
        for (model, kind), values in (samples or {}).items():
            for v in values:
                self.add(model, kind, v)

    @classmethod
    def from_records(cls, records: Iterable[Record], min_samples: int = 30) -> "QuantileTable":
        """
        Build from logged calls. Each record is a ``(model, kind, completion_tokens)``
        tuple or a mapping with keys ``model``, ``completion_tokens`` and optional ``kind``.
        """
        table = cls(min_samples=min_samples)
        for r in records:
            if isinstance(r, Mapping):
                table.add(r["model"], r.get("kind") or ANY_KIND, r["completion_tokens"])
            else:
                model, kind, tokens = r
                table.add(model, kind or ANY_KIND, tokens)
        return table

    def add(self, model: str, kind: str, completion_tokens: int) -> None:
        tokens = int(completion_tokens)
        if tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        bisect.insort(self._samples.setdefault((model, kind or ANY_KIND), []), tokens)

    def count(self, model: str, kind: str = ANY_KIND) -> int:
        return len(self._samples.get((model, kind), []))

    def _values(self, model: str, kind: str) -> Optional[List[int]]:
        values = self._samples.get((model, kind))
        if values is not None and len(values) >= self.min_samples:
            return values
        pooled = sorted(v for (m, _k), vs in self._samples.items() if m == model for v in vs)
        if len(pooled) >= self.min_samples:
            return pooled
        return None

    def bound(self, model: str, kind: str = ANY_KIND, beta: Optional[float] = None) -> Optional[int]:
        """
        Empirical (1 - beta) quantile of output tokens for (model, kind): the
        smallest logged value with at least a (1 - beta) fraction of samples at
        or below it. Returns None if beta is None or there are too few samples.
        """
        if beta is None:
            return None
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        values = self._values(model, kind or ANY_KIND)
        if values is None:
            return None
        idx = max(0, math.ceil((1.0 - beta) * len(values)) - 1)
        return values[min(idx, len(values) - 1)]

    def to_dict(self) -> dict:
        return {
            "min_samples": self.min_samples,
            "strata": [
                {"model": m, "kind": k, "completion_tokens": list(v)}
                for (m, k), v in sorted(self._samples.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "QuantileTable":
        table = cls(min_samples=int(data.get("min_samples", 30)))
        for s in data.get("strata", []):
            for v in s["completion_tokens"]:
                table.add(s["model"], s.get("kind") or ANY_KIND, v)
        return table

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "QuantileTable":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def resolve_beta(beta: Union[None, float, CalibrationResult]) -> Optional[float]:
    """Accept a raw beta or a CalibrationResult; None means "use preflight"."""
    if isinstance(beta, CalibrationResult):
        return beta.beta
    if beta is None:
        return None
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError("calibration_beta must be in [0, 1)")
    return beta
