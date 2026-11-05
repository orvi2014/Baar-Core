"""
baar/early_stop.py — certified early stopping for long-running agents.

Long agent runs that will cost more than their budget are refused only when
the money runs out, after most of it has been spent. At fixed checkpoints
(``step`` = calls completed; 0 = before the first call) an ``EarlyStopPolicy``
scores how likely the run is to exceed the budget and stops it when the score
reaches a threshold ``tau``.

The threshold is certified with fixed-sequence Learn-then-Test
(``calibrate_early_stop``): with ``harm`` = "the policy stopped a run that would
have finished (or resolved its task) within the budget", the certified policy
satisfies

    P( P(harm) <= eps ) >= 1 - delta

for i.i.d. runs from the deployment distribution. P(harm) is a share of *all*
runs; as a share of the runs that would have finished (or solved), the rate is
higher by the inverse of their frequency. Stopping never changes the
steps before it, so one batch of logged runs can be replayed under every
candidate threshold. The fallback, never stopping, has zero harm.

Fit the model on runs disjoint from the calibration runs, and fix the budget
before calibrating. This module has no third-party dependencies.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from baar.calibration import CalibrationResult, calibrate_from_failures
from baar.core.exceptions import BudgetExhausted

DEFAULT_CHECKPOINTS = (0, 5, 10, 15, 20, 30, 40, 50, 60, 80)
DEFAULT_TAUS = (0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5,
                0.4, 0.3, 0.2, 0.1)
MIN_ALIVE = 50


class RunStopped(BudgetExhausted):
    """
    Raised before a call when a certified ``EarlyStopPolicy`` stops the run.
    Subclass of ``BudgetExhausted`` so existing budget handlers catch it.
    """

    def __init__(self, message: str, remaining: float = 0.0, score: float = 0.0,
                 threshold: Optional[float] = None, step: int = 0):
        super().__init__(message, remaining=remaining)
        self.score = score
        self.threshold = threshold
        self.step = step


@dataclass
class RunState:
    """What the controller knows before the next call of a run."""

    step: int = 0                       # calls completed so far
    spent: float = 0.0                  # USD spent so far
    first_prompt_tokens: int = 0        # prompt tokens of the first call
    context_tokens: int = 0             # prompt tokens of the next call
    output_tokens: List[int] = field(default_factory=list)  # completion tokens so far
    counters: Dict[str, float] = field(default_factory=dict)  # behaviour counts (errors, repeats, ...)


def featurize(state: RunState, counter_keys: Sequence[str] = ()) -> List[float]:
    """Feature vector used by the per-checkpoint logistic models."""
    first = math.log1p(max(0, state.first_prompt_tokens))
    if state.step == 0:
        base = [1.0, first, 0.0, 0.0, 0.0, 0.0]
    else:
        outs = state.output_tokens or [0]
        base = [1.0, first, math.log1p(max(0.0, state.spent) * 1e4), math.log1p(max(0, state.context_tokens)),
                math.log1p(sum(outs) / len(outs)), math.log1p(max(outs))]
    return base + [math.log1p(max(0.0, float(state.counters.get(k, 0)))) for k in counter_keys]


# ── Logistic regression (pure Python) ─────────────────────────────────────────

def _solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-15:
            raise ValueError("singular system")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(X: Sequence[Sequence[float]], y: Sequence[float], l2: float = 1e-3,
                 iters: int = 50) -> List[float]:
    """Ridge-regularised logistic regression fitted by Newton's method."""
    d = len(X[0])
    w = [0.0] * d
    for _ in range(iters):
        g = [l2 * wj for wj in w]
        H = [[(l2 if i == j else 0.0) for j in range(d)] for i in range(d)]
        for xi, yi in zip(X, y):
            p = _sigmoid(sum(a * b for a, b in zip(xi, w)))
            r = p - yi
            s = p * (1 - p)
            for i in range(d):
                g[i] += r * xi[i]
                si = s * xi[i]
                if si:
                    row = H[i]
                    for j in range(d):
                        row[j] += si * xi[j]
        step = _solve(H, g)
        w = [a - b for a, b in zip(w, step)]
        if max(abs(v) for v in step) < 1e-8:
            break
    return w


# ── Policy ────────────────────────────────────────────────────────────────────

@dataclass
class EarlyStopPolicy:
    """
    Per-checkpoint logistic scores of "this run will cost more than ``budget``".
    ``should_stop`` is True only at a checkpoint with a model whose score is at
    least ``threshold``; ``threshold=None`` never stops.
    """

    checkpoints: List[int]
    weights: Dict[int, List[float]]
    threshold: Optional[float] = None
    counter_keys: List[str] = field(default_factory=list)
    budget: float = 0.0
    meta: Dict = field(default_factory=dict)

    def score(self, state: RunState) -> Optional[float]:
        w = self.weights.get(int(state.step))
        if w is None:
            return None
        x = featurize(state, self.counter_keys)
        return _sigmoid(sum(a * b for a, b in zip(x, w)))

    def should_stop(self, state: RunState) -> bool:
        if self.threshold is None or int(state.step) not in self.checkpoints:
            return False
        s = self.score(state)
        return s is not None and s >= self.threshold

    def to_dict(self) -> dict:
        d = asdict(self)
        d["weights"] = {str(k): v for k, v in self.weights.items()}
        d["meta"] = {k: (asdict(v) if isinstance(v, CalibrationResult) else v) for k, v in self.meta.items()}
        return d

    @classmethod
    def from_dict(cls, data: Mapping) -> "EarlyStopPolicy":
        meta = dict(data.get("meta", {}))
        if isinstance(meta.get("calibration"), Mapping):
            meta["calibration"] = CalibrationResult(**meta["calibration"])
        return cls(checkpoints=[int(c) for c in data["checkpoints"]],
                   weights={int(k): [float(x) for x in v] for k, v in data["weights"].items()},
                   threshold=data.get("threshold"), counter_keys=list(data.get("counter_keys", [])),
                   budget=float(data.get("budget", 0.0)), meta=meta)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "EarlyStopPolicy":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ── Logged runs → states ──────────────────────────────────────────────────────

def _run_cost(run: Mapping) -> float:
    return sum(float(s["cost"]) for s in run["steps"])


def state_at(run: Mapping, k: int) -> RunState:
    """State of a logged run before its call k+1 (k calls completed)."""
    steps = run["steps"]
    done = steps[:k]
    return RunState(
        step=k,
        spent=sum(float(s["cost"]) for s in done),
        first_prompt_tokens=int(steps[0]["prompt_tokens"]) if steps else 0,
        context_tokens=int(steps[k]["prompt_tokens"]) if k < len(steps) else (
            int(done[-1]["prompt_tokens"]) if done else 0),
        output_tokens=[int(s["completion_tokens"]) for s in done],
        counters=dict(done[-1].get("counters") or {}) if done else {},
    )


def fit_early_stop(runs: Iterable[Mapping], budget: float,
                   checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
                   counter_keys: Sequence[str] = ()) -> EarlyStopPolicy:
    """
    Fit one logistic model per checkpoint predicting "total cost > budget".

    Each run is a dict with ``steps`` (dicts with ``prompt_tokens``,
    ``completion_tokens``, ``cost`` and optionally ``counters``, the cumulative
    behaviour counts after that step), ``finished`` and optionally ``resolved``.
    Checkpoints with fewer than 50 runs still going, or with only one class, get
    no model (the policy never stops there). The returned policy has no
    threshold yet: certify one with ``calibrate_early_stop`` on other runs.
    """
    runs = list(runs)
    weights: Dict[int, List[float]] = {}
    for k in checkpoints:
        alive = [r for r in runs if len(r["steps"]) > k]
        y = [1.0 if _run_cost(r) > budget else 0.0 for r in alive]
        if len(alive) < MIN_ALIVE or not 0 < sum(y) < len(y):
            continue
        X = [featurize(state_at(r, k), counter_keys) for r in alive]
        weights[int(k)] = fit_logistic(X, y)
    return EarlyStopPolicy(checkpoints=[int(c) for c in checkpoints], weights=weights, threshold=None,
                           counter_keys=list(counter_keys), budget=float(budget),
                           meta={"fit_runs": len(runs)})


def _stops(policy: EarlyStopPolicy, run: Mapping, tau: float) -> bool:
    for k in policy.checkpoints:
        if k >= len(run["steps"]):
            break
        s = policy.score(state_at(run, k))
        if s is not None and s >= tau:
            return True
    return False


def calibrate_early_stop(policy: EarlyStopPolicy, runs: Iterable[Mapping], eps: float, delta: float,
                         harm: str = "finish", taus: Sequence[float] = DEFAULT_TAUS) -> EarlyStopPolicy:
    """
    Certify a stopping threshold with fixed-sequence Learn-then-Test.

    Each calibration run is replayed under every threshold in ``taus`` (stopping
    never changes earlier steps, so the same runs serve every threshold). Harm is
    "stopped a run that would have finished within the budget" (``harm="finish"``)
    or "... would have resolved its task within the budget" (``harm="resolve"``,
    needs ``resolved`` on each run). Thresholds are tested from the most
    conservative (never stop) toward the most aggressive.

    Guarantee: if calibration runs are i.i.d. draws from deployment, the returned
    threshold satisfies P(harm rate > eps) <= delta, where the harm rate is the
    share of all runs (not only the good ones) that are harmed. The fallback, never stopping
    (``threshold=None``), has zero harm. Requirements: fit the model on runs
    disjoint from these calibration runs, and fix the budget before calibrating.
    """
    if harm not in ("finish", "resolve"):
        raise ValueError('harm must be "finish" or "resolve"')
    runs = list(runs)
    taus = sorted(set(float(t) for t in taus), reverse=True)
    key = "finished" if harm == "finish" else "resolved"
    harmed = [bool(r.get(key)) and _run_cost(r) <= policy.budget for r in runs]
    failures = [sum(1 for r, h in zip(runs, harmed) if h and _stops(policy, r, t)) for t in taus]
    cr = calibrate_from_failures(list(range(len(taus))), failures, len(runs), eps, delta)
    threshold = taus[int(cr.beta)] if cr.beta is not None else None
    result = CalibrationResult(beta=threshold, eps=cr.eps, delta=cr.delta, n=cr.n,
                               tested=[taus[int(i)] for i in cr.tested], p_values=cr.p_values,
                               failures=cr.failures)
    meta = dict(policy.meta, calibration=result, harm=harm)
    return EarlyStopPolicy(checkpoints=list(policy.checkpoints), weights=dict(policy.weights),
                           threshold=threshold, counter_keys=list(policy.counter_keys),
                           budget=policy.budget, meta=meta)
