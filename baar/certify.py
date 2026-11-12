"""
baar/certify.py — from logged runs to a certified router setting.

``certify_calibrated`` picks the reservation level of
``on_output_overrun="calibrated"`` so that a whole run spends more than its
budget with probability at most ``eps`` (confidence ``1 - delta``).
``certify_early_stop`` fits and certifies an ``EarlyStopPolicy``. Both split the
logged runs themselves, replay them exactly, and return a ``Certificate`` that
can be saved, loaded and passed to the router::

    runs = load_runs("runs.jsonl").runs
    cert = certify_calibrated(runs, budget=0.50, eps=0.05)
    cert.save("baar_cert.json")
    router = BAARRouter(budget=0.50, **Certificate.load("baar_cert.json").router_kwargs())

Replay. Before each call the router admits it iff

    spent + price_in * ceil(est_prompt * (1 + margin)) + price_out * u <= budget

where ``u`` is the planned output (the calibrated quantile, never above the
call's worst case). A refused call ends the run; refusing never changes the
calls before it, so one batch of runs logged without a binding budget can be
replayed under every level. A run *overspends* when what it was admitted to
spend is more than the budget.

Requirements for the guarantee: runs are i.i.d. draws from deployment (same
agent, models, prompts and task mix), the budget and prompt margin are fixed
before calibrating and used unchanged by the router, and the fallback
(worst-case reservation) does not overspend, which needs prompt estimates that
are not below the billed prompt. The certificate reports how often the fallback
overspent on the calibration runs; a non-zero count means the last condition
fails.

This module has no third-party dependencies.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from baar.calibration import CalibrationResult, QuantileTable, calibrate_from_failures
from baar.early_stop import DEFAULT_TAUS, EarlyStopPolicy, calibrate_early_stop, fit_early_stop
from baar.runlog import split_runs

DEFAULT_BETAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16,
                 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SCHEMA = 1
_TOL = 1e-12

Prices = Mapping[str, Tuple[float, float]]


class NotEnoughRuns(ValueError):
    """Too few logged runs to certify anything at the requested eps and delta."""

    def __init__(self, message: str, have: int, need: int):
        super().__init__(message)
        self.have = have
        self.need = need


def min_calibration_runs(eps: float, delta: float) -> int:
    """Smallest n at which zero failures certify: (1 - eps)^n <= delta."""
    if not 0 < eps < 1 or not 0 < delta < 1:
        raise ValueError("eps and delta must be in (0, 1)")
    return math.ceil(math.log(delta) / math.log1p(-eps) - 1e-9)


def min_total_runs(eps: float, delta: float, holdout_share: float) -> int:
    """Logged runs needed when ``holdout_share`` of them are not used for calibration."""
    n = min_calibration_runs(eps, delta)
    total = n
    while total - int(round(holdout_share * total)) < n:   # same rounding as split_runs
        total += 1
    return total


# ── Exact replay ──────────────────────────────────────────────────────────────

def _prices(step: Mapping, prices: Optional[Prices]) -> Tuple[float, float]:
    if prices is not None and step.get("model") in prices:
        p_in, p_out = prices[step["model"]]
        return float(p_in), float(p_out)
    if step.get("price_in") is not None and step.get("price_out") is not None:
        return float(step["price_in"]), float(step["price_out"])
    raise ValueError(
        f"no price for model {step.get('model')!r} in the log (LiteLLM has no pricing for it); "
        "pass prices={model: (usd_per_input_token, usd_per_output_token)}"
    )


def replay_spend(run: Mapping, plan: Callable[[Mapping], Optional[float]], budget: float,
                 margin: float, prices: Optional[Prices] = None) -> Tuple[float, Optional[int]]:
    """
    Replay one logged run under an admit/refuse controller.

    ``plan(step)`` is the output reserved for the call (None: nothing can be
    reserved, the call is refused). Returns (spent, refused_at), where
    ``refused_at`` is the index of the first refused call or None.
    """
    spent = 0.0
    for t, step in enumerate(run["steps"]):
        u = plan(step)
        if u is None:
            return spent, t
        p_in, p_out = _prices(step, prices)
        est = int(step.get("est_prompt_tokens", step["prompt_tokens"]))
        need = spent + p_in * math.ceil(est * (1 + margin)) + p_out * float(u)
        if need > budget + _TOL:
            return spent, t
        spent += float(step["cost"])
    return spent, None


def _level_plan(table: QuantileTable, beta: Optional[float],
                default_max_output: Optional[int]) -> Callable[[Mapping], Optional[float]]:
    """Output reserved at level beta: the calibrated quantile, capped by the call's worst case."""
    cache: Dict[str, Optional[int]] = {}

    def plan(step: Mapping) -> Optional[float]:
        model = step["model"]
        if model not in cache:
            cache[model] = table.bound(model, "any", beta)
        worst = step.get("max_output_tokens") or default_max_output
        q = cache[model]
        if q is None:
            return None if worst is None else float(worst)
        return float(q if worst is None else min(q, int(worst)))

    return plan


def overspends(run: Mapping, plan: Callable[[Mapping], Optional[float]], budget: float,
               margin: float, prices: Optional[Prices] = None) -> bool:
    spent, _ = replay_spend(run, plan, budget, margin, prices)
    return spent > budget + _TOL


# ── Certificates ──────────────────────────────────────────────────────────────

@dataclass
class Certificate:
    """
    A certified router setting and the evidence for it.

    ``kind`` is "calibrated" or "early_stop". ``certified`` is False when no
    level could be certified; the router then falls back to the safe setting
    (worst-case reservation, or never stopping).
    """

    kind: str
    budget: float
    eps: float
    delta: float
    certified: bool
    level: Optional[float]                  # beta (calibrated) or threshold tau (early_stop)
    n_runs: int                             # runs read
    n_calibration: int                      # runs used by Learn-then-Test
    calibration: dict                       # CalibrationResult.to_dict()
    payload: dict                           # QuantileTable / EarlyStopPolicy as dict
    settings: dict = field(default_factory=dict)   # margin, harm, seed, shares
    diagnostics: dict = field(default_factory=dict)
    schema: int = SCHEMA
    created: str = ""

    def router_kwargs(self) -> dict:
        """Keyword arguments for ``BAARRouter`` (use the same ``budget``)."""
        if self.kind == "calibrated":
            return {
                "on_output_overrun": "calibrated",
                "output_quantiles": QuantileTable.from_dict(self.payload),
                "calibration_beta": CalibrationResult.from_dict(self.calibration),
                "prompt_margin": float(self.settings.get("margin", 0.2)),
            }
        if self.kind == "early_stop":
            return {"early_stop": EarlyStopPolicy.from_dict(self.payload)}
        raise ValueError(f"unknown certificate kind {self.kind!r}")

    def summary(self) -> str:
        what = ("a run spends more than its budget" if self.kind == "calibrated" else
                "a run is stopped although it would have "
                + ("finished" if self.settings.get("harm", "finish") == "finish" else "solved its task")
                + " within budget")
        lines = [f"Budget: ${self.budget:g} per run"]
        if self.certified:
            lvl = (f"reservation level beta = {self.level:g}" if self.kind == "calibrated"
                   else f"stopping threshold tau = {self.level:g}")
            lines.append(f"Certified: {lvl}")
            lines.append(f"Guarantee: P({what}) <= {self.eps:g}, with confidence {1 - self.delta:g}")
        else:
            safe = "worst-case reservation (preflight)" if self.kind == "calibrated" else "never stop"
            lines.append(f"Not certified: no level passed. The router will use the safe setting: {safe}.")
        lines.append(f"Runs: {self.n_runs} read, {self.n_calibration} used for calibration")
        for msg in self.diagnostics.get("warnings", []):
            lines.append(f"Warning: {msg}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "Certificate":
        if int(data.get("schema", SCHEMA)) > SCHEMA:
            raise ValueError("certificate was written by a newer version of baar-core")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Certificate":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _check_common(runs: Sequence[Mapping], budget: float, eps: float, delta: float) -> None:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0 < eps < 1 or not 0 < delta < 1:
        raise ValueError("eps and delta must be in (0, 1)")
    for r in runs:
        if not r.get("steps"):
            raise ValueError(f"run {r.get('run')!r} has no steps")


def _not_enough(have: int, eps: float, delta: float, holdout: float, what: str) -> NotEnoughRuns:
    need = min_total_runs(eps, delta, holdout)
    return NotEnoughRuns(
        f"{have} runs is too few to certify {what} at eps={eps:g}, delta={delta:g}: "
        f"at least {need} are needed ({min_calibration_runs(eps, delta)} for calibration), "
        f"so log {need - have} more.", have=have, need=need)


def certify_calibrated(runs: Sequence[Mapping], budget: float, eps: float, delta: float = 0.05, *,
                       margin: float = 0.2, betas: Sequence[float] = DEFAULT_BETAS,
                       table_share: float = 0.3, seed: int = 0, min_samples: int = 30,
                       prices: Optional[Prices] = None,
                       default_max_output: Optional[int] = None) -> Certificate:
    """
    Certify the reservation level of ``on_output_overrun="calibrated"``.

    ``table_share`` of the runs (chosen with ``seed``) build the output-length
    table; the rest calibrate the level with fixed-sequence Learn-then-Test,
    from the most conservative level to the most aggressive. Raises
    ``NotEnoughRuns`` when the calibration split is too small for anything to
    pass even with zero overspending runs.
    """
    runs = list(runs)
    _check_common(runs, budget, eps, delta)
    if not 0 < table_share < 1:
        raise ValueError("table_share must be in (0, 1)")
    tab_idx, cal_idx = split_runs(len(runs), (table_share, 1 - table_share), seed)
    if len(cal_idx) < min_calibration_runs(eps, delta):
        raise _not_enough(len(runs), eps, delta, table_share, "the calibrated mode")
    table = QuantileTable.from_records(
        ({"model": s["model"], "completion_tokens": s["completion_tokens"]}
         for i in tab_idx for s in runs[i]["steps"]), min_samples=min_samples)
    cal = [runs[i] for i in cal_idx]
    levels = sorted(set(float(b) for b in betas))   # small beta = large reservation = conservative
    failures = [sum(overspends(r, _level_plan(table, b, default_max_output), budget, margin, prices)
                    for r in cal) for b in levels]
    cr = calibrate_from_failures(levels, failures, len(cal), eps, delta)
    fallback = _level_plan(table, None, default_max_output)
    fb_fail = sum(overspends(r, fallback, budget, margin, prices) for r in cal)
    no_limit = sum(1 for r in cal for s in r["steps"]
                   if not (s.get("max_output_tokens") or default_max_output))
    warnings = []
    if fb_fail:
        warnings.append(
            f"{fb_fail} of {len(cal)} calibration runs overspent even with the worst-case reservation, "
            "so prompt estimates fell below the billed prompt. Raise the prompt margin; the guarantee "
            "assumes the fallback never overspends.")
    if no_limit:
        warnings.append(f"{no_limit} logged calls have no known maximum output; the fallback refuses them.")
    if cr.beta is None:
        warnings.append("No level was certified; the router will reserve the worst case.")
    models = sorted({s["model"] for r in runs for s in r["steps"]})
    thin = [m for m in models if table.bound(m, "any", 0.5) is None]
    if thin:
        warnings.append(f"Too few logged calls for {', '.join(thin)} (fewer than {min_samples} in the table "
                        "split); those models get the worst-case reservation.")
    return Certificate(
        kind="calibrated", budget=float(budget), eps=float(eps), delta=float(delta),
        certified=cr.beta is not None, level=cr.beta, n_runs=len(runs), n_calibration=len(cal),
        calibration=cr.to_dict(), payload=table.to_dict(),
        settings={"margin": float(margin), "table_share": float(table_share), "seed": int(seed),
                  "betas": levels},
        diagnostics={"failures_by_level": dict(zip([str(b) for b in levels], failures)),
                     "fallback_failures": fb_fail, "models": models, "warnings": warnings},
        created=_now())


def certify_early_stop(runs: Sequence[Mapping], budget: float, eps: float, delta: float = 0.05, *,
                       harm: str = "finish", fit_share: float = 0.5, seed: int = 0,
                       counter_keys: Sequence[str] = (), taus: Sequence[float] = DEFAULT_TAUS,
                       **fit_kwargs) -> Certificate:
    """
    Fit an ``EarlyStopPolicy`` on ``fit_share`` of the runs and certify its
    threshold on the rest. ``harm="resolve"`` needs ``resolved`` on every run.
    """
    runs = list(runs)
    _check_common(runs, budget, eps, delta)
    if harm not in ("finish", "resolve"):
        raise ValueError('harm must be "finish" or "resolve"')
    if harm == "resolve" and any(r.get("resolved") is None for r in runs):
        raise ValueError('harm="resolve" needs "resolved" (true/false) on every run')
    if not 0 < fit_share < 1:
        raise ValueError("fit_share must be in (0, 1)")
    fit_idx, cal_idx = split_runs(len(runs), (fit_share, 1 - fit_share), seed)
    if len(cal_idx) < min_calibration_runs(eps, delta):
        raise _not_enough(len(runs), eps, delta, fit_share, "early stopping")
    policy = fit_early_stop([runs[i] for i in fit_idx], budget, counter_keys=counter_keys, **fit_kwargs)
    cal = [runs[i] for i in cal_idx]
    certified = calibrate_early_stop(policy, cal, eps, delta, harm=harm, taus=taus)
    cr = certified.meta["calibration"]
    warnings = []
    if not policy.weights:
        warnings.append("No checkpoint had enough runs of both kinds (over and within budget) to fit a "
                        "model; the policy never stops.")
    over = sum(1 for r in runs if r["cost"] > budget + _TOL) if all("cost" in r for r in runs) else None
    return Certificate(
        kind="early_stop", budget=float(budget), eps=float(eps), delta=float(delta),
        certified=certified.threshold is not None, level=certified.threshold, n_runs=len(runs),
        n_calibration=len(cal), calibration=cr.to_dict(), payload=certified.to_dict(),
        settings={"harm": harm, "fit_share": float(fit_share), "seed": int(seed),
                  "counter_keys": list(counter_keys)},
        diagnostics={"runs_over_budget": over, "fitted_checkpoints": sorted(policy.weights),
                     "warnings": warnings},
        created=_now())
