"""
baar/monitor.py — is the certified guarantee still holding in deployment?

A certificate holds for runs that look like the calibration runs. When the
agent, its prompts, its models or its tasks change, it can stop holding without
any error. ``GuaranteeMonitor`` watches finished runs and raises an alarm when
the data show that the overspend rate is above ``eps``.

The test is anytime-valid: checking after every run never inflates the false
alarm rate. Under "the rate is at most eps" the evidence

    W_n = mean over f in FRACTIONS of  prod_{i<=n} (1 + (f / eps) * (X_i - eps))

(X_i = 1 if run i overspent) is a non-negative supermartingale with W_0 = 1, so by
Ville's inequality it ever reaches 1/alpha with probability at most alpha. The
alarm fires when it does.

This module has no third-party dependencies.
"""

from __future__ import annotations

import math
import threading
import warnings
from typing import Callable, List, Mapping, Optional

FRACTIONS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7)


class GuaranteeAlarm(UserWarning):
    """Warning category used when a monitor's alarm fires."""


class GuaranteeMonitor:
    """
    Sequential test of H0: P(run overspends) <= eps.

    Args:
        eps:      the certified risk level.
        alpha:    probability of a false alarm over the whole deployment.
        on_alarm: called once with the monitor when the alarm fires; by
                  default a ``GuaranteeAlarm`` warning is issued.
    """

    def __init__(self, eps: float, alpha: float = 0.05,
                 on_alarm: Optional[Callable[["GuaranteeMonitor"], None]] = None):
        if not 0 < eps < 1:
            raise ValueError("eps must be in (0, 1)")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.on_alarm = on_alarm
        self.n = 0
        self.failures = 0
        self._log_w: List[float] = [0.0] * len(FRACTIONS)
        self.alarm = False
        self.alarm_at: Optional[int] = None
        self._lock = threading.Lock()

    @classmethod
    def for_certificate(cls, cert, alpha: float = 0.05, **kwargs) -> "GuaranteeMonitor":
        """Monitor for the risk a ``Certificate`` of kind "calibrated" bounds."""
        if getattr(cert, "kind", None) != "calibrated":
            raise ValueError("only calibrated certificates have an observable risk "
                             "(an early-stopped run never shows whether it would have finished)")
        return cls(cert.eps, alpha=alpha, **kwargs)

    @property
    def evidence(self) -> float:
        """Current evidence against "rate <= eps" (alarm at 1/alpha)."""
        m = max(self._log_w)
        return math.exp(m) * sum(math.exp(v - m) for v in self._log_w) / len(self._log_w)

    @property
    def rate(self) -> float:
        return self.failures / self.n if self.n else 0.0

    def update(self, overspent: bool) -> bool:
        """Record one finished run; returns True if the alarm is (now or already) raised."""
        fire = False
        with self._lock:
            x = 1.0 if overspent else 0.0
            self.n += 1
            self.failures += int(x)
            for j, f in enumerate(FRACTIONS):
                self._log_w[j] += math.log1p((f / self.eps) * (x - self.eps))
            if not self.alarm and self.evidence >= 1.0 / self.alpha:
                self.alarm = True
                self.alarm_at = self.n
                fire = True
        if fire:
            if self.on_alarm is not None:
                self.on_alarm(self)
            else:
                warnings.warn(self.message(), GuaranteeAlarm, stacklevel=2)
        return self.alarm

    def message(self) -> str:
        return (f"Budget guarantee no longer holds: {self.failures} of {self.n} runs overspent "
                f"({self.rate:.1%}, certified at most {self.eps:.1%}). Runs have changed since "
                "calibration; log new runs and re-run baar-calibrate.")

    def to_dict(self) -> dict:
        return {"eps": self.eps, "alpha": self.alpha, "n": self.n, "failures": self.failures,
                "log_w": list(self._log_w), "alarm": self.alarm, "alarm_at": self.alarm_at}

    @classmethod
    def from_dict(cls, data: Mapping, **kwargs) -> "GuaranteeMonitor":
        mon = cls(float(data["eps"]), alpha=float(data["alpha"]), **kwargs)
        mon.n = int(data["n"])
        mon.failures = int(data["failures"])
        log_w = [float(v) for v in data.get("log_w", [])]
        if len(log_w) != len(FRACTIONS):
            raise ValueError("monitor state does not match this version's FRACTIONS")
        mon._log_w = log_w
        mon.alarm = bool(data.get("alarm", False))
        mon.alarm_at = data.get("alarm_at")
        return mon
