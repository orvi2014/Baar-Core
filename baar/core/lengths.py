"""
baar/core/lengths.py — learned output lengths for "preflight" reservations.

Output length is set by when the model emits its end token, so it is heavy-
tailed and a single point estimate is a poor fit. Instead of the router's guess
(or the model's full max output), "preflight" with ``overrun_risk=r`` reserves
the split-conformal (1 - r) quantile of the lengths each model actually
produced: with n past lengths sorted, the value at index ceil((n + 1)(1 - r)) - 1.
When calls are exchangeable, a new response exceeds it with probability at most
r. When the history is too short for that index, there is no prediction and the
worst case is reserved.

Lengths are kept per model in a sliding window, in memory. Share one instance
between routers (``BAARRouter(output_lengths=...)``) to pool what they learn.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Optional


class OutputLengths:
    """
    Thread-safe per-model history of completion-token counts.

    Args:
        min_samples: lengths needed before any prediction is made.
        window:      lengths kept per model (oldest dropped first).
    """

    def __init__(self, *, min_samples: int = 20, window: int = 500) -> None:
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if window < 1:
            raise ValueError("window must be at least 1")
        self.min_samples = min_samples
        self.window = window
        self._lengths: dict[str, deque] = {}
        self._lock = threading.Lock()

    def observe(self, model: str, completion_tokens: int) -> None:
        """Record one completed call's output length."""
        if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool) or completion_tokens < 0:
            return
        with self._lock:
            history = self._lengths.get(model)
            if history is None:
                history = self._lengths[model] = deque(maxlen=self.window)
            history.append(completion_tokens)

    def quantile(self, model: str, risk: float) -> Optional[int]:
        """
        The length a new response from ``model`` exceeds with probability at most
        ``risk``, or None (reserve the worst case) when ``risk`` is 0 or the
        history is too short to support it.
        """
        if not 0 <= risk < 1:
            raise ValueError("risk must be in [0, 1)")
        if risk == 0:
            return None
        with self._lock:
            history = list(self._lengths.get(model, ()))
        n = len(history)
        if n < self.min_samples:
            return None
        index = math.ceil((n + 1) * (1 - risk)) - 1
        if index > n - 1:
            return None
        return sorted(history)[index]

    def count(self, model: str) -> int:
        with self._lock:
            return len(self._lengths.get(model, ()))
