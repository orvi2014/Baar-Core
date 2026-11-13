"""
Budget tracker — real token-based cost using LiteLLM pricing.
This is the financial source of truth for the entire system.
"""

import math
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, List, Optional, Union
from litellm import completion_cost, cost_per_token

from baar.core.stores import BudgetStore, MemoryBudgetStore, FileBudgetStore, SQLiteBudgetStore

class BudgetWindow(str, Enum):
    """
    Time window for automatic budget reset.

    Pass to BudgetTracker to enforce daily, monthly, or hourly spend caps
    instead of a lifetime total. Requires a persistent store (SQLiteBudgetStore
    or FileBudgetStore) to survive process restarts across window boundaries.

    Example::

        tracker = BudgetTracker(
            total_budget=0.10,
            window=BudgetWindow.DAILY,
            store=SQLiteBudgetStore("budgets.db", namespace="user:alice"),
        )
    """
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


@dataclass
class Alert:
    """
    Budget threshold alert. Fires ``callback`` when utilization crosses ``threshold``.

    Args:
        threshold: Utilization fraction 0.0–1.0 that triggers the alert.
        callback:  Called with a dict: {threshold, utilization, spent, remaining, total_budget}.
        once:      If True (default), fires only the first time the threshold is crossed.

    Example::

        Alert(threshold=0.8, callback=lambda info: print("80% used!"))
    """
    threshold: float
    callback: Callable
    once: bool = True
    _fired: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.threshold = max(0.0, min(1.0, float(self.threshold)))


class WindowedBudgetStore(BudgetStore):
    """
    Wraps any BudgetStore to automatically reset spend each billing window.
    A fresh sub-store is created per time period using the same backend;
    historical period data is preserved (not deleted).

    Prefer constructing via BudgetTracker(window=...) rather than directly.
    """

    def __init__(self, store: BudgetStore, window: BudgetWindow) -> None:
        self._base_store = store
        self._window = window
        self._current_key: Optional[str] = None
        self._current_store: BudgetStore = store
        self._lock = threading.Lock()
        self._lease_stores: dict = {}

    def _period_key(self) -> str:
        now = datetime.now(timezone.utc)
        if self._window == BudgetWindow.HOURLY:
            return now.strftime("%Y-%m-%dT%H")
        elif self._window == BudgetWindow.DAILY:
            return now.strftime("%Y-%m-%d")
        return now.strftime("%Y-%m")

    def _get_store(self) -> BudgetStore:
        key = self._period_key()
        with self._lock:
            if key == self._current_key:
                return self._current_store
            self._current_store = self._make_store(key)
            self._current_key = key
            return self._current_store

    def _make_store(self, key: str) -> BudgetStore:
        base = self._base_store
        for_namespace = getattr(base, "for_namespace", None)
        if callable(for_namespace):  # RedisBudgetStore, PostgresBudgetStore
            return for_namespace(f"{base._namespace}:{key}")
        if isinstance(base, SQLiteBudgetStore):
            return SQLiteBudgetStore(base._db_path, namespace=f"{base._namespace}:{key}")
        if isinstance(base, FileBudgetStore):
            return FileBudgetStore(base._path, namespace=f"{base._namespace}:{key}")
        if isinstance(base, MemoryBudgetStore):
            return MemoryBudgetStore()
        raise TypeError(
            f"WindowedBudgetStore does not know how to create a period store for "
            f"{type(base).__name__}. Use a built-in store, give the store a "
            "for_namespace(name) method, or subclass WindowedBudgetStore and override _make_store."
        )

    def get_spent(self) -> float:
        return self._get_store().get_spent()

    def add_spent(self, amount: float) -> float:
        return self._get_store().add_spent(amount)

    def set_spent(self, value: float) -> None:
        self._get_store().set_spent(value)

    def reset(self) -> None:
        self._get_store().reset()

    def atomic_check_and_reserve(self, total_budget: float, amount: float) -> bool:
        return self._get_store().atomic_check_and_reserve(total_budget, amount)

    # A lease lives in the period store it was taken in, even after the window rolls over.

    @property
    def supports_leases(self) -> bool:  # type: ignore[override]
        return self._base_store.supports_leases

    def reserve_lease(self, total_budget: float, amount: float, ttl: float) -> Optional[str]:
        store = self._get_store()
        lease_id = store.reserve_lease(total_budget, amount, ttl)
        if lease_id is not None:
            with self._lock:
                self._lease_stores[lease_id] = store
        return lease_id

    def _lease_store(self, lease_id: str) -> BudgetStore:
        with self._lock:
            return self._lease_stores.get(lease_id) or self._current_store

    def extend_lease(self, lease_id: str, total_budget: float, extra: float, ttl: float) -> bool:
        return self._lease_store(lease_id).extend_lease(lease_id, total_budget, extra, ttl)

    def renew_lease(self, lease_id: str, ttl: float) -> bool:
        return self._lease_store(lease_id).renew_lease(lease_id, ttl)

    def release_lease(self, lease_id: str, amount: float) -> None:
        with self._lock:
            store = self._lease_stores.pop(lease_id, None) or self._current_store
        store.release_lease(lease_id, amount)

    def get_reserved(self) -> float:
        return self._get_store().get_reserved()


# Reservation/cancellation methods allow achat() to atomically claim budget
# before the async LLM call, preventing concurrent callers from both passing
# the affordability check and together overshooting the budget.


class BudgetExceeded(Exception):
    """Raised when a model call would exceed the remaining budget."""

    def __init__(self, requested: float, remaining: float, model: str):
        self.requested = requested
        self.remaining = remaining
        self.model = model
        super().__init__(
            f"Budget exceeded: model '{model}' would cost ~${requested:.6f} "
            f"but only ${remaining:.6f} remains."
        )


@dataclass
class SpendRecord:
    step: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    cumulative_cost: float


DEFAULT_LEASE_TTL = 900.0  # seconds; the router derives its own from call_timeout


@dataclass
class Reservation:
    """
    Money held for one call. With a store that supports leases it expires
    ``ttl`` seconds after it was taken or last renewed, so a worker that dies
    mid-call cannot hold the budget forever.
    """

    amount: float
    lease_id: str
    ttl: float
    renewed_at: float = field(default_factory=time.monotonic)
    released: bool = False


class BudgetTracker:
    """
    Tracks real spend using LiteLLM's live pricing data.
    Never uses hardcoded costs — always derives from actual token counts.

    Args:
        total_budget: Maximum allowed spend in USD.
        store:        Where to persist spend. Defaults to in-memory (no persistence).
                      Pass a FileBudgetStore or SQLiteBudgetStore for persistence
                      across process restarts.
    """

    def __init__(
        self,
        total_budget: float,
        store: Optional[BudgetStore] = None,
        max_records: Optional[int] = None,
        window: Optional[BudgetWindow] = None,
        alerts: Optional[List[Alert]] = None,
        lease_ttl: Optional[float] = None,
    ) -> None:
        self.total_budget = total_budget
        if lease_ttl is not None and not (lease_ttl > 0):
            raise ValueError("lease_ttl must be a positive number of seconds")
        self.lease_ttl = float(lease_ttl) if lease_ttl is not None else DEFAULT_LEASE_TTL
        base_store: BudgetStore = store if store is not None else MemoryBudgetStore()
        self._store: BudgetStore = (
            WindowedBudgetStore(base_store, window) if window is not None else base_store
        )
        self._records: Union[list, deque] = deque(maxlen=max_records) if max_records is not None else []
        self._step: int = 0
        self._alerts: List[Alert] = list(alerts) if alerts else []
        self._alert_lock = threading.Lock()
        self._last_window_key: Optional[str] = None

    # _spent is kept as a property so existing tests that do
    # `tracker._spent = X` continue to work — the assignment is forwarded
    # transparently to the store.

    @property
    def _spent(self) -> float:
        return self._store.get_spent()

    @_spent.setter
    def _spent(self, value: float) -> None:
        self._store.set_spent(float(value))

    @property
    def spent(self) -> float:
        return self._store.get_spent()

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self._store.get_spent())

    @property
    def utilization(self) -> float:
        """0.0 → 1.0 showing how much budget has been consumed."""
        if self.total_budget <= 0:
            return 1.0
        return min(1.0, self._store.get_spent() / self.total_budget)

    def cost_from_response(self, response) -> float:
        """
        Extract real cost from a LiteLLM completion response.
        Uses completion_cost() which reads live pricing from LiteLLM's
        model_prices_and_context_window.json — no hardcoding.
        """
        try:
            cost = completion_cost(completion_response=response)
            return float(cost)
        except Exception:
            usage = getattr(response, "usage", None)
            if usage:
                model = getattr(response, "model", None)
                if not model:
                    return 0.0
                try:
                    in_cost, out_cost = cost_per_token(
                        model=model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                    )
                    return float(in_cost + out_cost)
                except Exception:
                    pass
            return 0.0

    # Fallback estimate used when LiteLLM has no pricing data for a model.
    # $0.0001 ≈ a short gpt-4o-mini call — low enough not to block normal budgets
    # (e.g. $0.05) but nonzero so the kill-switch still fires at micro-budgets
    # (e.g. $0.00005). Register real pricing via litellm.register_model() to
    # replace this fallback with accurate per-model data.
    _UNKNOWN_MODEL_FALLBACK_COST = 0.0001

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int = 500) -> float:
        """
        Pre-flight cost estimate before making a call.
        Default completion_tokens=500 — more realistic than 200 for complex tasks.
        Pass a tighter estimate when you know the expected output length.
        """
        try:
            in_cost, out_cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            cost = float(in_cost + out_cost)
            if not math.isfinite(cost):
                raise ValueError(f"non-finite cost for model '{model}'")
            return cost
        except Exception:
            warnings.warn(
                f"BudgetTracker: no pricing data for model '{model}'. "
                f"Using ${self._UNKNOWN_MODEL_FALLBACK_COST} conservative estimate. "
                "For accurate tracking, register real pricing with "
                "litellm.register_model() before using BAARRouter.",
                UserWarning,
                stacklevel=2,
            )
            return self._UNKNOWN_MODEL_FALLBACK_COST

    def input_price_per_token(self, model: str, prompt_tokens: int = 1) -> Optional[float]:
        """
        USD price of one prompt token for ``model`` at a prompt of ``prompt_tokens``
        (rates can rise with prompt length), or None if LiteLLM has no pricing.
        """
        n = max(1, int(prompt_tokens))
        try:
            in_cost, _ = cost_per_token(model=model, prompt_tokens=n, completion_tokens=0)
            price = float(in_cost) / n
        except Exception:
            return None
        return price if math.isfinite(price) and price > 0 else None

    def output_price_per_token(self, model: str, prompt_tokens: int = 0) -> Optional[float]:
        """
        USD price of one completion token for ``model``, or None if LiteLLM has no pricing.
        Pass the real ``prompt_tokens`` for models whose rates rise with prompt length
        (e.g. Gemini 2.5 Pro above 200k tokens).
        """
        try:
            _, out_cost = cost_per_token(model=model, prompt_tokens=prompt_tokens, completion_tokens=1)
            price = float(out_cost)
        except Exception:
            return None
        return price if math.isfinite(price) and price > 0 else None

    def check_affordability(self, model: str, prompt_tokens: int, completion_tokens: int = 500) -> None:
        """
        Hard budget constraint (BCD — Budget-Constrained Decoding).
        Raises BudgetExceeded before the call is ever made.
        """
        estimated = self.estimate_cost(model, prompt_tokens, completion_tokens)
        if estimated > self.remaining:
            raise BudgetExceeded(
                requested=estimated,
                remaining=self.remaining,
                model=model,
            )

    def check_and_reserve(self, amount: float, ttl: Optional[float] = None) -> Reservation:
        """
        Atomically check the budget and hold ``amount`` in one operation.
        Raises BudgetExceeded if spent + held + amount > total_budget.
        This fixes both the double-call and cross-instance TOCTOU race.

        Returns a ``Reservation``; pass it to ``cancel_reservation()`` once the
        actual cost is recorded or the call failed. If the process dies first,
        a store with leases frees it after ``ttl`` seconds (default
        ``lease_ttl``).
        """
        ttl = float(ttl) if ttl is not None else self.lease_ttl
        lease_id = self._store.reserve_lease(self.total_budget, amount, ttl)
        if lease_id is None:
            raise BudgetExceeded(
                requested=amount,
                remaining=self.remaining,
                model="(reservation)",
            )
        return Reservation(amount=float(amount), lease_id=lease_id, ttl=ttl)

    def reserve(self, amount: float) -> Reservation:
        """
        Pre-deduct estimated cost so concurrent async calls cannot both pass
        the affordability check and together overshoot the budget (TOCTOU fix).
        Must be paired with a call to cancel_reservation() after recording actual cost.
        """
        return self.check_and_reserve(amount)

    def extend_reservation(self, reservation: Reservation, extra: float) -> None:
        """
        Hold ``extra`` more under the same reservation (a fallback model that
        costs more). Raises BudgetExceeded if it does not fit. If the lease had
        already expired, the whole amount is reserved again under a new lease.
        """
        if extra <= 0:
            return
        if self._store.extend_lease(reservation.lease_id, self.total_budget, extra, reservation.ttl):
            reservation.amount += extra
            reservation.renewed_at = time.monotonic()
            return
        total = reservation.amount + extra
        new_id = None
        expired = self._store.supports_leases and not self._store.renew_lease(
            reservation.lease_id, reservation.ttl)
        if expired:
            new_id = self._store.reserve_lease(self.total_budget, total, reservation.ttl)
        if new_id is None:
            raise BudgetExceeded(requested=extra, remaining=self.remaining, model="(reservation)")
        reservation.lease_id = new_id
        reservation.amount = total
        reservation.renewed_at = time.monotonic()

    def renew_reservation(self, reservation: Reservation, force: bool = False) -> None:
        """
        Keep a reservation alive during a long call (streaming). Cheap to call
        often: it only reaches the store once a third of the ttl has passed.
        If the lease had expired, it is taken again when the budget allows.
        """
        if not self._store.supports_leases or reservation.released:
            return
        now = time.monotonic()
        if not force and now - reservation.renewed_at < reservation.ttl / 3:
            return
        reservation.renewed_at = now
        if not self._store.renew_lease(reservation.lease_id, reservation.ttl):
            new_id = self._store.reserve_lease(self.total_budget, reservation.amount, reservation.ttl)
            if new_id is not None:
                reservation.lease_id = new_id

    def cancel_reservation(self, reservation: Union[Reservation, float]) -> None:
        """
        Release a reservation. Called after record() has added the actual
        cost, or when the LLM call failed and no cost was incurred. A plain
        float is the pre-0.14 form: that amount is subtracted from the spend.
        """
        if isinstance(reservation, Reservation):
            if reservation.released:
                return
            reservation.released = True
            self._store.release_lease(reservation.lease_id, reservation.amount)
        else:
            self._store.add_spent(-float(reservation))

    def settle_reservation(self, reservation: Union[Reservation, float]) -> None:
        """Turn a reservation into spend (the call's real cost is unknown, so charge what was held)."""
        if isinstance(reservation, Reservation):
            if reservation.released:
                return
            self._store.add_spent(reservation.amount)
            self.cancel_reservation(reservation)
        # A float reservation is already counted as spend.

    @property
    def reserved(self) -> float:
        """Money currently held by unexpired reservations (stores with leases)."""
        return self._store.get_reserved()

    def reset(self) -> None:
        """Reset spend to 0.0 and re-arm all once=True alerts."""
        self._store.reset()
        with self._alert_lock:
            for a in self._alerts:
                a._fired = False

    def _fire_alerts(self) -> None:
        if not self._alerts:
            return
        # Re-arm once=True alerts when the billing window rolls over.
        if isinstance(self._store, WindowedBudgetStore):
            current_key = self._store._current_key
            with self._alert_lock:
                if current_key is not None and current_key != self._last_window_key:
                    for a in self._alerts:
                        a._fired = False
                    self._last_window_key = current_key
        utilization = self.utilization
        with self._alert_lock:
            to_fire = [
                a for a in self._alerts
                if utilization >= a.threshold and not (a.once and a._fired)
            ]
            for a in to_fire:
                if a.once:
                    a._fired = True
        info = {
            "utilization": utilization,
            "spent": self.spent,
            "remaining": self.remaining,
            "total_budget": self.total_budget,
        }
        for a in to_fire:
            try:
                a.callback({**info, "threshold": a.threshold})
            except Exception:
                pass

    def record(self, response, model: str) -> SpendRecord:
        """Record actual spend after a successful call."""
        self._step += 1
        cost = self.cost_from_response(response)
        new_spent = self._store.add_spent(cost)

        usage = getattr(response, "usage", None)
        rec = SpendRecord(
            step=self._step,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cost=cost,
            cumulative_cost=new_spent,
        )
        self._records.append(rec)
        self._fire_alerts()
        return rec

    def record_manual(
        self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> SpendRecord:
        """
        Record spend from manually collected token counts.
        Used by stream_chat() where token counts are accumulated during streaming
        rather than read from a response object. ``cached_tokens`` of the prompt
        are billed at the cache-read rate.
        """
        self._step += 1
        try:
            kwargs = {"cache_read_input_tokens": min(cached_tokens, prompt_tokens)} if cached_tokens > 0 else {}
            in_cost, out_cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                **kwargs,
            )
            cost = float(in_cost + out_cost)
        except Exception:
            warnings.warn(
                f"BudgetTracker: no pricing data for model '{model}' in record_manual. "
                "Spend recorded as $0.00. For accurate tracking, register real pricing with "
                "litellm.register_model() before using BAARRouter.",
                UserWarning,
                stacklevel=2,
            )
            cost = 0.0
        new_spent = self._store.add_spent(cost)
        rec = SpendRecord(
            step=self._step,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            cumulative_cost=new_spent,
        )
        self._records.append(rec)
        self._fire_alerts()
        return rec

    @property
    def records(self) -> list:
        return self._records

    def summary(self) -> dict:
        spent = self._store.get_spent()
        return {
            "total_budget": self.total_budget,
            "spent": round(spent, 8),
            "reserved": round(self._store.get_reserved(), 8),
            "remaining": round(max(0.0, self.total_budget - spent), 8),
            "utilization_pct": round(min(1.0, spent / self.total_budget) * 100, 2) if self.total_budget > 0 else 0,
            "steps": self._step,
            "records": [
                {
                    "step": r.step,
                    "model": r.model,
                    "tokens": r.prompt_tokens + r.completion_tokens,
                    "cost": round(r.cost, 8),
                    "cumulative": round(r.cumulative_cost, 8),
                }
                for r in self._records
            ],
        }
