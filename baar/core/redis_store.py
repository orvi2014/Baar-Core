"""
baar/core/redis_store.py — a budget shared by workers on many machines, in Redis.

    from baar.core.stores import RedisBudgetStore
    store = RedisBudgetStore("redis://localhost:6379/0", namespace="tenant:acme")
    router = BAARRouter(budget=5.00, store=store)

Every worker that uses the same Redis and namespace draws on one budget. Each
check-and-reserve runs as one Lua script, so two workers can never both take
the last dollar. Reservations are leases timed by the Redis server's clock: if
a worker dies mid-call, its reservation expires after its ttl and the money is
available again.

Keys (the ``{namespace}`` hash tag keeps them in one Redis Cluster slot):

    <prefix>{<namespace>}:spent      committed spend (string float)
    <prefix>{<namespace>}:leases     sorted set: lease id -> expiry (ms)
    <prefix>{<namespace>}:amounts    hash: lease id -> amount held

Needs ``pip install baar-core[redis]``.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from baar.core.stores import BudgetStore

# Shared preamble: drop expired leases, then compute committed + held.
_PRELUDE = """
local spent_key, leases_key, amounts_key = KEYS[1], KEYS[2], KEYS[3]
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local expired = redis.call('ZRANGEBYSCORE', leases_key, '-inf', now)
if #expired > 0 then
  redis.call('ZREM', leases_key, unpack(expired))
  redis.call('HDEL', amounts_key, unpack(expired))
end
local held = 0
for _, v in ipairs(redis.call('HVALS', amounts_key)) do held = held + tonumber(v) end
local committed = tonumber(redis.call('GET', spent_key) or '0')
"""

# ARGV: budget, amount  -> 1 reserved (into committed spend) / 0 not
_CHECK_AND_RESERVE = _PRELUDE + """
if committed + held + tonumber(ARGV[2]) <= tonumber(ARGV[1]) then
  redis.call('INCRBYFLOAT', spent_key, ARGV[2])
  return 1
end
return 0
"""

# ARGV: budget, amount, ttl_ms, lease_id  -> 1 / 0
_RESERVE_LEASE = _PRELUDE + """
if committed + held + tonumber(ARGV[2]) <= tonumber(ARGV[1]) then
  redis.call('ZADD', leases_key, now + tonumber(ARGV[3]), ARGV[4])
  redis.call('HSET', amounts_key, ARGV[4], ARGV[2])
  return 1
end
return 0
"""

# ARGV: budget, extra, ttl_ms, lease_id  -> 1 / 0 (0 also when the lease expired)
_EXTEND_LEASE = _PRELUDE + """
if not redis.call('ZSCORE', leases_key, ARGV[4]) then return 0 end
if committed + held + tonumber(ARGV[2]) <= tonumber(ARGV[1]) then
  redis.call('ZADD', leases_key, now + tonumber(ARGV[3]), ARGV[4])
  redis.call('HINCRBYFLOAT', amounts_key, ARGV[4], ARGV[2])
  return 1
end
return 0
"""

# ARGV: ttl_ms, lease_id -> 1 / 0
_RENEW_LEASE = _PRELUDE + """
if not redis.call('ZSCORE', leases_key, ARGV[2]) then return 0 end
redis.call('ZADD', leases_key, now + tonumber(ARGV[1]), ARGV[2])
return 1
"""

# -> {committed, held} as strings with full precision
_READ = _PRELUDE + """
return {string.format('%.17g', committed), string.format('%.17g', held)}
"""


class RedisBudgetStore(BudgetStore):
    """
    Budget store in Redis, shared by every worker that uses the same namespace.

    Args:
        url_or_client: a Redis URL (``redis://host:6379/0``) or an existing
                       ``redis.Redis`` client (sync).
        namespace:     the budget's bucket, e.g. ``"tenant:acme"``.
        prefix:        key prefix (default ``"baar:"``).
    """

    supports_leases = True

    def __init__(self, url_or_client: Any, namespace: str = "default", prefix: str = "baar:") -> None:
        if isinstance(url_or_client, str):
            try:
                import redis
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "RedisBudgetStore needs the redis package: pip install 'baar-core[redis]'"
                ) from e
            client = redis.Redis.from_url(url_or_client)
        else:
            client = url_or_client
        self._client = client
        self._url = url_or_client if isinstance(url_or_client, str) else None
        self._namespace = namespace
        self._prefix = prefix
        base = f"{prefix}{{{namespace}}}"
        self._keys = [f"{base}:spent", f"{base}:leases", f"{base}:amounts"]
        self._check = client.register_script(_CHECK_AND_RESERVE)
        self._reserve = client.register_script(_RESERVE_LEASE)
        self._extend = client.register_script(_EXTEND_LEASE)
        self._renew = client.register_script(_RENEW_LEASE)
        self._read = client.register_script(_READ)

    def for_namespace(self, namespace: str) -> "RedisBudgetStore":
        """Same Redis connection, another budget (used by budget windows)."""
        return RedisBudgetStore(self._client, namespace=namespace, prefix=self._prefix)

    @staticmethod
    def _f(x: float) -> str:
        return repr(float(x))

    def _committed_and_held(self):
        committed, held = self._read(keys=self._keys)
        return float(committed), float(held)

    # ── BudgetStore ───────────────────────────────────────────────────────────

    def get_spent(self) -> float:
        committed, held = self._committed_and_held()
        return committed + held

    def get_reserved(self) -> float:
        return self._committed_and_held()[1]

    def add_spent(self, amount: float) -> float:
        self._client.incrbyfloat(self._keys[0], float(amount))
        return self.get_spent()

    def set_spent(self, value: float) -> None:
        """Overwrite the spend; drops this namespace's leases so get_spent() == value."""
        pipe = self._client.pipeline(transaction=True)
        pipe.delete(self._keys[1], self._keys[2])
        pipe.set(self._keys[0], self._f(value))
        pipe.execute()

    def reset(self) -> None:
        self._client.delete(*self._keys)

    def atomic_check_and_reserve(self, total_budget: float, amount: float) -> bool:
        return bool(self._check(keys=self._keys, args=[self._f(total_budget), self._f(amount)]))

    def reserve_lease(self, total_budget: float, amount: float, ttl: float) -> Optional[str]:
        lease_id = uuid.uuid4().hex
        ok = self._reserve(keys=self._keys, args=[
            self._f(total_budget), self._f(amount), int(ttl * 1000), lease_id])
        return lease_id if ok else None

    def extend_lease(self, lease_id: str, total_budget: float, extra: float, ttl: float) -> bool:
        return bool(self._extend(keys=self._keys, args=[
            self._f(total_budget), self._f(extra), int(ttl * 1000), lease_id]))

    def renew_lease(self, lease_id: str, ttl: float) -> bool:
        return bool(self._renew(keys=self._keys, args=[int(ttl * 1000), lease_id]))

    def release_lease(self, lease_id: str, amount: float) -> None:
        pipe = self._client.pipeline(transaction=True)
        pipe.zrem(self._keys[1], lease_id)
        pipe.hdel(self._keys[2], lease_id)
        pipe.execute()
