"""
Leases and the stores for many machines (Redis, PostgreSQL).

Redis tests run against fakeredis, and also against a real server when
BAAR_TEST_REDIS_URL is set. PostgreSQL tests run when BAAR_TEST_POSTGRES_URL
is set (CI starts a service container). The cross-process tests need a real
shared backend: SQLite always, Redis/PostgreSQL when their URL is set.
"""

import multiprocessing as mp
import os
import time
import uuid
from unittest.mock import patch

import pytest

from baar.core.budget import BudgetExceeded, BudgetTracker, BudgetWindow, Reservation
from baar.core.stores import BudgetStore, MemoryBudgetStore, SQLiteBudgetStore
from tests.test_output_overrun import completion_echo, make_router, priced  # noqa: F401  (fixture)

REDIS_URL = os.environ.get("BAAR_TEST_REDIS_URL")
PG_URL = os.environ.get("BAAR_TEST_POSTGRES_URL")
TTL = 0.3      # short leases so expiry tests stay fast
WAIT = 0.5     # longer than TTL


def _ns() -> str:
    return f"test:{uuid.uuid4().hex[:12]}"


def make_store(kind: str, tmp_path, namespace=None):
    ns = namespace or _ns()
    if kind == "sqlite":
        return SQLiteBudgetStore(str(tmp_path / "b.db"), namespace=ns)
    if kind == "fakeredis":
        fakeredis = pytest.importorskip("fakeredis")
        from baar.core.stores import RedisBudgetStore
        return RedisBudgetStore(fakeredis.FakeRedis(), namespace=ns)
    if kind == "redis":
        if not REDIS_URL:
            pytest.skip("BAAR_TEST_REDIS_URL not set")
        from baar.core.stores import RedisBudgetStore
        return RedisBudgetStore(REDIS_URL, namespace=ns)
    if kind == "postgres":
        if not PG_URL:
            pytest.skip("BAAR_TEST_POSTGRES_URL not set")
        from baar.core.stores import PostgresBudgetStore
        return PostgresBudgetStore(PG_URL, namespace=ns)
    raise AssertionError(kind)


LEASE_STORES = ["sqlite", "fakeredis", "redis", "postgres"]
SHARED_STORES = ["sqlite", "redis", "postgres"]   # visible to other processes


@pytest.fixture(params=LEASE_STORES)
def store(request, tmp_path):
    return make_store(request.param, tmp_path)


# ── store contract ────────────────────────────────────────────────────────────

def test_lease_counts_as_spent_until_released(store):
    assert store.supports_leases
    lease = store.reserve_lease(1.0, 0.4, ttl=60)
    assert lease is not None
    assert store.get_spent() == pytest.approx(0.4)
    assert store.get_reserved() == pytest.approx(0.4)
    store.add_spent(0.1)
    assert store.get_spent() == pytest.approx(0.5)
    store.release_lease(lease, 0.4)
    assert store.get_spent() == pytest.approx(0.1)
    assert store.get_reserved() == pytest.approx(0.0)


def test_lease_refused_when_it_does_not_fit(store):
    assert store.reserve_lease(1.0, 0.7, ttl=60) is not None
    assert store.reserve_lease(1.0, 0.31, ttl=60) is None
    assert store.reserve_lease(1.0, 0.3, ttl=60) is not None
    assert store.get_spent() == pytest.approx(1.0)


def test_committed_spend_and_plain_reservations_count_against_leases(store):
    store.add_spent(0.5)
    assert store.atomic_check_and_reserve(1.0, 0.3)
    assert store.reserve_lease(1.0, 0.3, ttl=60) is None
    assert store.reserve_lease(1.0, 0.2, ttl=60) is not None
    assert not store.atomic_check_and_reserve(1.0, 0.01)


def test_expired_lease_frees_the_budget(store):
    """A worker that dies holding a lease does not hold the budget forever."""
    assert store.reserve_lease(1.0, 0.9, ttl=TTL) is not None
    assert store.reserve_lease(1.0, 0.5, ttl=60) is None
    time.sleep(WAIT)
    assert store.get_reserved() == pytest.approx(0.0)
    assert store.get_spent() == pytest.approx(0.0)
    assert store.reserve_lease(1.0, 0.5, ttl=60) is not None


def test_renew_keeps_a_lease_alive(store):
    lease = store.reserve_lease(1.0, 0.5, ttl=TTL)
    for _ in range(3):
        time.sleep(TTL / 2)
        assert store.renew_lease(lease, TTL)
    assert store.get_reserved() == pytest.approx(0.5)
    time.sleep(WAIT)
    assert not store.renew_lease(lease, TTL)


def test_extend_lease(store):
    lease = store.reserve_lease(1.0, 0.4, ttl=60)
    assert store.extend_lease(lease, 1.0, 0.4, ttl=60)
    assert store.get_reserved() == pytest.approx(0.8)
    assert not store.extend_lease(lease, 1.0, 0.3, ttl=60)
    assert store.get_reserved() == pytest.approx(0.8)
    store.release_lease(lease, 0.8)
    assert store.get_spent() == pytest.approx(0.0)


def test_extend_fails_on_expired_lease(store):
    lease = store.reserve_lease(1.0, 0.4, ttl=TTL)
    time.sleep(WAIT)
    assert not store.extend_lease(lease, 1.0, 0.1, ttl=60)


def test_release_is_idempotent_and_ignores_unknown_ids(store):
    lease = store.reserve_lease(1.0, 0.4, ttl=60)
    store.release_lease(lease, 0.4)
    store.release_lease(lease, 0.4)
    store.release_lease("no-such-lease", 0.4)
    assert store.get_spent() == pytest.approx(0.0)


def test_set_spent_and_reset_drop_leases(store):
    store.reserve_lease(1.0, 0.4, ttl=60)
    store.set_spent(0.25)
    assert store.get_spent() == pytest.approx(0.25)
    assert store.get_reserved() == pytest.approx(0.0)
    store.reserve_lease(1.0, 0.4, ttl=60)
    store.reset()
    assert store.get_spent() == pytest.approx(0.0)


def test_namespaces_are_separate(request, tmp_path):
    for kind in ("sqlite", "fakeredis"):
        a = make_store(kind, tmp_path)
        b = make_store(kind, tmp_path)
        a.reserve_lease(1.0, 0.9, ttl=60)
        a.add_spent(0.05)
        assert b.get_spent() == 0.0
        assert b.reserve_lease(1.0, 0.9, ttl=60) is not None


def test_same_namespace_is_one_budget(tmp_path):
    fakeredis = pytest.importorskip("fakeredis")
    from baar.core.stores import RedisBudgetStore
    server = fakeredis.FakeServer()
    a = RedisBudgetStore(fakeredis.FakeRedis(server=server), namespace="shared")
    b = RedisBudgetStore(fakeredis.FakeRedis(server=server), namespace="shared")
    assert a.reserve_lease(1.0, 0.6, ttl=60) is not None
    assert b.reserve_lease(1.0, 0.6, ttl=60) is None
    assert b.get_reserved() == pytest.approx(0.6)


def test_fallback_store_without_leases():
    """A custom store that only implements the old methods still works."""
    store = MemoryBudgetStore()
    assert not store.supports_leases
    lease = store.reserve_lease(1.0, 0.4, ttl=60)
    assert lease is not None and store.get_spent() == pytest.approx(0.4)
    assert store.extend_lease(lease, 1.0, 0.2, ttl=60)
    assert store.get_spent() == pytest.approx(0.6)
    store.release_lease(lease, 0.6)
    assert store.get_spent() == pytest.approx(0.0)
    assert store.get_reserved() == 0.0


def test_postgres_rejects_bad_table_prefix():
    pytest.importorskip("psycopg")
    from baar.core.stores import PostgresBudgetStore
    with pytest.raises(ValueError):
        PostgresBudgetStore("postgresql://unused", table_prefix="x; DROP TABLE y")


def test_unknown_attribute_on_stores_module():
    import baar.core.stores as stores
    with pytest.raises(AttributeError):
        stores.NoSuchStore  # noqa: B018


# ── tracker ───────────────────────────────────────────────────────────────────

@pytest.fixture(params=["sqlite", "fakeredis"])
def lease_store(request, tmp_path):
    return make_store(request.param, tmp_path)


def test_tracker_reservation_round_trip(lease_store):
    t = BudgetTracker(1.0, store=lease_store, lease_ttl=60)
    r = t.check_and_reserve(0.3)
    assert isinstance(r, Reservation) and r.ttl == 60
    assert t.reserved == pytest.approx(0.3)
    assert t.summary()["reserved"] == pytest.approx(0.3)
    t.cancel_reservation(r)
    t.cancel_reservation(r)  # a second release is a no-op
    assert t.spent == pytest.approx(0.0)
    with pytest.raises(BudgetExceeded):
        t.check_and_reserve(1.5)


def test_tracker_extend_and_retake_after_expiry(lease_store):
    t = BudgetTracker(1.0, store=lease_store, lease_ttl=TTL)
    r = t.check_and_reserve(0.3)
    t.extend_reservation(r, 0.2)
    assert r.amount == pytest.approx(0.5) and t.reserved == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        t.extend_reservation(r, 0.6)
    t.extend_reservation(r, 0.0)
    time.sleep(WAIT)
    t.extend_reservation(r, 0.1)    # lease expired: the whole amount is taken again
    assert r.amount == pytest.approx(0.6) and t.reserved == pytest.approx(0.6)
    t.cancel_reservation(r)
    assert t.spent == pytest.approx(0.0)


def test_tracker_renew_is_rate_limited_and_retakes_expired(lease_store):
    t = BudgetTracker(1.0, store=lease_store, lease_ttl=TTL)
    r = t.check_and_reserve(0.4)
    with patch.object(lease_store, "renew_lease", wraps=lease_store.renew_lease) as renew:
        t.renew_reservation(r)
        assert renew.call_count == 0          # too soon
        time.sleep(TTL / 2)
        t.renew_reservation(r)
        assert renew.call_count == 1
    time.sleep(WAIT)
    assert t.reserved == 0.0
    t.renew_reservation(r, force=True)
    assert t.reserved == pytest.approx(0.4)   # taken again
    t.cancel_reservation(r)
    t.renew_reservation(r, force=True)        # released: nothing happens
    assert t.reserved == 0.0


def test_tracker_settle_turns_a_reservation_into_spend(lease_store):
    t = BudgetTracker(1.0, store=lease_store, lease_ttl=60)
    r = t.check_and_reserve(0.3)
    t.settle_reservation(r)
    t.settle_reservation(r)
    assert t.reserved == 0.0 and t.spent == pytest.approx(0.3)
    t.settle_reservation(0.1)  # legacy float: already counted, nothing to do
    assert t.spent == pytest.approx(0.3)


def test_tracker_legacy_float_cancel():
    t = BudgetTracker(1.0)
    t.check_and_reserve(0.3)
    t.cancel_reservation(0.3)
    assert t.spent == pytest.approx(0.0)


def test_tracker_rejects_bad_lease_ttl():
    with pytest.raises(ValueError):
        BudgetTracker(1.0, lease_ttl=0)


def test_windowed_redis_store_uses_period_namespaces():
    fakeredis = pytest.importorskip("fakeredis")
    from baar.core.stores import RedisBudgetStore
    client = fakeredis.FakeRedis()
    t = BudgetTracker(1.0, store=RedisBudgetStore(client, namespace="u1"), window=BudgetWindow.DAILY)
    assert t._store.supports_leases
    r = t.check_and_reserve(0.4)
    assert t.reserved == pytest.approx(0.4)
    keys = {k.decode() for k in client.keys("*")}
    assert keys and all(k.startswith("baar:{u1:20") for k in keys)   # baar:{u1:<today>}:...
    t._store._current_key = "2000-01-01"           # pretend the day rolled over
    t._store._current_store = RedisBudgetStore(client, namespace="u1:2000-01-01")
    t._store.renew_lease(r.lease_id, 60)
    t._store.extend_lease(r.lease_id, 1.0, 0.1, 60)
    t.cancel_reservation(r)                         # released in the period it was taken in
    assert client.keys("*") == []


def test_windowed_store_rejects_unknown_store():
    class Custom(BudgetStore):
        def get_spent(self): return 0.0
    with pytest.raises(TypeError):
        BudgetTracker(1.0, store=Custom(), window=BudgetWindow.DAILY).spent


# ── router ────────────────────────────────────────────────────────────────────

def test_router_lease_ttl_from_call_timeout():
    from baar.core.budget import DEFAULT_LEASE_TTL
    from baar.router import _lease_ttl
    assert _lease_ttl(None, None) == DEFAULT_LEASE_TTL
    assert _lease_ttl(None, 30) == 150
    assert _lease_ttl(12, 30) == 12
    assert make_router(budget=1.0, call_timeout=10)._tracker.lease_ttl == 90


def test_router_holds_a_lease_during_the_call(priced, tmp_path):
    store = make_store("fakeredis", tmp_path)
    router = make_router(budget=1.0, store=store)
    seen = []
    echo = completion_echo()

    def call(**kwargs):
        seen.append(store.get_reserved())
        return echo(**kwargs)

    with patch("baar.router.litellm.completion", side_effect=call):
        router.chat("hello")
    assert seen[0] > 0
    assert store.get_reserved() == 0.0
    assert store.get_spent() == pytest.approx(router.log.steps[-1].cost) and router.log.steps[-1].cost > 0


def test_router_streaming_renews_the_lease(priced, tmp_path):
    from tests.test_output_overrun import make_chunks
    store = make_store("fakeredis", tmp_path)
    router = make_router(budget=1.0, store=store, lease_ttl=60)
    with patch("baar.router.litellm.completion", return_value=iter(make_chunks(["a", "b", "c"], usage=(10, 3)))), \
         patch.object(router._tracker, "renew_reservation", wraps=router._tracker.renew_reservation) as renew:
        assert "".join(router.stream_chat("hi")) == "abc"
    assert renew.call_count == 4
    assert store.get_reserved() == 0.0


# ── across processes ──────────────────────────────────────────────────────────

def _store_from_spec(spec):
    kind, arg, ns = spec
    if kind == "sqlite":
        return SQLiteBudgetStore(arg, namespace=ns)
    if kind == "redis":
        from baar.core.stores import RedisBudgetStore
        return RedisBudgetStore(arg, namespace=ns)
    from baar.core.stores import PostgresBudgetStore
    return PostgresBudgetStore(arg, namespace=ns)


def _spend_worker(spec, budget, cost, tries, queue):
    store = _store_from_spec(spec)
    t = BudgetTracker(budget, store=store, lease_ttl=60)
    got = 0
    for _ in range(tries):
        try:
            r = t.check_and_reserve(cost)
        except BudgetExceeded:
            continue
        store.add_spent(cost)       # the call's real cost
        t.cancel_reservation(r)
        got += 1
    queue.put(got)


def _crash_worker(spec, queue):
    t = BudgetTracker(1.0, store=_store_from_spec(spec), lease_ttl=1.0)
    t.check_and_reserve(0.8)
    queue.put("reserved")
    queue.close()
    queue.join_thread()             # make sure the message left before dying
    os._exit(1)                     # die mid-call, never releasing


def _spec(kind, tmp_path):
    ns = _ns()
    if kind == "sqlite":
        return ("sqlite", str(tmp_path / "shared.db"), ns)
    url = REDIS_URL if kind == "redis" else PG_URL
    if not url:
        pytest.skip(f"no {kind} server configured")
    return (kind, url, ns)


@pytest.mark.parametrize("kind", SHARED_STORES)
def test_many_processes_never_overspend(kind, tmp_path):
    spec = _spec(kind, tmp_path)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_spend_worker, args=(spec, 1.0, 0.1, 6, queue)) for _ in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    granted = sum(queue.get(timeout=10) for _ in procs)
    store = _store_from_spec(spec)
    assert granted == 10                       # 48 attempts, room for exactly 10
    assert store.get_spent() == pytest.approx(1.0)
    assert store.get_spent() <= 1.0 + 1e-9
    assert store.get_reserved() == 0.0


@pytest.mark.parametrize("kind", SHARED_STORES)
def test_crashed_worker_releases_the_budget_after_its_ttl(kind, tmp_path):
    spec = _spec(kind, tmp_path)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_crash_worker, args=(spec, queue))
    p.start()
    assert queue.get(timeout=30) == "reserved"
    p.join(30)
    store = _store_from_spec(spec)
    t = BudgetTracker(1.0, store=store)
    assert store.get_reserved() == pytest.approx(0.8)
    with pytest.raises(BudgetExceeded):
        t.check_and_reserve(0.5)
    time.sleep(1.3)
    assert store.get_reserved() == 0.0
    t.cancel_reservation(t.check_and_reserve(0.5))


# ── PostgreSQL connection modes ───────────────────────────────────────────────

@pytest.fixture
def pg_url():
    if not PG_URL:
        pytest.skip("BAAR_TEST_POSTGRES_URL not set")
    return PG_URL


def test_postgres_with_an_open_connection(pg_url):
    import psycopg
    from baar.core.stores import PostgresBudgetStore
    conn = psycopg.connect(pg_url, autocommit=True)
    store = PostgresBudgetStore(conn, namespace=_ns())
    assert store.reserve_lease(1.0, 0.5, ttl=60) is not None
    other = store.for_namespace(_ns())
    assert other.get_spent() == 0.0 and store.get_reserved() == pytest.approx(0.5)
    conn.close()
    with pytest.raises(RuntimeError):
        store.get_spent()


def test_postgres_with_a_pool(pg_url):
    import contextlib
    import psycopg
    from baar.core.stores import PostgresBudgetStore

    class Pool:  # the two methods of psycopg_pool.ConnectionPool the store uses
        @contextlib.contextmanager
        def connection(self):
            with psycopg.connect(pg_url) as conn:
                yield conn

        def getconn(self):  # pragma: no cover
            raise AssertionError

    store = PostgresBudgetStore(Pool(), namespace=_ns())
    lease = store.reserve_lease(1.0, 0.5, ttl=60)
    assert store.reserve_lease(1.0, 0.6, ttl=60) is None
    store.release_lease(lease, 0.5)
    assert store.get_spent() == 0.0


def test_postgres_reconnects_after_a_broken_connection(pg_url):
    from baar.core.stores import PostgresBudgetStore
    store = PostgresBudgetStore(pg_url, namespace=_ns())
    store.add_spent(0.2)
    store._db._conn.close()
    assert store.get_spent() == pytest.approx(0.2)


def test_windowed_postgres_store(pg_url):
    from baar.core.stores import PostgresBudgetStore
    ns = _ns()
    t = BudgetTracker(1.0, store=PostgresBudgetStore(pg_url, namespace=ns), window=BudgetWindow.MONTHLY)
    r = t.check_and_reserve(0.4)
    assert t._store._current_store._namespace.startswith(ns + ":20")
    assert t.reserved == pytest.approx(0.4)
    t.cancel_reservation(r)
    assert t.spent == 0.0
