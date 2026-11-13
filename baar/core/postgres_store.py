"""
baar/core/postgres_store.py — a budget shared by workers on many machines, in PostgreSQL.

    from baar.core.stores import PostgresBudgetStore
    store = PostgresBudgetStore("postgresql://user:pw@db/app", namespace="tenant:acme")
    router = BAARRouter(budget=5.00, store=store)

Every worker that uses the same database and namespace draws on one budget.
Each check-and-reserve runs in one transaction that locks the namespace's row
(``SELECT ... FOR UPDATE``), so two workers can never both take the last
dollar. Reservations are leases timed by the database server's clock: if a
worker dies mid-call, its reservation expires after its ttl and the money is
available again.

Tables (created on first use unless ``create_tables=False``)::

    <prefix>budget_spend  (namespace TEXT PRIMARY KEY, spent DOUBLE PRECISION, updated TIMESTAMPTZ)
    <prefix>budget_leases (id TEXT PRIMARY KEY, namespace TEXT, amount DOUBLE PRECISION,
                           expires_at TIMESTAMPTZ)

Needs ``pip install baar-core[postgres]`` (psycopg 3).
"""

from __future__ import annotations

import contextlib
import re
import threading
import uuid
from typing import Any, Optional

from baar.core.stores import BudgetStore

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _Conn:
    """One connection shared under a lock, or a psycopg_pool pool."""

    def __init__(self, target: Any) -> None:
        self._lock = threading.Lock()
        self._pool = None
        self._conninfo = None
        self._conn = None
        if isinstance(target, str):
            self._conninfo = target
        elif hasattr(target, "connection") and hasattr(target, "getconn"):
            self._pool = target            # psycopg_pool.ConnectionPool
        else:
            self._conn = target            # an open psycopg.Connection

    def _own(self):
        if self._conn is None or self._conn.closed or (
                self._conninfo is not None and self._conn.broken):
            if self._conninfo is None:
                raise RuntimeError("PostgresBudgetStore: the connection it was given is closed")
            import psycopg
            self._conn = psycopg.connect(self._conninfo, autocommit=True)
        return self._conn

    @contextlib.contextmanager
    def tx(self):
        """A cursor inside one transaction (committed on success)."""
        if self._pool is not None:
            with self._pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        yield cur
            return
        with self._lock:
            conn = self._own()
            with conn.transaction():
                with conn.cursor() as cur:
                    yield cur


class PostgresBudgetStore(BudgetStore):
    """
    Budget store in PostgreSQL, shared by every worker that uses the same namespace.

    Args:
        conn:          a connection string, an open ``psycopg.Connection`` or a
                       ``psycopg_pool.ConnectionPool``. A connection string or
                       connection is shared by the store's threads under a lock;
                       pass a pool for many concurrent threads.
        namespace:     the budget's bucket, e.g. ``"tenant:acme"``.
        table_prefix:  prefix of the two tables (default ``"baar_"``).
        create_tables: create the tables if they do not exist (default True).
    """

    supports_leases = True

    def __init__(self, conn: Any, namespace: str = "default", table_prefix: str = "baar_",
                 create_tables: bool = True, _shared: Optional[_Conn] = None) -> None:
        if isinstance(conn, str) or _shared is None:
            try:
                import psycopg  # noqa: F401
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "PostgresBudgetStore needs psycopg 3: pip install 'baar-core[postgres]'"
                ) from e
        if not _IDENT.match(table_prefix + "x"):
            raise ValueError("table_prefix may contain only letters, digits and underscores")
        self._db = _shared if _shared is not None else _Conn(conn)
        self._namespace = namespace
        self._prefix = table_prefix
        self._spend = f"{table_prefix}budget_spend"
        self._leases = f"{table_prefix}budget_leases"
        if create_tables and _shared is None:
            self._create_tables()

    def for_namespace(self, namespace: str) -> "PostgresBudgetStore":
        """Same connection, another budget (used by budget windows)."""
        return PostgresBudgetStore(None, namespace=namespace, table_prefix=self._prefix,
                                   create_tables=False, _shared=self._db)

    def _create_tables(self) -> None:
        with self._db.tx() as cur:
            # Serialise concurrent first-time creation by several workers.
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (self._spend,))
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._spend} (
                    namespace TEXT PRIMARY KEY,
                    spent     DOUBLE PRECISION NOT NULL DEFAULT 0,
                    updated   TIMESTAMPTZ NOT NULL DEFAULT now()
                )""")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._leases} (
                    id         TEXT PRIMARY KEY,
                    namespace  TEXT NOT NULL,
                    amount     DOUBLE PRECISION NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL
                )""")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self._leases}_ns "
                        f"ON {self._leases} (namespace, expires_at)")

    # ── helpers (run inside a transaction) ────────────────────────────────────

    def _lock(self, cur) -> tuple:
        """Lock the namespace row, drop expired leases; return (committed, held)."""
        cur.execute(f"INSERT INTO {self._spend} (namespace) VALUES (%s) ON CONFLICT DO NOTHING",
                    (self._namespace,))
        cur.execute(f"SELECT spent FROM {self._spend} WHERE namespace = %s FOR UPDATE",
                    (self._namespace,))
        committed = float(cur.fetchone()[0])
        cur.execute(f"DELETE FROM {self._leases} WHERE namespace = %s AND expires_at <= clock_timestamp()",
                    (self._namespace,))
        cur.execute(f"SELECT COALESCE(SUM(amount), 0) FROM {self._leases} WHERE namespace = %s",
                    (self._namespace,))
        return committed, float(cur.fetchone()[0])

    def _read(self) -> tuple:
        with self._db.tx() as cur:
            cur.execute(f"""
                SELECT COALESCE((SELECT spent FROM {self._spend} WHERE namespace = %s), 0),
                       COALESCE((SELECT SUM(amount) FROM {self._leases}
                                 WHERE namespace = %s AND expires_at > clock_timestamp()), 0)
            """, (self._namespace, self._namespace))
            committed, held = cur.fetchone()
            return float(committed), float(held)

    # ── BudgetStore ───────────────────────────────────────────────────────────

    def get_spent(self) -> float:
        committed, held = self._read()
        return committed + held

    def get_reserved(self) -> float:
        return self._read()[1]

    def add_spent(self, amount: float) -> float:
        with self._db.tx() as cur:
            cur.execute(f"""
                INSERT INTO {self._spend} (namespace, spent) VALUES (%s, %s)
                ON CONFLICT (namespace) DO UPDATE
                SET spent = {self._spend}.spent + EXCLUDED.spent, updated = now()
            """, (self._namespace, float(amount)))
        return self.get_spent()

    def set_spent(self, value: float) -> None:
        """Overwrite the spend; drops this namespace's leases so get_spent() == value."""
        with self._db.tx() as cur:
            cur.execute(f"DELETE FROM {self._leases} WHERE namespace = %s", (self._namespace,))
            cur.execute(f"""
                INSERT INTO {self._spend} (namespace, spent) VALUES (%s, %s)
                ON CONFLICT (namespace) DO UPDATE SET spent = EXCLUDED.spent, updated = now()
            """, (self._namespace, float(value)))

    def reset(self) -> None:
        self.set_spent(0.0)

    def atomic_check_and_reserve(self, total_budget: float, amount: float) -> bool:
        with self._db.tx() as cur:
            committed, held = self._lock(cur)
            if committed + held + amount > total_budget:
                return False
            cur.execute(f"UPDATE {self._spend} SET spent = spent + %s, updated = now() WHERE namespace = %s",
                        (float(amount), self._namespace))
            return True

    def reserve_lease(self, total_budget: float, amount: float, ttl: float) -> Optional[str]:
        lease_id = uuid.uuid4().hex
        with self._db.tx() as cur:
            committed, held = self._lock(cur)
            if committed + held + amount > total_budget:
                return None
            cur.execute(f"""
                INSERT INTO {self._leases} (id, namespace, amount, expires_at)
                VALUES (%s, %s, %s, clock_timestamp() + make_interval(secs => %s))
            """, (lease_id, self._namespace, float(amount), float(ttl)))
        return lease_id

    def extend_lease(self, lease_id: str, total_budget: float, extra: float, ttl: float) -> bool:
        with self._db.tx() as cur:
            committed, held = self._lock(cur)
            cur.execute(f"SELECT 1 FROM {self._leases} WHERE id = %s AND namespace = %s",
                        (lease_id, self._namespace))
            if cur.fetchone() is None or committed + held + extra > total_budget:
                return False
            cur.execute(f"""
                UPDATE {self._leases}
                SET amount = amount + %s, expires_at = clock_timestamp() + make_interval(secs => %s)
                WHERE id = %s
            """, (float(extra), float(ttl), lease_id))
            return True

    def renew_lease(self, lease_id: str, ttl: float) -> bool:
        with self._db.tx() as cur:
            cur.execute(f"""
                UPDATE {self._leases}
                SET expires_at = clock_timestamp() + make_interval(secs => %s)
                WHERE id = %s AND namespace = %s AND expires_at > clock_timestamp()
            """, (float(ttl), lease_id, self._namespace))
            return cur.rowcount > 0

    def release_lease(self, lease_id: str, amount: float) -> None:
        with self._db.tx() as cur:
            cur.execute(f"DELETE FROM {self._leases} WHERE id = %s AND namespace = %s",
                        (lease_id, self._namespace))
