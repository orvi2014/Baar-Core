"""
baar/core/stores.py — Pluggable persistent budget storage backends.

All stores share the same three-method interface so BudgetTracker is
agnostic to the underlying storage technology.

Usage:
    # Default (in-memory, backward-compatible)
    tracker = BudgetTracker(total_budget=0.10)

    # JSON file — survives process restarts, good for single-process use
    from baar.core.stores import FileBudgetStore
    tracker = BudgetTracker(
        total_budget=0.10,
        store=FileBudgetStore("budgets.json", namespace="user_alice"),
    )

    # SQLite — thread-safe, no extra deps, good for per-user SaaS quotas
    from baar.core.stores import SQLiteBudgetStore
    tracker = BudgetTracker(
        total_budget=0.10,
        store=SQLiteBudgetStore("budgets.db", namespace="user_alice"),
    )
"""

import json
import os
import sqlite3
import threading
import time
import uuid
import weakref
from typing import Optional


# ── Base interface ─────────────────────────────────────────────────────────────

class BudgetStore:
    """
    Abstract base for budget storage backends.

    All implementations must be safe to call from multiple threads.
    Every method must be atomic from the caller's perspective.
    """

    def get_spent(self) -> float:
        """Return total spend recorded so far for this namespace."""
        raise NotImplementedError  # pragma: no cover

    def add_spent(self, amount: float) -> float:
        """
        Atomically add *amount* to the running total.
        Returns the new cumulative spend.
        """
        raise NotImplementedError  # pragma: no cover

    def set_spent(self, value: float) -> None:
        """
        Overwrite the stored spend with an absolute value.
        Used only for test-compatible direct assignment — prefer add_spent
        for normal operation.
        """
        raise NotImplementedError  # pragma: no cover

    def reset(self) -> None:
        """Reset spend to 0.0 for this namespace."""
        raise NotImplementedError  # pragma: no cover

    def atomic_check_and_reserve(self, total_budget: float, amount: float) -> bool:
        """
        Atomically check if current_spent + amount <= total_budget.
        If yes, add amount to the store and return True.
        If no, return False without modifying the store.

        Fixes the cross-instance TOCTOU race where two BAARRouter instances
        sharing the same store can both pass affordability checks.
        """
        raise NotImplementedError  # pragma: no cover

    # ── Leases ────────────────────────────────────────────────────────────────
    #
    # A lease is a reservation that expires. Stores that support leases count
    # an unexpired lease as spent; once it expires (the worker holding it
    # crashed or hung) the money is available again without anyone releasing
    # it. Stores that do not override these methods fall back to the plain
    # reserve/cancel behaviour below, where a reservation never expires.

    #: True when reservations expire after their ttl.
    supports_leases = False

    def reserve_lease(self, total_budget: float, amount: float, ttl: float) -> Optional[str]:
        """Reserve *amount* for *ttl* seconds if it fits; return a lease id, or None if it does not."""
        if not self.atomic_check_and_reserve(total_budget, amount):
            return None
        return uuid.uuid4().hex

    def extend_lease(self, lease_id: str, total_budget: float, extra: float, ttl: float) -> bool:
        """Add *extra* to a lease (and restart its ttl) if it fits; False if it does not."""
        return self.atomic_check_and_reserve(total_budget, extra)

    def renew_lease(self, lease_id: str, ttl: float) -> bool:
        """Restart a lease's ttl; False if it has already expired."""
        return True

    def release_lease(self, lease_id: str, amount: float) -> None:
        """Drop a lease. *amount* is what it holds; stores without leases subtract it."""
        self.add_spent(-amount)

    def get_reserved(self) -> float:
        """Total held by unexpired leases (0.0 for stores without leases)."""
        return 0.0


# ── In-memory (default, backward-compatible) ──────────────────────────────────

class MemoryBudgetStore(BudgetStore):
    """
    Thread-safe in-memory store.  Data is lost on process exit.
    This is the default store — existing code requires no changes.
    """

    def __init__(self) -> None:
        self._spent: float = 0.0
        self._lock = threading.Lock()

    def get_spent(self) -> float:
        with self._lock:
            return self._spent

    def add_spent(self, amount: float) -> float:
        with self._lock:
            self._spent += amount
            return self._spent

    def set_spent(self, value: float) -> None:
        with self._lock:
            self._spent = float(value)

    def reset(self) -> None:
        with self._lock:
            self._spent = 0.0

    def atomic_check_and_reserve(self, total_budget: float, amount: float) -> bool:
        with self._lock:
            if self._spent + amount <= total_budget:
                self._spent += amount
                return True
            return False


# ── JSON file ──────────────────────────────────────────────────────────────────

class FileBudgetStore(BudgetStore):
    """
    JSON file-backed store.  Survives process restarts.

    Suitable for single-process workloads. atomic_check_and_reserve uses an
    in-process threading.Lock — it is NOT safe across multiple OS processes
    (e.g., gunicorn/uvicorn workers). Use SQLiteBudgetStore for multi-process
    deployments.

    Args:
        path:      Path to the JSON file (created if absent).
        namespace: Key within the file — use one namespace per user/tenant.

    Example:
        store = FileBudgetStore("~/.baar/budgets.json", namespace="user_alice")
        tracker = BudgetTracker(total_budget=0.10, store=store)
    """

    def __init__(self, path: str, namespace: str = "default") -> None:
        self._path = os.path.expanduser(path)
        self._namespace = namespace
        self._lock = threading.Lock()
        self._ensure_file()

    # ── internal helpers ───────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        if not os.path.exists(self._path):
            self._write({})

    def _read(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict) -> None:
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self._path)

    # ── public interface ───────────────────────────────────────────────────────

    def get_spent(self) -> float:
        with self._lock:
            data = self._read()
            return float(data.get(self._namespace, 0.0))

    def add_spent(self, amount: float) -> float:
        with self._lock:
            data = self._read()
            new_val = float(data.get(self._namespace, 0.0)) + amount
            data[self._namespace] = new_val
            self._write(data)
            return new_val

    def set_spent(self, value: float) -> None:
        with self._lock:
            data = self._read()
            data[self._namespace] = float(value)
            self._write(data)

    def reset(self) -> None:
        with self._lock:
            data = self._read()
            data[self._namespace] = 0.0
            self._write(data)

    def atomic_check_and_reserve(self, total_budget: float, amount: float) -> bool:
        with self._lock:
            data = self._read()
            current = float(data.get(self._namespace, 0.0))
            if current + amount <= total_budget:
                data[self._namespace] = current + amount
                self._write(data)
                return True
            return False


# ── SQLite (recommended for multi-user / SaaS) ────────────────────────────────

# One lock per database path prevents multiple threads from stepping on
# each other when they share the same file.
_sqlite_locks: "weakref.WeakValueDictionary[str, threading.Lock]" = weakref.WeakValueDictionary()
_sqlite_locks_guard = threading.Lock()


def _sqlite_lock(path: str) -> threading.Lock:
    with _sqlite_locks_guard:
        lock = _sqlite_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _sqlite_locks[path] = lock
        return lock


class SQLiteBudgetStore(BudgetStore):
    """
    SQLite-backed budget store.  Thread-safe, no extra dependencies.

    Recommended for SaaS applications that need per-user budget enforcement
    that persists across process restarts and survives crashes. Safe across
    processes on one machine (gunicorn/uvicorn workers); for several machines
    use ``RedisBudgetStore`` or ``PostgresBudgetStore``.

    Reservations are leases: if a worker dies mid-call, its reservation
    expires after its ttl instead of holding the budget forever.

    Args:
        db_path:   Path to the SQLite database file (created if absent).
        namespace: Logical bucket for this budget — use one per user/tenant.
                   Convention: "user:<id>", "tenant:<id>", "session:<id>".

    Example (per-user daily quota):
        import datetime
        today = datetime.date.today().isoformat()   # "2026-05-08"
        store = SQLiteBudgetStore(
            "data/budgets.db",
            namespace=f"user:{user_id}:{today}",    # auto-resets tomorrow
        )
        tracker = BudgetTracker(total_budget=0.10, store=store)

    Example (per-tenant monthly cap):
        store = SQLiteBudgetStore(
            "data/budgets.db",
            namespace=f"tenant:{tenant_id}:2026-05",
        )
    """

    supports_leases = True

    def __init__(self, db_path: str, namespace: str = "default") -> None:
        self._db_path = os.path.expanduser(db_path)
        self._namespace = namespace
        self._lock = _sqlite_lock(self._db_path)
        self._init_db()

    # ── internal helpers ───────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS budget_spend (
                        namespace TEXT PRIMARY KEY,
                        spent     REAL    NOT NULL DEFAULT 0.0,
                        updated   TEXT    NOT NULL DEFAULT (datetime('now'))
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS budget_leases (
                        id        TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        amount    REAL NOT NULL,
                        expires   REAL NOT NULL
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS budget_leases_ns ON budget_leases (namespace)"
                )

    def _get(self, conn: sqlite3.Connection) -> float:
        row = conn.execute(
            "SELECT spent FROM budget_spend WHERE namespace = ?",
            (self._namespace,),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def _held(self, conn: sqlite3.Connection, now: float) -> float:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0.0) FROM budget_leases WHERE namespace = ? AND expires > ?",
            (self._namespace, now),
        ).fetchone()
        return float(row[0])

    def _upsert(self, conn: sqlite3.Connection, value: float) -> None:
        conn.execute("""
            INSERT INTO budget_spend (namespace, spent, updated)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(namespace) DO UPDATE SET
                spent   = excluded.spent,
                updated = excluded.updated
        """, (self._namespace, value))

    def _exclusive(self, fn):
        """Run fn(conn, now) in a BEGIN EXCLUSIVE transaction; commit unless it returns False."""
        with self._lock:
            # BEGIN EXCLUSIVE acquires an OS-level write lock on the database file,
            # blocking all other connections (in any process) until this transaction
            # ends. This makes check-and-reserve atomic across gunicorn/uvicorn
            # workers sharing the same DB file.
            conn = sqlite3.connect(
                self._db_path, check_same_thread=False, timeout=10, isolation_level=None
            )
            try:
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("BEGIN EXCLUSIVE")
                now = time.time()
                conn.execute(
                    "DELETE FROM budget_leases WHERE namespace = ? AND expires <= ?",
                    (self._namespace, now),
                )
                result = fn(conn, now)
                conn.execute("ROLLBACK" if result is False else "COMMIT")
                return result
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    # ── public interface ───────────────────────────────────────────────────────

    def get_spent(self) -> float:
        with self._lock:
            with self._connect() as conn:
                return self._get(conn) + self._held(conn, time.time())

    def get_reserved(self) -> float:
        with self._lock:
            with self._connect() as conn:
                return self._held(conn, time.time())

    def add_spent(self, amount: float) -> float:
        with self._lock:
            with self._connect() as conn:
                # Delta SQL — atomic across OS processes. Two concurrent writers
                # each get their own SQLite write lock, serialize naturally, and
                # each correctly adds their delta on top of the committed value.
                conn.execute("""
                    INSERT INTO budget_spend (namespace, spent, updated)
                    VALUES (?, ?, datetime('now'))
                    ON CONFLICT(namespace) DO UPDATE SET
                        spent   = budget_spend.spent + excluded.spent,
                        updated = excluded.updated
                """, (self._namespace, amount))
                return self._get(conn) + self._held(conn, time.time())

    def set_spent(self, value: float) -> None:
        """Overwrite the spend; drops this namespace's leases so get_spent() == value."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM budget_leases WHERE namespace = ?", (self._namespace,))
                self._upsert(conn, float(value))

    def reset(self) -> None:
        self.set_spent(0.0)

    def atomic_check_and_reserve(self, total_budget: float, amount: float) -> bool:
        def op(conn, now):
            current = self._get(conn)
            if current + self._held(conn, now) + amount <= total_budget:
                self._upsert(conn, current + amount)
                return True
            return False
        return self._exclusive(op)

    def reserve_lease(self, total_budget: float, amount: float, ttl: float) -> Optional[str]:
        lease_id = uuid.uuid4().hex

        def op(conn, now):
            if self._get(conn) + self._held(conn, now) + amount > total_budget:
                return False
            conn.execute(
                "INSERT INTO budget_leases (id, namespace, amount, expires) VALUES (?, ?, ?, ?)",
                (lease_id, self._namespace, float(amount), now + float(ttl)),
            )
            return True
        return lease_id if self._exclusive(op) else None

    def extend_lease(self, lease_id: str, total_budget: float, extra: float, ttl: float) -> bool:
        def op(conn, now):
            row = conn.execute(
                "SELECT 1 FROM budget_leases WHERE id = ? AND namespace = ?",
                (lease_id, self._namespace),
            ).fetchone()
            if row is None:  # expired: whoever extends must reserve the whole amount again
                return False
            if self._get(conn) + self._held(conn, now) + extra > total_budget:
                return False
            conn.execute(
                "UPDATE budget_leases SET amount = amount + ?, expires = ? WHERE id = ?",
                (float(extra), now + float(ttl), lease_id),
            )
            return True
        return bool(self._exclusive(op))

    def renew_lease(self, lease_id: str, ttl: float) -> bool:
        def op(conn, now):
            cur = conn.execute(
                "UPDATE budget_leases SET expires = ? WHERE id = ? AND namespace = ?",
                (now + float(ttl), lease_id, self._namespace),
            )
            return cur.rowcount > 0
        return bool(self._exclusive(op))

    def release_lease(self, lease_id: str, amount: float) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM budget_leases WHERE id = ? AND namespace = ?",
                    (lease_id, self._namespace),
                )


# ── Stores for many machines (optional dependencies, imported on first use) ──

def __getattr__(name: str):
    if name == "RedisBudgetStore":
        from baar.core.redis_store import RedisBudgetStore
        return RedisBudgetStore
    if name == "PostgresBudgetStore":
        from baar.core.postgres_store import PostgresBudgetStore
        return PostgresBudgetStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
