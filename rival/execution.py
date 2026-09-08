"""Durable per-attempt accounting for supervised model execution.

SQLite is the request authority. Unknown remote outcomes are quarantined rather
than replayed. Reservations are estimates, never represented as provider charges.
No client can guarantee a remote spending ceiling without provider-side limits.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .mathx import canonical_hash


class ExecutionError(RuntimeError):
    pass


class AuthorizationStopped(ExecutionError):
    pass


class ReconciliationRequired(ExecutionError):
    pass


class TerminalResponseError(ExecutionError):
    """A known unusable response must not trigger another paid retry."""


def reported_cost(payload: dict[str, Any]) -> float | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("cost", "total_cost", "cost_usd", "provider_cost_usd"):
        value = usage.get(key)
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0):
            return float(value)
    return None


class AttemptJournal:
    def __init__(self, path: str | Path, scope_sha256: str):
        self.connection = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS execution_scope (id INTEGER PRIMARY KEY, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
                work_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
                state TEXT NOT NULL, payload TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
                work_id TEXT NOT NULL, ordinal INTEGER NOT NULL, state TEXT NOT NULL,
                reservation REAL NOT NULL, cost REAL, response_id TEXT, payload TEXT,
                authorized_budget REAL NOT NULL, authorized_until TEXT NOT NULL,
                started_at TEXT NOT NULL, completed_at TEXT, reconciliation TEXT,
                PRIMARY KEY (work_id, ordinal)
            );
        """)
        try:
            with self._transaction():
                row = self.connection.execute("SELECT digest FROM execution_scope WHERE id=1").fetchone()
                if row and row["digest"] != scope_sha256:
                    raise ExecutionError("journal belongs to a different execution scope")
                self.connection.execute("INSERT OR IGNORE INTO execution_scope VALUES (1, ?)", (scope_sha256,))
        except BaseException:
            self.connection.close()
            raise

    @contextmanager
    def _transaction(self):
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def summary(self) -> dict[str, Any]:
        with self.lock:
            row = self.connection.execute("""
                SELECT COUNT(*) AS attempts, COALESCE(SUM(cost),0) AS known_cost_usd,
                COALESCE(SUM(CASE WHEN cost IS NULL THEN reservation ELSE 0 END),0) AS reserved_usd,
                COALESCE(SUM(CASE WHEN cost IS NULL THEN 1 ELSE 0 END),0) AS unresolved_attempts
                FROM attempts
            """).fetchone()
        result = dict(row)
        result["total_cost_usd"] = None if result["unresolved_attempts"] else result["known_cost_usd"]
        return result

    def attempts_for(self, work_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM attempts WHERE work_id=? ORDER BY ordinal", (work_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def execute(self, work_id: str, request: dict[str, Any],
                send: Callable[[], dict[str, Any]], accept: Callable[[dict[str, Any]], Any],
                *, budget_usd: float, not_after: datetime, reservation_usd: float,
                max_attempts: int = 3, retry_delay_seconds: float = 1.0,
                max_new_attempts: int | None = None,
                max_total_attempts: int | None = None,
                reserved_ordinals: list[int] | None = None) -> dict[str, Any]:
        for limit in (max_attempts, max_new_attempts, max_total_attempts):
            if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool)):
                raise ValueError("attempt limits must be integers")
        if (not math.isfinite(budget_usd) or budget_usd <= 0
                or not math.isfinite(reservation_usd) or reservation_usd <= 0
                or max_attempts < 1 or not_after.tzinfo is None
                or not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0
                or (max_new_attempts is not None and max_new_attempts < 0)
                or (max_total_attempts is not None and max_total_attempts < 1)):
            raise ValueError("execution requires finite positive limits and an aware expiry")
        request_hash = canonical_hash(request)
        with self._transaction():
            row = self.connection.execute("SELECT * FROM requests WHERE work_id=?", (work_id,)).fetchone()
            if row:
                if row["request_hash"] != request_hash:
                    raise ExecutionError("work ID was reused for a different request")
                if row["state"] == "DONE":
                    return json.loads(row["payload"])
                if row["state"] in {"RUNNING", "UNKNOWN"}:
                    raise ReconciliationRequired("request is in flight or requires reconciliation; no replay")
                if row["state"] == "FAILED":
                    raise ExecutionError("request already exhausted its attempts")
            self.connection.execute(
                "INSERT INTO requests VALUES (?, ?, 'RUNNING', NULL) "
                "ON CONFLICT(work_id) DO UPDATE SET state='RUNNING'", (work_id, request_hash)
            )

        for new_attempt in range(max_attempts):
            try:
                with self._transaction():
                    summary = self.summary()
                    if max_total_attempts is not None and summary["attempts"] >= max_total_attempts:
                        raise AuthorizationStopped("total physical attempt limit reached")
                    if max_new_attempts is not None and new_attempt >= max_new_attempts:
                        raise AuthorizationStopped("maximum new attempt count reached")
                    # Serialize remote attempts in this journal. A crashed or
                    # indeterminate attempt stops all further spending.
                    if summary["unresolved_attempts"]:
                        raise ReconciliationRequired("journal has unresolved provider attempts")
                    now = datetime.now(timezone.utc)
                    if now >= not_after:
                        raise AuthorizationStopped("authorization expired before provider attempt")
                    if summary["known_cost_usd"] + reservation_usd > budget_usd + 1e-12:
                        raise AuthorizationStopped("next attempt reservation exceeds budget")
                    ordinal = len(self.attempts_for(work_id)) + 1
                    if ordinal > max_attempts:
                        raise ExecutionError("request already exhausted its attempts")
                    self.connection.execute("""
                        INSERT INTO attempts
                        (work_id, ordinal, state, reservation, authorized_budget, authorized_until, started_at)
                        VALUES (?, ?, 'IN_FLIGHT', ?, ?, ?, ?)
                    """, (work_id, ordinal, reservation_usd, budget_usd, not_after.isoformat(), now.isoformat()))
                    if reserved_ordinals is not None:
                        reserved_ordinals.append(ordinal)
            except ExecutionError:
                with self._transaction():
                    self.connection.execute("UPDATE requests SET state='BLOCKED' WHERE work_id=?", (work_id,))
                raise

            # The reservation is durable before crossing the network boundary.
            # An interrupt here leaves IN_FLIGHT; restart must never infer failure.
            try:
                payload = send()
                if not isinstance(payload, dict):
                    raise ValueError("provider payload must be an object")
            except Exception as exc:
                with self._transaction():
                    self.connection.execute("UPDATE attempts SET state='UNKNOWN' WHERE work_id=? AND ordinal=?", (work_id, ordinal))
                    self.connection.execute("UPDATE requests SET state='UNKNOWN' WHERE work_id=?", (work_id,))
                raise ReconciliationRequired("remote outcome or billing unknown; reconcile before resuming") from exc

            cost = reported_cost(payload)
            terminal = False
            try:
                accept(payload)
                accepted = True
            except (ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
                accepted = False
                terminal = isinstance(exc, TerminalResponseError)
                if isinstance(payload.get("_rival_transport"), dict):
                    payload["_rival_transport"]["rejection_type"] = type(exc).__name__
            state = "UNKNOWN" if cost is None else ("DONE" if accepted else "REJECTED")
            with self._transaction():
                encoded = json.dumps(payload, sort_keys=True)
                self.connection.execute("""
                    UPDATE attempts SET state=?, cost=?, response_id=?, payload=?, completed_at=?
                    WHERE work_id=? AND ordinal=?
                """, (state, cost, str(payload.get("id", "")), encoded,
                      datetime.now(timezone.utc).isoformat(), work_id, ordinal))
                final_state = "UNKNOWN" if cost is None else ("DONE" if accepted else "RUNNING")
                if not accepted and (terminal or ordinal == max_attempts) and cost is not None:
                    final_state = "FAILED"
                self.connection.execute("UPDATE requests SET state=?, payload=? WHERE work_id=?",
                                        (final_state, encoded, work_id))
            if cost is None:
                raise ReconciliationRequired("provider omitted cost; response preserved for reconciliation")
            if accepted:
                return payload
            if terminal:
                raise TerminalResponseError("completion rejected; inspect execution audit before preparing a new study")
            if ordinal < max_attempts:
                retry_after = payload.get("_retry_after", 0)
                if not isinstance(retry_after, (float, int)) or not math.isfinite(retry_after):
                    retry_after = 0
                time.sleep(min(30.0, max(retry_after, retry_delay_seconds * 2 ** (ordinal - 1))))
        raise ExecutionError("request already exhausted its attempts")

    def reconcile(self, work_id: str, ordinal: int, *, cost_usd: float,
                  evidence_reference: str, completed_payload: dict[str, Any] | None = None) -> None:
        """Operator reconciliation after confirming the original worker has stopped.

        Provide the actual provider billing evidence. A failed/unknown completion
        remains terminal; this action does not authorize a replacement request.
        """
        if not evidence_reference.strip() or not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("reconciliation needs a nonnegative actual cost and evidence reference")
        with self._transaction():
            row = self.connection.execute("SELECT * FROM attempts WHERE work_id=? AND ordinal=?", (work_id, ordinal)).fetchone()
            if not row or row["state"] not in {"UNKNOWN", "IN_FLIGHT"}:
                raise ExecutionError("attempt is not awaiting reconciliation")
            payload = completed_payload
            encoded = json.dumps(payload, sort_keys=True) if payload is not None else None
            self.connection.execute("""
                UPDATE attempts SET state='RECONCILED', cost=?, reconciliation=?, completed_at=?
                WHERE work_id=? AND ordinal=?
            """, (cost_usd, evidence_reference, datetime.now(timezone.utc).isoformat(), work_id, ordinal))
            self.connection.execute("UPDATE requests SET state=?, payload=? WHERE work_id=?",
                                    ("DONE" if payload is not None else "FAILED", encoded, work_id))

    def release_interrupted_claim(self, work_id: str, *, worker_stopped: bool) -> None:
        """Release a stopped worker between attempts, never an ambiguous remote call.

        Use only after the operator has stopped the original worker. If an
        IN_FLIGHT or UNKNOWN attempt exists, reconcile its provider evidence.
        """
        if worker_stopped is not True:
            raise ExecutionError("confirm the original worker has stopped first")
        with self._transaction():
            row = self.connection.execute("SELECT state FROM requests WHERE work_id=?", (work_id,)).fetchone()
            if not row or row["state"] != "RUNNING":
                raise ExecutionError("request does not have an interrupted running claim")
            pending = self.connection.execute(
                "SELECT 1 FROM attempts WHERE work_id=? AND cost IS NULL", (work_id,)
            ).fetchone()
            if pending:
                raise ReconciliationRequired("remote attempt is unresolved; claim cannot be released")
            self.connection.execute("UPDATE requests SET state='BLOCKED' WHERE work_id=?", (work_id,))

    def close(self):
        self.connection.close()
