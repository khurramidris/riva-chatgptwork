from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .mathx import canonical_hash
from .schemas import (
    EvidenceSource,
    EvaluationResult,
    PhaseEvent,
    ScenarioSpec,
    SealedStudyManifest,
    SimulationResult,
    utc_now,
)


class ImmutableConflict(RuntimeError):
    pass


class EvidenceStore:
    """Append-only SQLite ledger for evidence, studies, runs and outcomes."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS evidence_sources (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sealed_manifests (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS phase_events (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(study_id, ordinal)
                );
                """
            )

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return value
        raise TypeError(f"cannot serialize {type(value)!r}")

    def _insert_immutable(
        self,
        table: str,
        identifier: str,
        payload: dict[str, Any],
        created_at: str,
        foreign_key: tuple[str, str] | None = None,
    ) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = canonical_hash(payload)
        columns = ["id"]
        values: list[Any] = [identifier]
        if foreign_key:
            columns.append(foreign_key[0])
            values.append(foreign_key[1])
        columns.extend(["payload", "sha256", "created_at"])
        values.extend([encoded, digest, created_at])
        placeholders = ",".join("?" for _ in values)
        with self.lock, self.connection:
            try:
                self.connection.execute(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                    values,
                )
            except sqlite3.IntegrityError as exc:
                existing = self.connection.execute(
                    f"SELECT sha256 FROM {table} WHERE id = ?", (identifier,)
                ).fetchone()
                if existing and existing["sha256"] == digest:
                    return digest
                raise ImmutableConflict(
                    f"{table} record {identifier!r} already exists with different content"
                ) from exc
        return digest

    def register_evidence(self, source: EvidenceSource) -> str:
        payload = self._payload(source)
        return self._insert_immutable(
            "evidence_sources",
            source.source_id,
            payload,
            source.collected_at.isoformat(),
        )

    def register_study(self, scenario: ScenarioSpec) -> str:
        payload = self._payload(scenario)
        return self._insert_immutable(
            "studies", scenario.scenario_id, payload, utc_now().isoformat()
        )

    def save_run(self, result: SimulationResult) -> str:
        payload = self._payload(result)
        return self._insert_immutable(
            "runs",
            result.run_id,
            payload,
            result.created_at.isoformat(),
            foreign_key=("study_id", result.scenario.scenario_id),
        )

    def save_evaluation(self, evaluation: EvaluationResult) -> str:
        payload = self._payload(evaluation)
        return self._insert_immutable(
            "evaluations",
            evaluation.evaluation_id,
            payload,
            evaluation.created_at.isoformat(),
            foreign_key=("run_id", evaluation.run_id),
        )

    def save_manifest(self, sealed: SealedStudyManifest) -> str:
        return self._insert_immutable(
            "sealed_manifests",
            sealed.manifest.manifest_id,
            self._payload(sealed),
            sealed.seal.sealed_at.isoformat(),
            foreign_key=("study_id", sealed.manifest.study_id),
        )

    def manifest_for_study(self, study_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT payload FROM sealed_manifests WHERE study_id = ?", (study_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def append_phase_event(self, event: PhaseEvent) -> str:
        allowed = {
            ("draft", "prediction_locked"),
            ("prediction_locked", "outcomes_revealed"),
            ("outcomes_revealed", "evaluated"),
        }
        if (event.from_phase, event.to_phase) not in allowed:
            raise ValueError(
                f"unsupported phase transition {event.from_phase!r} -> {event.to_phase!r}"
            )
        payload = event.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = canonical_hash(payload)
        with self.lock, self.connection:
            last = self.connection.execute(
                """
                SELECT ordinal, payload FROM phase_events
                WHERE study_id = ? ORDER BY ordinal DESC LIMIT 1
                """,
                (event.study_id,),
            ).fetchone()
            if last:
                previous = json.loads(last["payload"])
                ordinal = int(last["ordinal"]) + 1
                if event.from_phase != previous["to_phase"]:
                    raise ImmutableConflict("phase transition does not continue the chain")
                if event.previous_event_sha256 != previous["event_sha256"]:
                    raise ImmutableConflict("phase event previous hash does not match")
            else:
                ordinal = 0
                if event.from_phase != "draft" or event.previous_event_sha256 is not None:
                    raise ImmutableConflict("the first phase event must start at draft")
            try:
                self.connection.execute(
                    """
                    INSERT INTO phase_events
                    (id, study_id, ordinal, payload, sha256, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.study_id,
                        ordinal,
                        encoded,
                        digest,
                        event.created_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ImmutableConflict("phase event conflicts with the immutable chain") from exc
        return digest

    def last_phase_event(self, study_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                """
                SELECT payload FROM phase_events
                WHERE study_id = ? ORDER BY ordinal DESC LIMIT 1
                """,
                (study_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def verify_phase_chain(self, study_id: str) -> bool:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT payload, sha256 FROM phase_events
                WHERE study_id = ? ORDER BY ordinal ASC
                """,
                (study_id,),
            ).fetchall()
        previous_hash: str | None = None
        previous_phase = "draft"
        for row in rows:
            payload = json.loads(row["payload"])
            if canonical_hash(payload) != row["sha256"]:
                return False
            event_hash = payload.pop("event_sha256", None)
            if canonical_hash(payload) != event_hash:
                return False
            if payload["from_phase"] != previous_phase:
                return False
            if payload["previous_event_sha256"] != previous_hash:
                return False
            previous_hash = event_hash
            previous_phase = payload["to_phase"]
        return bool(rows)

    def get(self, table: str, identifier: str) -> dict[str, Any] | None:
        allowed = {
            "evidence_sources",
            "studies",
            "runs",
            "evaluations",
            "sealed_manifests",
            "phase_events",
        }
        if table not in allowed:
            raise ValueError(f"unsupported table {table!r}")
        with self.lock:
            row = self.connection.execute(
                f"SELECT payload FROM {table} WHERE id = ?", (identifier,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_rows(self, table: str, limit: int = 100) -> list[dict[str, Any]]:
        allowed = {
            "evidence_sources",
            "studies",
            "runs",
            "evaluations",
            "sealed_manifests",
            "phase_events",
        }
        if table not in allowed:
            raise ValueError(f"unsupported table {table!r}")
        limit = min(max(limit, 1), 1000)
        with self.lock:
            rows = self.connection.execute(
                f"SELECT payload FROM {table} ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def close(self) -> None:
        self.connection.close()
