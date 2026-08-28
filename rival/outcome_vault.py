from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .mathx import canonical_hash
from .schemas import OutcomeRevealReceipt, utc_now


class OutcomeVaultError(RuntimeError):
    pass


class OutcomeNotAvailable(OutcomeVaultError):
    pass


class OutcomeVaultAccessError(OutcomeVaultError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _key_bytes(key_material: bytes | str) -> bytes:
    material = (
        key_material.encode("utf-8")
        if isinstance(key_material, str)
        else bytes(key_material)
    )
    if len(material) < 16:
        raise ValueError("outcome vault key material must contain at least 16 bytes")
    return material


def _derive_key(key_material: bytes | str, salt: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"rival-outcome-vault-v1",
    ).derive(_key_bytes(key_material))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class OutcomeVault:
    """Separate AES-GCM encrypted store for not-yet-visible study outcomes."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.connection:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS encrypted_outcomes (
                    study_id TEXT PRIMARY KEY,
                    manifest_sha256 TEXT NOT NULL,
                    not_before TEXT NOT NULL,
                    salt BLOB NOT NULL,
                    nonce BLOB NOT NULL,
                    ciphertext BLOB NOT NULL,
                    outcome_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vault_access_events (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    detail_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _aad(study_id: str, manifest_sha256: str, not_before: str) -> bytes:
        return _canonical_bytes(
            {
                "schema": "rival.outcome-vault.v1",
                "study_id": study_id,
                "manifest_sha256": manifest_sha256,
                "not_before": not_before,
            }
        )

    def _audit(self, study_id: str, action: str, success: bool, detail: Any) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO vault_access_events
                (id, study_id, action, success, detail_sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"vlt_{uuid4().hex[:16]}",
                    study_id,
                    action,
                    int(success),
                    canonical_hash(detail),
                    utc_now().isoformat(),
                ),
            )

    def deposit(
        self,
        study_id: str,
        manifest_sha256: str,
        outcome: dict[str, Any],
        key_material: bytes | str,
        not_before: datetime,
    ) -> str:
        available_at = _as_utc(not_before).isoformat()
        plaintext = _canonical_bytes(outcome)
        outcome_sha256 = canonical_hash(outcome)
        salt = os.urandom(16)
        nonce = os.urandom(12)
        ciphertext = AESGCM(_derive_key(key_material, salt)).encrypt(
            nonce,
            plaintext,
            self._aad(study_id, manifest_sha256, available_at),
        )
        with self.lock, self.connection:
            existing = self.connection.execute(
                """
                SELECT manifest_sha256, outcome_sha256 FROM encrypted_outcomes
                WHERE study_id = ?
                """,
                (study_id,),
            ).fetchone()
            if existing:
                if (
                    existing["manifest_sha256"] == manifest_sha256
                    and existing["outcome_sha256"] == outcome_sha256
                ):
                    return outcome_sha256
                raise OutcomeVaultError("an immutable outcome already exists for this study")
            self.connection.execute(
                """
                INSERT INTO encrypted_outcomes
                (study_id, manifest_sha256, not_before, salt, nonce, ciphertext,
                 outcome_sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_id,
                    manifest_sha256,
                    available_at,
                    salt,
                    nonce,
                    ciphertext,
                    outcome_sha256,
                    utc_now().isoformat(),
                ),
            )
        self._audit(study_id, "deposit", True, {"outcome_sha256": outcome_sha256})
        return outcome_sha256

    def reveal(
        self,
        study_id: str,
        manifest_sha256: str,
        key_material: bytes | str,
        *,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], OutcomeRevealReceipt]:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM encrypted_outcomes WHERE study_id = ?", (study_id,)
            ).fetchone()
        if row is None:
            self._audit(study_id, "reveal", False, "missing")
            raise OutcomeVaultAccessError("no encrypted outcome exists for this study")
        if row["manifest_sha256"] != manifest_sha256:
            self._audit(study_id, "reveal", False, "manifest-mismatch")
            raise OutcomeVaultAccessError("manifest hash does not match the vault binding")
        current = _as_utc(now or utc_now())
        available_at = _as_utc(datetime.fromisoformat(row["not_before"]))
        if current < available_at:
            self._audit(study_id, "reveal", False, "not-before")
            raise OutcomeNotAvailable(
                f"outcome is unavailable until {available_at.isoformat()}"
            )
        try:
            plaintext = AESGCM(_derive_key(key_material, row["salt"])).decrypt(
                row["nonce"],
                row["ciphertext"],
                self._aad(study_id, manifest_sha256, row["not_before"]),
            )
        except InvalidTag as exc:
            self._audit(study_id, "reveal", False, "authentication-failed")
            raise OutcomeVaultAccessError("outcome decryption authentication failed") from exc
        outcome = json.loads(plaintext.decode("utf-8"))
        if canonical_hash(outcome) != row["outcome_sha256"]:
            self._audit(study_id, "reveal", False, "digest-mismatch")
            raise OutcomeVaultAccessError("decrypted outcome digest does not match")
        receipt = OutcomeRevealReceipt(
            study_id=study_id,
            manifest_sha256=manifest_sha256,
            outcome_sha256=row["outcome_sha256"],
            revealed_at=current,
        )
        self._audit(study_id, "reveal", True, receipt)
        return outcome, receipt

    def audit_events(self, study_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT id, study_id, action, success, detail_sha256, created_at
                FROM vault_access_events WHERE study_id = ? ORDER BY created_at, id
                """,
                (study_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.connection.close()
