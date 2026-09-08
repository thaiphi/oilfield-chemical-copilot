"""Durable, append-only decision ledger for a single Corpus V2 release."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .models import AcquisitionRecord, ApprovedSource, ExtractionRecord, ReleaseConfig, SourceDisposition, Stage


class CorpusV2LedgerError(RuntimeError):
    """Raised when a ledger operation would violate a Corpus V2 invariant."""


@dataclass(frozen=True)
class DispositionDecision:
    source_id: str
    disposition: SourceDisposition
    reviewer_id: str
    reason_code: str
    decided_at: str


@dataclass(frozen=True)
class LedgerEvent:
    event_type: str
    occurred_at: str
    payload: dict[str, object]


_STAGE_ORDER = tuple(Stage)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class CorpusV2Ledger:
    """SQLite-backed ledger that requires explicit source and stage decisions."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row

    @classmethod
    def create(cls, database_path: Path, *, release_config: ReleaseConfig) -> "CorpusV2Ledger":
        connection = cls._connect(database_path)
        ledger = cls(connection)
        try:
            with connection:
                ledger._create_schema()
                ledger._insert_release(release_config)
                ledger._event("release_created", {"release_id": release_config.release_id})
        except Exception:
            connection.close()
            raise
        return ledger

    @classmethod
    def open(cls, database_path: Path) -> "CorpusV2Ledger":
        connection = cls._connect(database_path)
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'release'"
        ).fetchone()
        if tables is None:
            connection.close()
            raise CorpusV2LedgerError("C2_LEDGER_INVALID")
        return cls(connection)

    @staticmethod
    def _connect(database_path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE release (
                release_id TEXT PRIMARY KEY,
                candidate_database_name TEXT NOT NULL,
                configured_database_name TEXT NOT NULL,
                legacy_database_names_json TEXT NOT NULL,
                expected_source_count INTEGER NOT NULL,
                source_register_sha256 TEXT NOT NULL
            );
            CREATE TABLE source_register (
                source_id TEXT PRIMARY KEY,
                source_sha256 TEXT NOT NULL
            );
            CREATE TABLE acquisitions (
                source_id TEXT PRIMARY KEY REFERENCES source_register(source_id),
                acquired_at TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                byte_count INTEGER NOT NULL
            );
            CREATE TABLE extractions (
                source_id TEXT PRIMARY KEY REFERENCES source_register(source_id),
                extracted_at TEXT NOT NULL,
                extractor TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                character_count INTEGER NOT NULL
            );
            CREATE TABLE disposition_decisions (
                decision_id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES source_register(source_id),
                disposition TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                decided_at TEXT NOT NULL
            );
            CREATE TABLE stage_checkpoints (
                stage TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            );
            CREATE TABLE event_history (
                event_id INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )

    def _insert_release(self, config: ReleaseConfig) -> None:
        self._connection.execute(
            """
            INSERT INTO release (
                release_id, candidate_database_name, configured_database_name,
                legacy_database_names_json, expected_source_count, source_register_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                config.release_id,
                config.candidate_database_name,
                config.configured_database_name,
                json.dumps(config.legacy_database_names),
                config.expected_source_count,
                config.source_register_sha256,
            ),
        )

    def _event(self, event_type: str, payload: dict[str, object]) -> None:
        self._connection.execute(
            "INSERT INTO event_history (event_type, occurred_at, payload_json) VALUES (?, ?, ?)",
            (event_type, _timestamp(), json.dumps(payload, sort_keys=True, separators=(",", ":"))),
        )

    def _source_exists(self, source_id: str) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM source_register WHERE source_id = ?", (source_id,)
        ).fetchone() is not None

    def record_source(self, source: ApprovedSource) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO source_register (source_id, source_sha256) VALUES (?, ?)",
                    (source.source_id, source.source_sha256),
                )
                self._event("source_registered", {"source_id": source.source_id})
        except sqlite3.IntegrityError as error:
            raise CorpusV2LedgerError("C2_SOURCE_DUPLICATE") from error

    def record_acquisition(self, acquisition: AcquisitionRecord) -> None:
        if not self._source_exists(acquisition.source_id):
            raise CorpusV2LedgerError("C2_SOURCE_UNKNOWN")
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO acquisitions (source_id, acquired_at, content_sha256, byte_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        acquisition.source_id,
                        acquisition.acquired_at,
                        acquisition.content_sha256,
                        acquisition.byte_count,
                    ),
                )
                self._event("source_acquired", {"source_id": acquisition.source_id})
        except sqlite3.IntegrityError as error:
            raise CorpusV2LedgerError("C2_ACQUISITION_DUPLICATE") from error

    def record_extraction(self, extraction: ExtractionRecord) -> None:
        if not self._source_exists(extraction.source_id):
            raise CorpusV2LedgerError("C2_SOURCE_UNKNOWN")
        if self._connection.execute(
            "SELECT 1 FROM acquisitions WHERE source_id = ?", (extraction.source_id,)
        ).fetchone() is None:
            raise CorpusV2LedgerError("C2_ACQUISITION_PREREQUISITE")
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO extractions (source_id, extracted_at, extractor, text_sha256, character_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        extraction.source_id,
                        extraction.extracted_at,
                        extraction.extractor,
                        extraction.text_sha256,
                        extraction.character_count,
                    ),
                )
                self._event("source_parsed", {"source_id": extraction.source_id})
        except sqlite3.IntegrityError as error:
            raise CorpusV2LedgerError("C2_EXTRACTION_DUPLICATE") from error

    def record_disposition(
        self,
        *,
        source_id: str,
        disposition: SourceDisposition,
        reviewer_id: str,
        reason_code: str,
    ) -> None:
        if not self._source_exists(source_id):
            raise CorpusV2LedgerError("C2_SOURCE_UNKNOWN")
        if not isinstance(disposition, SourceDisposition) or not reviewer_id or not reason_code:
            raise CorpusV2LedgerError("C2_DISPOSITION_INVALID")
        with self._connection:
            decided_at = _timestamp()
            self._connection.execute(
                """
                INSERT INTO disposition_decisions
                    (source_id, disposition, reviewer_id, reason_code, decided_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, disposition.value, reviewer_id, reason_code, decided_at),
            )
            self._event(
                "disposition_recorded",
                {"source_id": source_id, "disposition": disposition.value, "reviewer_id": reviewer_id},
            )

    def complete_stage(self, stage: Stage) -> None:
        if not isinstance(stage, Stage):
            raise CorpusV2LedgerError("C2_STAGE_INVALID")
        stage_index = _STAGE_ORDER.index(stage)
        if stage_index:
            prerequisite = _STAGE_ORDER[stage_index - 1]
            if prerequisite not in self.completed_stages():
                raise CorpusV2LedgerError("C2_STAGE_PREREQUISITE")
        if stage in self.completed_stages():
            raise CorpusV2LedgerError("C2_STAGE_DUPLICATE")
        with self._connection:
            self._connection.execute(
                "INSERT INTO stage_checkpoints (stage, completed_at) VALUES (?, ?)",
                (stage.value, _timestamp()),
            )
            self._event("stage_completed", {"stage": stage.value})

    def current_disposition(self, source_id: str) -> DispositionDecision | None:
        row = self._connection.execute(
            """
            SELECT source_id, disposition, reviewer_id, reason_code, decided_at
            FROM disposition_decisions WHERE source_id = ? ORDER BY decision_id DESC LIMIT 1
            """,
            (source_id,),
        ).fetchone()
        return self._decision(row) if row is not None else None

    def disposition_history(self, source_id: str) -> tuple[DispositionDecision, ...]:
        rows = self._connection.execute(
            """
            SELECT source_id, disposition, reviewer_id, reason_code, decided_at
            FROM disposition_decisions WHERE source_id = ? ORDER BY decision_id
            """,
            (source_id,),
        ).fetchall()
        return tuple(self._decision(row) for row in rows)

    @staticmethod
    def _decision(row: sqlite3.Row) -> DispositionDecision:
        return DispositionDecision(
            source_id=row["source_id"],
            disposition=SourceDisposition(row["disposition"]),
            reviewer_id=row["reviewer_id"],
            reason_code=row["reason_code"],
            decided_at=row["decided_at"],
        )

    def completed_stages(self) -> tuple[Stage, ...]:
        rows = self._connection.execute("SELECT stage FROM stage_checkpoints").fetchall()
        completed = {Stage(row["stage"]) for row in rows}
        return tuple(stage for stage in _STAGE_ORDER if stage in completed)

    def event_history(self) -> tuple[LedgerEvent, ...]:
        rows = self._connection.execute(
            "SELECT event_type, occurred_at, payload_json FROM event_history ORDER BY event_id"
        ).fetchall()
        return tuple(
            LedgerEvent(
                event_type=row["event_type"],
                occurred_at=row["occurred_at"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )

    def close(self) -> None:
        self._connection.close()
