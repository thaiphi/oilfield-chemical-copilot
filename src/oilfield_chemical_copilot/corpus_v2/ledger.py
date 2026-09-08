"""Durable, append-only decision ledger for a single Corpus V2 release."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .models import (
    AcquisitionRecord,
    ApprovedSource,
    ExtractionOutcome,
    ExtractionRecord,
    ReleaseConfig,
    SourceDisposition,
    StageArtifactKind,
    Stage,
)


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
_REQUIRED_TABLES = {
    "release": {"release_id", "candidate_database_name", "configured_database_name", "legacy_database_names_json", "expected_source_count", "source_register_sha256", "critical_source_register_sha256"},
    "source_register": {"source_id", "source_sha256"},
    "acquisitions": {"source_id", "acquired_at", "content_sha256", "byte_count"},
    "extractions": {"source_id", "extracted_at", "extractor", "text_sha256", "character_count", "outcome"},
    "disposition_decisions": {"decision_id", "source_id", "disposition", "reviewer_id", "reason_code", "decided_at"},
    "stage_checkpoints": {"stage", "completed_at"},
    "event_history": {"event_id", "event_type", "occurred_at", "payload_json"},
    "stage_artifacts": {"stage", "artifact_kind", "artifact_sha256", "recorded_at"},
}


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
            connection.execute("BEGIN IMMEDIATE")
            ledger._reject_existing_schema()
            ledger._create_schema()
            ledger._insert_release(release_config)
            ledger._event("release_created", {"release_id": release_config.release_id})
            connection.commit()
        except Exception:
            connection.rollback()
            connection.close()
            raise
        return ledger

    @classmethod
    def open(cls, database_path: Path) -> "CorpusV2Ledger":
        connection = cls._connect(database_path)
        ledger = cls(connection)
        try:
            connection.execute("BEGIN")
            ledger._validate_schema_and_metadata()
            connection.rollback()
        except (CorpusV2LedgerError, sqlite3.Error, ValueError, TypeError, json.JSONDecodeError):
            connection.rollback()
            connection.close()
            raise CorpusV2LedgerError("C2_LEDGER_INVALID") from None
        return ledger

    @staticmethod
    def _connect(database_path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _create_schema(self) -> None:
        statements = (
            """CREATE TABLE release (
                release_id TEXT PRIMARY KEY,
                candidate_database_name TEXT NOT NULL,
                configured_database_name TEXT NOT NULL,
                legacy_database_names_json TEXT NOT NULL,
                expected_source_count INTEGER NOT NULL,
                source_register_sha256 TEXT NOT NULL,
                critical_source_register_sha256 TEXT NOT NULL
            )""",
            """CREATE TABLE source_register (
                source_id TEXT PRIMARY KEY,
                source_sha256 TEXT NOT NULL
            )""",
            """CREATE TABLE acquisitions (
                source_id TEXT PRIMARY KEY REFERENCES source_register(source_id),
                acquired_at TEXT NOT NULL, content_sha256 TEXT NOT NULL, byte_count INTEGER NOT NULL
            )""",
            """CREATE TABLE extractions (
                source_id TEXT PRIMARY KEY REFERENCES source_register(source_id),
                extracted_at TEXT NOT NULL, extractor TEXT NOT NULL, text_sha256 TEXT,
                character_count INTEGER NOT NULL, outcome TEXT NOT NULL
            )""",
            """CREATE TABLE disposition_decisions (
                decision_id INTEGER PRIMARY KEY, source_id TEXT NOT NULL REFERENCES source_register(source_id),
                disposition TEXT NOT NULL, reviewer_id TEXT NOT NULL, reason_code TEXT NOT NULL,
                decided_at TEXT NOT NULL
            )""",
            """CREATE TABLE stage_checkpoints (
                stage TEXT PRIMARY KEY, completed_at TEXT NOT NULL
            )""",
            """CREATE TABLE event_history (
                event_id INTEGER PRIMARY KEY, event_type TEXT NOT NULL, occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )""",
            """CREATE TABLE stage_artifacts (
                stage TEXT PRIMARY KEY, artifact_kind TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            )""",
        )
        for statement in statements:
            self._connection.execute(statement)

    def _reject_existing_schema(self) -> None:
        tables = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        if tables:
            raise CorpusV2LedgerError("C2_LEDGER_INVALID")

    def _validate_schema_and_metadata(self) -> None:
        table_names = {
            row["name"]
            for row in self._connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if not set(_REQUIRED_TABLES).issubset(table_names):
            raise CorpusV2LedgerError("C2_LEDGER_INVALID")
        for table_name, columns in _REQUIRED_TABLES.items():
            actual_columns = {
                row["name"] for row in self._connection.execute(f"PRAGMA table_info({table_name})")
            }
            if not columns.issubset(actual_columns):
                raise CorpusV2LedgerError("C2_LEDGER_INVALID")
        rows = self._connection.execute("SELECT * FROM release").fetchall()
        if len(rows) != 1:
            raise CorpusV2LedgerError("C2_LEDGER_INVALID")
        release = rows[0]
        legacy_names = json.loads(release["legacy_database_names_json"])
        if (
            not isinstance(legacy_names, list)
            or not all(isinstance(name, str) and name for name in legacy_names)
            or not isinstance(release["candidate_database_name"], str)
            or not isinstance(release["configured_database_name"], str)
            or not release["candidate_database_name"]
            or release["candidate_database_name"] == release["configured_database_name"]
            or release["candidate_database_name"] in legacy_names
            or not isinstance(release["expected_source_count"], int)
            or release["expected_source_count"] <= 0
            or not isinstance(release["source_register_sha256"], str)
            or len(release["source_register_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in release["source_register_sha256"])
            or not isinstance(release["critical_source_register_sha256"], str)
            or len(release["critical_source_register_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in release["critical_source_register_sha256"]
            )
        ):
            raise CorpusV2LedgerError("C2_LEDGER_INVALID")

    def _insert_release(self, config: ReleaseConfig) -> None:
        self._connection.execute(
            """
            INSERT INTO release (
                release_id, candidate_database_name, configured_database_name,
                legacy_database_names_json, expected_source_count, source_register_sha256,
                critical_source_register_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.release_id,
                config.candidate_database_name,
                config.configured_database_name,
                json.dumps(config.legacy_database_names),
                config.expected_source_count,
                config.source_register_sha256,
                config.critical_source_register_sha256,
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

    def _ensure_mutable(self) -> None:
        if Stage.PROMOTED in self.completed_stages():
            raise CorpusV2LedgerError("C2_RELEASE_IMMUTABLE")

    def _require_completed(self, stage: Stage) -> None:
        if stage not in self.completed_stages():
            raise CorpusV2LedgerError("C2_STAGE_PREREQUISITE")

    def _require_incomplete(self, stage: Stage) -> None:
        if stage in self.completed_stages():
            raise CorpusV2LedgerError("C2_STAGE_STATE")

    def record_source(self, source: ApprovedSource) -> None:
        self._ensure_mutable()
        self._require_incomplete(Stage.REGISTERED)
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
        self._ensure_mutable()
        if not self._source_exists(acquisition.source_id):
            raise CorpusV2LedgerError("C2_SOURCE_UNKNOWN")
        self._require_completed(Stage.REGISTERED)
        self._require_incomplete(Stage.ACQUIRED)
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
        self._ensure_mutable()
        if not self._source_exists(extraction.source_id):
            raise CorpusV2LedgerError("C2_SOURCE_UNKNOWN")
        self._require_completed(Stage.ACQUIRED)
        self._require_incomplete(Stage.PARSED)
        if self._connection.execute(
            "SELECT 1 FROM acquisitions WHERE source_id = ?", (extraction.source_id,)
        ).fetchone() is None:
            raise CorpusV2LedgerError("C2_ACQUISITION_PREREQUISITE")
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO extractions
                        (source_id, extracted_at, extractor, text_sha256, character_count, outcome)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        extraction.source_id,
                        extraction.extracted_at,
                        extraction.extractor,
                        extraction.text_sha256,
                        extraction.character_count,
                        extraction.outcome.value,
                    ),
                )
                self._event("source_parsed", {"source_id": extraction.source_id})
        except sqlite3.IntegrityError as error:
            raise CorpusV2LedgerError("C2_EXTRACTION_DUPLICATE") from error

    def record_stage_artifact(
        self, stage: Stage, *, artifact_kind: StageArtifactKind, artifact_sha256: str
    ) -> None:
        self._ensure_mutable()
        if not isinstance(stage, Stage) or stage.value not in {
            "CHUNKED", "EMBEDDED", "INDEX_VALIDATED", "EVALUATED", "PROMOTION_READY", "PROMOTED",
        }:
            raise CorpusV2LedgerError("C2_STAGE_ARTIFACT_INVALID")
        stage_index = _STAGE_ORDER.index(stage)
        self._require_completed(_STAGE_ORDER[stage_index - 1])
        self._require_incomplete(stage)
        if not isinstance(artifact_kind, StageArtifactKind):
            raise CorpusV2LedgerError("C2_STAGE_ARTIFACT_INVALID")
        try:
            expected_kind = StageArtifactKind.for_stage(stage)
        except ValueError as error:
            raise CorpusV2LedgerError("C2_STAGE_ARTIFACT_INVALID") from error
        if artifact_kind is not expected_kind:
            raise CorpusV2LedgerError("C2_STAGE_ARTIFACT_INVALID")
        if (
            not isinstance(artifact_sha256, str)
            or len(artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in artifact_sha256)
        ):
            raise CorpusV2LedgerError("C2_STAGE_ARTIFACT_INVALID")
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO stage_artifacts (stage, artifact_kind, artifact_sha256, recorded_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (stage.value, artifact_kind.value, artifact_sha256, _timestamp()),
                )
                self._event(
                    "stage_artifact_recorded", {"stage": stage.value, "artifact_kind": artifact_kind.value}
                )
        except sqlite3.IntegrityError as error:
            raise CorpusV2LedgerError("C2_STAGE_ARTIFACT_DUPLICATE") from error

    def record_disposition(
        self,
        *,
        source_id: str,
        disposition: SourceDisposition,
        reviewer_id: str,
        reason_code: str,
    ) -> None:
        self._ensure_mutable()
        if not self._source_exists(source_id):
            raise CorpusV2LedgerError("C2_SOURCE_UNKNOWN")
        self._require_completed(Stage.PARSED)
        self._require_incomplete(Stage.REVIEWED)
        if not isinstance(disposition, SourceDisposition) or not reviewer_id or not reason_code:
            raise CorpusV2LedgerError("C2_DISPOSITION_INVALID")
        outcome_row = self._connection.execute(
            "SELECT outcome FROM extractions WHERE source_id = ?", (source_id,)
        ).fetchone()
        expected_disposition = {
            SourceDisposition.NON_TEXT: ExtractionOutcome.NON_TEXT,
            SourceDisposition.EMPTY: ExtractionOutcome.EMPTY,
            SourceDisposition.UNSUPPORTED: ExtractionOutcome.UNSUPPORTED,
            SourceDisposition.EXTRACTION_FAILED: ExtractionOutcome.FAILED,
        }.get(disposition)
        terminal_dispositions = {
            ExtractionOutcome.NON_TEXT: SourceDisposition.NON_TEXT,
            ExtractionOutcome.EMPTY: SourceDisposition.EMPTY,
            ExtractionOutcome.UNSUPPORTED: SourceDisposition.UNSUPPORTED,
            ExtractionOutcome.FAILED: SourceDisposition.EXTRACTION_FAILED,
        }
        if outcome_row is None:
            raise CorpusV2LedgerError("C2_DISPOSITION_PREREQUISITE")
        outcome = ExtractionOutcome(outcome_row["outcome"])
        if expected_disposition is not None and outcome is not expected_disposition:
            raise CorpusV2LedgerError("C2_DISPOSITION_PREREQUISITE")
        if outcome in terminal_dispositions and disposition is not terminal_dispositions[outcome]:
            raise CorpusV2LedgerError("C2_DISPOSITION_PREREQUISITE")
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
        self._validate_stage_state(stage)
        with self._connection:
            self._connection.execute(
                "INSERT INTO stage_checkpoints (stage, completed_at) VALUES (?, ?)",
                (stage.value, _timestamp()),
            )
            self._event("stage_completed", {"stage": stage.value})

    def _validate_stage_state(self, stage: Stage) -> None:
        source_count = self._connection.execute("SELECT COUNT(*) FROM source_register").fetchone()[0]
        expected_source_count = self._connection.execute(
            "SELECT expected_source_count FROM release"
        ).fetchone()[0]
        if stage is Stage.REGISTERED and source_count != expected_source_count:
            raise CorpusV2LedgerError("C2_STAGE_STATE")
        if stage is Stage.ACQUIRED:
            acquisition_count = self._connection.execute("SELECT COUNT(*) FROM acquisitions").fetchone()[0]
            if acquisition_count != source_count:
                raise CorpusV2LedgerError("C2_STAGE_STATE")
        if stage is Stage.PARSED:
            extraction_count = self._connection.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
            if extraction_count != source_count:
                raise CorpusV2LedgerError("C2_STAGE_STATE")
        if stage is Stage.REVIEWED:
            reviewed_count = self._connection.execute(
                "SELECT COUNT(DISTINCT source_id) FROM disposition_decisions"
            ).fetchone()[0]
            if reviewed_count != source_count:
                raise CorpusV2LedgerError("C2_STAGE_STATE")
        if stage.value in {"CHUNKED", "EMBEDDED", "INDEX_VALIDATED", "EVALUATED", "PROMOTION_READY", "PROMOTED"}:
            artifact = self._connection.execute(
                "SELECT artifact_sha256 FROM stage_artifacts WHERE stage = ?", (stage.value,)
            ).fetchone()
            if artifact is None:
                raise CorpusV2LedgerError("C2_STAGE_ARTIFACT_REQUIRED")

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
