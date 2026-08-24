"""Resumable metadata-only reconciliation for the private evaluation corpus."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
from tempfile import NamedTemporaryFile
from typing import Iterable, Mapping

from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
    E1A3SamplingError,
    E1A3SlotAllocation,
    E1A3SourceMetadata,
    allocate_sampling_slots,
    build_sampling_slots,
)


SCHEMA_VERSION = 1
RUN_STATUSES = frozenset({"IN_PROGRESS", "BLOCKED", "COMPLETE", "INVALID"})
CHECKPOINT_STATUSES = frozenset({"NOT_STARTED", "IN_PROGRESS", "COMPLETE", "BLOCKED"})
TOPICS = frozenset({"iron_sulfide", "scale", "corrosion", "paraffin"})
INDEX_SOURCE_TOPICS = TOPICS | {"unassigned"}
SOURCE_ROLES = frozenset({"foundational", "supporting"})
SUBSTANTIVE_STATUSES = frozenset({"SUBSTANTIVE", "TITLE_ONLY", "INELIGIBLE"})
SNAPSHOT_NAMES = (
    "drive-inventory.jsonl",
    "local-inventory.jsonl",
    "index-inventory.jsonl",
    "document-matches.jsonl",
    "review-decisions.jsonl",
    "locator-capacity.jsonl",
)


class CorpusReconciliationError(RuntimeError):
    """Raised when reconciliation metadata or state violates its contract."""


def _fail(code: str) -> None:
    raise CorpusReconciliationError(code)


def _exact_keys(mapping: Mapping[str, object], expected: frozenset[str], code: str) -> None:
    if not isinstance(mapping, Mapping) or frozenset(mapping) != expected:
        _fail(code)


def _string(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _optional_string(value: object, code: str) -> str | None:
    if value is None:
        return None
    return _string(value, code)


def _integer(value: object, code: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(code)
    return value


def _digest(value: object, code: str) -> str:
    digest = _string(value, code).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(code)
    return digest


def _relative_path(value: object, code: str) -> str:
    raw = _string(value, code).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw.startswith("/"):
        _fail(code)
    normalized = path.as_posix()
    if normalized in {"", "."}:
        _fail(code)
    return normalized


def require_private_reconciliation_root(path: Path, *, expected_root: Path) -> Path:
    """Require the exact controller-owned reconciliation root."""
    resolved = path.resolve()
    if resolved != expected_root.resolve():
        _fail("CORPUS_RECONCILIATION_PRIVATE_ROOT_INVALID")
    return resolved


@dataclass(frozen=True)
class DriveFileRecord:
    drive_file_id: str
    name: str
    mime_type: str
    size_bytes: int
    checksum_algorithm: str | None
    checksum: str | None
    modified_time: str
    parent_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> DriveFileRecord:
        code = "CORPUS_RECONCILIATION_DRIVE_RECORD_INVALID"
        _exact_keys(
            mapping,
            frozenset(
                {
                    "drive_file_id",
                    "name",
                    "mime_type",
                    "size_bytes",
                    "checksum_algorithm",
                    "checksum",
                    "modified_time",
                    "parent_ids",
                }
            ),
            code,
        )
        raw_algorithm = _optional_string(mapping["checksum_algorithm"], code)
        raw_checksum = _optional_string(mapping["checksum"], code)
        algorithm = raw_algorithm.lower() if raw_algorithm is not None else None
        checksum = raw_checksum.lower() if raw_checksum is not None else None
        if (algorithm is None) != (checksum is None):
            _fail(code)
        parents = mapping["parent_ids"]
        if not isinstance(parents, list):
            _fail(code)
        parent_ids = tuple(_string(item, code) for item in parents)
        if len(parent_ids) != len(set(parent_ids)):
            _fail(code)
        return cls(
            drive_file_id=_string(mapping["drive_file_id"], code),
            name=_string(mapping["name"], code),
            mime_type=_string(mapping["mime_type"], code),
            size_bytes=_integer(mapping["size_bytes"], code),
            checksum_algorithm=algorithm,
            checksum=checksum,
            modified_time=_string(mapping["modified_time"], code),
            parent_ids=parent_ids,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "drive_file_id": self.drive_file_id,
            "name": self.name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "checksum_algorithm": self.checksum_algorithm,
            "checksum": self.checksum,
            "modified_time": self.modified_time,
            "parent_ids": list(self.parent_ids),
        }


@dataclass(frozen=True)
class LocalFileRecord:
    relative_path: str
    sha256: str
    provider_checksum_algorithm: str | None
    provider_checksum: str | None
    size_bytes: int
    file_type: str
    parser_status: str
    page_or_sheet_count: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> LocalFileRecord:
        code = "CORPUS_RECONCILIATION_LOCAL_RECORD_INVALID"
        _exact_keys(
            mapping,
            frozenset(
                {
                    "relative_path",
                    "sha256",
                    "provider_checksum_algorithm",
                    "provider_checksum",
                    "size_bytes",
                    "file_type",
                    "parser_status",
                    "page_or_sheet_count",
                }
            ),
            code,
        )
        raw_algorithm = _optional_string(mapping["provider_checksum_algorithm"], code)
        raw_checksum = _optional_string(mapping["provider_checksum"], code)
        algorithm = raw_algorithm.lower() if raw_algorithm is not None else None
        checksum = raw_checksum.lower() if raw_checksum is not None else None
        if (algorithm is None) != (checksum is None):
            _fail(code)
        return cls(
            relative_path=_relative_path(mapping["relative_path"], code),
            sha256=_digest(mapping["sha256"], code),
            provider_checksum_algorithm=algorithm,
            provider_checksum=checksum,
            size_bytes=_integer(mapping["size_bytes"], code),
            file_type=_string(mapping["file_type"], code).lower(),
            parser_status=_string(mapping["parser_status"], code),
            page_or_sheet_count=_integer(mapping["page_or_sheet_count"], code),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "provider_checksum_algorithm": self.provider_checksum_algorithm,
            "provider_checksum": self.provider_checksum,
            "size_bytes": self.size_bytes,
            "file_type": self.file_type,
            "parser_status": self.parser_status,
            "page_or_sheet_count": self.page_or_sheet_count,
        }


@dataclass(frozen=True)
class LocalParserMetadata:
    parser_status: str
    page_or_sheet_count: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> LocalParserMetadata:
        code = "CORPUS_RECONCILIATION_LOCAL_METADATA_INVALID"
        _exact_keys(mapping, frozenset({"parser_status", "page_or_sheet_count"}), code)
        return cls(
            parser_status=_string(mapping["parser_status"], code),
            page_or_sheet_count=_integer(mapping["page_or_sheet_count"], code),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "parser_status": self.parser_status,
            "page_or_sheet_count": self.page_or_sheet_count,
        }


@dataclass(frozen=True)
class IndexSourceRecord:
    source_id: str
    source_path: str
    parser_type: str
    topic: str
    chunk_count: int
    embedding_model: str
    index_contract_sha256: str
    provenance_drive_file_id: str | None
    content_sha256: str | None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> IndexSourceRecord:
        code = "CORPUS_RECONCILIATION_INDEX_RECORD_INVALID"
        _exact_keys(
            mapping,
            frozenset(
                {
                    "source_id",
                    "source_path",
                    "parser_type",
                    "topic",
                    "chunk_count",
                    "embedding_model",
                    "index_contract_sha256",
                    "provenance_drive_file_id",
                    "content_sha256",
                }
            ),
            code,
        )
        topic = _string(mapping["topic"], code)
        if topic not in INDEX_SOURCE_TOPICS:
            _fail(code)
        return cls(
            source_id=_string(mapping["source_id"], code),
            source_path=_relative_path(mapping["source_path"], code),
            parser_type=_string(mapping["parser_type"], code),
            topic=topic,
            chunk_count=_integer(mapping["chunk_count"], code, minimum=1),
            embedding_model=_string(mapping["embedding_model"], code),
            index_contract_sha256=_digest(mapping["index_contract_sha256"], code),
            provenance_drive_file_id=_optional_string(
                mapping["provenance_drive_file_id"], code
            ),
            content_sha256=(
                _digest(mapping["content_sha256"], code)
                if mapping["content_sha256"] is not None
                else None
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_path": self.source_path,
            "parser_type": self.parser_type,
            "topic": self.topic,
            "chunk_count": self.chunk_count,
            "embedding_model": self.embedding_model,
            "index_contract_sha256": self.index_contract_sha256,
            "provenance_drive_file_id": self.provenance_drive_file_id,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class IndexLocatorRecord:
    source_id: str
    locator: str
    topic: str
    source_role: str
    substantive_status: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> IndexLocatorRecord:
        code = "CORPUS_RECONCILIATION_INDEX_RECORD_INVALID"
        _exact_keys(
            mapping,
            frozenset({"source_id", "locator", "topic", "source_role", "substantive_status"}),
            code,
        )
        topic = _string(mapping["topic"], code)
        source_role = _string(mapping["source_role"], code)
        substantive_status = _string(mapping["substantive_status"], code)
        topic_is_closed_unassigned = (
            topic == "unassigned" and substantive_status == "INELIGIBLE"
        )
        if (
            (topic not in TOPICS and not topic_is_closed_unassigned)
            or source_role not in SOURCE_ROLES
            or substantive_status not in SUBSTANTIVE_STATUSES
        ):
            _fail(code)
        return cls(
            source_id=_string(mapping["source_id"], code),
            locator=_string(mapping["locator"], code),
            topic=topic,
            source_role=source_role,
            substantive_status=substantive_status,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "locator": self.locator,
            "topic": self.topic,
            "source_role": self.source_role,
            "substantive_status": self.substantive_status,
        }


@dataclass(frozen=True)
class CheckpointRecord:
    stage: str
    status: str
    committed_records: int
    page_token: str | None
    error_code: str | None


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    committed_records: int


@dataclass(frozen=True)
class MatchSummary:
    exact_match: int
    duplicate_alias: int
    drive_only: int
    local_only: int
    parsed_not_indexed: int
    index_only: int
    ambiguous_review_required: int
    ineligible: int


@dataclass(frozen=True)
class CapacityStratum:
    topic: str
    source_role: str
    fresh_locator_count: int
    required_locators: int
    sufficient: bool


@dataclass(frozen=True)
class CapacityReport:
    strata: tuple[CapacityStratum, ...]
    all_sufficient: bool

    def to_public_mapping(self) -> dict[str, object]:
        return {
            "all_sufficient": self.all_sufficient,
            "strata": [
                {
                    "topic": item.topic,
                    "source_role": item.source_role,
                    "sufficient": item.sufficient,
                }
                for item in self.strata
            ],
        }


@dataclass(frozen=True)
class DryRunResult:
    status: str
    error_code: str | None
    allocations: tuple[E1A3SlotAllocation, ...]


@dataclass(frozen=True)
class SnapshotArtifact:
    name: str
    path: Path
    manifest_path: Path
    sha256: str
    record_count: int


@dataclass(frozen=True)
class SnapshotSet:
    artifacts: tuple[SnapshotArtifact, ...]


class ReconciliationStore:
    """Transactional controller-owned state for one reconciliation run."""

    def __init__(self, *, root: Path, run_id: str, connection: sqlite3.Connection) -> None:
        self.root = root
        self.run_id = run_id
        self._connection = connection
        self._connection.row_factory = sqlite3.Row

    @staticmethod
    def _connect(database_path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(database_path)
        connection.execute("pragma foreign_keys = on")
        connection.execute("pragma journal_mode = wal")
        connection.execute("pragma synchronous = full")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            create table if not exists runs (
                run_id text primary key,
                schema_version integer not null,
                status text not null,
                index_contract_sha256 text not null,
                e1a3_allocation_sha256 text not null,
                created_at text not null default current_timestamp,
                updated_at text not null default current_timestamp,
                unique(index_contract_sha256, e1a3_allocation_sha256)
            );
            create table if not exists checkpoints (
                run_id text not null references runs(run_id),
                stage text not null,
                status text not null,
                page_token text,
                committed_records integer not null,
                error_code text,
                updated_at text not null default current_timestamp,
                primary key(run_id, stage)
            );
            create table if not exists drive_files (
                run_id text not null references runs(run_id),
                drive_file_id text not null,
                name text not null,
                mime_type text not null,
                size_bytes integer not null,
                checksum_algorithm text,
                checksum text,
                modified_time text not null,
                parent_ids_json text not null,
                primary key(run_id, drive_file_id)
            );
            create table if not exists local_files (
                run_id text not null references runs(run_id),
                relative_path text not null,
                sha256 text not null,
                provider_checksum_algorithm text,
                provider_checksum text,
                size_bytes integer not null,
                file_type text not null,
                parser_status text not null,
                page_or_sheet_count integer not null,
                primary key(run_id, relative_path)
            );
            create table if not exists index_sources (
                run_id text not null references runs(run_id),
                source_id text not null,
                source_path text not null,
                parser_type text not null,
                topic text not null,
                chunk_count integer not null,
                embedding_model text not null,
                index_contract_sha256 text not null,
                provenance_drive_file_id text,
                content_sha256 text,
                primary key(run_id, source_id)
            );
            create table if not exists index_locators (
                run_id text not null references runs(run_id),
                source_id text not null,
                locator text not null,
                topic text not null,
                source_role text not null,
                substantive_status text not null,
                e1a3_used integer not null default 0,
                e1a4_available integer not null default 0,
                primary key(run_id, source_id, locator),
                foreign key(run_id, source_id) references index_sources(run_id, source_id)
            );
            create table if not exists document_matches (
                run_id text not null references runs(run_id),
                match_key text not null,
                drive_file_id text,
                relative_path text,
                source_id text,
                canonical_sha256 text,
                match_method text not null,
                match_status text not null,
                reason_code text not null,
                primary key(run_id, match_key),
                unique(run_id, drive_file_id)
            );
            create table if not exists review_decisions (
                run_id text not null references runs(run_id),
                decision_id text not null,
                match_key text not null,
                decision text not null,
                reviewer_id text not null,
                reason_code text not null,
                supersedes_decision_id text,
                decided_at text not null,
                primary key(run_id, decision_id)
            );
            """
        )

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        expected_root: Path,
        run_id: str,
        index_contract_sha256: str,
        e1a3_allocation_sha256: str,
    ) -> ReconciliationStore:
        resolved = require_private_reconciliation_root(root, expected_root=expected_root)
        safe_run_id = _string(run_id, "CORPUS_RECONCILIATION_RUN_INVALID")
        index_digest = _digest(index_contract_sha256, "CORPUS_RECONCILIATION_RUN_INVALID")
        allocation_digest = _digest(e1a3_allocation_sha256, "CORPUS_RECONCILIATION_RUN_INVALID")
        resolved.mkdir(parents=True, exist_ok=True)
        connection = cls._connect(resolved / "reconciliation.sqlite")
        cls._create_schema(connection)
        try:
            existing = connection.execute(
                "select run_id from runs where index_contract_sha256 = ? and e1a3_allocation_sha256 = ?",
                (index_digest, allocation_digest),
            ).fetchone()
            if existing is not None and existing[0] != safe_run_id:
                _fail("CORPUS_RECONCILIATION_RUN_CONFLICT")
            connection.execute(
                """
                insert into runs(run_id, schema_version, status, index_contract_sha256, e1a3_allocation_sha256)
                values (?, ?, 'IN_PROGRESS', ?, ?)
                on conflict(run_id) do nothing
                """,
                (safe_run_id, SCHEMA_VERSION, index_digest, allocation_digest),
            )
            connection.commit()
        except CorpusReconciliationError:
            connection.close()
            raise
        except sqlite3.Error as error:
            connection.close()
            raise CorpusReconciliationError("CORPUS_RECONCILIATION_STORE_FAILED") from error
        return cls(root=resolved, run_id=safe_run_id, connection=connection)

    @classmethod
    def open(
        cls, *, root: Path, expected_root: Path, run_id: str
    ) -> ReconciliationStore:
        resolved = require_private_reconciliation_root(root, expected_root=expected_root)
        database_path = resolved / "reconciliation.sqlite"
        if not database_path.is_file():
            _fail("CORPUS_RECONCILIATION_STORE_MISSING")
        connection = cls._connect(database_path)
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "select schema_version from runs where run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            connection.close()
            _fail("CORPUS_RECONCILIATION_RUN_INVALID")
        if type(row["schema_version"]) is not int or row["schema_version"] != SCHEMA_VERSION:
            connection.close()
            _fail("CORPUS_RECONCILIATION_SCHEMA_INVALID")
        return cls(root=resolved, run_id=run_id, connection=connection)

    def set_checkpoint(
        self,
        *,
        stage: str,
        status: str,
        committed_records: int,
        page_token: str | None = None,
        error_code: str | None = None,
    ) -> None:
        safe_stage = _string(stage, "CORPUS_RECONCILIATION_CHECKPOINT_INVALID")
        if status not in CHECKPOINT_STATUSES:
            _fail("CORPUS_RECONCILIATION_CHECKPOINT_INVALID")
        count = _integer(committed_records, "CORPUS_RECONCILIATION_CHECKPOINT_INVALID")
        token = _optional_string(page_token, "CORPUS_RECONCILIATION_CHECKPOINT_INVALID")
        error = _optional_string(error_code, "CORPUS_RECONCILIATION_CHECKPOINT_INVALID")
        with self._connection:
            self._connection.execute(
                """
                insert into checkpoints(run_id, stage, status, page_token, committed_records, error_code)
                values (?, ?, ?, ?, ?, ?)
                on conflict(run_id, stage) do update set
                    status = excluded.status,
                    page_token = excluded.page_token,
                    committed_records = excluded.committed_records,
                    error_code = excluded.error_code,
                    updated_at = current_timestamp
                """,
                (self.run_id, safe_stage, status, token, count, error),
            )

    def checkpoint(self, stage: str) -> CheckpointRecord:
        row = self._connection.execute(
            "select stage, status, committed_records, page_token, error_code from checkpoints where run_id = ? and stage = ?",
            (self.run_id, stage),
        ).fetchone()
        if row is None:
            _fail("CORPUS_RECONCILIATION_CHECKPOINT_MISSING")
        return CheckpointRecord(
            stage=str(row["stage"]),
            status=str(row["status"]),
            committed_records=int(row["committed_records"]),
            page_token=row["page_token"],
            error_code=row["error_code"],
        )

    def pragma(self, name: str) -> object:
        if name not in {"foreign_keys", "journal_mode", "synchronous"}:
            _fail("CORPUS_RECONCILIATION_STORE_FAILED")
        row = self._connection.execute(f"pragma {name}").fetchone()
        return row[0]

    def count(self, table: str) -> int:
        allowed = {
            "runs",
            "checkpoints",
            "drive_files",
            "local_files",
            "index_sources",
            "index_locators",
            "document_matches",
            "review_decisions",
        }
        if table not in allowed:
            _fail("CORPUS_RECONCILIATION_STORE_FAILED")
        row = self._connection.execute(
            f"select count(*) from {table} where run_id = ?", (self.run_id,)
        ).fetchone()
        return int(row[0])

    def index_contract_sha256(self) -> str:
        row = self._connection.execute(
            "select index_contract_sha256 from runs where run_id = ?", (self.run_id,)
        ).fetchone()
        if row is None:
            _fail("CORPUS_RECONCILIATION_RUN_INVALID")
        return str(row[0])

    def contract_digests(self) -> tuple[str, str]:
        row = self._connection.execute(
            """
            select index_contract_sha256, e1a3_allocation_sha256
            from runs where run_id = ?
            """,
            (self.run_id,),
        ).fetchone()
        if row is None:
            _fail("CORPUS_RECONCILIATION_RUN_INVALID")
        return str(row[0]), str(row[1])

    def local_file(self, relative_path: str) -> LocalFileRecord:
        safe_path = _relative_path(relative_path, "CORPUS_RECONCILIATION_LOCAL_RECORD_INVALID")
        row = self._connection.execute(
            """
            select relative_path, sha256, provider_checksum_algorithm, provider_checksum,
                   size_bytes, file_type, parser_status, page_or_sheet_count
            from local_files where run_id = ? and relative_path = ?
            """,
            (self.run_id, safe_path),
        ).fetchone()
        if row is None:
            _fail("CORPUS_RECONCILIATION_LOCAL_RECORD_MISSING")
        return LocalFileRecord.from_mapping(dict(row))

    def match_status(self, drive_file_id: str) -> str:
        row = self._connection.execute(
            """
            select match_status from document_matches
            where run_id = ? and drive_file_id = ?
            """,
            (self.run_id, drive_file_id),
        ).fetchone()
        if row is None:
            _fail("CORPUS_RECONCILIATION_MATCH_MISSING")
        return str(row[0])

    def match_method(self, drive_file_id: str) -> str:
        row = self._connection.execute(
            """
            select match_method from document_matches
            where run_id = ? and drive_file_id = ?
            """,
            (self.run_id, drive_file_id),
        ).fetchone()
        if row is None:
            _fail("CORPUS_RECONCILIATION_MATCH_MISSING")
        return str(row[0])

    def close(self) -> None:
        self._connection.close()


def _table_count(connection: sqlite3.Connection, *, run_id: str, table: str) -> int:
    row = connection.execute(
        f"select count(*) from {table} where run_id = ?", (run_id,)
    ).fetchone()
    return int(row[0])


def _block_stage(
    store: ReconciliationStore, *, stage: str, table: str, error_code: str
) -> None:
    try:
        committed = _table_count(store._connection, run_id=store.run_id, table=table)
        store.set_checkpoint(
            stage=stage,
            status="BLOCKED",
            committed_records=committed,
            error_code=error_code,
        )
    except sqlite3.Error:
        pass


def import_drive_page(
    *,
    store: ReconciliationStore,
    records: Iterable[DriveFileRecord],
    next_page_token: str | None,
) -> StageResult:
    """Commit one metadata-only Drive page using stable file IDs."""
    stage = "drive_inventory"
    safe_records = tuple(DriveFileRecord.from_mapping(record.to_mapping()) for record in records)
    token = _optional_string(
        next_page_token, "CORPUS_RECONCILIATION_DRIVE_PAGE_INVALID"
    )
    status = "IN_PROGRESS" if token is not None else "COMPLETE"
    try:
        with store._connection:
            for record in safe_records:
                parent_ids_json = json.dumps(
                    sorted(record.parent_ids), separators=(",", ":"), ensure_ascii=False
                )
                values = (
                    store.run_id,
                    record.drive_file_id,
                    record.name,
                    record.mime_type,
                    record.size_bytes,
                    record.checksum_algorithm,
                    record.checksum,
                    record.modified_time,
                    parent_ids_json,
                )
                existing = store._connection.execute(
                    """
                    select name, mime_type, size_bytes, checksum_algorithm, checksum,
                           modified_time, parent_ids_json
                    from drive_files where run_id = ? and drive_file_id = ?
                    """,
                    (store.run_id, record.drive_file_id),
                ).fetchone()
                expected = values[2:]
                if existing is not None and tuple(existing) != expected:
                    _fail("CORPUS_RECONCILIATION_DRIVE_CONFLICT")
                store._connection.execute(
                    """
                    insert into drive_files(
                        run_id, drive_file_id, name, mime_type, size_bytes,
                        checksum_algorithm, checksum, modified_time, parent_ids_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(run_id, drive_file_id) do nothing
                    """,
                    values,
                )
            committed = _table_count(
                store._connection, run_id=store.run_id, table="drive_files"
            )
            store._connection.execute(
                """
                insert into checkpoints(
                    run_id, stage, status, page_token, committed_records, error_code
                ) values (?, ?, ?, ?, ?, null)
                on conflict(run_id, stage) do update set
                    status = excluded.status,
                    page_token = excluded.page_token,
                    committed_records = excluded.committed_records,
                    error_code = null,
                    updated_at = current_timestamp
                """,
                (store.run_id, stage, status, token, committed),
            )
    except CorpusReconciliationError as error:
        code = str(error)
        _block_stage(store, stage=stage, table="drive_files", error_code=code)
        raise
    except sqlite3.Error as error:
        code = "CORPUS_RECONCILIATION_STORE_FAILED"
        _block_stage(store, stage=stage, table="drive_files", error_code=code)
        raise CorpusReconciliationError(code) from error
    return StageResult(stage=stage, status=status, committed_records=committed)


def _hash_local_file(path: Path) -> tuple[str, str, int]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size_bytes = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            sha256.update(block)
            md5.update(block)
            size_bytes += len(block)
    return sha256.hexdigest(), md5.hexdigest(), size_bytes


def inventory_local_files(
    *,
    store: ReconciliationStore,
    roots: Iterable[Path],
    parser_metadata: Mapping[str, LocalParserMetadata] | None = None,
) -> StageResult:
    """Hash approved local files without persisting or parsing their contents."""
    stage = "local_inventory"
    safe_roots = tuple(Path(root).resolve() for root in roots)
    if not safe_roots or len({root.name for root in safe_roots}) != len(safe_roots):
        _fail("CORPUS_RECONCILIATION_LOCAL_ROOT_INVALID")
    safe_metadata: dict[str, LocalParserMetadata] = {}
    for raw_path, raw_metadata in (parser_metadata or {}).items():
        relative_path = _relative_path(
            raw_path, "CORPUS_RECONCILIATION_LOCAL_METADATA_INVALID"
        )
        if relative_path in safe_metadata:
            _fail("CORPUS_RECONCILIATION_LOCAL_METADATA_INVALID")
        safe_metadata[relative_path] = LocalParserMetadata.from_mapping(
            raw_metadata.to_mapping()
        )
    seen_paths: set[str] = set()
    committed = store.count("local_files")
    try:
        for root in safe_roots:
            if not root.is_dir():
                _fail("CORPUS_RECONCILIATION_LOCAL_ROOT_INVALID")
            for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
                if path.is_symlink():
                    _fail("CORPUS_RECONCILIATION_LOCAL_ROOT_INVALID")
                relative_path = PurePosixPath(root.name, *path.relative_to(root).parts).as_posix()
                seen_paths.add(relative_path)
                sha256, md5, size_bytes = _hash_local_file(path)
                metadata = safe_metadata.get(
                    relative_path,
                    LocalParserMetadata(parser_status="UNKNOWN", page_or_sheet_count=0),
                )
                record = LocalFileRecord.from_mapping(
                    {
                        "relative_path": relative_path,
                        "sha256": sha256,
                        "provider_checksum_algorithm": "md5",
                        "provider_checksum": md5,
                        "size_bytes": size_bytes,
                        "file_type": path.suffix.removeprefix(".").lower() or "unknown",
                        "parser_status": metadata.parser_status,
                        "page_or_sheet_count": metadata.page_or_sheet_count,
                    }
                )
                values = (store.run_id, *record.to_mapping().values())
                with store._connection:
                    existing = store._connection.execute(
                        """
                        select relative_path, sha256, provider_checksum_algorithm,
                               provider_checksum, size_bytes, file_type, parser_status,
                               page_or_sheet_count
                        from local_files where run_id = ? and relative_path = ?
                        """,
                        (store.run_id, record.relative_path),
                    ).fetchone()
                    if existing is not None and tuple(existing) != values[1:]:
                        _fail("CORPUS_RECONCILIATION_LOCAL_CONFLICT")
                    store._connection.execute(
                        """
                        insert into local_files(
                            run_id, relative_path, sha256, provider_checksum_algorithm,
                            provider_checksum, size_bytes, file_type, parser_status,
                            page_or_sheet_count
                        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        on conflict(run_id, relative_path) do nothing
                        """,
                        values,
                    )
                    committed = _table_count(
                        store._connection, run_id=store.run_id, table="local_files"
                    )
                    store._connection.execute(
                        """
                        insert into checkpoints(
                            run_id, stage, status, committed_records, error_code
                        ) values (?, ?, 'IN_PROGRESS', ?, null)
                        on conflict(run_id, stage) do update set
                            status = 'IN_PROGRESS',
                            committed_records = excluded.committed_records,
                            error_code = null,
                            updated_at = current_timestamp
                        """,
                        (store.run_id, stage, committed),
                    )
        if set(safe_metadata) != seen_paths.intersection(safe_metadata):
            _fail("CORPUS_RECONCILIATION_LOCAL_METADATA_UNMATCHED")
        store.set_checkpoint(
            stage=stage, status="COMPLETE", committed_records=committed
        )
    except CorpusReconciliationError as error:
        _block_stage(store, stage=stage, table="local_files", error_code=str(error))
        raise
    except (OSError, sqlite3.Error) as error:
        code = "CORPUS_RECONCILIATION_LOCAL_INVENTORY_FAILED"
        _block_stage(store, stage=stage, table="local_files", error_code=code)
        raise CorpusReconciliationError(code) from error
    return StageResult(stage=stage, status="COMPLETE", committed_records=committed)


def import_index_inventory(
    *,
    store: ReconciliationStore,
    sources: Iterable[IndexSourceRecord],
    locators: Iterable[IndexLocatorRecord],
    expected_source_count: int,
    expected_chunk_count: int,
) -> StageResult:
    """Import a complete index metadata inventory bound to the run contract."""
    stage = "index_inventory"
    try:
        safe_sources = tuple(
            IndexSourceRecord.from_mapping(source.to_mapping()) for source in sources
        )
        safe_locators = tuple(
            IndexLocatorRecord.from_mapping(locator.to_mapping()) for locator in locators
        )
        source_count = _integer(
            expected_source_count, "CORPUS_RECONCILIATION_INDEX_TOTAL_INVALID", minimum=1
        )
        chunk_count = _integer(
            expected_chunk_count, "CORPUS_RECONCILIATION_INDEX_TOTAL_INVALID", minimum=1
        )
        if len(safe_sources) != source_count or sum(
            source.chunk_count for source in safe_sources
        ) != chunk_count:
            _fail("CORPUS_RECONCILIATION_INDEX_TOTAL_MISMATCH")
        source_ids = [source.source_id for source in safe_sources]
        if len(source_ids) != len(set(source_ids)):
            _fail("CORPUS_RECONCILIATION_INDEX_SOURCE_CONFLICT")
        locator_keys = [(locator.source_id, locator.locator) for locator in safe_locators]
        if len(locator_keys) != len(set(locator_keys)):
            _fail("CORPUS_RECONCILIATION_INDEX_LOCATOR_CONFLICT")
        if any(locator.source_id not in set(source_ids) for locator in safe_locators):
            _fail("CORPUS_RECONCILIATION_INDEX_LOCATOR_INVALID")
        if len({source.embedding_model for source in safe_sources}) != 1:
            _fail("CORPUS_RECONCILIATION_INDEX_MODEL_MIXED")
        bound_digest = store.index_contract_sha256()
        if any(source.index_contract_sha256 != bound_digest for source in safe_sources):
            _fail("CORPUS_RECONCILIATION_INDEX_CONTRACT_MISMATCH")

        with store._connection:
            for source in safe_sources:
                values = (store.run_id, *source.to_mapping().values())
                existing = store._connection.execute(
                    """
                    select source_id, source_path, parser_type, topic, chunk_count,
                           embedding_model, index_contract_sha256,
                           provenance_drive_file_id, content_sha256
                    from index_sources where run_id = ? and source_id = ?
                    """,
                    (store.run_id, source.source_id),
                ).fetchone()
                if existing is not None and tuple(existing) != values[1:]:
                    _fail("CORPUS_RECONCILIATION_INDEX_SOURCE_CONFLICT")
                store._connection.execute(
                    """
                    insert into index_sources(
                        run_id, source_id, source_path, parser_type, topic, chunk_count,
                        embedding_model, index_contract_sha256,
                        provenance_drive_file_id, content_sha256
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(run_id, source_id) do nothing
                    """,
                    values,
                )
            for locator in safe_locators:
                values = (store.run_id, *locator.to_mapping().values())
                existing = store._connection.execute(
                    """
                    select source_id, locator, topic, source_role, substantive_status
                    from index_locators
                    where run_id = ? and source_id = ? and locator = ?
                    """,
                    (store.run_id, locator.source_id, locator.locator),
                ).fetchone()
                if existing is not None and tuple(existing) != values[1:]:
                    _fail("CORPUS_RECONCILIATION_INDEX_LOCATOR_CONFLICT")
                store._connection.execute(
                    """
                    insert into index_locators(
                        run_id, source_id, locator, topic, source_role, substantive_status
                    ) values (?, ?, ?, ?, ?, ?)
                    on conflict(run_id, source_id, locator) do nothing
                    """,
                    values,
                )
            committed = _table_count(
                store._connection, run_id=store.run_id, table="index_sources"
            )
            store._connection.execute(
                """
                insert into checkpoints(
                    run_id, stage, status, committed_records, error_code
                ) values (?, ?, 'COMPLETE', ?, null)
                on conflict(run_id, stage) do update set
                    status = 'COMPLETE',
                    committed_records = excluded.committed_records,
                    error_code = null,
                    updated_at = current_timestamp
                """,
                (store.run_id, stage, committed),
            )
    except CorpusReconciliationError as error:
        _block_stage(store, stage=stage, table="index_sources", error_code=str(error))
        raise
    except sqlite3.Error as error:
        code = "CORPUS_RECONCILIATION_STORE_FAILED"
        _block_stage(store, stage=stage, table="index_sources", error_code=code)
        raise CorpusReconciliationError(code) from error
    return StageResult(stage=stage, status="COMPLETE", committed_records=committed)


def _require_complete_checkpoint(
    store: ReconciliationStore, stage: str, *, error_code: str
) -> None:
    try:
        checkpoint = store.checkpoint(stage)
    except CorpusReconciliationError as error:
        raise CorpusReconciliationError(error_code) from error
    if checkpoint.status != "COMPLETE":
        _fail(error_code)


def _load_drive_records(store: ReconciliationStore) -> tuple[DriveFileRecord, ...]:
    rows = store._connection.execute(
        """
        select drive_file_id, name, mime_type, size_bytes, checksum_algorithm,
               checksum, modified_time, parent_ids_json
        from drive_files where run_id = ? order by drive_file_id
        """,
        (store.run_id,),
    ).fetchall()
    return tuple(
        DriveFileRecord.from_mapping(
            {
                "drive_file_id": row["drive_file_id"],
                "name": row["name"],
                "mime_type": row["mime_type"],
                "size_bytes": row["size_bytes"],
                "checksum_algorithm": row["checksum_algorithm"],
                "checksum": row["checksum"],
                "modified_time": row["modified_time"],
                "parent_ids": json.loads(row["parent_ids_json"]),
            }
        )
        for row in rows
    )


def _load_local_records(store: ReconciliationStore) -> tuple[LocalFileRecord, ...]:
    rows = store._connection.execute(
        """
        select relative_path, sha256, provider_checksum_algorithm, provider_checksum,
               size_bytes, file_type, parser_status, page_or_sheet_count
        from local_files where run_id = ? order by relative_path
        """,
        (store.run_id,),
    ).fetchall()
    return tuple(LocalFileRecord.from_mapping(dict(row)) for row in rows)


def _load_index_sources(store: ReconciliationStore) -> tuple[IndexSourceRecord, ...]:
    rows = store._connection.execute(
        """
        select source_id, source_path, parser_type, topic, chunk_count,
               embedding_model, index_contract_sha256, provenance_drive_file_id,
               content_sha256
        from index_sources where run_id = ? order by source_id
        """,
        (store.run_id,),
    ).fetchall()
    return tuple(IndexSourceRecord.from_mapping(dict(row)) for row in rows)


def _source_local_bindings(
    *, sources: tuple[IndexSourceRecord, ...], locals_: tuple[LocalFileRecord, ...]
) -> dict[str, str]:
    local_by_path = {record.relative_path: record for record in locals_}
    locals_by_sha: dict[str, list[LocalFileRecord]] = {}
    for record in locals_:
        locals_by_sha.setdefault(record.sha256, []).append(record)
    bindings: dict[str, str] = {}
    for source in sources:
        path_match = local_by_path.get(source.source_path)
        if source.content_sha256 is not None:
            if path_match is not None and path_match.sha256 != source.content_sha256:
                _fail("CORPUS_RECONCILIATION_INDEX_HASH_CONFLICT")
            hash_matches = sorted(
                locals_by_sha.get(source.content_sha256, ()),
                key=lambda record: record.relative_path,
            )
            if path_match is not None:
                bindings[source.source_id] = path_match.relative_path
            elif hash_matches:
                bindings[source.source_id] = hash_matches[0].relative_path
        elif path_match is not None:
            bindings[source.source_id] = path_match.relative_path
    return bindings


def _match_row(
    *,
    match_key: str,
    drive_file_id: str | None,
    relative_path: str | None,
    source_id: str | None,
    canonical_sha256: str | None,
    match_method: str,
    match_status: str,
    reason_code: str,
) -> tuple[object, ...]:
    return (
        match_key,
        drive_file_id,
        relative_path,
        source_id,
        canonical_sha256,
        match_method,
        match_status,
        reason_code,
    )


def reconcile_document_matches(*, store: ReconciliationStore) -> MatchSummary:
    """Reconcile exact identities without confidence scores or content access."""
    stage = "document_matching"
    for prerequisite in ("drive_inventory", "local_inventory", "index_inventory"):
        _require_complete_checkpoint(
            store,
            prerequisite,
            error_code="CORPUS_RECONCILIATION_MATCH_INPUT_INCOMPLETE",
        )
    try:
        drives = _load_drive_records(store)
        locals_ = _load_local_records(store)
        sources = _load_index_sources(store)
        local_by_path = {record.relative_path: record for record in locals_}
        source_to_local = _source_local_bindings(sources=sources, locals_=locals_)
        sources_by_local: dict[str, list[str]] = {}
        for source_id, relative_path in source_to_local.items():
            sources_by_local.setdefault(relative_path, []).append(source_id)

        checksum_to_locals: dict[tuple[str, str], list[LocalFileRecord]] = {}
        for local in locals_:
            if (
                local.provider_checksum_algorithm is not None
                and local.provider_checksum is not None
            ):
                checksum_to_locals.setdefault(
                    (local.provider_checksum_algorithm, local.provider_checksum), []
                ).append(local)

        provenance_sources: dict[str, list[IndexSourceRecord]] = {}
        for source in sources:
            if source.provenance_drive_file_id is not None:
                provenance_sources.setdefault(source.provenance_drive_file_id, []).append(source)

        rows: list[tuple[object, ...]] = []
        exact_local_paths: set[str] = set()
        represented_sources: set[str] = set()
        canonical_drive_by_identity: dict[str, str] = {}
        for drive in drives:
            provenance = sorted(
                provenance_sources.get(drive.drive_file_id, ()),
                key=lambda source: source.source_id,
            )
            provenance_paths = {
                source_to_local[source.source_id]
                for source in provenance
                if source.source_id in source_to_local
            }
            provenance_hashes = {
                local_by_path[path].sha256 for path in provenance_paths
            }
            if len(provenance_hashes) > 1:
                _fail("CORPUS_RECONCILIATION_MATCH_CONFLICT")

            checksum_matches: list[LocalFileRecord] = []
            if drive.checksum_algorithm is not None and drive.checksum is not None:
                checksum_matches = sorted(
                    checksum_to_locals.get(
                        (drive.checksum_algorithm, drive.checksum), ()
                    ),
                    key=lambda record: record.relative_path,
                )
                if len({record.sha256 for record in checksum_matches}) > 1:
                    _fail("CORPUS_RECONCILIATION_MATCH_CONFLICT")

            selected_local: LocalFileRecord | None = None
            method: str | None = None
            selected_source_id: str | None = None
            if provenance:
                if provenance_paths:
                    selected_path = sorted(provenance_paths)[0]
                    selected_local = local_by_path[selected_path]
                if checksum_matches and selected_local is not None and all(
                    match.sha256 != selected_local.sha256 for match in checksum_matches
                ):
                    _fail("CORPUS_RECONCILIATION_MATCH_CONFLICT")
                if selected_local is None and checksum_matches:
                    selected_local = checksum_matches[0]
                selected_source_id = provenance[0].source_id
                method = "DRIVE_ID_PROVENANCE"
            elif checksum_matches:
                selected_local = checksum_matches[0]
                method = "PROVIDER_CHECKSUM"

            if provenance and selected_local is None and selected_source_id is not None:
                selected_source = provenance[0]
                identity_key = selected_source.content_sha256 or f"source:{selected_source_id}"
                canonical_drive = canonical_drive_by_identity.get(identity_key)
                status = "EXACT_MATCH" if canonical_drive is None else "DUPLICATE_ALIAS"
                reason = (
                    "EXACT_IDENTITY_CONFIRMED"
                    if canonical_drive is None
                    else "DUPLICATE_CONTENT_ALIAS"
                )
                canonical_drive_by_identity.setdefault(identity_key, drive.drive_file_id)
                represented_sources.add(selected_source_id)
                rows.append(
                    _match_row(
                        match_key=f"drive:{drive.drive_file_id}",
                        drive_file_id=drive.drive_file_id,
                        relative_path=None,
                        source_id=selected_source_id,
                        canonical_sha256=selected_source.content_sha256,
                        match_method="DRIVE_ID_PROVENANCE",
                        match_status=status,
                        reason_code=reason,
                    )
                )
                continue

            if selected_local is not None and method is not None:
                if selected_source_id is None:
                    linked_sources = sorted(
                        sources_by_local.get(selected_local.relative_path, ())
                    )
                    selected_source_id = linked_sources[0] if linked_sources else None
                exact_local_paths.add(selected_local.relative_path)
                if selected_source_id is not None:
                    represented_sources.add(selected_source_id)
                canonical_drive = canonical_drive_by_identity.get(selected_local.sha256)
                status = "EXACT_MATCH" if canonical_drive is None else "DUPLICATE_ALIAS"
                reason = (
                    "EXACT_IDENTITY_CONFIRMED"
                    if canonical_drive is None
                    else "DUPLICATE_CONTENT_ALIAS"
                )
                canonical_drive_by_identity.setdefault(
                    selected_local.sha256, drive.drive_file_id
                )
                rows.append(
                    _match_row(
                        match_key=f"drive:{drive.drive_file_id}",
                        drive_file_id=drive.drive_file_id,
                        relative_path=selected_local.relative_path,
                        source_id=selected_source_id,
                        canonical_sha256=selected_local.sha256,
                        match_method=method,
                        match_status=status,
                        reason_code=reason,
                    )
                )
                continue

            filename_candidates = sorted(
                (
                    local
                    for local in locals_
                    if PurePosixPath(local.relative_path).name.casefold()
                    == drive.name.casefold()
                    and local.size_bytes == drive.size_bytes
                ),
                key=lambda record: record.relative_path,
            )
            if filename_candidates:
                candidate = filename_candidates[0]
                rows.append(
                    _match_row(
                        match_key=f"drive:{drive.drive_file_id}",
                        drive_file_id=drive.drive_file_id,
                        relative_path=candidate.relative_path,
                        source_id=None,
                        canonical_sha256=None,
                        match_method="FILENAME_AND_SIZE",
                        match_status="AMBIGUOUS_REVIEW_REQUIRED",
                        reason_code="FILENAME_SIZE_NOT_IDENTITY",
                    )
                )
            else:
                rows.append(
                    _match_row(
                        match_key=f"drive:{drive.drive_file_id}",
                        drive_file_id=drive.drive_file_id,
                        relative_path=None,
                        source_id=None,
                        canonical_sha256=None,
                        match_method="NONE",
                        match_status="DRIVE_ONLY",
                        reason_code="NO_EXACT_LOCAL_MATCH",
                    )
                )

        for local in locals_:
            if local.relative_path in exact_local_paths:
                continue
            linked_sources = sorted(sources_by_local.get(local.relative_path, ()))
            source_id = linked_sources[0] if linked_sources else None
            if source_id is not None:
                represented_sources.update(linked_sources)
                status = "LOCAL_ONLY"
                reason = "NO_EXACT_DRIVE_MATCH"
            elif local.parser_status == "PARSED":
                status = "PARSED_NOT_INDEXED"
                reason = "PARSED_SOURCE_ABSENT_FROM_INDEX"
            else:
                status = "LOCAL_ONLY"
                reason = "LOCAL_SOURCE_UNMATCHED"
            rows.append(
                _match_row(
                    match_key=f"local:{local.relative_path}",
                    drive_file_id=None,
                    relative_path=local.relative_path,
                    source_id=source_id,
                    canonical_sha256=local.sha256,
                    match_method="INGESTION_PROVENANCE" if source_id else "NONE",
                    match_status=status,
                    reason_code=reason,
                )
            )

        for source in sources:
            if source.source_id in represented_sources:
                continue
            rows.append(
                _match_row(
                    match_key=f"index:{source.source_id}",
                    drive_file_id=None,
                    relative_path=None,
                    source_id=source.source_id,
                    canonical_sha256=source.content_sha256,
                    match_method="NONE",
                    match_status="INDEX_ONLY",
                    reason_code="INDEX_SOURCE_ABSENT_FROM_LOCAL_INVENTORY",
                )
            )

        with store._connection:
            store._connection.execute(
                "delete from document_matches where run_id = ?", (store.run_id,)
            )
            store._connection.executemany(
                """
                insert into document_matches(
                    run_id, match_key, drive_file_id, relative_path, source_id,
                    canonical_sha256, match_method, match_status, reason_code
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ((store.run_id, *row) for row in rows),
            )
            store._connection.execute(
                """
                insert into checkpoints(
                    run_id, stage, status, committed_records, error_code
                ) values (?, ?, 'COMPLETE', ?, null)
                on conflict(run_id, stage) do update set
                    status = 'COMPLETE',
                    committed_records = excluded.committed_records,
                    error_code = null,
                    updated_at = current_timestamp
                """,
                (store.run_id, stage, len(rows)),
            )
    except CorpusReconciliationError as error:
        _block_stage(store, stage=stage, table="document_matches", error_code=str(error))
        raise
    except (json.JSONDecodeError, sqlite3.Error) as error:
        code = "CORPUS_RECONCILIATION_MATCH_FAILED"
        _block_stage(store, stage=stage, table="document_matches", error_code=code)
        raise CorpusReconciliationError(code) from error

    counts = {
        status: sum(1 for row in rows if row[6] == status)
        for status in (
            "EXACT_MATCH",
            "DUPLICATE_ALIAS",
            "DRIVE_ONLY",
            "LOCAL_ONLY",
            "PARSED_NOT_INDEXED",
            "INDEX_ONLY",
            "AMBIGUOUS_REVIEW_REQUIRED",
            "INELIGIBLE",
        )
    }
    return MatchSummary(
        exact_match=counts["EXACT_MATCH"],
        duplicate_alias=counts["DUPLICATE_ALIAS"],
        drive_only=counts["DRIVE_ONLY"],
        local_only=counts["LOCAL_ONLY"],
        parsed_not_indexed=counts["PARSED_NOT_INDEXED"],
        index_only=counts["INDEX_ONLY"],
        ambiguous_review_required=counts["AMBIGUOUS_REVIEW_REQUIRED"],
        ineligible=counts["INELIGIBLE"],
    )


def _locator_inventory_rows(store: ReconciliationStore) -> tuple[sqlite3.Row, ...]:
    return tuple(
        store._connection.execute(
            """
            select locator.source_id, locator.locator, locator.topic,
                   locator.source_role, locator.substantive_status,
                   source.parser_type
            from index_locators as locator
            join index_sources as source
              on source.run_id = locator.run_id and source.source_id = locator.source_id
            where locator.run_id = ?
            order by locator.source_id, locator.locator, locator.topic,
                     locator.source_role
            """,
            (store.run_id,),
        ).fetchall()
    )


def _validated_prior_locator_keys(
    prior_locator_keys: Iterable[str], *, inventory_keys: set[str]
) -> frozenset[str]:
    values = tuple(
        _string(key, "CORPUS_RECONCILIATION_PRIOR_LOCATOR_INVALID")
        for key in prior_locator_keys
    )
    if len(values) != len(set(values)):
        _fail("CORPUS_RECONCILIATION_PRIOR_LOCATOR_INVALID")
    prior = frozenset(values)
    if not prior.issubset(inventory_keys):
        _fail("CORPUS_RECONCILIATION_PRIOR_LOCATOR_MISSING")
    return prior


def _calculate_locator_capacity(
    *, store: ReconciliationStore, prior_locator_keys: Iterable[str]
) -> CapacityReport:
    """Count fresh substantive locators in all eight E1a-4 strata."""
    _require_complete_checkpoint(
        store,
        "index_inventory",
        error_code="CORPUS_RECONCILIATION_CAPACITY_INPUT_INCOMPLETE",
    )
    rows = _locator_inventory_rows(store)
    inventory_keys = {f"{row['source_id']}:{row['locator']}" for row in rows}
    prior = _validated_prior_locator_keys(
        prior_locator_keys, inventory_keys=inventory_keys
    )
    counts = {
        (topic, role): 0
        for topic in ("iron_sulfide", "scale", "corrosion", "paraffin")
        for role in ("foundational", "supporting")
    }
    available_keys: list[tuple[str, str]] = []
    for row in rows:
        locator = IndexLocatorRecord.from_mapping(
            {
                "source_id": row["source_id"],
                "locator": row["locator"],
                "topic": row["topic"],
                "source_role": row["source_role"],
                "substantive_status": row["substantive_status"],
            }
        )
        locator_key = f"{locator.source_id}:{locator.locator}"
        if locator.substantive_status != "SUBSTANTIVE" or locator_key in prior:
            continue
        counts[(locator.topic, locator.source_role)] += 1
        available_keys.append((locator.source_id, locator.locator))

    strata = tuple(
        CapacityStratum(
            topic=topic,
            source_role=role,
            fresh_locator_count=counts[(topic, role)],
            required_locators=12,
            sufficient=counts[(topic, role)] >= 12,
        )
        for topic in ("iron_sulfide", "scale", "corrosion", "paraffin")
        for role in ("foundational", "supporting")
    )
    with store._connection:
        store._connection.execute(
            """
            update index_locators
            set e1a3_used = 0, e1a4_available = 0
            where run_id = ?
            """,
            (store.run_id,),
        )
        for row in rows:
            locator_key = f"{row['source_id']}:{row['locator']}"
            if locator_key in prior:
                store._connection.execute(
                    """
                    update index_locators set e1a3_used = 1
                    where run_id = ? and source_id = ? and locator = ?
                    """,
                    (store.run_id, row["source_id"], row["locator"]),
                )
        store._connection.executemany(
            """
            update index_locators set e1a4_available = 1
            where run_id = ? and source_id = ? and locator = ?
            """,
            (
                (store.run_id, source_id, locator)
                for source_id, locator in available_keys
            ),
        )
        store._connection.execute(
            """
            insert into checkpoints(
                run_id, stage, status, committed_records, error_code
            ) values (?, 'locator_capacity', 'COMPLETE', ?, null)
            on conflict(run_id, stage) do update set
                status = 'COMPLETE',
                committed_records = excluded.committed_records,
                error_code = null,
                updated_at = current_timestamp
            """,
            (store.run_id, len(available_keys)),
        )
    return CapacityReport(
        strata=strata,
        all_sufficient=all(item.sufficient for item in strata),
    )


def calculate_locator_capacity(
    *, store: ReconciliationStore, prior_locator_keys: Iterable[str]
) -> CapacityReport:
    """Fail closed around the private capacity state boundary."""
    try:
        return _calculate_locator_capacity(
            store=store, prior_locator_keys=prior_locator_keys
        )
    except CorpusReconciliationError:
        raise
    except sqlite3.Error as error:
        code = "CORPUS_RECONCILIATION_CAPACITY_FAILED"
        _block_stage(
            store,
            stage="locator_capacity",
            table="index_locators",
            error_code=code,
        )
        raise CorpusReconciliationError(code) from error


def _available_sampling_sources(
    *, rows: tuple[sqlite3.Row, ...], prior: frozenset[str]
) -> tuple[E1A3SourceMetadata, ...]:
    grouped: dict[tuple[str, str, str, str], list[str]] = {}
    for row in rows:
        locator = IndexLocatorRecord.from_mapping(
            {
                "source_id": row["source_id"],
                "locator": row["locator"],
                "topic": row["topic"],
                "source_role": row["source_role"],
                "substantive_status": row["substantive_status"],
            }
        )
        locator_key = f"{locator.source_id}:{locator.locator}"
        if locator.substantive_status != "SUBSTANTIVE" or locator_key in prior:
            continue
        grouped.setdefault(
            (
                locator.source_id,
                locator.topic,
                locator.source_role,
                str(row["parser_type"]),
            ),
            [],
        ).append(locator.locator)
    return tuple(
        E1A3SourceMetadata(
            source_id=source_id,
            topic=topic,  # type: ignore[arg-type]
            source_role=source_role,  # type: ignore[arg-type]
            parser_type=parser_type,
            locators=tuple(sorted(locators)),
            eligibility_status="eligible",
        )
        for (source_id, topic, source_role, parser_type), locators in sorted(
            grouped.items()
        )
    )


def dry_run_e1a4_allocation(
    *, store: ReconciliationStore, prior_locator_keys: Iterable[str]
) -> DryRunResult:
    """Run the existing deterministic 96-slot allocator without publishing files."""
    prior_values = tuple(prior_locator_keys)
    report = calculate_locator_capacity(
        store=store, prior_locator_keys=prior_values
    )
    if not report.all_sufficient:
        error_code = "CORPUS_RECONCILIATION_E1A4_ALLOCATION_UNAVAILABLE"
        store.set_checkpoint(
            stage="e1a4_dry_run",
            status="BLOCKED",
            committed_records=0,
            error_code=error_code,
        )
        return DryRunResult(
            status="BLOCKED", error_code=error_code, allocations=()
        )
    rows = _locator_inventory_rows(store)
    inventory_keys = {f"{row['source_id']}:{row['locator']}" for row in rows}
    prior = _validated_prior_locator_keys(
        prior_values, inventory_keys=inventory_keys
    )
    sources = _available_sampling_sources(rows=rows, prior=prior)
    try:
        allocations = allocate_sampling_slots(
            slots=build_sampling_slots(), sources=sources
        )
    except E1A3SamplingError:
        error_code = "CORPUS_RECONCILIATION_E1A4_ALLOCATION_UNAVAILABLE"
        store.set_checkpoint(
            stage="e1a4_dry_run",
            status="BLOCKED",
            committed_records=0,
            error_code=error_code,
        )
        return DryRunResult(
            status="BLOCKED", error_code=error_code, allocations=()
        )
    if len(allocations) != 96 or len(
        {(item.source_id, item.locator) for item in allocations}
    ) != 96:
        _fail("CORPUS_RECONCILIATION_E1A4_DRY_RUN_INVALID")
    store.set_checkpoint(
        stage="e1a4_dry_run",
        status="COMPLETE",
        committed_records=96,
    )
    return DryRunResult(
        status="COMPLETE", error_code=None, allocations=allocations
    )


def _snapshot_root(root: Path) -> Path:
    resolved = root.resolve()
    if not (
        resolved.name == "v1"
        and resolved.parent.name == "corpus-reconciliation"
        and resolved.parent.parent.name == ".private"
    ):
        _fail("CORPUS_RECONCILIATION_PRIVATE_ROOT_INVALID")
    return resolved


def _canonical_jsonl(records: Iterable[Mapping[str, object]]) -> bytes:
    lines = sorted(
        json.dumps(dict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for record in records
    )
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def _snapshot_payloads(store: ReconciliationStore) -> dict[str, bytes]:
    for stage in (
        "drive_inventory",
        "local_inventory",
        "index_inventory",
        "document_matching",
        "locator_capacity",
    ):
        _require_complete_checkpoint(
            store,
            stage,
            error_code="CORPUS_RECONCILIATION_SNAPSHOT_INPUT_INCOMPLETE",
        )

    drives = tuple(record.to_mapping() for record in _load_drive_records(store))
    locals_ = tuple(record.to_mapping() for record in _load_local_records(store))
    source_rows = tuple(
        {"record_type": "source", **record.to_mapping()}
        for record in _load_index_sources(store)
    )
    locator_rows = tuple(
        {
            "record_type": "locator",
            "source_id": row["source_id"],
            "locator": row["locator"],
            "topic": row["topic"],
            "source_role": row["source_role"],
            "substantive_status": row["substantive_status"],
            "e1a3_used": bool(row["e1a3_used"]),
            "e1a4_available": bool(row["e1a4_available"]),
        }
        for row in store._connection.execute(
            """
            select source_id, locator, topic, source_role, substantive_status,
                   e1a3_used, e1a4_available
            from index_locators where run_id = ?
            """,
            (store.run_id,),
        ).fetchall()
    )
    matches = tuple(
        dict(row)
        for row in store._connection.execute(
            """
            select match_key, drive_file_id, relative_path, source_id,
                   canonical_sha256, match_method, match_status, reason_code
            from document_matches where run_id = ?
            """,
            (store.run_id,),
        ).fetchall()
    )
    decisions = tuple(
        dict(row)
        for row in store._connection.execute(
            """
            select decision_id, match_key, decision, reviewer_id, reason_code,
                   supersedes_decision_id, decided_at
            from review_decisions where run_id = ?
            """,
            (store.run_id,),
        ).fetchall()
    )
    capacity_counts = {
        (topic, role): 0
        for topic in ("iron_sulfide", "scale", "corrosion", "paraffin")
        for role in ("foundational", "supporting")
    }
    for row in locator_rows:
        if row["e1a4_available"] is True:
            capacity_counts[(str(row["topic"]), str(row["source_role"]))] += 1
    capacity = tuple(
        {
            "topic": topic,
            "source_role": role,
            "fresh_locator_count": capacity_counts[(topic, role)],
            "required_locators": 12,
            "sufficient": capacity_counts[(topic, role)] >= 12,
        }
        for topic in ("iron_sulfide", "scale", "corrosion", "paraffin")
        for role in ("foundational", "supporting")
    )
    return {
        "drive-inventory.jsonl": _canonical_jsonl(drives),
        "local-inventory.jsonl": _canonical_jsonl(locals_),
        "index-inventory.jsonl": _canonical_jsonl((*source_rows, *locator_rows)),
        "document-matches.jsonl": _canonical_jsonl(matches),
        "review-decisions.jsonl": _canonical_jsonl(decisions),
        "locator-capacity.jsonl": _canonical_jsonl(capacity),
    }


_DRIVE_SNAPSHOT_KEYS = frozenset(
    {
        "drive_file_id",
        "name",
        "mime_type",
        "size_bytes",
        "checksum_algorithm",
        "checksum",
        "modified_time",
        "parent_ids",
    }
)
_LOCAL_SNAPSHOT_KEYS = frozenset(
    {
        "relative_path",
        "sha256",
        "provider_checksum_algorithm",
        "provider_checksum",
        "size_bytes",
        "file_type",
        "parser_status",
        "page_or_sheet_count",
    }
)
_INDEX_SOURCE_SNAPSHOT_KEYS = frozenset(
    {
        "record_type",
        "source_id",
        "source_path",
        "parser_type",
        "topic",
        "chunk_count",
        "embedding_model",
        "index_contract_sha256",
        "provenance_drive_file_id",
        "content_sha256",
    }
)
_INDEX_LOCATOR_SNAPSHOT_KEYS = frozenset(
    {
        "record_type",
        "source_id",
        "locator",
        "topic",
        "source_role",
        "substantive_status",
        "e1a3_used",
        "e1a4_available",
    }
)
_MATCH_SNAPSHOT_KEYS = frozenset(
    {
        "match_key",
        "drive_file_id",
        "relative_path",
        "source_id",
        "canonical_sha256",
        "match_method",
        "match_status",
        "reason_code",
    }
)
_DECISION_SNAPSHOT_KEYS = frozenset(
    {
        "decision_id",
        "match_key",
        "decision",
        "reviewer_id",
        "reason_code",
        "supersedes_decision_id",
        "decided_at",
    }
)
_CAPACITY_SNAPSHOT_KEYS = frozenset(
    {
        "topic",
        "source_role",
        "fresh_locator_count",
        "required_locators",
        "sufficient",
    }
)


def _validate_snapshot_record(name: str, record: object) -> None:
    code = "CORPUS_RECONCILIATION_SNAPSHOT_SCHEMA_INVALID"
    if not isinstance(record, Mapping):
        _fail(code)
    if name == "drive-inventory.jsonl":
        _exact_keys(record, _DRIVE_SNAPSHOT_KEYS, code)
        DriveFileRecord.from_mapping(record)
        return
    if name == "local-inventory.jsonl":
        _exact_keys(record, _LOCAL_SNAPSHOT_KEYS, code)
        LocalFileRecord.from_mapping(record)
        return
    if name == "index-inventory.jsonl":
        record_type = record.get("record_type")
        if record_type == "source":
            _exact_keys(record, _INDEX_SOURCE_SNAPSHOT_KEYS, code)
            IndexSourceRecord.from_mapping(
                {key: value for key, value in record.items() if key != "record_type"}
            )
            return
        if record_type == "locator":
            _exact_keys(record, _INDEX_LOCATOR_SNAPSHOT_KEYS, code)
            if type(record["e1a3_used"]) is not bool or type(
                record["e1a4_available"]
            ) is not bool:
                _fail(code)
            IndexLocatorRecord.from_mapping(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"record_type", "e1a3_used", "e1a4_available"}
                }
            )
            return
        _fail(code)
    expected = {
        "document-matches.jsonl": _MATCH_SNAPSHOT_KEYS,
        "review-decisions.jsonl": _DECISION_SNAPSHOT_KEYS,
        "locator-capacity.jsonl": _CAPACITY_SNAPSHOT_KEYS,
    }.get(name)
    if expected is None:
        _fail(code)
    _exact_keys(record, expected, code)
    if name == "locator-capacity.jsonl":
        topic = _string(record["topic"], code)
        role = _string(record["source_role"], code)
        fresh = _integer(record["fresh_locator_count"], code)
        required = _integer(record["required_locators"], code, minimum=1)
        if (
            topic not in TOPICS
            or role not in SOURCE_ROLES
            or type(record["sufficient"]) is not bool
            or record["sufficient"] != (fresh >= required)
        ):
            _fail(code)
    else:
        for key, value in record.items():
            if value is not None and not isinstance(value, str):
                _fail(code)


def _verified_snapshot_records(name: str, content: bytes) -> int:
    code = "CORPUS_RECONCILIATION_SNAPSHOT_SCHEMA_INVALID"
    if content and not content.endswith(b"\n"):
        _fail(code)
    try:
        text = content.decode("utf-8")
        records = tuple(json.loads(line) for line in text.splitlines())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusReconciliationError(code) from error
    for record in records:
        _validate_snapshot_record(name, record)
    if _canonical_jsonl(records) != content:
        _fail(code)
    return len(records)


def verify_reconciliation_snapshots(*, root: Path) -> SnapshotSet:
    """Verify a complete canonical snapshot and manifest set without rewriting it."""
    resolved = _snapshot_root(root)
    snapshots = resolved / "snapshots"
    expected_paths = tuple(snapshots / name for name in SNAPSHOT_NAMES)
    expected_manifests = tuple(
        path.with_name(f"{path.name}.sha256") for path in expected_paths
    )
    all_paths = (*expected_paths, *expected_manifests)
    present = tuple(path.is_file() for path in all_paths)
    if not any(present):
        _fail("CORPUS_RECONCILIATION_SNAPSHOT_MISSING")
    if not all(present):
        _fail("CORPUS_RECONCILIATION_SNAPSHOT_PARTIAL")
    artifacts: list[SnapshotArtifact] = []
    for path, manifest in zip(expected_paths, expected_manifests, strict=True):
        try:
            content = path.read_bytes()
            manifest_text = manifest.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise CorpusReconciliationError(
                "CORPUS_RECONCILIATION_SNAPSHOT_VERIFY_FAILED"
            ) from error
        if (
            len(manifest_text) != 65
            or not manifest_text.endswith("\n")
            or any(character not in "0123456789abcdef" for character in manifest_text[:64])
        ):
            _fail("CORPUS_RECONCILIATION_SNAPSHOT_MANIFEST_INVALID")
        digest = hashlib.sha256(content).hexdigest()
        if manifest_text[:64] != digest:
            _fail("CORPUS_RECONCILIATION_SNAPSHOT_DIGEST_MISMATCH")
        record_count = _verified_snapshot_records(path.name, content)
        artifacts.append(
            SnapshotArtifact(
                name=path.name,
                path=path,
                manifest_path=manifest,
                sha256=digest,
                record_count=record_count,
            )
        )
    return SnapshotSet(artifacts=tuple(artifacts))


def _write_fsynced_temporary(*, destination: Path, content: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        return Path(temporary.name)


def seal_reconciliation_snapshots(
    *, store: ReconciliationStore, root: Path
) -> SnapshotSet:
    """Atomically publish the complete canonical reconciliation snapshot set."""
    resolved = _snapshot_root(root)
    if resolved != store.root.resolve():
        _fail("CORPUS_RECONCILIATION_PRIVATE_ROOT_INVALID")
    snapshots = resolved / "snapshots"
    destinations = tuple(
        destination
        for name in SNAPSHOT_NAMES
        for destination in (
            snapshots / name,
            snapshots / f"{name}.sha256",
        )
    )
    presence = tuple(path.exists() for path in destinations)
    if any(presence):
        if not all(presence):
            _fail("CORPUS_RECONCILIATION_SNAPSHOT_PARTIAL")
        return verify_reconciliation_snapshots(root=resolved)

    try:
        payloads = _snapshot_payloads(store)
    except CorpusReconciliationError:
        raise
    except (json.JSONDecodeError, sqlite3.Error) as error:
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_SNAPSHOT_BUILD_FAILED"
        ) from error
    prepared: list[tuple[Path, bytes]] = []
    for name in SNAPSHOT_NAMES:
        content = payloads[name]
        _verified_snapshot_records(name, content)
        snapshot_path = snapshots / name
        prepared.append((snapshot_path, content))
        prepared.append(
            (
                snapshots / f"{name}.sha256",
                (hashlib.sha256(content).hexdigest() + "\n").encode("ascii"),
            )
        )

    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        if any(destination.exists() for destination, _ in prepared):
            _fail("CORPUS_RECONCILIATION_SNAPSHOT_PARTIAL")
        for destination, content in prepared:
            staged.append(
                (
                    _write_fsynced_temporary(
                        destination=destination, content=content
                    ),
                    destination,
                )
            )
        for temporary, destination in staged:
            os.replace(temporary, destination)
            published.append(destination)
        return verify_reconciliation_snapshots(root=resolved)
    except (CorpusReconciliationError, OSError) as error:
        for temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        for destination in published:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_SNAPSHOT_WRITE_FAILED"
        ) from error
