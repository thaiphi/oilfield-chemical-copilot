"""Resumable metadata-only reconciliation for the private evaluation corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import sqlite3
from typing import Mapping


SCHEMA_VERSION = 1
RUN_STATUSES = frozenset({"IN_PROGRESS", "BLOCKED", "COMPLETE", "INVALID"})
CHECKPOINT_STATUSES = frozenset({"NOT_STARTED", "IN_PROGRESS", "COMPLETE", "BLOCKED"})
TOPICS = frozenset({"iron_sulfide", "scale", "corrosion", "paraffin"})
SOURCE_ROLES = frozenset({"foundational", "supporting"})


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
        algorithm = _optional_string(mapping["checksum_algorithm"], code)
        checksum = _optional_string(mapping["checksum"], code)
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
        algorithm = _optional_string(mapping["provider_checksum_algorithm"], code)
        checksum = _optional_string(mapping["provider_checksum"], code)
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
class IndexSourceRecord:
    source_id: str
    source_path: str
    parser_type: str
    topic: str
    chunk_count: int
    embedding_model: str
    index_contract_sha256: str

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
                }
            ),
            code,
        )
        topic = _string(mapping["topic"], code)
        if topic not in TOPICS:
            _fail(code)
        return cls(
            source_id=_string(mapping["source_id"], code),
            source_path=_string(mapping["source_path"], code),
            parser_type=_string(mapping["parser_type"], code),
            topic=topic,
            chunk_count=_integer(mapping["chunk_count"], code, minimum=1),
            embedding_model=_string(mapping["embedding_model"], code),
            index_contract_sha256=_digest(mapping["index_contract_sha256"], code),
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
        if topic not in TOPICS or source_role not in SOURCE_ROLES:
            _fail(code)
        return cls(
            source_id=_string(mapping["source_id"], code),
            locator=_string(mapping["locator"], code),
            topic=topic,
            source_role=source_role,
            substantive_status=_string(mapping["substantive_status"], code),
        )


@dataclass(frozen=True)
class CheckpointRecord:
    stage: str
    status: str
    committed_records: int
    page_token: str | None
    error_code: str | None


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
                primary key(run_id, source_id, locator),
                foreign key(run_id, source_id) references index_sources(run_id, source_id)
            );
            create table if not exists document_matches (
                run_id text not null references runs(run_id),
                drive_file_id text,
                relative_path text,
                source_id text,
                match_method text not null,
                match_status text not null,
                reason_code text not null,
                primary key(run_id, drive_file_id, relative_path, source_id)
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

    def close(self) -> None:
        self._connection.close()
