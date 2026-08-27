"""Private, no-write foundational-role audit for Iron Sulfide supplements."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from tempfile import mkdtemp
from typing import Mapping

from pypdf import PdfReader

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
    CorpusReconciliationError,
    ReconciliationStore,
    TOPICS,
    verify_reconciliation_snapshots,
)
from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
    E1A3SamplingError,
    E1A3SourceMetadata,
    allocate_sampling_slots,
    build_sampling_slots,
)
from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
    FoundationalAuditStore,
    FoundationalLocatorAuditError,
    HypotheticalCapacityReport,
    HypotheticalCapacityStratum,
    LocatorAuditDecision,
    verify_correction_proposal,
)


SUPPLEMENT_ARTIFACT_VERSION = "v2"
SUPPLEMENT_CORRECTION_NAME = "iron-sulfide-supplement-corrections.v2.jsonl"
SUPPLEMENT_BINDING_NAME = "audit-binding.v2.json"


class IronSulfideSupplementAuditError(RuntimeError):
    """Raised when supplement audit state violates its frozen contract."""


def _fail(code: str) -> None:
    raise IronSulfideSupplementAuditError(code)


def _string(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _digest(value: object, code: str) -> str:
    text = _string(value, code)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(code)
    return text


def _positive_integer(value: object, code: str) -> int:
    if type(value) is not int or value < 1:
        _fail(code)
    return value


def _basename(value: object) -> str:
    return PurePosixPath(str(value).replace("\\", "/")).name


def _canonical_digest(records: tuple[Mapping[str, object], ...]) -> str:
    payload = json.dumps(
        [dict(record) for record in records],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_root_digest(source_root: Path) -> str:
    resolved = source_root.resolve()
    if not resolved.is_dir():
        _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_ROOT_INVALID")
    normalized = os.path.normcase(str(resolved))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SupplementAuditStatus:
    status: str
    source_count: int
    candidate_count: int
    reviewed_count: int
    promotion_count: int
    remaining_count: int
    needs_second_review_count: int


SUPPLEMENT_DECISION_CONTRACTS = frozenset(
    {
        ("PROMOTE_FOUNDATIONAL", "GENERALIZABLE_FOUNDATIONAL_EVIDENCE"),
        ("KEEP_SUPPORTING", "CASE_OR_APPLICATION_SPECIFIC"),
        ("KEEP_SUPPORTING", "PRODUCT_OR_VENDOR_SPECIFIC"),
        ("KEEP_SUPPORTING", "PROCEDURAL_OR_JOB_SPECIFIC"),
        ("KEEP_SUPPORTING", "DATA_OR_EXAMPLE_WITHOUT_GENERAL_PRINCIPLE"),
        ("KEEP_SUPPORTING", "TITLE_INDEX_OR_REFERENCE_ONLY"),
        ("KEEP_SUPPORTING", "INSUFFICIENT_STANDALONE_CONTEXT"),
        ("KEEP_SUPPORTING", "DUPLICATE_PAGE_CONTENT"),
        ("KEEP_SUPPORTING", "WRONG_TOPIC"),
        ("NEEDS_SECOND_REVIEW", "AMBIGUOUS_FOUNDATIONAL_ROLE"),
    }
)


@dataclass(frozen=True)
class SupplementLocatorDecision:
    decision_id: str
    source_id: str
    locator: str
    decision: str
    reason_code: str
    page_text_sha256: str
    reviewer_id: str
    supersedes_decision_id: str | None
    decided_at: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> SupplementLocatorDecision:
        code = "IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID"
        expected = {
            "decision_id",
            "source_id",
            "locator",
            "decision",
            "reason_code",
            "page_text_sha256",
            "reviewer_id",
            "supersedes_decision_id",
            "decided_at",
        }
        if not isinstance(mapping, Mapping) or set(mapping) != expected:
            _fail(code)
        decision = _string(mapping["decision"], code)
        reason = _string(mapping["reason_code"], code)
        if (decision, reason) not in SUPPLEMENT_DECISION_CONTRACTS:
            _fail(code)
        supersedes = mapping["supersedes_decision_id"]
        if supersedes is not None:
            supersedes = _string(supersedes, code)
        return cls(
            decision_id=_string(mapping["decision_id"], code),
            source_id=_string(mapping["source_id"], code),
            locator=_string(mapping["locator"], code),
            decision=decision,
            reason_code=reason,
            page_text_sha256=_digest(mapping["page_text_sha256"], code),
            reviewer_id=_string(mapping["reviewer_id"], code),
            supersedes_decision_id=supersedes,
            decided_at=_string(mapping["decided_at"], code),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "source_id": self.source_id,
            "locator": self.locator,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "page_text_sha256": self.page_text_sha256,
            "reviewer_id": self.reviewer_id,
            "supersedes_decision_id": self.supersedes_decision_id,
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class SupplementCandidate:
    source_id: str
    locator: str
    page_number: int
    review_order: int
    page_text_sha256: str


@dataclass(frozen=True)
class SupplementReviewPacket:
    source_id: str
    locator: str
    page_number: int
    page_text: str
    page_text_sha256: str


@dataclass(frozen=True)
class SupplementAuditArtifact:
    name: str
    path: Path
    manifest_path: Path
    sha256: str
    record_count: int


@dataclass(frozen=True)
class SupplementAuditSeal:
    artifacts: tuple[SupplementAuditArtifact, ...]
    binding_sha256: str


class IronSulfideSupplementAuditStore:
    """Handle to one supplement audit stored in reconciliation SQLite."""

    def __init__(
        self,
        *,
        database_path: Path,
        run_id: str,
        audit_id: str,
        connection: sqlite3.Connection,
    ) -> None:
        self.database_path = database_path
        self.run_id = run_id
        self.audit_id = audit_id
        self._connection = connection
        self._connection.row_factory = sqlite3.Row

    @classmethod
    def open(
        cls, *, database_path: Path, run_id: str, audit_id: str
    ) -> IronSulfideSupplementAuditStore:
        safe_path = database_path.resolve()
        if not safe_path.is_file():
            _fail("IRON_SULFIDE_SUPPLEMENT_STORE_MISSING")
        connection = sqlite3.connect(safe_path)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma foreign_keys = on")
        connection.execute("pragma journal_mode = wal")
        connection.execute("pragma synchronous = full")
        row = connection.execute(
            """
            select audit_id from iron_sulfide_supplement_audit_runs
            where run_id = ? and audit_id = ?
            """,
            (run_id, audit_id),
        ).fetchone()
        if row is None:
            connection.close()
            _fail("IRON_SULFIDE_SUPPLEMENT_RUN_MISSING")
        return cls(
            database_path=safe_path,
            run_id=_string(run_id, "IRON_SULFIDE_SUPPLEMENT_RUN_INVALID"),
            audit_id=_string(audit_id, "IRON_SULFIDE_SUPPLEMENT_RUN_INVALID"),
            connection=connection,
        )

    def close(self) -> None:
        self._connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        create table if not exists iron_sulfide_supplement_audit_runs (
            run_id text not null,
            audit_id text not null,
            snapshot_binding_sha256 text not null,
            source_set_sha256 text not null,
            candidate_set_sha256 text not null,
            source_count integer not null,
            candidate_count integer not null,
            promotion_target integer not null,
            status text not null,
            source_root_sha256 text not null,
            primary key (run_id, audit_id)
        );
        create table if not exists iron_sulfide_supplement_audit_sources (
            run_id text not null,
            audit_id text not null,
            source_id text not null,
            drive_file_id text not null,
            relative_path text not null,
            file_sha256 text not null,
            page_count integer not null,
            primary key (run_id, audit_id, source_id)
        );
        create table if not exists iron_sulfide_supplement_audit_candidates (
            run_id text not null,
            audit_id text not null,
            source_id text not null,
            locator text not null,
            page_number integer not null,
            review_order integer not null,
            page_text_sha256 text,
            primary key (run_id, audit_id, source_id, locator),
            unique (run_id, audit_id, review_order)
        );
        create table if not exists iron_sulfide_supplement_audit_decisions (
            run_id text not null,
            audit_id text not null,
            decision_id text not null,
            source_id text not null,
            locator text not null,
            decision text not null,
            reason_code text not null,
            page_text_sha256 text not null,
            reviewer_id text not null,
            supersedes_decision_id text,
            decided_at text not null,
            primary key (run_id, audit_id, decision_id)
        );
        """
    )
    columns = {
        str(row[1])
        for row in connection.execute(
            "pragma table_info(iron_sulfide_supplement_audit_runs)"
        ).fetchall()
    }
    if "source_root_sha256" not in columns:
        connection.execute(
            "alter table iron_sulfide_supplement_audit_runs "
            "add column source_root_sha256 text"
        )


def _current_accept(connection: sqlite3.Connection, *, run_id: str, match_key: str) -> bool:
    rows = connection.execute(
        """
        select decision_id, decision, supersedes_decision_id
        from review_decisions where run_id = ? and match_key = ?
        """,
        (run_id, match_key),
    ).fetchall()
    superseded = {
        str(row["supersedes_decision_id"])
        for row in rows
        if row["supersedes_decision_id"] is not None
    }
    current = tuple(row for row in rows if str(row["decision_id"]) not in superseded)
    return len(current) == 1 and str(current[0]["decision"]) == "ACCEPT"


def _candidate_inventory(
    *, store: ReconciliationStore, source_root: Path
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    resolved_root = source_root.resolve()
    if not resolved_root.is_dir():
        _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_ROOT_INVALID")
    root_label = resolved_root.name.casefold()
    mounted_by_name: dict[str, list[Path]] = {}
    for path in resolved_root.rglob("*"):
        if path.is_file():
            mounted_by_name.setdefault(path.name.casefold(), []).append(path.resolve())

    source_rows = store._connection.execute(
        """
        select source_id, source_path, parser_type,
               provenance_drive_file_id, content_sha256
        from index_sources where run_id = ?
        """,
        (store.run_id,),
    ).fetchall()
    locator_rows = store._connection.execute(
        """
        select source_id, locator, topic, source_role, substantive_status,
               e1a3_used, e1a4_available
        from index_locators where run_id = ?
        """,
        (store.run_id,),
    ).fetchall()
    drives = store._connection.execute(
        """
        select drive_file_id, name, mime_type, size_bytes
        from drive_files where run_id = ?
        """,
        (store.run_id,),
    ).fetchall()
    locals_ = store._connection.execute(
        """
        select relative_path, sha256, size_bytes, file_type, page_or_sheet_count
        from local_files where run_id = ?
        """,
        (store.run_id,),
    ).fetchall()
    drive_by_name: dict[str, list[sqlite3.Row]] = {}
    local_by_name: dict[str, list[sqlite3.Row]] = {}
    for row in drives:
        drive_by_name.setdefault(str(row["name"]).casefold(), []).append(row)
    for row in locals_:
        local_by_name.setdefault(_basename(row["relative_path"]).casefold(), []).append(row)

    locators_by_source: dict[str, list[sqlite3.Row]] = {}
    for row in locator_rows:
        if (
            str(row["topic"]) == "iron_sulfide"
            and str(row["source_role"]) == "supporting"
            and str(row["substantive_status"]) == "SUBSTANTIVE"
            and int(row["e1a3_used"]) == 0
            and int(row["e1a4_available"]) == 1
        ):
            locators_by_source.setdefault(str(row["source_id"]), []).append(row)

    admitted_sources: list[Mapping[str, object]] = []
    admitted_candidates: list[Mapping[str, object]] = []
    for source in source_rows:
        source_id = str(source["source_id"])
        candidate_rows = locators_by_source.get(source_id, ())
        source_parts = tuple(
            part.casefold()
            for part in PurePosixPath(
                str(source["source_path"]).replace("\\", "/")
            ).parts
        )
        if not candidate_rows or root_label not in source_parts:
            continue
        name = _basename(source["source_path"])
        name_key = name.casefold()
        mounted = mounted_by_name.get(name_key, ())
        drive = drive_by_name.get(name_key, ())
        local = local_by_name.get(name_key, ())
        if (
            str(source["parser_type"]).lower() != "pdf"
            or len(mounted) != 1
            or len(drive) != 1
            or len(local) != 1
            or str(drive[0]["mime_type"]) != "application/pdf"
            or str(local[0]["file_type"]).lower() != "pdf"
        ):
            continue
        match = store._connection.execute(
            """
            select match_key, match_method, match_status
            from document_matches
            where run_id = ? and drive_file_id = ? and relative_path = ?
            """,
            (
                store.run_id,
                drive[0]["drive_file_id"],
                local[0]["relative_path"],
            ),
        ).fetchall()
        if (
            len(match) != 1
            or str(match[0]["match_method"]) != "FILENAME_AND_SIZE"
            or str(match[0]["match_status"]) != "AMBIGUOUS_REVIEW_REQUIRED"
            or not _current_accept(
                store._connection,
                run_id=store.run_id,
                match_key=str(match[0]["match_key"]),
            )
        ):
            continue
        try:
            content = mounted[0].read_bytes()
            reader = PdfReader(BytesIO(content))
        except Exception as error:
            raise IronSulfideSupplementAuditError(
                "IRON_SULFIDE_SUPPLEMENT_SOURCE_PROVENANCE_INVALID"
            ) from error
        file_digest = hashlib.sha256(content).hexdigest()
        if (
            file_digest != str(local[0]["sha256"])
            or len(content) != int(local[0]["size_bytes"])
            or len(content) != int(drive[0]["size_bytes"])
            or int(local[0]["page_or_sheet_count"]) not in (0, len(reader.pages))
            or source["provenance_drive_file_id"]
            not in (None, drive[0]["drive_file_id"])
            or source["content_sha256"] not in (None, file_digest)
        ):
            continue
        parsed_candidates: list[tuple[int, str]] = []
        for row in candidate_rows:
            locator = str(row["locator"])
            if not locator.startswith("page:") or not locator[5:].isdigit():
                parsed_candidates = []
                break
            page_number = int(locator[5:])
            if page_number < 1 or page_number > len(reader.pages):
                parsed_candidates = []
                break
            parsed_candidates.append((page_number, locator))
        if not parsed_candidates:
            continue
        admitted_sources.append(
            {
                "source_id": source_id,
                "drive_file_id": str(drive[0]["drive_file_id"]),
                "relative_path": str(local[0]["relative_path"]),
                "file_sha256": file_digest,
                "page_count": len(reader.pages),
                "source_path": str(source["source_path"]),
            }
        )
        admitted_candidates.extend(
            {
                "source_id": source_id,
                "locator": locator,
                "page_number": page_number,
                "source_path": str(source["source_path"]),
            }
            for page_number, locator in parsed_candidates
        )

    if not admitted_sources or not admitted_candidates:
        _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_PROVENANCE_INVALID")
    ordered_sources = tuple(
        sorted(admitted_sources, key=lambda row: (str(row["source_path"]).casefold(), str(row["source_id"])))
    )
    ordered_candidates = tuple(
        {
            **candidate,
            "review_order": index,
        }
        for index, candidate in enumerate(
            sorted(
                admitted_candidates,
                key=lambda row: (
                    str(row["source_path"]).casefold(),
                    int(row["page_number"]),
                    str(row["source_id"]),
                    str(row["locator"]),
                ),
            ),
            start=1,
        )
    )
    return ordered_sources, ordered_candidates


def initialize_supplement_audit(
    *,
    store: ReconciliationStore,
    audit_id: str,
    snapshot_binding_sha256: str,
    source_root: Path,
    promotion_target: int,
) -> IronSulfideSupplementAuditStore:
    """Create or resume a supplement audit from authenticated reconciliation state."""
    safe_audit_id = _string(audit_id, "IRON_SULFIDE_SUPPLEMENT_RUN_INVALID")
    binding_digest = _digest(
        snapshot_binding_sha256, "IRON_SULFIDE_SUPPLEMENT_RUN_INVALID"
    )
    target = _positive_integer(
        promotion_target, "IRON_SULFIDE_SUPPLEMENT_RUN_INVALID"
    )
    try:
        verify_reconciliation_snapshots(
            root=store.root,
            store=store,
            expected_binding_sha256=binding_digest,
        )
    except CorpusReconciliationError as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_RECONCILIATION_UNTRUSTED"
        ) from error
    sources, candidates = _candidate_inventory(store=store, source_root=source_root)
    source_payload = tuple(
        {
            key: row[key]
            for key in (
                "source_id",
                "drive_file_id",
                "relative_path",
                "file_sha256",
                "page_count",
            )
        }
        for row in sources
    )
    candidate_payload = tuple(
        {
            key: row[key]
            for key in ("source_id", "locator", "page_number", "review_order")
        }
        for row in candidates
    )
    source_digest = _canonical_digest(source_payload)
    candidate_digest = _canonical_digest(candidate_payload)
    source_root_digest = _source_root_digest(source_root)
    _create_schema(store._connection)
    existing = store._connection.execute(
        """
        select snapshot_binding_sha256, source_set_sha256, candidate_set_sha256,
               source_count, candidate_count, promotion_target, source_root_sha256
        from iron_sulfide_supplement_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (store.run_id, safe_audit_id),
    ).fetchone()
    expected_without_root = (
        binding_digest,
        source_digest,
        candidate_digest,
        len(source_payload),
        len(candidate_payload),
        target,
    )
    if existing is not None:
        if tuple(existing)[:6] != expected_without_root:
            _fail("IRON_SULFIDE_SUPPLEMENT_BINDING_CONFLICT")
        stored_root = existing["source_root_sha256"]
        if stored_root is not None and str(stored_root) != source_root_digest:
            _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_ROOT_MISMATCH")
    with store._connection:
        if existing is not None and existing["source_root_sha256"] is None:
            store._connection.execute(
                """
                update iron_sulfide_supplement_audit_runs
                set source_root_sha256 = ? where run_id = ? and audit_id = ?
                """,
                (source_root_digest, store.run_id, safe_audit_id),
            )
        store._connection.execute(
            """
            insert into iron_sulfide_supplement_audit_runs(
                run_id, audit_id, snapshot_binding_sha256, source_set_sha256,
                candidate_set_sha256, source_count, candidate_count,
                promotion_target, status, source_root_sha256
            ) values (?, ?, ?, ?, ?, ?, ?, ?, 'IN_PROGRESS', ?)
            on conflict(run_id, audit_id) do nothing
            """,
            (
                store.run_id,
                safe_audit_id,
                *expected_without_root,
                source_root_digest,
            ),
        )
        if existing is None:
            store._connection.executemany(
                """
                insert into iron_sulfide_supplement_audit_sources(
                    run_id, audit_id, source_id, drive_file_id, relative_path,
                    file_sha256, page_count
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        store.run_id,
                        safe_audit_id,
                        row["source_id"],
                        row["drive_file_id"],
                        row["relative_path"],
                        row["file_sha256"],
                        row["page_count"],
                    )
                    for row in source_payload
                ),
            )
            store._connection.executemany(
                """
                insert into iron_sulfide_supplement_audit_candidates(
                    run_id, audit_id, source_id, locator, page_number,
                    review_order, page_text_sha256
                ) values (?, ?, ?, ?, ?, ?, null)
                """,
                (
                    (
                        store.run_id,
                        safe_audit_id,
                        row["source_id"],
                        row["locator"],
                        row["page_number"],
                        row["review_order"],
                    )
                    for row in candidate_payload
                ),
            )
    return IronSulfideSupplementAuditStore(
        database_path=store.root / "reconciliation.sqlite",
        run_id=store.run_id,
        audit_id=safe_audit_id,
        connection=store._connection,
    )


def supplement_audit_status(
    audit: IronSulfideSupplementAuditStore,
) -> SupplementAuditStatus:
    row = audit._connection.execute(
        """
        select status, source_count, candidate_count
        from iron_sulfide_supplement_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if row is None:
        _fail("IRON_SULFIDE_SUPPLEMENT_RUN_MISSING")
    current = _current_supplement_decisions(audit)
    promotion_count = sum(
        decision.decision == "PROMOTE_FOUNDATIONAL" for decision in current.values()
    )
    needs_second_review_count = sum(
        decision.decision == "NEEDS_SECOND_REVIEW" for decision in current.values()
    )
    return SupplementAuditStatus(
        status=str(row["status"]),
        source_count=int(row["source_count"]),
        candidate_count=int(row["candidate_count"]),
        reviewed_count=len(current),
        promotion_count=promotion_count,
        remaining_count=int(row["candidate_count"]) - len(current),
        needs_second_review_count=needs_second_review_count,
    )


def bind_supplement_pages(
    *, audit: IronSulfideSupplementAuditStore, source_root: Path
) -> int:
    """Bind every candidate page to text from the same verified PDF bytes."""
    resolved_root = source_root.resolve()
    expected_root = audit._connection.execute(
        """
        select source_root_sha256 from iron_sulfide_supplement_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if (
        expected_root is None
        or expected_root["source_root_sha256"] is None
        or _source_root_digest(resolved_root)
        != str(expected_root["source_root_sha256"])
    ):
        _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_ROOT_MISMATCH")
    mounted_by_name: dict[str, list[Path]] = {}
    for path in resolved_root.rglob("*"):
        if path.is_file():
            mounted_by_name.setdefault(path.name.casefold(), []).append(path.resolve())
    sources = audit._connection.execute(
        """
        select audited.source_id, audited.file_sha256, source.source_path
        from iron_sulfide_supplement_audit_sources as audited
        join index_sources as source
          on source.run_id = audited.run_id and source.source_id = audited.source_id
        where audited.run_id = ? and audited.audit_id = ?
        order by audited.source_id
        """,
        (audit.run_id, audit.audit_id),
    ).fetchall()
    bindings: list[tuple[str, str, str, object]] = []
    try:
        for source in sources:
            paths = mounted_by_name.get(_basename(source["source_path"]).casefold(), ())
            if len(paths) != 1:
                _fail("IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH")
            content = paths[0].read_bytes()
            if hashlib.sha256(content).hexdigest() != str(source["file_sha256"]):
                _fail("IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH")
            reader = PdfReader(BytesIO(content))
            candidates = audit._connection.execute(
                """
                select locator, page_number, page_text_sha256
                from iron_sulfide_supplement_audit_candidates
                where run_id = ? and audit_id = ? and source_id = ?
                order by review_order
                """,
                (audit.run_id, audit.audit_id, source["source_id"]),
            ).fetchall()
            for candidate in candidates:
                page_number = int(candidate["page_number"])
                if page_number < 1 or page_number > len(reader.pages):
                    _fail("IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH")
                extracted = reader.pages[page_number - 1].extract_text()
                text = (extracted or "").replace("\r\n", "\n").replace("\r", "\n")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                prior = candidate["page_text_sha256"]
                if prior is not None and str(prior) != digest:
                    _fail("IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH")
                bindings.append(
                    (
                        digest,
                        str(source["source_id"]),
                        str(candidate["locator"]),
                        prior,
                    )
                )
    except IronSulfideSupplementAuditError:
        raise
    except Exception as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH"
        ) from error
    with audit._connection:
        audit._connection.executemany(
            """
            update iron_sulfide_supplement_audit_candidates
            set page_text_sha256 = ?
            where run_id = ? and audit_id = ? and source_id = ? and locator = ?
            """,
            (
                (
                    digest,
                    audit.run_id,
                    audit.audit_id,
                    source_id,
                    locator,
                )
                for digest, source_id, locator, _ in bindings
            ),
        )
    return len(bindings)


def _supplement_decision_records(
    audit: IronSulfideSupplementAuditStore,
) -> tuple[SupplementLocatorDecision, ...]:
    rows = audit._connection.execute(
        """
        select decision_id, source_id, locator, decision, reason_code,
               page_text_sha256, reviewer_id, supersedes_decision_id, decided_at
        from iron_sulfide_supplement_audit_decisions
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchall()
    return tuple(SupplementLocatorDecision.from_mapping(dict(row)) for row in rows)


def _current_supplement_decisions(
    audit: IronSulfideSupplementAuditStore,
) -> dict[tuple[str, str], SupplementLocatorDecision]:
    decisions = _supplement_decision_records(audit)
    by_id = {decision.decision_id: decision for decision in decisions}
    if len(by_id) != len(decisions):
        _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID")
    superseded: set[str] = set()
    for decision in decisions:
        if decision.supersedes_decision_id is None:
            continue
        prior = by_id.get(decision.supersedes_decision_id)
        if (
            prior is None
            or prior.source_id != decision.source_id
            or prior.locator != decision.locator
        ):
            _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID")
        superseded.add(prior.decision_id)
    current: dict[tuple[str, str], SupplementLocatorDecision] = {}
    for decision in decisions:
        if decision.decision_id in superseded:
            continue
        key = (decision.source_id, decision.locator)
        if key in current:
            _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID")
        current[key] = decision
    candidates = audit._connection.execute(
        """
        select source_id, locator, review_order
        from iron_sulfide_supplement_audit_candidates
        where run_id = ? and audit_id = ? order by review_order
        """,
        (audit.run_id, audit.audit_id),
    ).fetchall()
    order_by_key = {
        (str(row["source_id"]), str(row["locator"])): int(row["review_order"])
        for row in candidates
    }
    if any(key not in order_by_key for key in current):
        _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID")
    current_orders = sorted(order_by_key[key] for key in current)
    if current_orders != list(range(1, len(current_orders) + 1)):
        _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_OUT_OF_ORDER")
    return current


def _candidate_from_row(row: sqlite3.Row) -> SupplementCandidate:
    digest = row["page_text_sha256"]
    if digest is None:
        _fail("IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH")
    return SupplementCandidate(
        source_id=str(row["source_id"]),
        locator=str(row["locator"]),
        page_number=int(row["page_number"]),
        review_order=int(row["review_order"]),
        page_text_sha256=_digest(
            digest, "IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH"
        ),
    )


def next_supplement_candidate(
    audit: IronSulfideSupplementAuditStore,
) -> SupplementCandidate | None:
    """Return only the next candidate in the frozen prefix."""
    status = supplement_audit_status(audit)
    if status.status in {"TARGET_MET", "EXHAUSTED_INSUFFICIENT"}:
        return None
    current = _current_supplement_decisions(audit)
    needs = tuple(
        decision
        for decision in current.values()
        if decision.decision == "NEEDS_SECOND_REVIEW"
    )
    if len(needs) > 1:
        _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID")
    if needs:
        row = audit._connection.execute(
            """
            select source_id, locator, page_number, review_order, page_text_sha256
            from iron_sulfide_supplement_audit_candidates
            where run_id = ? and audit_id = ? and source_id = ? and locator = ?
            """,
            (
                audit.run_id,
                audit.audit_id,
                needs[0].source_id,
                needs[0].locator,
            ),
        ).fetchone()
    else:
        row = audit._connection.execute(
            """
            select source_id, locator, page_number, review_order, page_text_sha256
            from iron_sulfide_supplement_audit_candidates
            where run_id = ? and audit_id = ? and review_order = ?
            """,
            (audit.run_id, audit.audit_id, len(current) + 1),
        ).fetchone()
    return None if row is None else _candidate_from_row(row)


def _set_supplement_status(audit: IronSulfideSupplementAuditStore) -> None:
    current = _current_supplement_decisions(audit)
    run = audit._connection.execute(
        """
        select candidate_count, promotion_target
        from iron_sulfide_supplement_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if run is None:
        _fail("IRON_SULFIDE_SUPPLEMENT_RUN_MISSING")
    promotions = sum(
        decision.decision == "PROMOTE_FOUNDATIONAL" for decision in current.values()
    )
    unresolved = any(
        decision.decision == "NEEDS_SECOND_REVIEW" for decision in current.values()
    )
    if not unresolved and promotions >= int(run["promotion_target"]):
        status = "TARGET_MET"
    elif not unresolved and len(current) == int(run["candidate_count"]):
        status = "EXHAUSTED_INSUFFICIENT"
    else:
        status = "IN_PROGRESS"
    audit._connection.execute(
        """
        update iron_sulfide_supplement_audit_runs set status = ?
        where run_id = ? and audit_id = ?
        """,
        (status, audit.run_id, audit.audit_id),
    )


def record_supplement_decision(
    *, audit: IronSulfideSupplementAuditStore, record: SupplementLocatorDecision
) -> SupplementAuditStatus:
    """Append one exact next-prefix decision in one SQLite transaction."""
    safe = SupplementLocatorDecision.from_mapping(record.to_mapping())
    candidate = next_supplement_candidate(audit)
    if candidate is None or (
        safe.source_id,
        safe.locator,
    ) != (candidate.source_id, candidate.locator):
        _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_OUT_OF_ORDER")
    if safe.page_text_sha256 != candidate.page_text_sha256:
        _fail("IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH")
    current = _current_supplement_decisions(audit)
    prior = current.get((candidate.source_id, candidate.locator))
    if prior is None:
        if safe.supersedes_decision_id is not None:
            _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID")
    elif (
        prior.decision != "NEEDS_SECOND_REVIEW"
        or safe.supersedes_decision_id != prior.decision_id
        or safe.decision == "NEEDS_SECOND_REVIEW"
    ):
        _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID")
    try:
        with audit._connection:
            audit._connection.execute(
                """
                insert into iron_sulfide_supplement_audit_decisions(
                    run_id, audit_id, decision_id, source_id, locator, decision,
                    reason_code, page_text_sha256, reviewer_id,
                    supersedes_decision_id, decided_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit.run_id,
                    audit.audit_id,
                    safe.decision_id,
                    safe.source_id,
                    safe.locator,
                    safe.decision,
                    safe.reason_code,
                    safe.page_text_sha256,
                    safe.reviewer_id,
                    safe.supersedes_decision_id,
                    safe.decided_at,
                ),
            )
            _set_supplement_status(audit)
    except sqlite3.IntegrityError as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_DECISION_INVALID"
        ) from error
    return supplement_audit_status(audit)


def extract_supplement_page(
    *,
    audit: IronSulfideSupplementAuditStore,
    source_root: Path,
    source_id: str,
    locator: str,
) -> SupplementReviewPacket:
    """Extract only the exact bound next candidate from verified PDF bytes."""
    candidate = next_supplement_candidate(audit)
    if candidate is None or (
        _string(source_id, "IRON_SULFIDE_SUPPLEMENT_CANDIDATE_INVALID"),
        _string(locator, "IRON_SULFIDE_SUPPLEMENT_CANDIDATE_INVALID"),
    ) != (candidate.source_id, candidate.locator):
        _fail("IRON_SULFIDE_SUPPLEMENT_DECISION_OUT_OF_ORDER")
    source = audit._connection.execute(
        """
        select audited.file_sha256, indexed.source_path
        from iron_sulfide_supplement_audit_sources as audited
        join index_sources as indexed
          on indexed.run_id = audited.run_id and indexed.source_id = audited.source_id
        where audited.run_id = ? and audited.audit_id = ? and audited.source_id = ?
        """,
        (audit.run_id, audit.audit_id, candidate.source_id),
    ).fetchone()
    if source is None:
        _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_PROVENANCE_INVALID")
    resolved_root = source_root.resolve()
    paths = tuple(
        path.resolve()
        for path in resolved_root.rglob("*")
        if path.is_file()
        and path.name.casefold() == _basename(source["source_path"]).casefold()
    )
    try:
        if len(paths) != 1 or not paths[0].is_relative_to(resolved_root):
            _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_PROVENANCE_INVALID")
        content = paths[0].read_bytes()
        if hashlib.sha256(content).hexdigest() != str(source["file_sha256"]):
            _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_PROVENANCE_INVALID")
        reader = PdfReader(BytesIO(content))
        extracted = reader.pages[candidate.page_number - 1].extract_text()
    except IronSulfideSupplementAuditError:
        raise
    except Exception as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_PAGE_EXTRACTION_FAILED"
        ) from error
    text = (extracted or "").replace("\r\n", "\n").replace("\r", "\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != candidate.page_text_sha256:
        _fail("IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH")
    return SupplementReviewPacket(
        source_id=candidate.source_id,
        locator=candidate.locator,
        page_number=candidate.page_number,
        page_text=text,
        page_text_sha256=digest,
    )


def _canonical_jsonl(records: tuple[Mapping[str, object], ...]) -> bytes:
    if not records:
        return b""
    return (
        "\n".join(
            json.dumps(
                dict(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            for record in records
        )
        + "\n"
    ).encode("utf-8")


def _terminal_decisions(
    audit: IronSulfideSupplementAuditStore,
) -> tuple[SupplementLocatorDecision, ...]:
    status = supplement_audit_status(audit)
    if (
        status.status not in {"TARGET_MET", "EXHAUSTED_INSUFFICIENT"}
        or status.needs_second_review_count != 0
    ):
        _fail("IRON_SULFIDE_SUPPLEMENT_AUDIT_INCOMPLETE")
    run = audit._connection.execute(
        """
        select promotion_target from iron_sulfide_supplement_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if run is None:
        _fail("IRON_SULFIDE_SUPPLEMENT_RUN_MISSING")
    target = int(run["promotion_target"])
    if (
        status.status == "TARGET_MET" and status.promotion_count != target
    ) or (
        status.status == "EXHAUSTED_INSUFFICIENT"
        and (
            status.reviewed_count != status.candidate_count
            or status.promotion_count >= target
        )
    ):
        _fail("IRON_SULFIDE_SUPPLEMENT_AUDIT_INCOMPLETE")
    current = _current_supplement_decisions(audit)
    rows = audit._connection.execute(
        """
        select source_id, locator, review_order, page_text_sha256
        from iron_sulfide_supplement_audit_candidates
        where run_id = ? and audit_id = ? and review_order <= ?
        order by review_order
        """,
        (audit.run_id, audit.audit_id, status.reviewed_count),
    ).fetchall()
    decisions: list[SupplementLocatorDecision] = []
    for row in rows:
        key = (str(row["source_id"]), str(row["locator"]))
        decision = current.get(key)
        if (
            decision is None
            or row["page_text_sha256"] is None
            or decision.page_text_sha256 != str(row["page_text_sha256"])
        ):
            _fail("IRON_SULFIDE_SUPPLEMENT_AUDIT_INCOMPLETE")
        decisions.append(decision)
    if len(decisions) != status.reviewed_count:
        _fail("IRON_SULFIDE_SUPPLEMENT_AUDIT_INCOMPLETE")
    return tuple(decisions)


def _page_binding_digest(audit: IronSulfideSupplementAuditStore) -> str:
    rows = audit._connection.execute(
        """
        select source_id, locator, page_number, review_order, page_text_sha256
        from iron_sulfide_supplement_audit_candidates
        where run_id = ? and audit_id = ? order by review_order
        """,
        (audit.run_id, audit.audit_id),
    ).fetchall()
    if not rows or any(row["page_text_sha256"] is None for row in rows):
        _fail("IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH")
    return _canonical_digest(tuple(dict(row) for row in rows))


def _supplement_binding_payload(
    *,
    audit: IronSulfideSupplementAuditStore,
    decision_payload_sha256: str,
    core_binding_sha256: str,
) -> bytes:
    run = audit._connection.execute(
        """
        select snapshot_binding_sha256, source_set_sha256, candidate_set_sha256,
               source_count, candidate_count, promotion_target, status,
               source_root_sha256
        from iron_sulfide_supplement_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if run is None:
        _fail("IRON_SULFIDE_SUPPLEMENT_RUN_MISSING")
    decisions = _terminal_decisions(audit)
    binding = {
        "schema_version": 2,
        "run_id": audit.run_id,
        "audit_id": audit.audit_id,
        "snapshot_binding_sha256": _digest(
            run["snapshot_binding_sha256"],
            "IRON_SULFIDE_SUPPLEMENT_BINDING_INVALID",
        ),
        "source_set_sha256": _digest(
            run["source_set_sha256"], "IRON_SULFIDE_SUPPLEMENT_BINDING_INVALID"
        ),
        "candidate_set_sha256": _digest(
            run["candidate_set_sha256"],
            "IRON_SULFIDE_SUPPLEMENT_BINDING_INVALID",
        ),
        "source_root_sha256": _digest(
            run["source_root_sha256"],
            "IRON_SULFIDE_SUPPLEMENT_BINDING_INVALID",
        ),
        "page_binding_sha256": _page_binding_digest(audit),
        "source_count": int(run["source_count"]),
        "candidate_count": int(run["candidate_count"]),
        "promotion_target": int(run["promotion_target"]),
        "terminal_status": str(run["status"]),
        "reviewed_prefix_count": len(decisions),
        "decision_payload_sha256": _digest(
            decision_payload_sha256,
            "IRON_SULFIDE_SUPPLEMENT_BINDING_INVALID",
        ),
        "core_audit_binding_sha256": _digest(
            core_binding_sha256,
            "IRON_SULFIDE_SUPPLEMENT_BINDING_INVALID",
        ),
    }
    return (
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _supplement_sealed_directory(audit: IronSulfideSupplementAuditStore) -> Path:
    return (
        audit.database_path.parent
        / "iron-sulfide-supplement-audit"
        / SUPPLEMENT_ARTIFACT_VERSION
        / "sealed"
    )


def _write_fsynced_file(*, destination: Path, content: bytes) -> None:
    with destination.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _verified_manifest(path: Path) -> str:
    manifest = path.with_name(f"{path.name}.sha256")
    try:
        content = path.read_bytes()
        text = manifest.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_SEAL_VERIFY_FAILED"
        ) from error
    if (
        len(text) != 65
        or not text.endswith("\n")
        or any(character not in "0123456789abcdef" for character in text[:64])
    ):
        _fail("IRON_SULFIDE_SUPPLEMENT_MANIFEST_INVALID")
    digest = hashlib.sha256(content).hexdigest()
    if digest != text[:64]:
        _fail("IRON_SULFIDE_SUPPLEMENT_DIGEST_MISMATCH")
    return digest


def _authenticate_core_proposal(
    *,
    core_audit: FoundationalAuditStore,
    supplement_audit: IronSulfideSupplementAuditStore,
    expected_core_binding_sha256: str,
) -> str:
    trusted = _digest(
        expected_core_binding_sha256,
        "IRON_SULFIDE_SUPPLEMENT_CORE_AUDIT_INVALID",
    )
    if (
        core_audit.database_path.resolve()
        != supplement_audit.database_path.resolve()
        or core_audit.run_id != supplement_audit.run_id
    ):
        _fail("IRON_SULFIDE_SUPPLEMENT_CORE_AUDIT_INVALID")
    core_run = core_audit._connection.execute(
        """
        select snapshot_binding_sha256 from foundational_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (core_audit.run_id, core_audit.audit_id),
    ).fetchone()
    supplement_run = supplement_audit._connection.execute(
        """
        select snapshot_binding_sha256
        from iron_sulfide_supplement_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (supplement_audit.run_id, supplement_audit.audit_id),
    ).fetchone()
    if (
        core_run is None
        or supplement_run is None
        or str(core_run["snapshot_binding_sha256"])
        != str(supplement_run["snapshot_binding_sha256"])
    ):
        _fail("IRON_SULFIDE_SUPPLEMENT_CORE_AUDIT_INVALID")
    try:
        verify_correction_proposal(
            audit=core_audit,
            expected_binding_sha256=trusted,
        )
    except FoundationalLocatorAuditError as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_CORE_AUDIT_INVALID"
        ) from error
    return trusted


def verify_supplement_proposal(
    *,
    audit: IronSulfideSupplementAuditStore,
    core_audit: FoundationalAuditStore,
    expected_binding_sha256: str,
    expected_core_binding_sha256: str,
) -> SupplementAuditSeal:
    """Verify the exact four-file supplement proposal without rewriting it."""
    trusted = _digest(
        expected_binding_sha256, "IRON_SULFIDE_SUPPLEMENT_BINDING_MISMATCH"
    )
    core_trusted = _authenticate_core_proposal(
        core_audit=core_audit,
        supplement_audit=audit,
        expected_core_binding_sha256=expected_core_binding_sha256,
    )
    sealed = _supplement_sealed_directory(audit)
    correction_path = sealed / SUPPLEMENT_CORRECTION_NAME
    binding_path = sealed / SUPPLEMENT_BINDING_NAME
    paths = (correction_path, binding_path)
    manifests = tuple(path.with_name(f"{path.name}.sha256") for path in paths)
    presence = tuple(path.is_file() for path in (*paths, *manifests))
    if not any(presence):
        _fail("IRON_SULFIDE_SUPPLEMENT_SEAL_MISSING")
    if not all(presence):
        _fail("IRON_SULFIDE_SUPPLEMENT_SEAL_PARTIAL")
    try:
        if {path.name for path in sealed.iterdir()} != {
            path.name for path in (*paths, *manifests)
        }:
            _fail("IRON_SULFIDE_SUPPLEMENT_SEAL_PARTIAL")
    except OSError as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_SEAL_VERIFY_FAILED"
        ) from error
    correction_digest = _verified_manifest(correction_path)
    binding_digest = _verified_manifest(binding_path)
    if binding_digest != trusted:
        _fail("IRON_SULFIDE_SUPPLEMENT_BINDING_MISMATCH")
    expected_correction = _canonical_jsonl(
        tuple(decision.to_mapping() for decision in _terminal_decisions(audit))
    )
    if correction_path.read_bytes() != expected_correction:
        _fail("IRON_SULFIDE_SUPPLEMENT_BINDING_MISMATCH")
    expected_binding = _supplement_binding_payload(
        audit=audit,
        decision_payload_sha256=correction_digest,
        core_binding_sha256=core_trusted,
    )
    if binding_path.read_bytes() != expected_binding:
        _fail("IRON_SULFIDE_SUPPLEMENT_BINDING_MISMATCH")
    return SupplementAuditSeal(
        artifacts=(
            SupplementAuditArtifact(
                name=SUPPLEMENT_CORRECTION_NAME,
                path=correction_path,
                manifest_path=manifests[0],
                sha256=correction_digest,
                record_count=len(_terminal_decisions(audit)),
            ),
            SupplementAuditArtifact(
                name=SUPPLEMENT_BINDING_NAME,
                path=binding_path,
                manifest_path=manifests[1],
                sha256=binding_digest,
                record_count=1,
            ),
        ),
        binding_sha256=binding_digest,
    )


def seal_supplement_proposal(
    *,
    audit: IronSulfideSupplementAuditStore,
    core_audit: FoundationalAuditStore,
    core_binding_sha256: str,
) -> SupplementAuditSeal:
    """Atomically publish a terminal supplement correction proposal."""
    core_trusted = _authenticate_core_proposal(
        core_audit=core_audit,
        supplement_audit=audit,
        expected_core_binding_sha256=core_binding_sha256,
    )
    correction = _canonical_jsonl(
        tuple(decision.to_mapping() for decision in _terminal_decisions(audit))
    )
    correction_digest = hashlib.sha256(correction).hexdigest()
    binding = _supplement_binding_payload(
        audit=audit,
        decision_payload_sha256=correction_digest,
        core_binding_sha256=core_trusted,
    )
    binding_digest = hashlib.sha256(binding).hexdigest()
    sealed = _supplement_sealed_directory(audit)
    if sealed.exists():
        if not sealed.is_dir():
            _fail("IRON_SULFIDE_SUPPLEMENT_SEAL_PARTIAL")
        return verify_supplement_proposal(
            audit=audit,
            core_audit=core_audit,
            expected_binding_sha256=binding_digest,
            expected_core_binding_sha256=core_trusted,
        )
    version_root = sealed.parent
    version_root.mkdir(parents=True, exist_ok=True)
    for stale in version_root.glob(".sealed.*.tmp"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    staged = Path(mkdtemp(prefix=".sealed.", suffix=".tmp", dir=version_root))
    try:
        for name, content in {
            SUPPLEMENT_CORRECTION_NAME: correction,
            SUPPLEMENT_BINDING_NAME: binding,
        }.items():
            _write_fsynced_file(destination=staged / name, content=content)
            _write_fsynced_file(
                destination=staged / f"{name}.sha256",
                content=(hashlib.sha256(content).hexdigest() + "\n").encode("ascii"),
            )
        os.replace(staged, sealed)
        return verify_supplement_proposal(
            audit=audit,
            core_audit=core_audit,
            expected_binding_sha256=binding_digest,
            expected_core_binding_sha256=core_trusted,
        )
    except (IronSulfideSupplementAuditError, OSError) as error:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_SEAL_WRITE_FAILED"
        ) from error


def _core_sealed_decisions(
    *, core_audit: FoundationalAuditStore, expected_binding_sha256: str
) -> tuple[LocatorAuditDecision, ...]:
    seal = verify_correction_proposal(
        audit=core_audit,
        expected_binding_sha256=expected_binding_sha256,
    )
    correction = next(
        artifact for artifact in seal.artifacts if artifact.name.endswith(".jsonl")
    )
    try:
        return tuple(
            LocatorAuditDecision.from_mapping(json.loads(line))
            for line in correction.path.read_text(encoding="utf-8").splitlines()
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_CORE_AUDIT_INVALID"
        ) from error


def calculate_combined_hypothetical_capacity(
    *,
    core_audit: FoundationalAuditStore,
    supplement_audit: IronSulfideSupplementAuditStore,
    expected_core_binding_sha256: str,
    expected_supplement_binding_sha256: str,
) -> HypotheticalCapacityReport:
    """Project both approved proposals without mutating the inventory."""
    core_trusted = _authenticate_core_proposal(
        core_audit=core_audit,
        supplement_audit=supplement_audit,
        expected_core_binding_sha256=expected_core_binding_sha256,
    )
    core_decisions = _core_sealed_decisions(
        core_audit=core_audit,
        expected_binding_sha256=core_trusted,
    )
    verify_supplement_proposal(
        audit=supplement_audit,
        core_audit=core_audit,
        expected_binding_sha256=expected_supplement_binding_sha256,
        expected_core_binding_sha256=core_trusted,
    )
    supplement_decisions = _terminal_decisions(supplement_audit)
    core_promotions = tuple(
        decision for decision in core_decisions if decision.decision == "PROMOTE_FOUNDATIONAL"
    )
    supplement_promotions = tuple(
        decision
        for decision in supplement_decisions
        if decision.decision == "PROMOTE_FOUNDATIONAL"
    )
    core_keys = {(item.source_id, item.locator) for item in core_promotions}
    supplement_keys = {
        (item.source_id, item.locator) for item in supplement_promotions
    }
    if core_keys & supplement_keys:
        _fail("IRON_SULFIDE_SUPPLEMENT_CAPACITY_INVALID")
    promoted_keys = core_keys | supplement_keys
    rows = supplement_audit._connection.execute(
        """
        select locator.source_id, locator.locator, locator.topic,
               locator.source_role, source.parser_type
        from index_locators as locator
        join index_sources as source
          on source.run_id = locator.run_id and source.source_id = locator.source_id
        where locator.run_id = ? and locator.e1a4_available = 1
        order by locator.source_id, locator.locator
        """,
        (supplement_audit.run_id,),
    ).fetchall()
    parser_rows = supplement_audit._connection.execute(
        """
        select source_id, parser_type from index_sources where run_id = ?
        """,
        (supplement_audit.run_id,),
    ).fetchall()
    parser_by_source = {
        str(row["source_id"]): str(row["parser_type"]) for row in parser_rows
    }
    grouped: dict[tuple[str, str, str, str], list[str]] = {}
    for row in rows:
        key = (str(row["source_id"]), str(row["locator"]))
        if key in promoted_keys:
            continue
        grouped.setdefault(
            (
                key[0],
                str(row["topic"]),
                str(row["source_role"]),
                str(row["parser_type"]),
            ),
            [],
        ).append(key[1])
    for decision in core_promotions:
        parser_type = parser_by_source.get(decision.source_id)
        if parser_type is None or decision.proposed_topic is None:
            _fail("IRON_SULFIDE_SUPPLEMENT_CAPACITY_INVALID")
        grouped.setdefault(
            (
                decision.source_id,
                decision.proposed_topic,
                "foundational",
                parser_type,
            ),
            [],
        ).append(decision.locator)
    for decision in supplement_promotions:
        parser_type = parser_by_source.get(decision.source_id)
        if parser_type is None:
            _fail("IRON_SULFIDE_SUPPLEMENT_CAPACITY_INVALID")
        grouped.setdefault(
            (decision.source_id, "iron_sulfide", "foundational", parser_type),
            [],
        ).append(decision.locator)
    counts = {
        (topic, role): 0
        for topic in TOPICS
        for role in ("foundational", "supporting")
    }
    for (_, topic, role, _), locators in grouped.items():
        counts[(topic, role)] += len(set(locators))
    strata = tuple(
        HypotheticalCapacityStratum(
            topic=topic,
            source_role=role,
            fresh_locator_count=counts[(topic, role)],
            required_locators=12,
            sufficient=counts[(topic, role)] >= 12,
        )
        for topic in ("iron_sulfide", "scale", "corrosion", "paraffin")
        for role in ("foundational", "supporting")
    )
    all_sufficient = all(item.sufficient for item in strata)
    allocation_count = 0
    if all_sufficient:
        sources = tuple(
            E1A3SourceMetadata(
                source_id=source_id,
                topic=topic,  # type: ignore[arg-type]
                source_role=role,  # type: ignore[arg-type]
                parser_type=parser_type,
                locators=tuple(sorted(set(locators))),
                eligibility_status="eligible",
            )
            for (source_id, topic, role, parser_type), locators in sorted(
                grouped.items()
            )
        )
        try:
            allocations = allocate_sampling_slots(
                slots=build_sampling_slots(),
                sources=sources,
            )
            if len(allocations) == 96:
                allocation_count = 96
        except E1A3SamplingError:
            allocation_count = 0
    return HypotheticalCapacityReport(
        strata=strata,
        all_sufficient=all_sufficient,
        allocation_available=allocation_count == 96,
        allocation_count=allocation_count,
    )
