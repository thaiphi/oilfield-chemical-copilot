"""Durable, fail-closed review of unclassified foundational PDF locators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
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


AUDIT_DECISION_CONTRACTS = frozenset(
    {
        ("PROMOTE_FOUNDATIONAL", "SUBSTANTIVE_TARGET_EVIDENCE"),
        ("KEEP_INELIGIBLE", "NO_TARGET_TOPIC"),
        ("KEEP_INELIGIBLE", "TITLE_OR_INDEX_ONLY"),
        ("KEEP_INELIGIBLE", "INSUFFICIENT_CONTEXT"),
        ("KEEP_INELIGIBLE", "SUPPORTING_ONLY"),
        ("KEEP_INELIGIBLE", "DUPLICATE_PAGE_CONTENT"),
        ("NEEDS_SECOND_REVIEW", "AMBIGUOUS_OR_NONEXTRACTABLE"),
    }
)
AUDIT_ARTIFACT_VERSION = "v2"
CORRECTION_NAME = "foundational-locator-corrections.v2.jsonl"
AUDIT_BINDING_NAME = "audit-binding.v2.json"


class FoundationalLocatorAuditError(RuntimeError):
    """Raised when foundational locator audit state violates its contract."""


def _fail(code: str) -> None:
    raise FoundationalLocatorAuditError(code)


def _exact_keys(
    mapping: Mapping[str, object], expected: frozenset[str], code: str
) -> None:
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


def _digest(value: object, code: str) -> str:
    text = _string(value, code)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(code)
    return text


def _page_number(locator: object, code: str) -> int:
    safe = _string(locator, code)
    match = re.fullmatch(r"page:([1-9][0-9]*)", safe)
    if match is None:
        _fail(code)
    return int(match.group(1))


@dataclass(frozen=True)
class LocatorAuditDecision:
    decision_id: str
    source_id: str
    locator: str
    decision: str
    proposed_topic: str | None
    reason_code: str
    page_text_sha256: str
    reviewer_id: str
    supersedes_decision_id: str | None
    decided_at: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> LocatorAuditDecision:
        code = "FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID"
        _exact_keys(
            mapping,
            frozenset(
                {
                    "decision_id",
                    "source_id",
                    "locator",
                    "decision",
                    "proposed_topic",
                    "reason_code",
                    "page_text_sha256",
                    "reviewer_id",
                    "supersedes_decision_id",
                    "decided_at",
                }
            ),
            code,
        )
        decision_id = _string(mapping["decision_id"], code)
        decision = _string(mapping["decision"], code)
        reason = _string(mapping["reason_code"], code)
        topic = _optional_string(mapping["proposed_topic"], code)
        supersedes = _optional_string(mapping["supersedes_decision_id"], code)
        if (
            (decision, reason) not in AUDIT_DECISION_CONTRACTS
            or (decision == "PROMOTE_FOUNDATIONAL") != (topic in TOPICS)
            or supersedes == decision_id
        ):
            _fail(code)
        return cls(
            decision_id=decision_id,
            source_id=_string(mapping["source_id"], code),
            locator=_string(mapping["locator"], code),
            decision=decision,
            proposed_topic=topic,
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
            "proposed_topic": self.proposed_topic,
            "reason_code": self.reason_code,
            "page_text_sha256": self.page_text_sha256,
            "reviewer_id": self.reviewer_id,
            "supersedes_decision_id": self.supersedes_decision_id,
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class AuditStatus:
    status: str
    candidate_count: int
    current_decision_count: int
    remaining_count: int
    needs_second_review_count: int


@dataclass(frozen=True)
class VerifiedSourcePdf:
    path: Path
    sha256: str
    page_count: int


@dataclass(frozen=True)
class LocatorReviewPacket:
    source_id: str
    locator: str
    page_number: int
    page_text: str
    page_text_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "locator": self.locator,
            "page_number": self.page_number,
            "page_text": self.page_text,
            "page_text_sha256": self.page_text_sha256,
        }


@dataclass(frozen=True)
class AuditArtifact:
    name: str
    path: Path
    manifest_path: Path
    sha256: str
    record_count: int


@dataclass(frozen=True)
class AuditSeal:
    artifacts: tuple[AuditArtifact, ...]
    binding_sha256: str


@dataclass(frozen=True)
class HypotheticalCapacityStratum:
    topic: str
    source_role: str
    fresh_locator_count: int
    required_locators: int
    sufficient: bool


@dataclass(frozen=True)
class HypotheticalCapacityReport:
    strata: tuple[HypotheticalCapacityStratum, ...]
    all_sufficient: bool
    allocation_available: bool
    allocation_count: int


class FoundationalAuditStore:
    """Connection to one audit run in the reconciliation SQLite database."""

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
    ) -> FoundationalAuditStore:
        safe_path = database_path.resolve()
        if not safe_path.is_file():
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_STORE_MISSING")
        connection = sqlite3.connect(safe_path)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma foreign_keys = on")
        connection.execute("pragma journal_mode = wal")
        connection.execute("pragma synchronous = full")
        row = connection.execute(
            """
            select audit_id from foundational_audit_runs
            where run_id = ? and audit_id = ?
            """,
            (run_id, audit_id),
        ).fetchone()
        if row is None:
            connection.close()
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_RUN_MISSING")
        return cls(
            database_path=safe_path,
            run_id=_string(run_id, "FOUNDATIONAL_LOCATOR_AUDIT_RUN_INVALID"),
            audit_id=_string(audit_id, "FOUNDATIONAL_LOCATOR_AUDIT_RUN_INVALID"),
            connection=connection,
        )

    def close(self) -> None:
        self._connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        create table if not exists foundational_audit_runs (
            run_id text not null references runs(run_id),
            audit_id text not null,
            snapshot_binding_sha256 text not null,
            source_id text not null,
            source_drive_file_id text not null,
            source_file_sha256 text not null,
            candidate_set_sha256 text not null,
            candidate_count integer not null,
            status text not null,
            created_at text not null default current_timestamp,
            updated_at text not null default current_timestamp,
            primary key (run_id, audit_id)
        );
        create table if not exists foundational_audit_candidates (
            run_id text not null,
            audit_id text not null,
            source_id text not null,
            locator text not null,
            page_number integer not null,
            page_text_sha256 text,
            primary key (run_id, audit_id, source_id, locator),
            foreign key (run_id, audit_id)
                references foundational_audit_runs(run_id, audit_id)
        );
        create table if not exists foundational_audit_decisions (
            run_id text not null,
            audit_id text not null,
            decision_id text not null,
            source_id text not null,
            locator text not null,
            decision text not null,
            proposed_topic text,
            reason_code text not null,
            page_text_sha256 text not null,
            reviewer_id text not null,
            supersedes_decision_id text,
            decided_at text not null,
            primary key (run_id, audit_id, decision_id),
            foreign key (run_id, audit_id, source_id, locator)
                references foundational_audit_candidates(
                    run_id, audit_id, source_id, locator
                )
        );
        """
    )
    columns = {
        str(row[1])
        for row in connection.execute(
            "pragma table_info(foundational_audit_candidates)"
        ).fetchall()
    }
    if "page_text_sha256" not in columns:
        connection.execute(
            "alter table foundational_audit_candidates "
            "add column page_text_sha256 text"
        )


def _candidate_records(
    store: ReconciliationStore,
) -> tuple[tuple[str, str, int], ...]:
    rows = store._connection.execute(
        """
        select source_id, locator
        from index_locators
        where run_id = ? and source_role = 'foundational'
          and substantive_status = 'INELIGIBLE'
        order by source_id, locator
        """,
        (store.run_id,),
    ).fetchall()
    records = tuple(
        (
            _string(row["source_id"], "FOUNDATIONAL_LOCATOR_AUDIT_CANDIDATE_INVALID"),
            _string(row["locator"], "FOUNDATIONAL_LOCATOR_AUDIT_CANDIDATE_INVALID"),
            _page_number(
                row["locator"], "FOUNDATIONAL_LOCATOR_AUDIT_CANDIDATE_INVALID"
            ),
        )
        for row in rows
    )
    if not records or len({record[:2] for record in records}) != len(records):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_CANDIDATE_INVALID")
    if len({record[0] for record in records}) != 1:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_SCOPE_INVALID")
    return records


def _candidate_set_sha256(records: tuple[tuple[str, str, int], ...]) -> str:
    payload = json.dumps(
        [
            {"source_id": source_id, "locator": locator, "page_number": page_number}
            for source_id, locator, page_number in records
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trusted_source_sha256(
    *,
    store: ReconciliationStore,
    source_id: str,
    source_name: str,
    source_provenance_drive_file_id: object,
    source_content_sha256: object,
    drive_id: str,
    drive_size_bytes: int,
) -> str:
    """Derive the expected PDF digest from authenticated reconciliation evidence."""
    match = store._connection.execute(
        """
        select match_key, relative_path, source_id, canonical_sha256,
               match_method, match_status
        from document_matches
        where run_id = ? and drive_file_id = ?
        """,
        (store.run_id, drive_id),
    ).fetchone()
    if match is None or (
        match["source_id"] is not None and str(match["source_id"]) != source_id
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")

    relative_path = match["relative_path"]
    local = None
    if relative_path is not None:
        local = store._connection.execute(
            """
            select sha256, size_bytes, file_type from local_files
            where run_id = ? and relative_path = ?
            """,
            (store.run_id, relative_path),
        ).fetchone()
        if (
            local is None
            or Path(str(relative_path).replace("\\", "/")).name.casefold()
            != source_name.casefold()
            or int(local["size_bytes"]) != drive_size_bytes
            or str(local["file_type"]).lower() != "pdf"
        ):
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")

    status = str(match["match_status"])
    method = str(match["match_method"])
    if status in {"EXACT_MATCH", "DUPLICATE_ALIAS"}:
        if match["canonical_sha256"] is None:
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")
        expected_digest = _digest(
            match["canonical_sha256"],
            "FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID",
        )
        if local is not None and str(local["sha256"]) != expected_digest:
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")
        if local is None and not (
            method == "DRIVE_ID_PROVENANCE"
            and source_provenance_drive_file_id == drive_id
            and source_content_sha256 == expected_digest
        ):
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")
    elif status == "AMBIGUOUS_REVIEW_REQUIRED" and method == "FILENAME_AND_SIZE":
        decisions = store._connection.execute(
            """
            select decision_id, decision, supersedes_decision_id
            from review_decisions
            where run_id = ? and match_key = ?
            """,
            (store.run_id, match["match_key"]),
        ).fetchall()
        superseded = {
            str(row["supersedes_decision_id"])
            for row in decisions
            if row["supersedes_decision_id"] is not None
        }
        current = tuple(
            row for row in decisions if str(row["decision_id"]) not in superseded
        )
        if (
            local is None
            or len(current) != 1
            or str(current[0]["decision"]) != "ACCEPT"
        ):
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")
        expected_digest = _digest(
            local["sha256"],
            "FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID",
        )
    else:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")

    if source_provenance_drive_file_id not in (None, drive_id) or (
        source_content_sha256 is not None
        and str(source_content_sha256) != expected_digest
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")
    return expected_digest


def initialize_audit(
    *,
    store: ReconciliationStore,
    audit_id: str,
    snapshot_binding_sha256: str,
    source_drive_file_id: str,
    source_file_sha256: str,
) -> FoundationalAuditStore:
    """Create or resume one audit bound to the exact reconciliation state."""
    safe_audit_id = _string(audit_id, "FOUNDATIONAL_LOCATOR_AUDIT_RUN_INVALID")
    binding_digest = _digest(
        snapshot_binding_sha256, "FOUNDATIONAL_LOCATOR_AUDIT_RUN_INVALID"
    )
    drive_id = _string(
        source_drive_file_id, "FOUNDATIONAL_LOCATOR_AUDIT_RUN_INVALID"
    )
    source_digest = _digest(
        source_file_sha256, "FOUNDATIONAL_LOCATOR_AUDIT_RUN_INVALID"
    )
    candidates = _candidate_records(store)
    source_id = candidates[0][0]
    candidate_digest = _candidate_set_sha256(candidates)
    try:
        verify_reconciliation_snapshots(
            root=store.root,
            store=store,
            expected_binding_sha256=binding_digest,
        )
    except CorpusReconciliationError as error:
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_RECONCILIATION_UNTRUSTED"
        ) from error
    source = store._connection.execute(
        """
        select source_path, parser_type, provenance_drive_file_id, content_sha256
        from index_sources
        where run_id = ? and source_id = ?
        """,
        (store.run_id, source_id),
    ).fetchone()
    drive = store._connection.execute(
        """
        select name, mime_type, size_bytes from drive_files
        where run_id = ? and drive_file_id = ?
        """,
        (store.run_id, drive_id),
    ).fetchone()
    if source is None or drive is None:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")
    source_name = Path(str(source["source_path"]).replace("\\", "/")).name
    duplicate_drive_names = store._connection.execute(
        "select count(*) from drive_files where run_id = ? and name = ?",
        (store.run_id, drive["name"]),
    ).fetchone()
    if (
        str(source["parser_type"]).lower() != "pdf"
        or str(drive["mime_type"]) != "application/pdf"
        or source_name != str(drive["name"])
        or duplicate_drive_names is None
        or int(duplicate_drive_names[0]) != 1
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")
    trusted_source_digest = _trusted_source_sha256(
        store=store,
        source_id=source_id,
        source_name=source_name,
        source_provenance_drive_file_id=source["provenance_drive_file_id"],
        source_content_sha256=source["content_sha256"],
        drive_id=drive_id,
        drive_size_bytes=int(drive["size_bytes"]),
    )
    if source_digest != trusted_source_digest:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID")
    _create_schema(store._connection)
    existing = store._connection.execute(
        """
        select snapshot_binding_sha256, source_id, source_drive_file_id,
               source_file_sha256, candidate_set_sha256, candidate_count
        from foundational_audit_runs where run_id = ? and audit_id = ?
        """,
        (store.run_id, safe_audit_id),
    ).fetchone()
    expected = (
        binding_digest,
        source_id,
        drive_id,
        source_digest,
        candidate_digest,
        len(candidates),
    )
    if existing is not None and tuple(existing) != expected:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_BINDING_CONFLICT")
    with store._connection:
        store._connection.execute(
            """
            insert into foundational_audit_runs(
                run_id, audit_id, snapshot_binding_sha256, source_id,
                source_drive_file_id, source_file_sha256,
                candidate_set_sha256, candidate_count, status
            ) values (?, ?, ?, ?, ?, ?, ?, ?, 'IN_PROGRESS')
            on conflict(run_id, audit_id) do nothing
            """,
            (store.run_id, safe_audit_id, *expected),
        )
        store._connection.executemany(
            """
            insert into foundational_audit_candidates(
                run_id, audit_id, source_id, locator, page_number
            ) values (?, ?, ?, ?, ?)
            on conflict(run_id, audit_id, source_id, locator) do nothing
            """,
            (
                (store.run_id, safe_audit_id, source, locator, page)
                for source, locator, page in candidates
            ),
        )
    return FoundationalAuditStore.open(
        database_path=store.root / "reconciliation.sqlite",
        run_id=store.run_id,
        audit_id=safe_audit_id,
    )


def _decision_records(audit: FoundationalAuditStore) -> tuple[LocatorAuditDecision, ...]:
    rows = audit._connection.execute(
        """
        select decision_id, source_id, locator, decision, proposed_topic,
               reason_code, page_text_sha256, reviewer_id,
               supersedes_decision_id, decided_at
        from foundational_audit_decisions
        where run_id = ? and audit_id = ?
        order by decision_id
        """,
        (audit.run_id, audit.audit_id),
    ).fetchall()
    return tuple(LocatorAuditDecision.from_mapping(dict(row)) for row in rows)


def _current_decisions(
    records: tuple[LocatorAuditDecision, ...],
) -> dict[tuple[str, str], LocatorAuditDecision]:
    by_id = {record.decision_id: record for record in records}
    if len(by_id) != len(records):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID")
    superseded: set[str] = set()
    for record in records:
        if record.supersedes_decision_id is None:
            continue
        prior = by_id.get(record.supersedes_decision_id)
        if (
            prior is None
            or prior.source_id != record.source_id
            or prior.locator != record.locator
        ):
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID")
        superseded.add(prior.decision_id)
    current: dict[tuple[str, str], LocatorAuditDecision] = {}
    for record in records:
        if record.decision_id in superseded:
            continue
        key = (record.source_id, record.locator)
        if key in current:
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID")
        current[key] = record
    return current


def audit_status(audit: FoundationalAuditStore) -> AuditStatus:
    candidate_row = audit._connection.execute(
        """
        select candidate_count from foundational_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if candidate_row is None:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_RUN_MISSING")
    candidate_count = int(candidate_row["candidate_count"])
    current = _current_decisions(_decision_records(audit))
    second_review = sum(
        decision.decision == "NEEDS_SECOND_REVIEW" for decision in current.values()
    )
    if len(current) == candidate_count and second_review == 0:
        status = "COMPLETE"
    elif second_review:
        status = "BLOCKED"
    else:
        status = "IN_PROGRESS"
    return AuditStatus(
        status=status,
        candidate_count=candidate_count,
        current_decision_count=len(current),
        remaining_count=candidate_count - len(current),
        needs_second_review_count=second_review,
    )


def record_locator_decision(
    *, audit: FoundationalAuditStore, record: LocatorAuditDecision
) -> AuditStatus:
    """Append one validated decision and commit it before returning."""
    safe = LocatorAuditDecision.from_mapping(record.to_mapping())
    candidate = audit._connection.execute(
        """
        select page_text_sha256 from foundational_audit_candidates
        where run_id = ? and audit_id = ? and source_id = ? and locator = ?
        """,
        (audit.run_id, audit.audit_id, safe.source_id, safe.locator),
    ).fetchone()
    if candidate is None:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID")
    if (
        candidate["page_text_sha256"] is None
        or safe.page_text_sha256 != str(candidate["page_text_sha256"])
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_PAGE_BINDING_MISMATCH")
    current = _current_decisions(_decision_records(audit))
    prior = current.get((safe.source_id, safe.locator))
    if (prior is None) != (safe.supersedes_decision_id is None) or (
        prior is not None and safe.supersedes_decision_id != prior.decision_id
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID")
    with audit._connection:
        try:
            audit._connection.execute(
                """
                insert into foundational_audit_decisions(
                    run_id, audit_id, decision_id, source_id, locator,
                    decision, proposed_topic, reason_code, page_text_sha256,
                    reviewer_id, supersedes_decision_id, decided_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit.run_id,
                    audit.audit_id,
                    safe.decision_id,
                    safe.source_id,
                    safe.locator,
                    safe.decision,
                    safe.proposed_topic,
                    safe.reason_code,
                    safe.page_text_sha256,
                    safe.reviewer_id,
                    safe.supersedes_decision_id,
                    safe.decided_at,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise FoundationalLocatorAuditError(
                "FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID"
            ) from error
        status = audit_status(audit)
        audit._connection.execute(
            """
            update foundational_audit_runs
            set status = ?, updated_at = current_timestamp
            where run_id = ? and audit_id = ?
            """,
            (status.status, audit.run_id, audit.audit_id),
        )
        return status


def _verified_pdf_reader(
    *, audit: FoundationalAuditStore, pdf_path: Path
) -> tuple[VerifiedSourcePdf, PdfReader]:
    path = pdf_path.resolve()
    if not path.is_file():
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_PDF_MISSING")
    expected = audit._connection.execute(
        """
        select source_file_sha256 from foundational_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if expected is None:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_RUN_MISSING")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_PDF_READ_FAILED"
        ) from error
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected["source_file_sha256"]:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_PDF_MISMATCH")
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as error:
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_PDF_INVALID"
        ) from error
    page_count = len(reader.pages)
    required = audit._connection.execute(
        """
        select max(page_number) from foundational_audit_candidates
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if required is None or required[0] is None or int(required[0]) > page_count:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_PDF_PAGE_RANGE_INVALID")
    return VerifiedSourcePdf(path=path, sha256=digest, page_count=page_count), reader


def verify_source_pdf(
    *, audit: FoundationalAuditStore, pdf_path: Path
) -> VerifiedSourcePdf:
    """Require the exact bound PDF bytes and complete candidate page range."""
    verified, _ = _verified_pdf_reader(audit=audit, pdf_path=pdf_path)
    return verified


def _normalized_page_text(reader: PdfReader, page_number: int) -> str:
    try:
        extracted = reader.pages[page_number - 1].extract_text()
    except Exception as error:
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_PAGE_EXTRACTION_FAILED"
        ) from error
    return (extracted or "").replace("\r\n", "\n").replace("\r", "\n")


def bind_candidate_pages(
    *, audit: FoundationalAuditStore, pdf_path: Path
) -> int:
    """Bind every candidate to text parsed from the same verified PDF bytes."""
    _, reader = _verified_pdf_reader(audit=audit, pdf_path=pdf_path)
    candidates = audit._connection.execute(
        """
        select source_id, locator, page_number, page_text_sha256
        from foundational_audit_candidates
        where run_id = ? and audit_id = ?
        order by page_number, source_id, locator
        """,
        (audit.run_id, audit.audit_id),
    ).fetchall()
    bindings = tuple(
        (
            hashlib.sha256(
                _normalized_page_text(reader, int(row["page_number"])).encode("utf-8")
            ).hexdigest(),
            str(row["source_id"]),
            str(row["locator"]),
            row["page_text_sha256"],
        )
        for row in candidates
    )
    if any(
        prior is not None and str(prior) != digest
        for digest, _, _, prior in bindings
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_PAGE_BINDING_MISMATCH")
    expected_by_key = {
        (source_id, locator): digest for digest, source_id, locator, _ in bindings
    }
    if any(
        decision.page_text_sha256
        != expected_by_key.get((decision.source_id, decision.locator))
        for decision in _decision_records(audit)
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_PAGE_BINDING_MISMATCH")
    with audit._connection:
        audit._connection.executemany(
            """
            update foundational_audit_candidates
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


def extract_candidate_page(
    *, audit: FoundationalAuditStore, pdf_path: Path, locator: str
) -> LocatorReviewPacket:
    """Extract one exact bound candidate page without changing audit state."""
    safe_locator = _string(locator, "FOUNDATIONAL_LOCATOR_AUDIT_CANDIDATE_INVALID")
    candidate = audit._connection.execute(
        """
        select source_id, locator, page_number
        from foundational_audit_candidates
        where run_id = ? and audit_id = ? and locator = ?
        """,
        (audit.run_id, audit.audit_id, safe_locator),
    ).fetchone()
    if candidate is None:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_CANDIDATE_INVALID")
    _, reader = _verified_pdf_reader(audit=audit, pdf_path=pdf_path)
    text = _normalized_page_text(reader, int(candidate["page_number"]))
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return LocatorReviewPacket(
        source_id=str(candidate["source_id"]),
        locator=str(candidate["locator"]),
        page_number=int(candidate["page_number"]),
        page_text=text,
        page_text_sha256=text_digest,
    )


def next_unreviewed_locator(audit: FoundationalAuditStore) -> str | None:
    """Return the next candidate locator with no current decision."""
    current = _current_decisions(_decision_records(audit))
    rows = audit._connection.execute(
        """
        select source_id, locator from foundational_audit_candidates
        where run_id = ? and audit_id = ?
        order by page_number, source_id, locator
        """,
        (audit.run_id, audit.audit_id),
    ).fetchall()
    for row in rows:
        key = (str(row["source_id"]), str(row["locator"]))
        if key not in current:
            return key[1]
    return None


def _canonical_jsonl(records: tuple[Mapping[str, object], ...]) -> bytes:
    if not records:
        return b""
    lines = tuple(
        json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for record in records
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _complete_current_decisions(
    audit: FoundationalAuditStore,
) -> tuple[LocatorAuditDecision, ...]:
    status = audit_status(audit)
    if status.status != "COMPLETE" or status.remaining_count != 0:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_INCOMPLETE")
    current = _current_decisions(_decision_records(audit))
    candidate_bindings = {
        (str(row["source_id"]), str(row["locator"])): row["page_text_sha256"]
        for row in audit._connection.execute(
            """
            select source_id, locator, page_text_sha256
            from foundational_audit_candidates
            where run_id = ? and audit_id = ?
            """,
            (audit.run_id, audit.audit_id),
        ).fetchall()
    }
    if len(candidate_bindings) != status.candidate_count or any(
        digest is None
        or current[key].page_text_sha256 != str(digest)
        for key, digest in candidate_bindings.items()
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_PAGE_BINDING_MISMATCH")
    return tuple(
        current[key]
        for key in sorted(current, key=lambda item: (item[0], _page_number(item[1], "FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID")))
    )


def _audit_binding_row(audit: FoundationalAuditStore) -> sqlite3.Row:
    row = audit._connection.execute(
        """
        select snapshot_binding_sha256, source_drive_file_id, source_file_sha256,
               candidate_set_sha256, candidate_count
        from foundational_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (audit.run_id, audit.audit_id),
    ).fetchone()
    if row is None:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_RUN_MISSING")
    return row


def _correction_payload(audit: FoundationalAuditStore) -> bytes:
    decisions = _complete_current_decisions(audit)
    return _canonical_jsonl(tuple(decision.to_mapping() for decision in decisions))


def _binding_payload(audit: FoundationalAuditStore, correction_sha256: str) -> bytes:
    row = _audit_binding_row(audit)
    binding = {
        "schema_version": 2,
        "run_id": audit.run_id,
        "audit_id": audit.audit_id,
        "snapshot_binding_sha256": _digest(
            row["snapshot_binding_sha256"],
            "FOUNDATIONAL_LOCATOR_AUDIT_BINDING_INVALID",
        ),
        "source_drive_file_id": _string(
            row["source_drive_file_id"],
            "FOUNDATIONAL_LOCATOR_AUDIT_BINDING_INVALID",
        ),
        "source_file_sha256": _digest(
            row["source_file_sha256"],
            "FOUNDATIONAL_LOCATOR_AUDIT_BINDING_INVALID",
        ),
        "candidate_set_sha256": _digest(
            row["candidate_set_sha256"],
            "FOUNDATIONAL_LOCATOR_AUDIT_BINDING_INVALID",
        ),
        "candidate_count": int(row["candidate_count"]),
        "correction_payload_sha256": _digest(
            correction_sha256,
            "FOUNDATIONAL_LOCATOR_AUDIT_BINDING_INVALID",
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


def _sealed_directory(audit: FoundationalAuditStore) -> Path:
    return (
        audit.database_path.parent
        / "foundational-locator-audit"
        / AUDIT_ARTIFACT_VERSION
        / "sealed"
    )


def _write_fsynced_file(*, destination: Path, content: bytes) -> None:
    with destination.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _manifest_digest(path: Path) -> str:
    manifest = path.with_name(f"{path.name}.sha256")
    try:
        content = path.read_bytes()
        text = manifest.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_SEAL_VERIFY_FAILED"
        ) from error
    if (
        len(text) != 65
        or not text.endswith("\n")
        or any(character not in "0123456789abcdef" for character in text[:64])
    ):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_MANIFEST_INVALID")
    digest = hashlib.sha256(content).hexdigest()
    if digest != text[:64]:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_DIGEST_MISMATCH")
    return digest


def verify_correction_proposal(
    *, audit: FoundationalAuditStore, expected_binding_sha256: str
) -> AuditSeal:
    """Verify the sealed proposal against its trust anchor and current audit state."""
    trusted = _digest(
        expected_binding_sha256,
        "FOUNDATIONAL_LOCATOR_AUDIT_BINDING_MISMATCH",
    )
    sealed = _sealed_directory(audit)
    correction_path = sealed / CORRECTION_NAME
    binding_path = sealed / AUDIT_BINDING_NAME
    paths = (correction_path, binding_path)
    manifests = tuple(path.with_name(f"{path.name}.sha256") for path in paths)
    presence = tuple(path.is_file() for path in (*paths, *manifests))
    if not any(presence):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SEAL_MISSING")
    if not all(presence):
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SEAL_PARTIAL")
    expected_names = {path.name for path in (*paths, *manifests)}
    try:
        actual_names = {path.name for path in sealed.iterdir()}
    except OSError as error:
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_SEAL_VERIFY_FAILED"
        ) from error
    if actual_names != expected_names:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_SEAL_PARTIAL")
    correction_digest = _manifest_digest(correction_path)
    binding_digest = _manifest_digest(binding_path)
    if binding_digest != trusted:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_BINDING_MISMATCH")
    expected_corrections = _correction_payload(audit)
    if correction_path.read_bytes() != expected_corrections:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_BINDING_MISMATCH")
    expected_binding = _binding_payload(audit, correction_digest)
    if binding_path.read_bytes() != expected_binding:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_BINDING_MISMATCH")
    decisions = tuple(
        LocatorAuditDecision.from_mapping(json.loads(line))
        for line in expected_corrections.decode("utf-8").splitlines()
    )
    return AuditSeal(
        artifacts=(
            AuditArtifact(
                name=CORRECTION_NAME,
                path=correction_path,
                manifest_path=manifests[0],
                sha256=correction_digest,
                record_count=len(decisions),
            ),
            AuditArtifact(
                name=AUDIT_BINDING_NAME,
                path=binding_path,
                manifest_path=manifests[1],
                sha256=binding_digest,
                record_count=1,
            ),
        ),
        binding_sha256=binding_digest,
    )


def seal_correction_proposal(*, audit: FoundationalAuditStore) -> AuditSeal:
    """Atomically publish an immutable proposal without changing index rows."""
    correction = _correction_payload(audit)
    correction_digest = hashlib.sha256(correction).hexdigest()
    binding = _binding_payload(audit, correction_digest)
    binding_digest = hashlib.sha256(binding).hexdigest()
    sealed = _sealed_directory(audit)
    payloads = {
        CORRECTION_NAME: correction,
        AUDIT_BINDING_NAME: binding,
    }
    if sealed.exists():
        if not sealed.is_dir():
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_SEAL_PARTIAL")
        return verify_correction_proposal(
            audit=audit,
            expected_binding_sha256=binding_digest,
        )
    version_root = sealed.parent
    version_root.mkdir(parents=True, exist_ok=True)
    for stale in version_root.glob(".sealed.*.tmp"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    staged = Path(mkdtemp(prefix=".sealed.", suffix=".tmp", dir=version_root))
    try:
        for name, content in payloads.items():
            _write_fsynced_file(destination=staged / name, content=content)
            _write_fsynced_file(
                destination=staged / f"{name}.sha256",
                content=(hashlib.sha256(content).hexdigest() + "\n").encode(
                    "ascii"
                ),
            )
        os.replace(staged, sealed)
        return verify_correction_proposal(
            audit=audit,
            expected_binding_sha256=binding_digest,
        )
    except (FoundationalLocatorAuditError, OSError) as error:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_SEAL_WRITE_FAILED"
        ) from error


def calculate_hypothetical_capacity(
    *, audit: FoundationalAuditStore
) -> HypotheticalCapacityReport:
    """Project current promotions into capacity without mutating inventory."""
    decisions = _complete_current_decisions(audit)
    rows = audit._connection.execute(
        """
        select locator.source_id, locator.locator, locator.topic,
               locator.source_role, source.parser_type
        from index_locators as locator
        join index_sources as source
          on source.run_id = locator.run_id and source.source_id = locator.source_id
        where locator.run_id = ? and locator.e1a4_available = 1
        order by locator.source_id, locator.locator
        """,
        (audit.run_id,),
    ).fetchall()
    parser_rows = audit._connection.execute(
        """
        select source_id, parser_type from index_sources where run_id = ?
        """,
        (audit.run_id,),
    ).fetchall()
    parser_by_source = {str(row["source_id"]): str(row["parser_type"]) for row in parser_rows}
    grouped: dict[tuple[str, str, str, str], list[str]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["source_id"]),
                str(row["topic"]),
                str(row["source_role"]),
                str(row["parser_type"]),
            ),
            [],
        ).append(str(row["locator"]))
    for decision in decisions:
        if decision.decision != "PROMOTE_FOUNDATIONAL":
            continue
        parser_type = parser_by_source.get(decision.source_id)
        if parser_type is None or decision.proposed_topic is None:
            _fail("FOUNDATIONAL_LOCATOR_AUDIT_CAPACITY_INVALID")
        grouped.setdefault(
            (
                decision.source_id,
                decision.proposed_topic,
                "foundational",
                parser_type,
            ),
            [],
        ).append(decision.locator)
    counts = {
        (topic, role): 0
        for topic in ("iron_sulfide", "scale", "corrosion", "paraffin")
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
            for (source_id, topic, role, parser_type), locators in sorted(grouped.items())
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
