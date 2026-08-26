"""Durable, fail-closed review of unclassified foundational PDF locators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Mapping

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
    ReconciliationStore,
    TOPICS,
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
        select 1 from foundational_audit_candidates
        where run_id = ? and audit_id = ? and source_id = ? and locator = ?
        """,
        (audit.run_id, audit.audit_id, safe.source_id, safe.locator),
    ).fetchone()
    if candidate is None:
        _fail("FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID")
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
