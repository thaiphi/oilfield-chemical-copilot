"""Private, no-write foundational-role audit for Iron Sulfide supplements."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import sqlite3
from typing import Mapping

from pypdf import PdfReader

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
    CorpusReconciliationError,
    ReconciliationStore,
    verify_reconciliation_snapshots,
)


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


@dataclass(frozen=True)
class SupplementAuditStatus:
    status: str
    source_count: int
    candidate_count: int
    reviewed_count: int
    promotion_count: int
    remaining_count: int
    needs_second_review_count: int


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
    _create_schema(store._connection)
    existing = store._connection.execute(
        """
        select snapshot_binding_sha256, source_set_sha256, candidate_set_sha256,
               source_count, candidate_count, promotion_target
        from iron_sulfide_supplement_audit_runs
        where run_id = ? and audit_id = ?
        """,
        (store.run_id, safe_audit_id),
    ).fetchone()
    expected = (
        binding_digest,
        source_digest,
        candidate_digest,
        len(source_payload),
        len(candidate_payload),
        target,
    )
    if existing is not None and tuple(existing) != expected:
        _fail("IRON_SULFIDE_SUPPLEMENT_BINDING_CONFLICT")
    with store._connection:
        store._connection.execute(
            """
            insert into iron_sulfide_supplement_audit_runs(
                run_id, audit_id, snapshot_binding_sha256, source_set_sha256,
                candidate_set_sha256, source_count, candidate_count,
                promotion_target, status
            ) values (?, ?, ?, ?, ?, ?, ?, ?, 'IN_PROGRESS')
            on conflict(run_id, audit_id) do nothing
            """,
            (store.run_id, safe_audit_id, *expected),
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
    return SupplementAuditStatus(
        status=str(row["status"]),
        source_count=int(row["source_count"]),
        candidate_count=int(row["candidate_count"]),
        reviewed_count=0,
        promotion_count=0,
        remaining_count=int(row["candidate_count"]),
        needs_second_review_count=0,
    )


def bind_supplement_pages(
    *, audit: IronSulfideSupplementAuditStore, source_root: Path
) -> int:
    """Bind every candidate page to text from the same verified PDF bytes."""
    resolved_root = source_root.resolve()
    if not resolved_root.is_dir():
        _fail("IRON_SULFIDE_SUPPLEMENT_SOURCE_ROOT_INVALID")
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
