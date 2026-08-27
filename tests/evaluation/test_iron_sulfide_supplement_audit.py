from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _pdf(path: Path, *, pages: int = 6) -> Path:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _sealed_reconciliation(tmp_path: Path):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        DriveFileRecord,
        IndexLocatorRecord,
        IndexSourceRecord,
        ReconciliationStore,
        ReviewDecisionRecord,
        calculate_locator_capacity,
        dry_run_e1a4_allocation,
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        record_review_decision,
        reconcile_document_matches,
        seal_reconciliation_snapshots,
    )

    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    store = ReconciliationStore.create(
        root=root,
        expected_root=root,
        run_id="run-001",
        index_contract_sha256="a" * 64,
        e1a3_allocation_sha256="b" * 64,
    )
    handouts = tmp_path / "approved-handouts"
    source_root = handouts / "1.0 Iron Sulfide"
    pdf = _pdf(source_root / "supplement.pdf")
    inventory_local_files(store=store, roots=(handouts,))
    relative_path = str(
        store._connection.execute(
            "select relative_path from local_files where run_id = ?",
            (store.run_id,),
        ).fetchone()[0]
    )
    import_drive_page(
        store=store,
        records=(
            DriveFileRecord.from_mapping(
                {
                    "drive_file_id": "drive-supplement-1",
                    "name": "supplement.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": pdf.stat().st_size,
                    "checksum_algorithm": None,
                    "checksum": None,
                    "modified_time": "2026-08-27T00:00:00Z",
                    "parent_ids": ["iron-folder"],
                }
            ),
        ),
        next_page_token=None,
    )
    source = IndexSourceRecord.from_mapping(
        {
            "source_id": "source-supplement-1",
            "source_path": relative_path,
            "parser_type": "pdf",
            "topic": "iron_sulfide",
            "chunk_count": 6,
            "embedding_model": "model",
            "index_contract_sha256": "a" * 64,
            "provenance_drive_file_id": None,
            "content_sha256": None,
        }
    )
    definitions = (
        ("page:1", "iron_sulfide", "supporting", "SUBSTANTIVE"),
        ("page:2", "iron_sulfide", "supporting", "SUBSTANTIVE"),
        ("page:3", "iron_sulfide", "supporting", "SUBSTANTIVE"),
        ("page:4", "iron_sulfide", "supporting", "INELIGIBLE"),
        ("page:5", "scale", "supporting", "SUBSTANTIVE"),
        ("page:6", "iron_sulfide", "foundational", "SUBSTANTIVE"),
    )
    import_index_inventory(
        store=store,
        sources=(source,),
        locators=tuple(
            IndexLocatorRecord.from_mapping(
                {
                    "source_id": source.source_id,
                    "locator": locator,
                    "topic": topic,
                    "source_role": role,
                    "substantive_status": status,
                }
            )
            for locator, topic, role, status in definitions
        ),
        expected_source_count=1,
        expected_chunk_count=6,
    )
    reconcile_document_matches(store=store)
    record_review_decision(
        store=store,
        record=ReviewDecisionRecord.from_mapping(
            {
                "decision_id": "accept-supplement",
                "match_key": "drive:drive-supplement-1",
                "decision": "ACCEPT",
                "reviewer_id": "reviewer-1",
                "reason_code": "REVIEWED",
                "supersedes_decision_id": None,
                "decided_at": "2026-08-27T00:00:00Z",
            }
        ),
    )
    calculate_locator_capacity(store=store, prior_locator_keys=())
    assert dry_run_e1a4_allocation(store=store, prior_locator_keys=()).status == "BLOCKED"
    sealed = seal_reconciliation_snapshots(store=store, root=root)
    binding = next(
        artifact.sha256
        for artifact in sealed.artifacts
        if artifact.name == "snapshot-binding.json"
    )
    return store, binding, source_root


def test_initialize_freezes_only_fresh_substantive_iron_sulfide_supporting_pages(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        initialize_supplement_audit,
        supplement_audit_status,
    )

    store, binding, source_root = _sealed_reconciliation(tmp_path)

    audit = initialize_supplement_audit(
        store=store,
        audit_id="iron-sulfide-supplement-audit-v1",
        snapshot_binding_sha256=binding,
        source_root=source_root,
        promotion_target=7,
    )

    status = supplement_audit_status(audit)
    assert status.candidate_count == 3
    assert status.source_count == 1
    assert status.status == "IN_PROGRESS"
    audit.close()
    store.close()


def test_initialize_rejects_same_name_pdf_with_untrusted_bytes(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        IronSulfideSupplementAuditError,
        initialize_supplement_audit,
    )

    store, binding, source_root = _sealed_reconciliation(tmp_path)
    pdf = source_root / "supplement.pdf"
    pdf.write_bytes(pdf.read_bytes() + b"untrusted")

    with pytest.raises(
        IronSulfideSupplementAuditError,
        match="IRON_SULFIDE_SUPPLEMENT_SOURCE_PROVENANCE_INVALID",
    ):
        initialize_supplement_audit(
            store=store,
            audit_id="iron-sulfide-supplement-audit-v1",
            snapshot_binding_sha256=binding,
            source_root=source_root,
            promotion_target=7,
        )

    store.close()


def test_initialize_rejects_untrusted_reconciliation_binding(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        IronSulfideSupplementAuditError,
        initialize_supplement_audit,
    )

    store, _, source_root = _sealed_reconciliation(tmp_path)

    with pytest.raises(
        IronSulfideSupplementAuditError,
        match="IRON_SULFIDE_SUPPLEMENT_RECONCILIATION_UNTRUSTED",
    ):
        initialize_supplement_audit(
            store=store,
            audit_id="iron-sulfide-supplement-audit-v1",
            snapshot_binding_sha256=hashlib.sha256(b"wrong").hexdigest(),
            source_root=source_root,
            promotion_target=7,
        )

    store.close()


def test_bind_supplement_pages_persists_every_verified_page_digest(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        bind_supplement_pages,
        initialize_supplement_audit,
    )

    store, binding, source_root = _sealed_reconciliation(tmp_path)
    audit = initialize_supplement_audit(
        store=store,
        audit_id="iron-sulfide-supplement-audit-v1",
        snapshot_binding_sha256=binding,
        source_root=source_root,
        promotion_target=7,
    )

    assert bind_supplement_pages(audit=audit, source_root=source_root) == 3
    assert (
        audit._connection.execute(
            """
            select count(*) from iron_sulfide_supplement_audit_candidates
            where page_text_sha256 is not null
            """
        ).fetchone()[0]
        == 3
    )
    audit.close()


def test_bind_supplement_pages_rejects_pdf_changed_after_initialization(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        IronSulfideSupplementAuditError,
        bind_supplement_pages,
        initialize_supplement_audit,
    )

    store, binding, source_root = _sealed_reconciliation(tmp_path)
    audit = initialize_supplement_audit(
        store=store,
        audit_id="iron-sulfide-supplement-audit-v1",
        snapshot_binding_sha256=binding,
        source_root=source_root,
        promotion_target=7,
    )
    pdf = source_root / "supplement.pdf"
    pdf.write_bytes(pdf.read_bytes() + b"changed")

    with pytest.raises(
        IronSulfideSupplementAuditError,
        match="IRON_SULFIDE_SUPPLEMENT_PAGE_BINDING_MISMATCH",
    ):
        bind_supplement_pages(audit=audit, source_root=source_root)

    audit.close()
