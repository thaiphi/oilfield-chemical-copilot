from __future__ import annotations

from pathlib import Path

import pytest


def _reconciliation_with_foundational_candidates(tmp_path: Path):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        IndexLocatorRecord,
        IndexSourceRecord,
        ReconciliationStore,
        import_index_inventory,
    )

    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    store = ReconciliationStore.create(
        root=root,
        expected_root=root,
        run_id="run-001",
        index_contract_sha256="a" * 64,
        e1a3_allocation_sha256="b" * 64,
    )
    source = IndexSourceRecord.from_mapping(
        {
            "source_id": "source-foundational",
            "source_path": "approved/foundational.pdf",
            "parser_type": "pdf",
            "topic": "unassigned",
            "chunk_count": 3,
            "embedding_model": "model",
            "index_contract_sha256": "a" * 64,
            "provenance_drive_file_id": None,
            "content_sha256": None,
        }
    )
    locators = tuple(
        IndexLocatorRecord.from_mapping(
            {
                "source_id": source.source_id,
                "locator": f"page:{page}",
                "topic": "unassigned",
                "source_role": "foundational",
                "substantive_status": "INELIGIBLE",
            }
        )
        for page in range(1, 4)
    )
    import_index_inventory(
        store=store,
        sources=(source,),
        locators=locators,
        expected_source_count=1,
        expected_chunk_count=3,
    )
    return store


def _initialize(tmp_path: Path):
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        initialize_audit,
    )

    store = _reconciliation_with_foundational_candidates(tmp_path)
    audit = initialize_audit(
        store=store,
        audit_id="foundational-locator-audit-v1",
        snapshot_binding_sha256="c" * 64,
        source_drive_file_id="drive-source-1",
        source_file_sha256="d" * 64,
    )
    store.close()
    return audit


def _decision(*, locator: str, decision: str = "KEEP_INELIGIBLE"):
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        LocatorAuditDecision,
    )

    promoted = decision == "PROMOTE_FOUNDATIONAL"
    return LocatorAuditDecision.from_mapping(
        {
            "decision_id": f"decision-{locator.replace(':', '-')}",
            "source_id": "source-foundational",
            "locator": locator,
            "decision": decision,
            "proposed_topic": "scale" if promoted else None,
            "reason_code": (
                "SUBSTANTIVE_TARGET_EVIDENCE" if promoted else "NO_TARGET_TOPIC"
            ),
            "page_text_sha256": "e" * 64,
            "reviewer_id": "reviewer-1",
            "supersedes_decision_id": None,
            "decided_at": "2026-08-26T00:00:00Z",
        }
    )


def test_initialize_audit_binds_exact_candidate_set(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        audit_status,
    )

    audit = _initialize(tmp_path)

    status = audit_status(audit)

    assert status.status == "IN_PROGRESS"
    assert status.candidate_count == 3
    assert status.current_decision_count == 0
    assert status.remaining_count == 3
    audit.close()


def test_initialize_audit_rejects_binding_change(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalLocatorAuditError,
        initialize_audit,
    )

    audit = _initialize(tmp_path)
    audit.close()
    store = _reconciliation_with_foundational_candidates(tmp_path)

    with pytest.raises(
        FoundationalLocatorAuditError,
        match="FOUNDATIONAL_LOCATOR_AUDIT_BINDING_CONFLICT",
    ):
        initialize_audit(
            store=store,
            audit_id="foundational-locator-audit-v1",
            snapshot_binding_sha256="c" * 64,
            source_drive_file_id="drive-source-1",
            source_file_sha256="f" * 64,
        )

    store.close()


def test_record_locator_decision_commits_one_append_only_transaction(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalAuditStore,
        audit_status,
        record_locator_decision,
    )

    audit = _initialize(tmp_path)

    result = record_locator_decision(
        audit=audit,
        record=_decision(locator="page:1"),
    )

    assert result.current_decision_count == 1
    database_path = audit.database_path
    run_id = audit.run_id
    audit_id = audit.audit_id
    audit.close()
    reopened = FoundationalAuditStore.open(
        database_path=database_path,
        run_id=run_id,
        audit_id=audit_id,
    )
    assert audit_status(reopened).current_decision_count == 1
    reopened.close()


def test_record_locator_decision_rejects_unknown_locator(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalLocatorAuditError,
        record_locator_decision,
    )

    audit = _initialize(tmp_path)

    with pytest.raises(
        FoundationalLocatorAuditError,
        match="FOUNDATIONAL_LOCATOR_AUDIT_DECISION_INVALID",
    ):
        record_locator_decision(
            audit=audit,
            record=_decision(locator="page:4"),
        )

    audit.close()
