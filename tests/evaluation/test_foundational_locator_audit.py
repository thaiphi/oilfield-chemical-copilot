from __future__ import annotations

import hashlib
import importlib.util
import json
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


def _initialize(tmp_path: Path, *, source_file_sha256: str = "d" * 64):
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        initialize_audit,
    )

    store = _reconciliation_with_foundational_candidates(tmp_path)
    audit = initialize_audit(
        store=store,
        audit_id="foundational-locator-audit-v1",
        snapshot_binding_sha256="c" * 64,
        source_drive_file_id="drive-source-1",
        source_file_sha256=source_file_sha256,
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


def _three_page_pdf(tmp_path: Path) -> Path:
    from pypdf import PdfWriter

    path = tmp_path / "source.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _runner_module():
    path = Path(__file__).resolve().parents[2] / "eval" / "audit_foundational_locators.py"
    spec = importlib.util.spec_from_file_location("audit_foundational_locators", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_candidate_page_requires_bound_pdf_and_exact_locator(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        extract_candidate_page,
    )

    pdf = _three_page_pdf(tmp_path)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    audit = _initialize(tmp_path, source_file_sha256=digest)

    packet = extract_candidate_page(
        audit=audit,
        pdf_path=pdf,
        locator="page:2",
    )

    assert packet.page_number == 2
    assert packet.locator == "page:2"
    assert packet.page_text == ""
    assert packet.page_text_sha256 == hashlib.sha256(b"").hexdigest()
    audit.close()


def test_extract_candidate_page_rejects_pdf_hash_mismatch(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalLocatorAuditError,
        extract_candidate_page,
    )

    pdf = _three_page_pdf(tmp_path)
    audit = _initialize(tmp_path)

    with pytest.raises(
        FoundationalLocatorAuditError,
        match="FOUNDATIONAL_LOCATOR_AUDIT_PDF_MISMATCH",
    ):
        extract_candidate_page(audit=audit, pdf_path=pdf, locator="page:2")

    audit.close()


def test_cli_status_never_prints_private_identifiers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit = _initialize(tmp_path)
    root = audit.database_path.parent
    audit.close()
    runner = _runner_module()

    assert (
        runner.cli(
            [
                "status",
                "--private-root",
                str(root),
                "--run-id",
                "run-001",
                "--audit-id",
                "foundational-locator-audit-v1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "IN_PROGRESS",
        "candidate_count": 3,
        "current_decision_count": 0,
        "remaining_count": 3,
    }


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


def _completed_audit(tmp_path: Path):
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        record_locator_decision,
    )

    audit = _initialize(tmp_path)
    record_locator_decision(
        audit=audit,
        record=_decision(locator="page:1", decision="PROMOTE_FOUNDATIONAL"),
    )
    record_locator_decision(audit=audit, record=_decision(locator="page:2"))
    record_locator_decision(audit=audit, record=_decision(locator="page:3"))
    return audit


def _locator_rows(audit) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in audit._connection.execute(
            """
            select source_id, locator, topic, source_role,
                   substantive_status, e1a3_used, e1a4_available
            from index_locators where run_id = ?
            order by source_id, locator
            """,
            (audit.run_id,),
        ).fetchall()
    )


def test_seal_requires_one_current_closed_decision_per_candidate(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalLocatorAuditError,
        seal_correction_proposal,
    )

    audit = _initialize(tmp_path)

    with pytest.raises(
        FoundationalLocatorAuditError,
        match="FOUNDATIONAL_LOCATOR_AUDIT_INCOMPLETE",
    ):
        seal_correction_proposal(audit=audit)

    audit.close()


def test_correction_proposal_is_canonical_complete_and_idempotent(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        seal_correction_proposal,
        verify_correction_proposal,
    )

    audit = _completed_audit(tmp_path)

    sealed = seal_correction_proposal(audit=audit)
    verified = verify_correction_proposal(
        audit=audit,
        expected_binding_sha256=sealed.binding_sha256,
    )
    resealed = seal_correction_proposal(audit=audit)

    assert len(sealed.artifacts) == 2
    assert all(artifact.manifest_path.is_file() for artifact in sealed.artifacts)
    assert verified == sealed
    assert resealed == sealed
    audit.close()


def test_hypothetical_capacity_does_not_update_index_locators(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        calculate_hypothetical_capacity,
    )

    audit = _completed_audit(tmp_path)
    before = _locator_rows(audit)

    report = calculate_hypothetical_capacity(audit=audit)

    assert _locator_rows(audit) == before
    assert len(report.strata) == 8
    assert report.all_sufficient is False
    assert report.allocation_available is False
    assert report.allocation_count == 0
    audit.close()


def test_cli_seal_verify_and_capacity_outputs_are_aggregate_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit = _completed_audit(tmp_path)
    root = audit.database_path.parent
    audit.close()
    common = [
        "--private-root",
        str(root),
        "--run-id",
        "run-001",
        "--audit-id",
        "foundational-locator-audit-v1",
    ]
    runner = _runner_module()

    assert runner.cli(["seal", *common]) == 0
    sealed = json.loads(capsys.readouterr().out)
    assert sealed == {
        "status": "SEALED",
        "artifact_count": 2,
        "manifest_count": 2,
    }
    binding_manifest = (
        root
        / "foundational-locator-audit"
        / "v1"
        / "sealed"
        / "audit-binding.v1.json.sha256"
    )
    trusted = binding_manifest.read_text(encoding="ascii").strip()
    assert runner.cli(["verify", *common, "--expected-binding-sha256", trusted]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified == {
        "status": "VERIFIED",
        "artifact_count": 2,
        "manifest_count": 2,
    }
    assert runner.cli(["capacity", *common]) == 0
    capacity = json.loads(capsys.readouterr().out)
    assert capacity == {
        "status": "INSUFFICIENT",
        "all_sufficient": False,
        "allocation_available": False,
        "allocation_count": 0,
        "sufficient_strata_count": 0,
    }
