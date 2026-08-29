from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


def _reconciliation_with_foundational_candidates(
    tmp_path: Path, *, source_pdf: Path
):
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
    approved = tmp_path / "approved"
    approved.mkdir(exist_ok=True)
    local_pdf = approved / "foundational.pdf"
    shutil.copyfile(source_pdf, local_pdf)
    inventory_local_files(store=store, roots=(approved,))
    local_relative_path = str(
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
                    "drive_file_id": "drive-source-1",
                    "name": "foundational.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": source_pdf.stat().st_size,
                    "checksum_algorithm": None,
                    "checksum": None,
                    "modified_time": "2026-08-26T00:00:00Z",
                    "parent_ids": ["approved-folder"],
                }
            ),
        ),
        next_page_token=None,
    )
    source = IndexSourceRecord.from_mapping(
        {
            "source_id": "source-foundational",
            "source_path": local_relative_path,
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
    reconcile_document_matches(store=store)
    record_review_decision(
        store=store,
        record=ReviewDecisionRecord.from_mapping(
            {
                "decision_id": "accept-drive-source-1",
                "match_key": "drive:drive-source-1",
                "decision": "ACCEPT",
                "reviewer_id": "reviewer-1",
                "reason_code": "REVIEWED",
                "supersedes_decision_id": None,
                "decided_at": "2026-08-26T00:00:00Z",
            }
        ),
    )
    calculate_locator_capacity(store=store, prior_locator_keys=())
    dry_run = dry_run_e1a4_allocation(store=store, prior_locator_keys=())
    assert dry_run.status == "BLOCKED"
    sealed = seal_reconciliation_snapshots(store=store, root=store.root)
    trusted_binding_sha256 = next(
        artifact.sha256
        for artifact in sealed.artifacts
        if artifact.name == "snapshot-binding.json"
    )
    return store, trusted_binding_sha256


def _initialize(
    tmp_path: Path,
    *,
    source_file_sha256: str | None = None,
    bind_pages: bool = True,
):
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        bind_candidate_pages,
        initialize_audit,
    )

    pdf = _three_page_pdf(tmp_path)
    digest = source_file_sha256 or hashlib.sha256(pdf.read_bytes()).hexdigest()
    store, trusted_binding_sha256 = _reconciliation_with_foundational_candidates(
        tmp_path,
        source_pdf=pdf,
    )
    audit = initialize_audit(
        store=store,
        audit_id="foundational-locator-audit-v1",
        snapshot_binding_sha256=trusted_binding_sha256,
        source_drive_file_id="drive-source-1",
        source_file_sha256=digest,
    )
    if bind_pages:
        bind_candidate_pages(audit=audit, pdf_path=pdf)
    store.close()
    return audit


def _decision(
    *,
    locator: str,
    decision: str = "KEEP_INELIGIBLE",
    page_text_sha256: str | None = None,
    supersedes_decision_id: str | None = None,
    decision_id: str | None = None,
):
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        LocatorAuditDecision,
    )

    promoted = decision == "PROMOTE_FOUNDATIONAL"
    return LocatorAuditDecision.from_mapping(
        {
            "decision_id": decision_id or f"decision-{locator.replace(':', '-')}",
            "source_id": "source-foundational",
            "locator": locator,
            "decision": decision,
            "proposed_topic": "scale" if promoted else None,
            "reason_code": (
                "SUBSTANTIVE_TARGET_EVIDENCE" if promoted else "NO_TARGET_TOPIC"
            ),
            "page_text_sha256": page_text_sha256 or hashlib.sha256(b"").hexdigest(),
            "reviewer_id": "reviewer-1",
            "supersedes_decision_id": supersedes_decision_id,
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
    with pdf.open("ab") as stream:
        stream.write(b"untrusted")

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


def test_initialize_audit_rejects_source_hash_change(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalLocatorAuditError,
        initialize_audit,
    )

    audit = _initialize(tmp_path)
    audit.close()
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        ReconciliationStore,
    )
    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    store = ReconciliationStore.open(
        root=root,
        expected_root=root,
        run_id="run-001",
    )
    trusted_binding_sha256 = hashlib.sha256(
        (root / "snapshots" / "snapshot-binding.json").read_bytes()
    ).hexdigest()

    with pytest.raises(
        FoundationalLocatorAuditError,
        match="FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID",
    ):
        initialize_audit(
            store=store,
            audit_id="foundational-locator-audit-v1",
            snapshot_binding_sha256=trusted_binding_sha256,
            source_drive_file_id="drive-source-1",
            source_file_sha256="f" * 64,
        )

    store.close()


def test_initialize_audit_authenticates_trusted_active_reconciliation_seal(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalLocatorAuditError,
        initialize_audit,
    )

    pdf = _three_page_pdf(tmp_path)
    store, _ = _reconciliation_with_foundational_candidates(
        tmp_path, source_pdf=pdf
    )

    with pytest.raises(
        FoundationalLocatorAuditError,
        match="FOUNDATIONAL_LOCATOR_AUDIT_RECONCILIATION_UNTRUSTED",
    ):
        initialize_audit(
            store=store,
            audit_id="foundational-locator-audit-v1",
            snapshot_binding_sha256="c" * 64,
            source_drive_file_id="drive-source-1",
            source_file_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        )

    store.close()


def test_initialize_audit_rejects_unrelated_pdf_bytes_for_trusted_drive_record(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalLocatorAuditError,
        initialize_audit,
    )

    pdf = _three_page_pdf(tmp_path)
    store, trusted_binding_sha256 = _reconciliation_with_foundational_candidates(
        tmp_path, source_pdf=pdf
    )

    with pytest.raises(
        FoundationalLocatorAuditError,
        match="FOUNDATIONAL_LOCATOR_AUDIT_SOURCE_PROVENANCE_INVALID",
    ):
        initialize_audit(
            store=store,
            audit_id="foundational-locator-audit-v1",
            snapshot_binding_sha256=trusted_binding_sha256,
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


def test_record_locator_decision_rejects_page_hash_not_bound_to_verified_pdf(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        FoundationalLocatorAuditError,
        record_locator_decision,
    )

    audit = _initialize(tmp_path)

    with pytest.raises(
        FoundationalLocatorAuditError,
        match="FOUNDATIONAL_LOCATOR_AUDIT_PAGE_BINDING_MISMATCH",
    ):
        record_locator_decision(
            audit=audit,
            record=_decision(locator="page:1", page_text_sha256="e" * 64),
        )

    audit.close()


def test_second_review_resolution_is_append_only_and_requires_current_supersession(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        audit_status,
        record_locator_decision,
    )

    audit = _initialize(tmp_path)
    first = _decision(locator="page:1", decision_id="review-page-1")
    first = first.__class__.from_mapping(
        {
            **first.to_mapping(),
            "decision": "NEEDS_SECOND_REVIEW",
            "reason_code": "AMBIGUOUS_OR_NONEXTRACTABLE",
        }
    )
    record_locator_decision(audit=audit, record=first)
    assert audit_status(audit).needs_second_review_count == 1

    record_locator_decision(
        audit=audit,
        record=_decision(
            locator="page:1",
            decision_id="resolved-page-1",
            supersedes_decision_id="review-page-1",
        ),
    )

    assert audit_status(audit).needs_second_review_count == 0
    assert (
        audit._connection.execute(
            "select count(*) from foundational_audit_decisions where locator = ?",
            ("page:1",),
        ).fetchone()[0]
        == 2
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
    binding = json.loads(
        next(
            artifact.path.read_text(encoding="utf-8")
            for artifact in sealed.artifacts
            if artifact.name == "audit-binding.v2.json"
        )
    )
    assert binding["source_drive_file_id"] == "drive-source-1"
    audit.close()


def test_correction_proposal_publish_is_single_rename_and_recovers_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oilfield_chemical_copilot.evaluation.foundational_locator_audit as module

    audit = _completed_audit(tmp_path)
    real_replace = module.os.replace

    with monkeypatch.context() as patch:
        patch.setattr(
            module.os,
            "replace",
            lambda _source, _destination: (_ for _ in ()).throw(OSError("stop")),
        )
        with pytest.raises(
            module.FoundationalLocatorAuditError,
            match="FOUNDATIONAL_LOCATOR_AUDIT_SEAL_WRITE_FAILED",
        ):
            module.seal_correction_proposal(audit=audit)

    version_root = (
        audit.database_path.parent / "foundational-locator-audit" / "v2"
    )
    assert not (version_root / "sealed").exists()
    assert not tuple(version_root.glob(".sealed.*.tmp"))
    monkeypatch.setattr(module.os, "replace", real_replace)
    assert len(module.seal_correction_proposal(audit=audit).artifacts) == 2
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
    binding_path = (
        root
        / "foundational-locator-audit"
        / "v2"
        / "sealed"
        / "audit-binding.v2.json"
    )
    trusted = hashlib.sha256(binding_path.read_bytes()).hexdigest()
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


def test_cli_sanitizes_unexpected_operational_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    root.mkdir(parents=True)
    private_path = root / "secret.pdf"
    monkeypatch.setattr(
        runner,
        "_open_audit",
        lambda _args: (_ for _ in ()).throw(OSError(str(private_path))),
    )

    assert runner.cli(["status", "--private-root", str(root)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(private_path) not in captured.err
    assert json.loads(captured.err) == {
        "status": "FOUNDATIONAL_LOCATOR_AUDIT_BLOCKED",
        "error_code": "FOUNDATIONAL_LOCATOR_AUDIT_OPERATION_FAILED",
    }
