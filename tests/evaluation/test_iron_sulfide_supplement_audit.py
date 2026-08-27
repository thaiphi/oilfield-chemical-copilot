from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


def _runner_module():
    script = Path(__file__).parents[2] / "eval" / "audit_iron_sulfide_supplement.py"
    spec = importlib.util.spec_from_file_location("audit_iron_sulfide_supplement", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        ("page:4", "iron_sulfide", "foundational", "INELIGIBLE"),
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


def _initialized_audit(tmp_path: Path, *, promotion_target: int = 7):
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
        promotion_target=promotion_target,
    )
    bind_supplement_pages(audit=audit, source_root=source_root)
    return audit, source_root


def _decision(
    *,
    locator: str,
    decision: str = "KEEP_SUPPORTING",
    decision_id: str | None = None,
    supersedes_decision_id: str | None = None,
):
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        SupplementLocatorDecision,
    )

    reason = {
        "PROMOTE_FOUNDATIONAL": "GENERALIZABLE_FOUNDATIONAL_EVIDENCE",
        "KEEP_SUPPORTING": "CASE_OR_APPLICATION_SPECIFIC",
        "NEEDS_SECOND_REVIEW": "AMBIGUOUS_FOUNDATIONAL_ROLE",
    }[decision]
    return SupplementLocatorDecision.from_mapping(
        {
            "decision_id": decision_id or f"decision-{locator.replace(':', '-')}",
            "source_id": "source-supplement-1",
            "locator": locator,
            "decision": decision,
            "reason_code": reason,
            "page_text_sha256": hashlib.sha256(b"").hexdigest(),
            "reviewer_id": "reviewer-1",
            "supersedes_decision_id": supersedes_decision_id,
            "decided_at": "2026-08-27T00:00:00Z",
        }
    )


def test_decisions_must_follow_frozen_candidate_order(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        IronSulfideSupplementAuditError,
        record_supplement_decision,
    )

    audit, _ = _initialized_audit(tmp_path)

    with pytest.raises(
        IronSulfideSupplementAuditError,
        match="IRON_SULFIDE_SUPPLEMENT_DECISION_OUT_OF_ORDER",
    ):
        record_supplement_decision(audit=audit, record=_decision(locator="page:2"))

    audit.close()


def test_target_promotions_close_review_without_using_unreviewed_suffix(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        next_supplement_candidate,
        record_supplement_decision,
        supplement_audit_status,
    )

    audit, _ = _initialized_audit(tmp_path, promotion_target=2)
    record_supplement_decision(
        audit=audit,
        record=_decision(locator="page:1", decision="PROMOTE_FOUNDATIONAL"),
    )
    record_supplement_decision(
        audit=audit,
        record=_decision(locator="page:2", decision="PROMOTE_FOUNDATIONAL"),
    )

    status = supplement_audit_status(audit)
    assert status.status == "TARGET_MET"
    assert status.reviewed_count == 2
    assert status.promotion_count == 2
    assert status.remaining_count == 1
    assert next_supplement_candidate(audit) is None
    audit.close()


def test_second_review_blocks_progress_until_append_only_resolution(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        next_supplement_candidate,
        record_supplement_decision,
        supplement_audit_status,
    )

    audit, _ = _initialized_audit(tmp_path)
    record_supplement_decision(
        audit=audit,
        record=_decision(
            locator="page:1",
            decision="NEEDS_SECOND_REVIEW",
            decision_id="review-page-1",
        ),
    )
    assert supplement_audit_status(audit).needs_second_review_count == 1
    assert next_supplement_candidate(audit).locator == "page:1"

    record_supplement_decision(
        audit=audit,
        record=_decision(
            locator="page:1",
            decision_id="resolve-page-1",
            supersedes_decision_id="review-page-1",
        ),
    )

    assert supplement_audit_status(audit).needs_second_review_count == 0
    assert next_supplement_candidate(audit).locator == "page:2"
    assert (
        audit._connection.execute(
            "select count(*) from iron_sulfide_supplement_audit_decisions"
        ).fetchone()[0]
        == 2
    )
    audit.close()


def test_extract_supplement_page_returns_only_the_bound_next_page(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        extract_supplement_page,
        next_supplement_candidate,
    )

    audit, source_root = _initialized_audit(tmp_path)
    candidate = next_supplement_candidate(audit)

    packet = extract_supplement_page(
        audit=audit,
        source_root=source_root,
        source_id=candidate.source_id,
        locator=candidate.locator,
    )

    assert packet.source_id == "source-supplement-1"
    assert packet.locator == "page:1"
    assert packet.page_number == 1
    assert packet.page_text == ""
    assert packet.page_text_sha256 == hashlib.sha256(b"").hexdigest()
    audit.close()


def test_cli_next_and_record_keep_private_packet_out_of_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, source_root = _initialized_audit(tmp_path)
    root = audit.database_path.parent
    audit.close()
    runner = _runner_module()
    common = [
        "--private-root",
        str(root),
        "--run-id",
        "run-001",
        "--audit-id",
        "iron-sulfide-supplement-audit-v1",
    ]
    packet = root / "iron-sulfide-supplement-audit" / "v1" / "packet.json"

    assert (
        runner.cli(
            [
                "next",
                *common,
                "--source-root",
                str(source_root),
                "--packet-output",
                str(packet),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == {
        "status": "IN_PROGRESS",
        "source_count": 1,
        "candidate_count": 3,
        "reviewed_count": 0,
        "promotion_count": 0,
        "remaining_count": 3,
        "needs_second_review_count": 0,
    }
    private_packet = json.loads(packet.read_text(encoding="utf-8"))
    assert private_packet["locator"] == "page:1"
    assert private_packet["source_id"] == "source-supplement-1"

    monkeypatch.setattr(
        runner.sys,
        "stdin",
        io.StringIO(json.dumps(_decision(locator="page:1").to_mapping())),
    )
    assert runner.cli(["record", *common]) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["reviewed_count"] == 1
    assert recorded["remaining_count"] == 2
    assert "source_id" not in recorded
    assert "locator" not in recorded


def test_cli_sanitizes_unexpected_private_path_errors(
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
        "status": "IRON_SULFIDE_SUPPLEMENT_AUDIT_BLOCKED",
        "error_code": "IRON_SULFIDE_SUPPLEMENT_OPERATION_FAILED",
    }


def _closed_audits(tmp_path: Path):
    from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
        LocatorAuditDecision,
        bind_candidate_pages,
        initialize_audit,
        record_locator_decision,
        seal_correction_proposal,
    )
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        bind_supplement_pages,
        initialize_supplement_audit,
        record_supplement_decision,
        seal_supplement_proposal,
    )

    store, binding, source_root = _sealed_reconciliation(tmp_path)
    pdf = source_root / "supplement.pdf"
    core = initialize_audit(
        store=store,
        audit_id="foundational-locator-audit-v1",
        snapshot_binding_sha256=binding,
        source_drive_file_id="drive-supplement-1",
        source_file_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
    )
    bind_candidate_pages(audit=core, pdf_path=pdf)
    record_locator_decision(
        audit=core,
        record=LocatorAuditDecision.from_mapping(
            {
                "decision_id": "core-page-4",
                "source_id": "source-supplement-1",
                "locator": "page:4",
                "decision": "PROMOTE_FOUNDATIONAL",
                "proposed_topic": "iron_sulfide",
                "reason_code": "SUBSTANTIVE_TARGET_EVIDENCE",
                "page_text_sha256": hashlib.sha256(b"").hexdigest(),
                "reviewer_id": "reviewer-1",
                "supersedes_decision_id": None,
                "decided_at": "2026-08-27T00:00:00Z",
            }
        ),
    )
    core_seal = seal_correction_proposal(audit=core)

    supplement = initialize_supplement_audit(
        store=store,
        audit_id="iron-sulfide-supplement-audit-v1",
        snapshot_binding_sha256=binding,
        source_root=source_root,
        promotion_target=2,
    )
    bind_supplement_pages(audit=supplement, source_root=source_root)
    for page in (1, 2):
        record_supplement_decision(
            audit=supplement,
            record=_decision(
                locator=f"page:{page}", decision="PROMOTE_FOUNDATIONAL"
            ),
        )
    supplement_seal = seal_supplement_proposal(
        audit=supplement,
        core_binding_sha256=core_seal.binding_sha256,
    )
    return core, supplement, core_seal, supplement_seal


def test_supplement_seal_requires_a_registered_terminal_condition(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        IronSulfideSupplementAuditError,
        record_supplement_decision,
        seal_supplement_proposal,
    )

    audit, _ = _initialized_audit(tmp_path, promotion_target=2)
    record_supplement_decision(
        audit=audit,
        record=_decision(locator="page:1", decision="PROMOTE_FOUNDATIONAL"),
    )

    with pytest.raises(
        IronSulfideSupplementAuditError,
        match="IRON_SULFIDE_SUPPLEMENT_AUDIT_INCOMPLETE",
    ):
        seal_supplement_proposal(audit=audit, core_binding_sha256="c" * 64)

    audit.close()


def test_supplement_proposal_is_atomic_complete_and_independently_verifiable(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        seal_supplement_proposal,
        verify_supplement_proposal,
    )

    core, supplement, core_seal, sealed = _closed_audits(tmp_path)
    verified = verify_supplement_proposal(
        audit=supplement,
        expected_binding_sha256=sealed.binding_sha256,
        expected_core_binding_sha256=core_seal.binding_sha256,
    )
    resealed = seal_supplement_proposal(
        audit=supplement,
        core_binding_sha256=core_seal.binding_sha256,
    )

    assert len(sealed.artifacts) == 2
    assert all(artifact.manifest_path.is_file() for artifact in sealed.artifacts)
    assert verified == sealed
    assert resealed == sealed
    supplement.close()


def test_supplement_seal_publish_failure_leaves_no_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit as module

    audit, _ = _initialized_audit(tmp_path, promotion_target=2)
    for page in (1, 2):
        module.record_supplement_decision(
            audit=audit,
            record=_decision(
                locator=f"page:{page}", decision="PROMOTE_FOUNDATIONAL"
            ),
        )
    sealed = module._supplement_sealed_directory(audit)
    original_replace = module.os.replace
    with monkeypatch.context() as context:
        context.setattr(
            module.os,
            "replace",
            lambda _source, _destination: (_ for _ in ()).throw(OSError("crash")),
        )
        with pytest.raises(
            module.IronSulfideSupplementAuditError,
            match="IRON_SULFIDE_SUPPLEMENT_SEAL_WRITE_FAILED",
        ):
            module.seal_supplement_proposal(
                audit=audit,
                core_binding_sha256="c" * 64,
            )

    assert not sealed.exists()
    assert not tuple(sealed.parent.glob(".sealed.*.tmp"))
    assert module.os.replace is original_replace
    result = module.seal_supplement_proposal(
        audit=audit,
        core_binding_sha256="c" * 64,
    )
    assert result.binding_sha256
    audit.close()


def test_combined_capacity_moves_promotions_without_mutating_index_locators(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
        calculate_combined_hypothetical_capacity,
    )

    core, supplement, core_seal, supplement_seal = _closed_audits(tmp_path)
    before = tuple(
        tuple(row)
        for row in supplement._connection.execute(
            """
            select source_id, locator, topic, source_role,
                   substantive_status, e1a3_used, e1a4_available
            from index_locators order by source_id, locator
            """
        ).fetchall()
    )

    report = calculate_combined_hypothetical_capacity(
        core_audit=core,
        supplement_audit=supplement,
        expected_core_binding_sha256=core_seal.binding_sha256,
        expected_supplement_binding_sha256=supplement_seal.binding_sha256,
    )

    after = tuple(
        tuple(row)
        for row in supplement._connection.execute(
            """
            select source_id, locator, topic, source_role,
                   substantive_status, e1a3_used, e1a4_available
            from index_locators order by source_id, locator
            """
        ).fetchall()
    )
    iron_foundational = next(
        item
        for item in report.strata
        if item.topic == "iron_sulfide" and item.source_role == "foundational"
    )
    assert iron_foundational.fresh_locator_count == 4
    assert after == before
    supplement.close()


def test_cli_seal_verify_and_combined_capacity_are_aggregate_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    core, supplement, core_seal, supplement_seal = _closed_audits(tmp_path)
    root = supplement.database_path.parent
    supplement.close()
    runner = _runner_module()
    common = [
        "--private-root",
        str(root),
        "--run-id",
        "run-001",
        "--audit-id",
        "iron-sulfide-supplement-audit-v1",
    ]

    assert (
        runner.cli(
            [
                "seal",
                *common,
                "--core-binding-sha256",
                core_seal.binding_sha256,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "SEALED",
        "artifact_count": 2,
        "manifest_count": 2,
    }
    assert (
        runner.cli(
            [
                "verify",
                *common,
                "--expected-binding-sha256",
                supplement_seal.binding_sha256,
                "--expected-core-binding-sha256",
                core_seal.binding_sha256,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "VERIFIED",
        "artifact_count": 2,
        "manifest_count": 2,
    }
    assert (
        runner.cli(
            [
                "capacity",
                *common,
                "--core-audit-id",
                "foundational-locator-audit-v1",
                "--expected-binding-sha256",
                supplement_seal.binding_sha256,
                "--expected-core-binding-sha256",
                core_seal.binding_sha256,
            ]
        )
        == 0
    )
    capacity = json.loads(capsys.readouterr().out)
    assert capacity == {
        "status": "INSUFFICIENT",
        "all_sufficient": False,
        "allocation_available": False,
        "allocation_count": 0,
        "sufficient_strata_count": 0,
    }
