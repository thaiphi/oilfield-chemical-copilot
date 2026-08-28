from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import json
import shutil
import sqlite3
from tempfile import mkdtemp
from types import SimpleNamespace

import pytest

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
    reconcile_document_matches,
    record_review_decision,
    seal_reconciliation_snapshots,
)
from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
    E1A3SamplingError,
    build_sampling_slots,
    private_sampling_payload_digest,
)
from oilfield_chemical_copilot.evaluation.e1a4_mapping_application import (
    E1A4MappingApplicationError,
    build_e1a4_role_mapping,
    seal_e1a4_role_mapping,
    verify_e1a4_role_mapping,
)
from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
    FoundationalAuditStore,
    LocatorAuditDecision,
    bind_candidate_pages,
    initialize_audit,
    record_locator_decision,
    seal_correction_proposal,
)
from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
    IronSulfideSupplementAuditStore,
    SupplementLocatorDecision,
    bind_supplement_pages,
    initialize_supplement_audit,
    record_supplement_decision,
    seal_supplement_proposal,
)


TOPICS = ("iron_sulfide", "scale", "corrosion", "paraffin")
ROLES = ("foundational", "supporting")


@dataclass(frozen=True)
class _TrustChain:
    store: ReconciliationStore
    core: FoundationalAuditStore
    supplement: IronSulfideSupplementAuditStore
    reconciliation_binding: str
    core_binding: str
    supplement_binding: str
    allocation_path: Path
    allocation_manifest_path: Path
    private_root: Path
    output_root: Path

    def inputs(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "store": self.store,
            "core_audit": self.core,
            "supplement_audit": self.supplement,
            "expected_reconciliation_binding_sha256": self.reconciliation_binding,
            "expected_core_binding_sha256": self.core_binding,
            "expected_supplement_binding_sha256": self.supplement_binding,
            "e1a3_allocation_path": self.allocation_path,
            "e1a3_allocation_manifest_path": self.allocation_manifest_path,
            "e1a3_private_root": self.private_root,
        }
        values.update(overrides)
        return values


def _source_id(topic: str, role: str) -> str:
    return f"baseline-{topic}-{role}"


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _blank_pdf(path: Path, *, pages: int) -> Path:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _allocation_payload() -> dict[str, object]:
    counters = {(topic, role): 0 for topic in TOPICS for role in ROLES}
    allocations: list[dict[str, object]] = []
    for slot in build_sampling_slots():
        key = (slot.topic, slot.source_role)
        counters[key] += 1
        allocations.append(
            {
                **slot.to_mapping(),
                "source_id": _source_id(*key),
                "parser_type": "synthetic",
                "locator": f"prior:{counters[key]:02d}",
            }
        )
    return {"schema_version": 1, "allocations": allocations}


def _write_allocation(
    *, private_root: Path, payload: dict[str, object]
) -> tuple[Path, Path, str]:
    allocation_path = private_root / "e1a3" / "sampling-allocation.v1.json"
    manifest_path = allocation_path.with_name(f"{allocation_path.name}.sha256")
    content = _canonical_json(payload)
    digest = private_sampling_payload_digest(payload)
    allocation_path.parent.mkdir(parents=True, exist_ok=True)
    allocation_path.write_bytes(content)
    manifest_path.write_text(f"{digest}\n", encoding="ascii")
    return allocation_path, manifest_path, digest


def _binding_digest(snapshot: object) -> str:
    return next(
        artifact.sha256
        for artifact in snapshot.artifacts  # type: ignore[attr-defined]
        if artifact.name == "snapshot-binding.json"
    )


def _locator_rows(store: ReconciliationStore) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in store._connection.execute(
            """
            select source_id, locator, topic, source_role,
                   substantive_status, e1a3_used, e1a4_available
            from index_locators where run_id = ? order by source_id, locator
            """,
            (store.run_id,),
        ).fetchall()
    )


def _make_trust_chain(tmp_path: Path) -> _TrustChain:
    private_root = tmp_path / ".private"
    allocation_path, allocation_manifest_path, allocation_digest = _write_allocation(
        private_root=private_root,
        payload=_allocation_payload(),
    )
    reconciliation_root = private_root / "corpus-reconciliation" / "v1"
    store = ReconciliationStore.create(
        root=reconciliation_root,
        expected_root=reconciliation_root,
        run_id="synthetic-run",
        index_contract_sha256="a" * 64,
        e1a3_allocation_sha256=allocation_digest,
    )

    approved = tmp_path / "approved"
    core_pdf = _blank_pdf(approved / "core.pdf", pages=2)
    supplement_root = approved / "1.0 Iron Sulfide"
    supplement_pdf = _blank_pdf(supplement_root / "supplement.pdf", pages=2)
    inventory_local_files(store=store, roots=(approved,))
    relative_paths = {
        Path(str(row["relative_path"])).name: str(row["relative_path"])
        for row in store._connection.execute(
            "select relative_path from local_files where run_id = ?",
            (store.run_id,),
        ).fetchall()
    }
    import_drive_page(
        store=store,
        records=tuple(
            DriveFileRecord.from_mapping(
                {
                    "drive_file_id": drive_id,
                    "name": path.name,
                    "mime_type": "application/pdf",
                    "size_bytes": path.stat().st_size,
                    "checksum_algorithm": None,
                    "checksum": None,
                    "modified_time": "2026-08-27T00:00:00Z",
                    "parent_ids": ["synthetic-folder"],
                }
            )
            for drive_id, path in (
                ("drive-core", core_pdf),
                ("drive-supplement", supplement_pdf),
            )
        ),
        next_page_token=None,
    )

    sources: list[IndexSourceRecord] = []
    locators: list[IndexLocatorRecord] = []
    for topic in TOPICS:
        for role in ROLES:
            source_id = _source_id(topic, role)
            fresh_count = 11 if topic == "iron_sulfide" else 12
            locator_values = tuple(
                (f"prior:{index:02d}", "SUBSTANTIVE") for index in range(1, 13)
            ) + tuple(
                (f"fresh:{index:02d}", "SUBSTANTIVE")
                for index in range(1, fresh_count + 1)
            )
            sources.append(
                IndexSourceRecord.from_mapping(
                    {
                        "source_id": source_id,
                        "source_path": f"index-only/{source_id}.synthetic",
                        "parser_type": "synthetic",
                        "topic": topic,
                        "chunk_count": len(locator_values),
                        "embedding_model": "synthetic-model",
                        "index_contract_sha256": "a" * 64,
                        "provenance_drive_file_id": None,
                        "content_sha256": None,
                    }
                )
            )
            locators.extend(
                IndexLocatorRecord.from_mapping(
                    {
                        "source_id": source_id,
                        "locator": locator,
                        "topic": topic,
                        "source_role": role,
                        "substantive_status": status,
                    }
                )
                for locator, status in locator_values
            )

    sources.extend(
        (
            IndexSourceRecord.from_mapping(
                {
                    "source_id": "core-source",
                    "source_path": relative_paths[core_pdf.name],
                    "parser_type": "pdf",
                    "topic": "unassigned",
                    "chunk_count": 2,
                    "embedding_model": "synthetic-model",
                    "index_contract_sha256": "a" * 64,
                    "provenance_drive_file_id": None,
                    "content_sha256": None,
                }
            ),
            IndexSourceRecord.from_mapping(
                {
                    "source_id": "supplement-source",
                    "source_path": relative_paths[supplement_pdf.name],
                    "parser_type": "pdf",
                    "topic": "iron_sulfide",
                    "chunk_count": 2,
                    "embedding_model": "synthetic-model",
                    "index_contract_sha256": "a" * 64,
                    "provenance_drive_file_id": None,
                    "content_sha256": None,
                }
            ),
        )
    )
    locators.extend(
        IndexLocatorRecord.from_mapping(record)
        for record in (
            {
                "source_id": "core-source",
                "locator": "page:1",
                "topic": "unassigned",
                "source_role": "foundational",
                "substantive_status": "INELIGIBLE",
            },
            {
                "source_id": "core-source",
                "locator": "page:2",
                "topic": "unassigned",
                "source_role": "foundational",
                "substantive_status": "INELIGIBLE",
            },
            {
                "source_id": "supplement-source",
                "locator": "page:1",
                "topic": "iron_sulfide",
                "source_role": "supporting",
                "substantive_status": "SUBSTANTIVE",
            },
            {
                "source_id": "supplement-source",
                "locator": "page:2",
                "topic": "iron_sulfide",
                "source_role": "supporting",
                "substantive_status": "SUBSTANTIVE",
            },
        )
    )
    import_index_inventory(
        store=store,
        sources=sources,
        locators=locators,
        expected_source_count=len(sources),
        expected_chunk_count=sum(source.chunk_count for source in sources),
    )
    reconcile_document_matches(store=store)
    for drive_id in ("drive-core", "drive-supplement"):
        row = store._connection.execute(
            "select match_key from document_matches where run_id = ? and drive_file_id = ?",
            (store.run_id, drive_id),
        ).fetchone()
        assert row is not None
        record_review_decision(
            store=store,
            record=ReviewDecisionRecord.from_mapping(
                {
                    "decision_id": f"accept-{drive_id}",
                    "match_key": str(row["match_key"]),
                    "decision": "ACCEPT",
                    "reviewer_id": "synthetic-reviewer",
                    "reason_code": "REVIEWED",
                    "supersedes_decision_id": None,
                    "decided_at": "2026-08-27T00:00:00Z",
                }
            ),
        )
    allocations = _allocation_payload()["allocations"]
    assert isinstance(allocations, list)
    prior_keys = {
        f"{allocation['source_id']}:{allocation['locator']}"
        for allocation in allocations
    }
    calculate_locator_capacity(store=store, prior_locator_keys=prior_keys)
    dry_run_e1a4_allocation(store=store, prior_locator_keys=prior_keys)
    reconciliation_binding = _binding_digest(
        seal_reconciliation_snapshots(store=store, root=reconciliation_root)
    )

    core = initialize_audit(
        store=store,
        audit_id="core-audit",
        snapshot_binding_sha256=reconciliation_binding,
        source_drive_file_id="drive-core",
        source_file_sha256=hashlib.sha256(core_pdf.read_bytes()).hexdigest(),
    )
    bind_candidate_pages(audit=core, pdf_path=core_pdf)
    core_digests = {
        str(row["locator"]): str(row["page_text_sha256"])
        for row in core._connection.execute(
            """
            select locator, page_text_sha256 from foundational_audit_candidates
            where run_id = ? and audit_id = ?
            """,
            (core.run_id, core.audit_id),
        ).fetchall()
    }
    for locator, decision, topic, reason in (
        (
            "page:1",
            "PROMOTE_FOUNDATIONAL",
            "scale",
            "SUBSTANTIVE_TARGET_EVIDENCE",
        ),
        ("page:2", "KEEP_INELIGIBLE", None, "NO_TARGET_TOPIC"),
    ):
        record_locator_decision(
            audit=core,
            record=LocatorAuditDecision.from_mapping(
                {
                    "decision_id": f"core-{locator}",
                    "source_id": "core-source",
                    "locator": locator,
                    "decision": decision,
                    "proposed_topic": topic,
                    "reason_code": reason,
                    "page_text_sha256": core_digests[locator],
                    "reviewer_id": "synthetic-reviewer",
                    "supersedes_decision_id": None,
                    "decided_at": "2026-08-27T00:00:00Z",
                }
            ),
        )
    core_binding = seal_correction_proposal(audit=core).binding_sha256

    supplement = initialize_supplement_audit(
        store=store,
        audit_id="supplement-audit",
        snapshot_binding_sha256=reconciliation_binding,
        source_root=supplement_root,
        promotion_target=1,
    )
    bind_supplement_pages(audit=supplement, source_root=supplement_root)
    supplement_row = supplement._connection.execute(
        """
        select page_text_sha256 from iron_sulfide_supplement_audit_candidates
        where run_id = ? and audit_id = ? and source_id = ? and locator = ?
        """,
        (
            supplement.run_id,
            supplement.audit_id,
            "supplement-source",
            "page:1",
        ),
    ).fetchone()
    assert supplement_row is not None
    record_supplement_decision(
        audit=supplement,
        record=SupplementLocatorDecision.from_mapping(
            {
                "decision_id": "supplement-page-1",
                "source_id": "supplement-source",
                "locator": "page:1",
                "decision": "PROMOTE_FOUNDATIONAL",
                "reason_code": "GENERALIZABLE_FOUNDATIONAL_EVIDENCE",
                "page_text_sha256": str(supplement_row["page_text_sha256"]),
                "reviewer_id": "synthetic-reviewer",
                "supersedes_decision_id": None,
                "decided_at": "2026-08-27T00:00:00Z",
            }
        ),
    )
    supplement_binding = seal_supplement_proposal(
        audit=supplement,
        core_audit=core,
        core_binding_sha256=core_binding,
    ).binding_sha256
    return _TrustChain(
        store=store,
        core=core,
        supplement=supplement,
        reconciliation_binding=reconciliation_binding,
        core_binding=core_binding,
        supplement_binding=supplement_binding,
        allocation_path=allocation_path,
        allocation_manifest_path=allocation_manifest_path,
        private_root=private_root,
        output_root=private_root / "mapping-output",
    )


@pytest.fixture
def trust_chain() -> _TrustChain:
    fixture_root = Path(mkdtemp(prefix="e1a4-chain-"))
    chain = _make_trust_chain(fixture_root)
    try:
        yield chain
    finally:
        chain.core.close()
        chain.supplement.close()
        shutil.rmtree(fixture_root, ignore_errors=True)


@pytest.fixture
def mapping_root() -> Path:
    """Use a short synthetic directory so Windows staging paths stay below MAX_PATH."""
    root = Path(mkdtemp(prefix="e1a4-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_mapping_application_module_exposes_the_public_contract() -> None:
    """Removing the authenticated mapping boundary must break this test."""
    from oilfield_chemical_copilot.evaluation.e1a4_mapping_application import (
        E1A4MappingApplicationError,
        E1A4MappingArtifact,
        E1A4MappingSeal,
        build_e1a4_role_mapping,
        seal_e1a4_role_mapping,
        verify_e1a4_role_mapping,
    )

    assert issubclass(E1A4MappingApplicationError, RuntimeError)
    assert E1A4MappingArtifact.__name__ == "E1A4MappingArtifact"
    assert E1A4MappingSeal.__name__ == "E1A4MappingSeal"
    assert callable(build_e1a4_role_mapping)
    assert callable(seal_e1a4_role_mapping)
    assert callable(verify_e1a4_role_mapping)


def test_mapping_application_rejects_missing_output_artifact(tmp_path: Path) -> None:
    """Changing missing-artifact handling to accept absent output must fail."""
    from oilfield_chemical_copilot.evaluation.e1a4_mapping_application import (
        E1A4MappingApplicationError,
        _verify_mapping_directory,
    )

    with pytest.raises(E1A4MappingApplicationError, match="E1A4_MAPPING_SEAL_MISSING"):
        _verify_mapping_directory({}, {}, tmp_path, "a" * 64)


def _payloads() -> tuple[dict[str, object], dict[str, object]]:
    mapping = {
        "schema_version": 1,
        "sources": [
            {
                "source_id": "synthetic-source",
                "topic": "scale",
                "source_role": "foundational",
                "parser_type": "pdf",
                "locators": ["page:1"],
            }
        ],
    }
    return mapping, {"schema_version": 1, "mapping_payload_sha256": "a" * 64}


def test_mapping_seal_is_exact_idempotent_and_verifies_without_writing(
    mapping_root: Path,
) -> None:
    """Accepting an altered/extra seal or rewriting during verify must fail."""
    from oilfield_chemical_copilot.evaluation.e1a4_mapping_application import (
        _mapping_directory,
        _publish_mapping_directory,
        _verify_mapping_directory,
    )

    mapping, binding = _payloads()
    sealed = _publish_mapping_directory(mapping, binding, mapping_root)
    before = tuple((path.name, path.stat().st_mtime_ns) for path in sealed.artifacts[0].path.parent.iterdir())
    verified = _verify_mapping_directory(mapping, binding, mapping_root, sealed.binding_sha256)

    assert verified == sealed
    assert tuple((path.name, path.stat().st_mtime_ns) for path in _mapping_directory(mapping_root).iterdir()) == before


@pytest.mark.parametrize("mutation", ["extra", "payload", "manifest", "binding"])
def test_mapping_seal_fails_closed_for_tampering(mapping_root: Path, mutation: str) -> None:
    """Removing exact-file or byte checks must make this test fail."""
    from oilfield_chemical_copilot.evaluation.e1a4_mapping_application import (
        E1A4MappingApplicationError,
        _mapping_directory,
        _publish_mapping_directory,
        _verify_mapping_directory,
    )

    mapping, binding = _payloads()
    seal = _publish_mapping_directory(mapping, binding, mapping_root)
    directory = _mapping_directory(mapping_root)
    if mutation == "extra":
        (directory / "unexpected").write_text("x", encoding="utf-8")
    elif mutation == "payload":
        (directory / "role-mapping.v1.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "manifest":
        (directory / "role-mapping.v1.json.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    else:
        (directory / "mapping-binding.v1.json").write_text(json.dumps({"altered": True}) + "\n", encoding="utf-8")
    with pytest.raises(E1A4MappingApplicationError, match="E1A4_MAPPING_(SEAL_PARTIAL|BINDING_MISMATCH)"):
        _verify_mapping_directory(mapping, binding, mapping_root, seal.binding_sha256)


def test_mapping_seal_cleans_staging_after_rename_crash(
    mapping_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaving a partial visible seal after a crash must fail this test."""
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    mapping, binding = _payloads()
    monkeypatch.setattr(module.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(module.E1A4MappingApplicationError, match="E1A4_MAPPING_SEAL_WRITE_FAILED"):
        module._publish_mapping_directory(mapping, binding, mapping_root)
    assert not module._mapping_directory(mapping_root).exists()
    assert not tuple(module._mapping_directory(mapping_root).parent.glob(".sealed.*.tmp"))


@pytest.mark.parametrize(
    ("code", "operation"),
    [
        ("E1A4_MAPPING_BINDING_MISMATCH", "wrong-anchor"),
        ("E1A4_MAPPING_SEAL_PARTIAL", "partial"),
    ],
)
def test_mapping_verification_fails_closed_for_anchor_and_partial_artifacts(
    mapping_root: Path, code: str, operation: str
) -> None:
    """Accepting a wrong trust anchor or partial artifact must break this test."""
    from oilfield_chemical_copilot.evaluation.e1a4_mapping_application import (
        E1A4MappingApplicationError,
        _mapping_directory,
        _publish_mapping_directory,
        _verify_mapping_directory,
    )

    mapping, binding = _payloads()
    seal = _publish_mapping_directory(mapping, binding, mapping_root)
    if operation == "partial":
        (_mapping_directory(mapping_root) / "mapping-binding.v1.json.sha256").unlink()
    with pytest.raises(E1A4MappingApplicationError, match=code):
        _verify_mapping_directory(
            mapping, binding, mapping_root, "b" * 64 if operation == "wrong-anchor" else seal.binding_sha256
        )


def _role_for(mapping: dict[str, object], source_id: str, locator: str) -> str:
    sources = mapping["sources"]
    assert isinstance(sources, list)
    return next(
        str(source["source_role"])
        for source in sources
        if source["source_id"] == source_id and locator in source["locators"]
    )


def _artifact_state(root: Path) -> tuple[tuple[str, bytes, int], ...]:
    return tuple(
        (path.name, path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.iterdir())
    )


def test_mapping_application_preserves_inventory_and_mixed_roles(
    trust_chain: _TrustChain,
) -> None:
    before = _locator_rows(trust_chain.store)
    allocation_before = (
        trust_chain.allocation_path.read_bytes(),
        trust_chain.allocation_manifest_path.read_bytes(),
    )

    mapping, binding = build_e1a4_role_mapping(**trust_chain.inputs())  # type: ignore[arg-type]

    assert _locator_rows(trust_chain.store) == before
    assert allocation_before == (
        trust_chain.allocation_path.read_bytes(),
        trust_chain.allocation_manifest_path.read_bytes(),
    )
    assert _role_for(mapping, "supplement-source", "page:1") == "foundational"
    assert _role_for(mapping, "supplement-source", "page:2") == "supporting"
    sources = mapping["sources"]
    assert isinstance(sources, list)
    assert not any(
        source["source_id"] == "core-source" and "page:2" in source["locators"]
        for source in sources
    )
    assert binding == {
        "schema_version": 1,
        "reconciliation_run_id": "synthetic-run",
        "reconciliation_binding_sha256": trust_chain.reconciliation_binding,
        "core_binding_sha256": trust_chain.core_binding,
        "supplement_binding_sha256": trust_chain.supplement_binding,
        "e1a3_allocation_sha256": private_sampling_payload_digest(
            _allocation_payload()
        ),
        "mapping_payload_sha256": hashlib.sha256(_canonical_json(mapping)).hexdigest(),
        "source_record_count": len(sources),
        "unique_locator_count": 97,
        "stratum_locator_counts": {
            "iron_sulfide:foundational": 12,
            "iron_sulfide:supporting": 12,
            "scale:foundational": 13,
            "scale:supporting": 12,
            "corrosion:foundational": 12,
            "corrosion:supporting": 12,
            "paraffin:foundational": 12,
            "paraffin:supporting": 12,
        },
        "allocator_available": True,
        "allocator_slot_count": 96,
        "e1a3_excluded_before_allocation": True,
    }


def test_mapping_application_rejects_fake_reconciliation_digest(
    trust_chain: _TrustChain,
) -> None:
    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_AUTHENTICATION_FAILED",
    ):
        build_e1a4_role_mapping(
            **trust_chain.inputs(expected_reconciliation_binding_sha256="f" * 64)
        )


@pytest.mark.parametrize(
    "field",
    ["expected_core_binding_sha256", "expected_supplement_binding_sha256"],
)
def test_mapping_application_rejects_fake_proposal_binding(
    trust_chain: _TrustChain, field: str
) -> None:
    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_AUTHENTICATION_FAILED",
    ):
        build_e1a4_role_mapping(**trust_chain.inputs(**{field: "f" * 64}))


def test_mapping_application_rejects_a_different_sqlite_database(
    trust_chain: _TrustChain,
) -> None:
    foreign_root = Path(mkdtemp(prefix="e1a4-foreign-"))
    foreign = _make_trust_chain(foreign_root)
    try:
        with pytest.raises(
            E1A4MappingApplicationError,
            match="E1A4_MAPPING_AUTHENTICATION_FAILED",
        ):
            build_e1a4_role_mapping(
                **trust_chain.inputs(
                    core_audit=foreign.core,
                    supplement_audit=foreign.supplement,
                    expected_core_binding_sha256=foreign.core_binding,
                    expected_supplement_binding_sha256=foreign.supplement_binding,
                )
            )
    finally:
        foreign.core.close()
        foreign.supplement.close()
        shutil.rmtree(foreign_root, ignore_errors=True)


def test_mapping_application_rejects_reconciliation_snapshot_mismatch(
    trust_chain: _TrustChain,
) -> None:
    with trust_chain.core._connection:
        trust_chain.core._connection.execute(
            """
            update foundational_audit_runs set snapshot_binding_sha256 = ?
            where run_id = ? and audit_id = ?
            """,
            ("f" * 64, trust_chain.core.run_id, trust_chain.core.audit_id),
        )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_AUTHENTICATION_FAILED",
    ):
        build_e1a4_role_mapping(**trust_chain.inputs())  # type: ignore[arg-type]


@pytest.mark.parametrize("mutation", ["payload", "manifest"])
def test_mapping_application_rejects_altered_e1a3_allocation_or_manifest(
    trust_chain: _TrustChain, mutation: str
) -> None:
    if mutation == "payload":
        payload = json.loads(trust_chain.allocation_path.read_text(encoding="utf-8"))
        payload["allocations"][0]["locator"] = "fresh:01"
        trust_chain.allocation_path.write_bytes(_canonical_json(payload))
        trust_chain.allocation_manifest_path.write_text(
            f"{private_sampling_payload_digest(payload)}\n", encoding="ascii"
        )
    else:
        trust_chain.allocation_manifest_path.write_text(
            f"{'f' * 64}\n", encoding="ascii"
        )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_AUTHENTICATION_FAILED",
    ):
        build_e1a4_role_mapping(**trust_chain.inputs())  # type: ignore[arg-type]


def _reseal_changed_reconciliation(chain: _TrustChain) -> _TrustChain:
    shutil.rmtree(chain.store.root / "snapshots")
    reconciliation_binding = _binding_digest(
        seal_reconciliation_snapshots(store=chain.store, root=chain.store.root)
    )
    with chain.core._connection:
        chain.core._connection.execute(
            """
            update foundational_audit_runs set snapshot_binding_sha256 = ?
            where run_id = ? and audit_id = ?
            """,
            (reconciliation_binding, chain.core.run_id, chain.core.audit_id),
        )
    with chain.supplement._connection:
        chain.supplement._connection.execute(
            """
            update iron_sulfide_supplement_audit_runs set snapshot_binding_sha256 = ?
            where run_id = ? and audit_id = ?
            """,
            (
                reconciliation_binding,
                chain.supplement.run_id,
                chain.supplement.audit_id,
            ),
        )
    shutil.rmtree(
        chain.store.root / "foundational-locator-audit" / "v2" / "sealed"
    )
    shutil.rmtree(
        chain.store.root / "iron-sulfide-supplement-audit" / "v2" / "sealed"
    )
    core_binding = seal_correction_proposal(audit=chain.core).binding_sha256
    supplement_binding = seal_supplement_proposal(
        audit=chain.supplement,
        core_audit=chain.core,
        core_binding_sha256=core_binding,
    ).binding_sha256
    return replace(
        chain,
        reconciliation_binding=reconciliation_binding,
        core_binding=core_binding,
        supplement_binding=supplement_binding,
    )


def test_mapping_application_requires_exact_e1a3_used_exclusion_set(
    trust_chain: _TrustChain,
) -> None:
    with trust_chain.store._connection:
        trust_chain.store._connection.execute(
            """
            update index_locators set e1a3_used = 0, e1a4_available = 1
            where run_id = ? and source_id = ? and locator = 'prior:01'
            """,
            (
                trust_chain.store.run_id,
                _source_id("scale", "supporting"),
            ),
        )
        trust_chain.store._connection.execute(
            """
            update index_locators set e1a3_used = 1, e1a4_available = 0
            where run_id = ? and source_id = ? and locator = 'fresh:01'
            """,
            (
                trust_chain.store.run_id,
                _source_id("scale", "supporting"),
            ),
        )
    changed = _reseal_changed_reconciliation(trust_chain)

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_E1A3_EXCLUSION_MISMATCH",
    ):
        build_e1a4_role_mapping(**changed.inputs())  # type: ignore[arg-type]


def _authenticated(chain: _TrustChain):
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    return module._authenticate_mapping_inputs(**chain.inputs())


def _replace_inventory_row(
    authenticated: object,
    *,
    source_id: str,
    locator: str,
    **changes: object,
):
    rows = tuple(
        replace(row, **changes)
        if (row.source_id, row.locator) == (source_id, locator)
        else row
        for row in authenticated.inventory  # type: ignore[attr-defined]
    )
    return replace(authenticated, inventory=rows)


def test_mapping_application_rejects_overlapping_proposal_keys(
    trust_chain: _TrustChain,
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    authenticated = _authenticated(trust_chain)
    core_promotion = next(
        decision
        for decision in authenticated.core
        if decision.decision == "PROMOTE_FOUNDATIONAL"
    )
    supplement_overlap = replace(
        authenticated.supplement[0],
        source_id=core_promotion.source_id,
        locator=core_promotion.locator,
    )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_PROMOTION_OVERLAP",
    ):
        module._project_mapping_sources(
            replace(authenticated, supplement=(supplement_overlap,))
        )


@pytest.mark.parametrize("proposal", ["core", "supplement"])
def test_mapping_application_rejects_promotion_outside_frozen_candidates(
    trust_chain: _TrustChain, proposal: str
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    authenticated = _authenticated(trust_chain)
    if proposal == "core":
        altered = replace(
            authenticated,
            core=(replace(authenticated.core[0], locator="page:999"),),
        )
    else:
        altered = replace(
            authenticated,
            supplement=(
                replace(
                    authenticated.supplement[0],
                    source_id=_source_id("iron_sulfide", "supporting"),
                    locator="prior:01",
                ),
            ),
        )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_PROMOTION_INVALID",
    ):
        module._project_mapping_sources(altered)


@pytest.mark.parametrize("mutation", ["role", "status", "topic"])
def test_mapping_application_enforces_core_pre_application_rules(
    trust_chain: _TrustChain, mutation: str
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    authenticated = _authenticated(trust_chain)
    if mutation == "topic":
        authenticated = replace(
            authenticated,
            core=(replace(authenticated.core[0], proposed_topic="unassigned"),)
            + authenticated.core[1:],
        )
    else:
        field, value = (
            ("source_role", "supporting")
            if mutation == "role"
            else ("substantive_status", "SUBSTANTIVE")
        )
        authenticated = _replace_inventory_row(
            authenticated,
            source_id="core-source",
            locator="page:1",
            **{field: value},
        )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_PROMOTION_INVALID",
    ):
        module._project_mapping_sources(authenticated)


@pytest.mark.parametrize("mutation", ["role", "status", "topic"])
def test_mapping_application_enforces_supplement_pre_application_rules(
    trust_chain: _TrustChain, mutation: str
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    authenticated = _authenticated(trust_chain)
    field, value = {
        "role": ("source_role", "foundational"),
        "status": ("substantive_status", "INELIGIBLE"),
        "topic": ("topic", "scale"),
    }[mutation]
    authenticated = _replace_inventory_row(
        authenticated,
        source_id="supplement-source",
        locator="page:1",
        **{field: value},
    )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_PROMOTION_INVALID",
    ):
        module._project_mapping_sources(authenticated)


@pytest.mark.parametrize("proposal", ["core", "supplement"])
def test_mapping_application_rejects_unresolved_proposal_decision(
    trust_chain: _TrustChain, proposal: str
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    authenticated = _authenticated(trust_chain)
    if proposal == "core":
        altered = replace(
            authenticated,
            core=(
                replace(
                    authenticated.core[0],
                    decision="NEEDS_SECOND_REVIEW",
                    proposed_topic=None,
                ),
            )
            + authenticated.core[1:],
        )
    else:
        altered = replace(
            authenticated,
            supplement=(
                replace(
                    authenticated.supplement[0],
                    decision="NEEDS_SECOND_REVIEW",
                ),
            ),
        )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_PROPOSAL_UNRESOLVED",
    ):
        module._project_mapping_sources(altered)


@pytest.mark.parametrize(
    ("topic", "role"), [(topic, role) for topic in TOPICS for role in ROLES]
)
def test_mapping_application_rejects_each_insufficient_required_stratum(
    trust_chain: _TrustChain, topic: str, role: str
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    authenticated = _authenticated(trust_chain)
    if (topic, role) == ("scale", "foundational"):
        authenticated = replace(
            authenticated,
            core=(
                replace(
                    authenticated.core[0],
                    decision="KEEP_INELIGIBLE",
                    proposed_topic=None,
                ),
            )
            + authenticated.core[1:],
        )
    authenticated = _replace_inventory_row(
        authenticated,
        source_id=_source_id(topic, role),
        locator="fresh:01",
        substantive_status="INELIGIBLE",
    )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_STRATUM_INSUFFICIENT",
    ):
        module._project_mapping_sources(authenticated)


def test_mapping_application_sanitizes_allocator_failure_and_calls_once(
    trust_chain: _TrustChain, monkeypatch: pytest.MonkeyPatch
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    calls = 0

    def fail_allocator(**_kwargs: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        raise E1A3SamplingError("private allocator detail")

    monkeypatch.setattr(module, "allocate_sampling_slots", fail_allocator)
    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_ALLOCATION_UNAVAILABLE",
    ):
        build_e1a4_role_mapping(**trust_chain.inputs())  # type: ignore[arg-type]
    assert calls == 1


@pytest.mark.parametrize("mode", ["duplicate", "short"])
def test_mapping_application_rejects_invalid_allocator_uniqueness(
    trust_chain: _TrustChain,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    rows = tuple(
        SimpleNamespace(source_id="same", locator="same") for _ in range(96)
    )
    monkeypatch.setattr(
        module,
        "allocate_sampling_slots",
        lambda **_kwargs: rows if mode == "duplicate" else rows[:95],
    )
    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_ALLOCATION_INVALID",
    ):
        build_e1a4_role_mapping(**trust_chain.inputs())  # type: ignore[arg-type]


def test_mapping_seal_is_atomic_exact_and_publicly_idempotent(
    trust_chain: _TrustChain,
) -> None:
    inventory_before = _locator_rows(trust_chain.store)
    sealed = seal_e1a4_role_mapping(
        output_root=trust_chain.output_root,
        **trust_chain.inputs(),  # type: ignore[arg-type]
    )
    sealed_root = sealed.artifacts[0].path.parent
    before = _artifact_state(sealed_root)

    verified_once = verify_e1a4_role_mapping(
        output_root=trust_chain.output_root,
        expected_mapping_binding_sha256=sealed.binding_sha256,
        **trust_chain.inputs(),  # type: ignore[arg-type]
    )
    verified_twice = verify_e1a4_role_mapping(
        output_root=trust_chain.output_root,
        expected_mapping_binding_sha256=sealed.binding_sha256,
        **trust_chain.inputs(),  # type: ignore[arg-type]
    )

    assert {name for name, _, _ in before} == {
        "role-mapping.v1.json",
        "role-mapping.v1.json.sha256",
        "mapping-binding.v1.json",
        "mapping-binding.v1.json.sha256",
    }
    assert verified_once == sealed
    assert verified_twice == sealed
    assert _artifact_state(sealed_root) == before
    assert _locator_rows(trust_chain.store) == inventory_before


def test_mapping_seal_crash_removes_staging_and_visible_output(
    trust_chain: _TrustChain, monkeypatch: pytest.MonkeyPatch
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic crash")),
    )
    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_SEAL_WRITE_FAILED",
    ):
        seal_e1a4_role_mapping(
            output_root=trust_chain.output_root,
            **trust_chain.inputs(),  # type: ignore[arg-type]
        )
    sealed = trust_chain.output_root / "e1a4-role-mapping" / "v1" / "sealed"
    assert not sealed.exists()
    assert not tuple(sealed.parent.glob(".sealed.*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "partial",
        "mapping",
        "mapping-manifest",
        "binding",
        "binding-manifest",
    ],
)
def test_mapping_verification_rejects_nonexact_or_altered_file_set(
    trust_chain: _TrustChain, mutation: str
) -> None:
    sealed = seal_e1a4_role_mapping(
        output_root=trust_chain.output_root,
        **trust_chain.inputs(),  # type: ignore[arg-type]
    )
    root = sealed.artifacts[0].path.parent
    target = {
        "mapping": root / "role-mapping.v1.json",
        "mapping-manifest": root / "role-mapping.v1.json.sha256",
        "binding": root / "mapping-binding.v1.json",
        "binding-manifest": root / "mapping-binding.v1.json.sha256",
    }.get(mutation)
    if mutation == "extra":
        (root / "extra.private").write_text("extra", encoding="utf-8")
    elif mutation == "partial":
        (root / "mapping-binding.v1.json.sha256").unlink()
    elif target is not None:
        target.write_bytes(b"altered\n")

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_(SEAL_PARTIAL|BINDING_MISMATCH)",
    ):
        verify_e1a4_role_mapping(
            output_root=trust_chain.output_root,
            expected_mapping_binding_sha256=sealed.binding_sha256,
            **trust_chain.inputs(),  # type: ignore[arg-type]
        )


def test_mapping_verification_rejects_wrong_mapping_binding_anchor(
    trust_chain: _TrustChain,
) -> None:
    seal_e1a4_role_mapping(
        output_root=trust_chain.output_root,
        **trust_chain.inputs(),  # type: ignore[arg-type]
    )
    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_BINDING_MISMATCH",
    ):
        verify_e1a4_role_mapping(
            output_root=trust_chain.output_root,
            expected_mapping_binding_sha256="f" * 64,
            **trust_chain.inputs(),  # type: ignore[arg-type]
        )


def test_mapping_verification_rejects_sqlite_state_drift_without_rewriting_seal(
    trust_chain: _TrustChain,
) -> None:
    sealed = seal_e1a4_role_mapping(
        output_root=trust_chain.output_root,
        **trust_chain.inputs(),  # type: ignore[arg-type]
    )
    root = sealed.artifacts[0].path.parent
    before = _artifact_state(root)
    with trust_chain.store._connection:
        trust_chain.store._connection.execute(
            """
            update index_locators set substantive_status = 'INELIGIBLE'
            where run_id = ? and source_id = ? and locator = 'fresh:01'
            """,
            (
                trust_chain.store.run_id,
                _source_id("corrosion", "supporting"),
            ),
        )

    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_AUTHENTICATION_FAILED",
    ):
        verify_e1a4_role_mapping(
            output_root=trust_chain.output_root,
            expected_mapping_binding_sha256=sealed.binding_sha256,
            **trust_chain.inputs(),  # type: ignore[arg-type]
        )
    assert _artifact_state(root) == before


def _invoke_public_mapping_operation(
    chain: _TrustChain,
    operation: str,
    *,
    mapping_binding: str | None = None,
) -> object:
    if operation == "build":
        return build_e1a4_role_mapping(**chain.inputs())  # type: ignore[arg-type]
    if operation == "seal":
        return seal_e1a4_role_mapping(
            output_root=chain.output_root,
            **chain.inputs(),  # type: ignore[arg-type]
        )
    assert operation == "verify" and mapping_binding is not None
    return verify_e1a4_role_mapping(
        output_root=chain.output_root,
        expected_mapping_binding_sha256=mapping_binding,
        **chain.inputs(),  # type: ignore[arg-type]
    )


def _prepare_public_verify(chain: _TrustChain, operation: str) -> str | None:
    if operation != "verify":
        return None
    return seal_e1a4_role_mapping(
        output_root=chain.output_root,
        **chain.inputs(),  # type: ignore[arg-type]
    ).binding_sha256


def _swap_verified_correction_file(chain: _TrustChain, proposal: str) -> None:
    directory = (
        chain.store.root
        / (
            "foundational-locator-audit"
            if proposal == "core"
            else "iron-sulfide-supplement-audit"
        )
        / "v2"
        / "sealed"
    )
    correction_path = next(directory.glob("*.jsonl"))
    records = [
        json.loads(line)
        for line in correction_path.read_text(encoding="utf-8").splitlines()
    ]
    if proposal == "core":
        records[0].update(
            {
                "decision": "KEEP_INELIGIBLE",
                "proposed_topic": None,
                "reason_code": "NO_TARGET_TOPIC",
            }
        )
    else:
        records[0].update(
            {
                "decision": "KEEP_SUPPORTING",
                "reason_code": "CASE_OR_APPLICATION_SPECIFIC",
            }
        )
    correction_path.write_bytes(
        b"".join(_canonical_json(record) for record in records)
    )


@pytest.mark.parametrize("operation", ["build", "seal", "verify"])
@pytest.mark.parametrize("proposal", ["core", "supplement"])
def test_public_mapping_operations_reject_post_verification_correction_swap(
    trust_chain: _TrustChain,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    proposal: str,
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    mapping_binding = _prepare_public_verify(trust_chain, operation)
    real_verify = module.verify_supplement_proposal

    def verify_then_swap(**kwargs: object):
        result = real_verify(**kwargs)  # type: ignore[arg-type]
        _swap_verified_correction_file(trust_chain, proposal)
        return result

    monkeypatch.setattr(module, "verify_supplement_proposal", verify_then_swap)
    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_AUTHENTICATION_FAILED",
    ):
        _invoke_public_mapping_operation(
            trust_chain,
            operation,
            mapping_binding=mapping_binding,
        )


@pytest.mark.parametrize("operation", ["build", "seal", "verify"])
def test_public_mapping_operations_reject_post_verification_sqlite_drift(
    trust_chain: _TrustChain,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import oilfield_chemical_copilot.evaluation.e1a4_mapping_application as module

    mapping_binding = _prepare_public_verify(trust_chain, operation)
    real_verify = module.verify_supplement_proposal

    def verify_then_drift(**kwargs: object):
        result = real_verify(**kwargs)  # type: ignore[arg-type]
        with sqlite3.connect(
            trust_chain.store.root / "reconciliation.sqlite", timeout=0
        ) as connection:
            connection.execute(
                """
                update index_locators set substantive_status = 'INELIGIBLE'
                where run_id = ? and source_id = ? and locator = 'fresh:01'
                """,
                (
                    trust_chain.store.run_id,
                    _source_id("scale", "supporting"),
                ),
            )
        return result

    monkeypatch.setattr(module, "verify_supplement_proposal", verify_then_drift)
    with pytest.raises(
        E1A4MappingApplicationError,
        match="E1A4_MAPPING_AUTHENTICATION_FAILED",
    ):
        _invoke_public_mapping_operation(
            trust_chain,
            operation,
            mapping_binding=mapping_binding,
        )


def _mapping_runner_args(
    chain: _TrustChain, command: str, *, mapping_binding: str | None = None
) -> list[str]:
    arguments = [
        command,
        "--reconciliation-root",
        str(chain.store.root),
        "--run-id",
        chain.store.run_id,
        "--core-audit-id",
        chain.core.audit_id,
        "--supplement-audit-id",
        chain.supplement.audit_id,
        "--expected-reconciliation-binding-sha256",
        chain.reconciliation_binding,
        "--expected-core-binding-sha256",
        chain.core_binding,
        "--expected-supplement-binding-sha256",
        chain.supplement_binding,
        "--e1a3-allocation-path",
        str(chain.allocation_path),
        "--e1a3-allocation-manifest-path",
        str(chain.allocation_manifest_path),
        "--e1a3-private-root",
        str(chain.private_root),
        "--output-root",
        str(chain.output_root),
    ]
    if mapping_binding is not None:
        arguments.extend(
            ["--expected-mapping-binding-sha256", mapping_binding]
        )
    return arguments


def test_mapping_cli_apply_and_verify_emit_exact_aggregate_json(
    trust_chain: _TrustChain, capsys: pytest.CaptureFixture[str]
) -> None:
    import eval.apply_e1a4_role_corrections as runner

    assert runner.cli(_mapping_runner_args(trust_chain, "apply")) == 0
    applied = json.loads(capsys.readouterr().out)
    binding_path = (
        trust_chain.output_root
        / "e1a4-role-mapping"
        / "v1"
        / "sealed"
        / "mapping-binding.v1.json"
    )
    binding = hashlib.sha256(binding_path.read_bytes()).hexdigest()

    assert applied == {
        "status": "E1A4_ROLE_MAPPING_SEALED",
        "source_record_count": 11,
        "sufficient_strata_count": 8,
        "allocator_slot_count": 96,
    }
    assert runner.cli(
        _mapping_runner_args(
            trust_chain, "verify", mapping_binding=binding
        )
    ) == 0
    assert json.loads(capsys.readouterr().out) == {
        **applied,
        "status": "E1A4_ROLE_MAPPING_VERIFIED",
    }


def test_mapping_cli_uses_one_sqlite_path_and_closes_every_opened_store(
    trust_chain: _TrustChain,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eval.apply_e1a4_role_corrections as runner

    opened: list[object] = []
    closed: list[object] = []
    database_paths: list[Path] = []
    real_store_open = runner.ReconciliationStore.open
    real_core_open = runner.FoundationalAuditStore.open
    real_supplement_open = runner.IronSulfideSupplementAuditStore.open
    real_store_close = runner.ReconciliationStore.close
    real_core_close = runner.FoundationalAuditStore.close
    real_supplement_close = runner.IronSulfideSupplementAuditStore.close

    def open_store(_cls: object, **kwargs: object) -> object:
        result = real_store_open(**kwargs)  # type: ignore[arg-type]
        opened.append(result)
        database_paths.append((Path(kwargs["root"]) / "reconciliation.sqlite").resolve())
        return result

    def open_core(_cls: object, **kwargs: object) -> object:
        result = real_core_open(**kwargs)  # type: ignore[arg-type]
        opened.append(result)
        database_paths.append(Path(kwargs["database_path"]).resolve())
        return result

    def open_supplement(_cls: object, **kwargs: object) -> object:
        result = real_supplement_open(**kwargs)  # type: ignore[arg-type]
        opened.append(result)
        database_paths.append(Path(kwargs["database_path"]).resolve())
        return result

    def close_store(instance: object) -> None:
        closed.append(instance)
        real_store_close(instance)  # type: ignore[arg-type]

    def close_core(instance: object) -> None:
        closed.append(instance)
        real_core_close(instance)  # type: ignore[arg-type]

    def close_supplement(instance: object) -> None:
        closed.append(instance)
        real_supplement_close(instance)  # type: ignore[arg-type]

    monkeypatch.setattr(
        runner.ReconciliationStore, "open", classmethod(open_store)
    )
    monkeypatch.setattr(
        runner.FoundationalAuditStore, "open", classmethod(open_core)
    )
    monkeypatch.setattr(
        runner.IronSulfideSupplementAuditStore,
        "open",
        classmethod(open_supplement),
    )
    monkeypatch.setattr(runner.ReconciliationStore, "close", close_store)
    monkeypatch.setattr(runner.FoundationalAuditStore, "close", close_core)
    monkeypatch.setattr(
        runner.IronSulfideSupplementAuditStore, "close", close_supplement
    )

    assert runner.cli(_mapping_runner_args(trust_chain, "apply")) == 0
    assert len(opened) == 3
    assert {id(item) for item in closed} == {id(item) for item in opened}
    assert len(set(database_paths)) == 1
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("operation_fails", [False, True])
def test_mapping_cli_attempts_every_close_and_sanitizes_close_failures(
    trust_chain: _TrustChain,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    operation_fails: bool,
) -> None:
    import eval.apply_e1a4_role_corrections as runner

    closed: list[str] = []
    real_store_close = runner.ReconciliationStore.close
    real_core_close = runner.FoundationalAuditStore.close
    real_supplement_close = runner.IronSulfideSupplementAuditStore.close

    def close_store(instance: object) -> None:
        closed.append("store")
        real_store_close(instance)  # type: ignore[arg-type]
        raise RuntimeError("private store close detail")

    def close_core(instance: object) -> None:
        closed.append("core")
        real_core_close(instance)  # type: ignore[arg-type]
        raise RuntimeError("private core close detail")

    def close_supplement(instance: object) -> None:
        closed.append("supplement")
        real_supplement_close(instance)  # type: ignore[arg-type]
        raise RuntimeError("private supplement close detail")

    monkeypatch.setattr(runner.ReconciliationStore, "close", close_store)
    monkeypatch.setattr(runner.FoundationalAuditStore, "close", close_core)
    monkeypatch.setattr(
        runner.IronSulfideSupplementAuditStore, "close", close_supplement
    )
    if operation_fails:
        monkeypatch.setattr(
            runner,
            "seal_e1a4_role_mapping",
            lambda **_kwargs: (_ for _ in ()).throw(
                runner.E1A4MappingApplicationError(
                    "E1A4_MAPPING_AUTHENTICATION_FAILED"
                )
            ),
        )

    assert runner.cli(_mapping_runner_args(trust_chain, "apply")) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "E1A4_ROLE_MAPPING_BLOCKED",
        "error_code": "E1A4_ROLE_MAPPING_CLOSE_FAILED",
    }
    assert closed == ["supplement", "core", "store"]
    assert "private" not in captured.err


@pytest.mark.parametrize("mode", ["malformed", "unexpected"])
def test_mapping_cli_sanitizes_failures_without_private_values(
    trust_chain: _TrustChain,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    import eval.apply_e1a4_role_corrections as runner

    private_values = (
        str(trust_chain.store.root),
        trust_chain.store.run_id,
        trust_chain.reconciliation_binding,
        "prior:01",
    )
    arguments = _mapping_runner_args(trust_chain, "apply")
    if mode == "malformed":
        arguments.append("--unexpected-private-option")
    else:
        monkeypatch.setattr(
            runner,
            "seal_e1a4_role_mapping",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError(" ".join(private_values))
            ),
        )

    assert runner.cli(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "E1A4_ROLE_MAPPING_BLOCKED",
        "error_code": (
            "E1A4_ROLE_MAPPING_ARGUMENT_INVALID"
            if mode == "malformed"
            else "E1A4_ROLE_MAPPING_OPERATION_FAILED"
        ),
    }
    assert not any(value in captured.out + captured.err for value in private_values)
