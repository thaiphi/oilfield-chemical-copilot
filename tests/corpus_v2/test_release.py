from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import sqlite3
import json

import pytest
from ingestion.corpus_v2_legacy_guard import legacy_guard

from oilfield_chemical_copilot.corpus_v2 import release
from oilfield_chemical_copilot.corpus_v2.ledger import CorpusV2Ledger
from oilfield_chemical_copilot.corpus_v2.models import (
    AcquisitionRecord, ApprovedSource, ExtractionRecord, ReleaseConfig,
    SourceDisposition, Stage, StageArtifactKind,
)


class MemoryPublication:
    """External capability double: locked, exclusive, exact tree semantics."""

    def __init__(self):
        self.members = {}
        self.final = False
        self.name = ".release.synthetic.tmp"

    def ensure_no_staging(self, prefix, suffix):
        if self.members and not self.final:
            raise ValueError("residue")

    def final_exists(self, name):
        return self.final

    def create_staging(self, prefix, suffix):
        return self

    def mkdir(self, name):
        pass

    def write_exclusive(self, name, content):
        assert name not in self.members
        self.members[name] = content

    def sync_directory(self, name):
        pass

    def sync_root(self):
        pass

    def publish_no_replace(self, stage, name):
        if self.final:
            raise ValueError("exists")
        self.final = True

    def sync_parent(self):
        pass

    def read_exact_tree(self, name, layout):
        expected = {f"{folder}/{member}" for folder, members in layout.items() for member in members}
        if set(self.members) != expected:
            raise ValueError("tree")
        return dict(self.members)


@pytest.fixture
def candidate(tmp_path):
    config = ReleaseConfig("corpus-v2-test", tmp_path, "candidate", "legacy", ("legacy",),
                           1, sha256(b"source").hexdigest(), "a" * 64)
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config)
    ledger.record_source(ApprovedSource("doc-1", "a" * 64))
    ledger.complete_stage(Stage.REGISTERED)
    artifacts = {key: key.encode() for key in release.REQUIRED_DIGESTS}
    artifacts["source_register"] = b"source"
    artifacts["chunks"] = release._json([{"chunk_id": "chunk-1", "source_id": "doc-1"}])
    artifacts["embeddings"] = release._json([{"chunk_id": "chunk-1",
        "embedding_provider": "deterministic", "embedding_model": "deterministic-token-hash-384",
        "embedding_dimension": 384}])
    artifacts["index_validation"] = release._json({
        "database_identity": sha256(b"postgresql://localhost/candidate").hexdigest(),
        "database_name": "candidate", "release_id": config.release_id,
        "chunks_sha256": sha256(artifacts["chunks"]).hexdigest(),
        "embeddings_sha256": sha256(artifacts["embeddings"]).hexdigest()})
    artifacts["index_contract"] = release.index_contract_bytes(release.IndexContract(
        mode="v2", release_id=config.release_id, database_name="candidate",
        database_identity=sha256(b"postgresql://localhost/candidate").hexdigest(),
        embedding_provider="deterministic", embedding_model="deterministic-token-hash-384",
        embedding_dimension=384, index_manifest_sha256=sha256(artifacts["chunks"]).hexdigest(),
        source_register_sha256=config.source_register_sha256, source_count=1, chunk_count=1,
    ))
    @contextmanager
    def reader(*, default_transaction_read_only):
        assert default_transaction_read_only is True
        yield {"source_count": 198, "chunk_count": 4797, "read_only": True}
    attestation = legacy_guard(reader, config=config, operator_evidence=release.OperatorRestoreEvidence(
        "b" * 64, True, True, release.LegacyFingerprint(198, 4797, True)))
    artifacts["legacy_guard"] = release.legacy_guard_bytes(attestation)
    ledger.record_snapshot_acquisition(
        AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", "a" * 64, 10),
        snapshot_relative_path="snapshots/doc-1.blob", mime_type="application/pdf",
        revision_token="synthetic",
    )
    for stage in tuple(Stage)[1:-1]:
        if stage is Stage.PARSED:
            ledger.record_extraction(ExtractionRecord(
                "doc-1", "2026-09-07T00:00:00Z", "synthetic", "a" * 64, 10))
        if stage is Stage.REVIEWED:
            ledger.record_disposition(source_id="doc-1", disposition=SourceDisposition.INDEXED,
                                      reviewer_id="reviewer", reason_code="APPROVED")
        if stage in release.STAGE_DIGESTS:
            ledger.record_stage_artifact(stage, artifact_kind=StageArtifactKind.for_stage(stage),
                artifact_sha256=sha256(artifacts[release.STAGE_DIGESTS[stage]]).hexdigest())
        ledger.complete_stage(stage)
    artifacts["dispositions"] = release.final_dispositions_bytes(ledger)
    artifacts["ledger_projection"] = release.ledger_projection_bytes(ledger)
    yield config, ledger, artifacts, MemoryPublication()
    ledger.close()


def seal(candidate):
    config, ledger, artifacts, publication = candidate
    return release.seal_release_binding(config=config, ledger=ledger, artifacts=artifacts,
                                        publication=publication)


def test_seal_roundtrip_and_no_replace(candidate):
    binding = seal(candidate)
    assert len(binding) == 64
    config, ledger, _, publication = candidate
    assert release.verify_release_binding(config=config, ledger=ledger, publication=publication,
                                          expected_sha256=binding) == binding
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)


def test_task6_publication_is_consumable_by_runtime(candidate):
    from oilfield_chemical_copilot.corpus_v2.runtime import (
        AuthenticatedRuntimeBinding, load_runtime_release,
    )
    digest = seal(candidate)
    config, _, artifacts, publication = candidate
    env = dict(CORPUS_MODE="v2", CORPUS_RELEASE_ID=config.release_id,
               CORPUS_DATABASE_URL="postgresql://localhost/candidate",
               CORPUS_EMBEDDING_PROVIDER="deterministic",
               CORPUS_EMBEDDING_MODEL="deterministic-token-hash-384",
               CORPUS_EMBEDDING_DIMENSION="384", CORPUS_RELEASE_MANIFEST_SHA256=digest,
               CORPUS_RELEASE_BINDING_PATH="synthetic")
    selected = load_runtime_release(env, binding_reader=lambda _: AuthenticatedRuntimeBinding(
        publication.members["sealed/binding.json"], artifacts["index_contract"]))
    assert selected.release_id == "corpus-v2-test"


@pytest.mark.parametrize("content", [b"index_contract", b"{}\n"])
def test_task6_rejects_unstructured_index_contract(candidate, content):
    candidate[2]["index_contract"] = content
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)


@pytest.mark.parametrize("field,value", [("source_count", 2), ("chunk_count", 2),
    ("embedding_provider", "ollama"), ("embedding_model", "wrong"),
    ("embedding_dimension", 768), ("database_identity", "f" * 64),
    ("index_manifest_sha256", "f" * 64)])
def test_structured_false_contract_cannot_seal(candidate, monkeypatch, field, value):
    _, ledger, artifacts, publication = candidate
    contract = json.loads(artifacts["index_contract"])
    contract[field] = value
    if field == "source_count":
        contract["chunk_count"] = 2
    if field == "embedding_model":
        contract["embedding_provider"] = "ollama"
    artifacts["index_contract"] = release._json(contract)
    projection = ledger.release_projection()
    for row in projection["stage_artifacts"]:
        if row["stage"] == Stage.INDEX_VALIDATED.value:
            row["artifact_sha256"] = sha256(artifacts["index_contract"]).hexdigest()
    monkeypatch.setattr(ledger, "release_projection", lambda: projection)
    artifacts["ledger_projection"] = release._json(projection)
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not publication.final


@pytest.mark.parametrize("damage", ["missing", "extra", "stale", "stage_digest", "disposition"])
def test_binding_rejects_invalid_inputs_before_publication(candidate, damage):
    _, _, artifacts, publication = candidate
    if damage == "missing":
        del artifacts["evaluation_specification"]
    elif damage == "extra":
        artifacts["unexpected"] = b"secret"
    elif damage == "stale":
        artifacts["ledger_projection"] = b"stale"
    elif damage == "stage_digest":
        artifacts["chunks"] = b"changed"
    else:
        artifacts["dispositions"] = b"[]"
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not publication.final


@pytest.mark.parametrize("damage", ["extra", "partial", "tamper", "stale"])
def test_verification_rejects_changed_tree_or_ledger(candidate, damage):
    binding = seal(candidate)
    config, ledger, _, publication = candidate
    if damage == "extra":
        publication.members["sealed/extra"] = b"secret"
    elif damage == "partial":
        del publication.members["sealed/chunks.bin"]
    elif damage == "tamper":
        publication.members["sealed/chunks.bin"] = b"changed"
    else:
        ledger.record_stage_artifact(Stage.PROMOTED,
            artifact_kind=StageArtifactKind.PROMOTION_RECEIPT, artifact_sha256="f" * 64)
    with pytest.raises(release.CorpusV2ReleaseError):
        release.verify_release_binding(config=config, ledger=ledger, publication=publication,
                                       expected_sha256=binding)


def test_legacy_requires_read_only_boundary_and_exact_counts():
    @contextmanager
    def reader(*, default_transaction_read_only):
        assert default_transaction_read_only is True
        yield {"source_count": 198, "chunk_count": 4797, "read_only": True}
    assert release.capture_legacy_fingerprint(reader).chunk_count == 4797
    for source_count, chunk_count, readonly in [(197, 4797, True), (198, 4796, True),
                                               (198, 4797, False), (198.0, 4797, True)]:
        with pytest.raises(release.CorpusV2ReleaseError):
            release.LegacyFingerprint(source_count, chunk_count, readonly)


def test_restore_evidence_requires_matching_fingerprint_and_disposable_target():
    evidence = release.OperatorRestoreEvidence("a" * 64, True, True,
                                               release.LegacyFingerprint(198, 4797, True))
    release.validate_operator_restore_evidence(evidence)
    for bad in (replace(evidence, backup_sha256="bad"), replace(evidence, restore_succeeded=False),
                replace(evidence, distinct_disposable_target=False)):
        with pytest.raises(release.CorpusV2ReleaseError):
            release.validate_operator_restore_evidence(bad)


def test_public_status_rejects_free_text_and_private_fields():
    report = release.build_public_status()
    assert "legacy-index provenance gap" in report
    assert "No V2 build has occurred" in report
    for field in ["source_id", "path", "drive_id", "sha256", "source_details"]:
        with pytest.raises(release.CorpusV2ReleaseError):
            release.build_public_status({field: "private-secret"})
    assert "private-secret" not in report


def test_extra_staged_member_is_rejected_before_atomic_publish(candidate):
    publication = candidate[3]
    publication.sync_root = lambda: publication.members.update({"sealed/extra": b"secret"})
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not publication.final


@pytest.mark.parametrize("state", ["UNRESOLVED", "CANDIDATE", "", None])
def test_non_final_ledger_disposition_rejected(candidate, monkeypatch, state):
    _, ledger, artifacts, publication = candidate
    projection = ledger.release_projection()
    projection["disposition_decisions"][0]["disposition"] = state
    monkeypatch.setattr(ledger, "release_projection", lambda: projection)
    artifacts["ledger_projection"] = release.ledger_projection_bytes(ledger)
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not publication.final


def test_missing_legacy_preflight_cannot_publish(candidate):
    del candidate[2]["legacy_guard"]
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not candidate[3].final


def test_mutation_at_publication_boundary_cannot_publish_stale_evidence(candidate):
    config, _, _, publication = candidate
    original_publish = publication.publish_no_replace

    def race(stage, name):
        # An independent SQLite connection exercises the real concurrency boundary.
        with sqlite3.connect(config.release_root / "ledger.sqlite", timeout=0) as writer:
            writer.execute("INSERT INTO event_history(event_type, occurred_at, payload_json) "
                           "VALUES ('race', 'synthetic', '{}')")
        original_publish(stage, name)

    publication.publish_no_replace = race
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not publication.final


def test_same_connection_mutation_during_publication_is_rejected(candidate):
    _, ledger, _, publication = candidate

    def race(stage, name):
        ledger.record_stage_artifact(Stage.PROMOTED,
            artifact_kind=StageArtifactKind.PROMOTION_RECEIPT, artifact_sha256="f" * 64)
        publication.final = True

    publication.publish_no_replace = race
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not publication.final


def test_successful_legacy_guard_is_bound_and_failed_restore_never_reads(candidate):
    config, _, artifacts, publication = candidate
    @contextmanager
    def reader(*, default_transaction_read_only):
        pytest.fail("invalid restore evidence must fail before reader entry")
        yield
    evidence = release.OperatorRestoreEvidence("b" * 64, False, True,
                                               release.LegacyFingerprint(198, 4797, True))
    with pytest.raises(release.CorpusV2ReleaseError):
        legacy_guard(reader, config=config, operator_evidence=evidence)
    binding = seal(candidate)
    assert len(binding) == 64
    assert publication.members["sealed/legacy_guard.bin"] == artifacts["legacy_guard"]


def test_legacy_attestation_for_another_release_cannot_publish(candidate):
    candidate[2]["legacy_guard"] = candidate[2]["legacy_guard"].replace(
        b'corpus-v2-test', b'corpus-v2-other')
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not candidate[3].final


def test_contending_writer_cannot_change_successfully_published_projection(candidate):
    config, ledger, artifacts, publication = candidate
    original_publish = publication.publish_no_replace
    blocked = []

    def race(stage, name):
        writer = sqlite3.connect(config.release_root / "ledger.sqlite", timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("INSERT INTO event_history(event_type, occurred_at, payload_json) "
                               "VALUES ('race', 'synthetic', '{}')")
            blocked.append(True)
        finally:
            writer.close()
        original_publish(stage, name)

    publication.publish_no_replace = race
    binding = seal(candidate)
    assert blocked == [True]
    assert publication.final
    assert release.ledger_projection_bytes(ledger) == artifacts["ledger_projection"]
    assert release.verify_release_binding(config=config, ledger=ledger, publication=publication,
                                          expected_sha256=binding) == binding
    # The reservation is released after sealing, so subsequent promotion remains possible.
    ledger.record_stage_artifact(Stage.PROMOTED,
        artifact_kind=StageArtifactKind.PROMOTION_RECEIPT, artifact_sha256="f" * 64)


@pytest.mark.parametrize("old,new", [(b'"restore_succeeded":true', b'"restore_succeeded":false'),
                                    (b'"source_count":198', b'"source_count":197')])
def test_invalid_guard_evidence_cannot_publish(candidate, old, new):
    candidate[2]["legacy_guard"] = candidate[2]["legacy_guard"].replace(old, new)
    with pytest.raises(release.CorpusV2ReleaseError):
        seal(candidate)
    assert not candidate[3].final
