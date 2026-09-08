from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from oilfield_chemical_copilot.corpus_v2.ledger import CorpusV2Ledger, CorpusV2LedgerError
from oilfield_chemical_copilot.corpus_v2.models import (
    AcquisitionRecord,
    ApprovedSource,
    ExtractionOutcome,
    ExtractionRecord,
    ReleaseConfig,
    SourceDisposition,
    StageArtifactKind,
    Stage,
)


SHA = "a" * 64


def config(*, expected_source_count: int = 1) -> ReleaseConfig:
    return ReleaseConfig.from_mapping(
        {
            "release_id": "corpus-v2-test",
            "release_root": "C:/private/corpus-v2",
            "candidate_database_name": "oilfield_copilot_v2_candidate",
            "configured_database_name": "oilfield_copilot_v1",
            "legacy_database_names": ["oilfield_copilot"],
            "expected_source_count": expected_source_count,
            "source_register_sha256": SHA,
        }
    )


def test_ledger_rejects_stage_skip_and_preserves_decision_history(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config())
    with pytest.raises(CorpusV2LedgerError, match="C2_STAGE_PREREQUISITE"):
        ledger.complete_stage(Stage.CHUNKED)
    ledger.record_source(ApprovedSource(source_id="doc-1", source_sha256=SHA))
    ledger.complete_stage(Stage.REGISTERED)
    ledger.record_acquisition(AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", SHA, 10))
    ledger.complete_stage(Stage.ACQUIRED)
    ledger.record_extraction(ExtractionRecord("doc-1", "2026-09-07T00:00:00Z", "synthetic", SHA, 10))
    ledger.complete_stage(Stage.PARSED)
    ledger.record_disposition(
        source_id="doc-1", disposition=SourceDisposition.INDEXED,
        reviewer_id="reviewer-a", reason_code="APPROVED",
    )
    ledger.record_disposition(
        source_id="doc-1", disposition=SourceDisposition.UNRESOLVED,
        reviewer_id="reviewer-b", reason_code="RECHECK_REQUIRED",
    )
    assert ledger.current_disposition("doc-1").reviewer_id == "reviewer-b"
    assert len(ledger.disposition_history("doc-1")) == 2


def test_ledger_rejects_stage_completion_without_required_state(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config())
    with pytest.raises(CorpusV2LedgerError, match="C2_STAGE_STATE"):
        ledger.complete_stage(Stage.REGISTERED)


def test_ledger_requires_the_configured_source_count_before_registration(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config(expected_source_count=2))
    ledger.record_source(ApprovedSource(source_id="doc-1", source_sha256=SHA))
    with pytest.raises(CorpusV2LedgerError, match="C2_STAGE_STATE"):
        ledger.complete_stage(Stage.REGISTERED)


def test_ledger_enforces_foreign_keys_and_survives_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "ledger.sqlite"
    ledger = CorpusV2Ledger.create(database_path, release_config=config())
    with pytest.raises(CorpusV2LedgerError, match="C2_SOURCE_UNKNOWN"):
        ledger.record_acquisition(
            AcquisitionRecord("missing", "2026-09-07T00:00:00Z", SHA, 10)
        )
    ledger.record_source(ApprovedSource(source_id="doc-1", source_sha256=SHA))
    ledger.complete_stage(Stage.REGISTERED)
    ledger.close()

    reopened = CorpusV2Ledger.open(database_path)
    assert reopened.completed_stages() == (Stage.REGISTERED,)
    assert reopened.event_history()[-1].event_type == "stage_completed"


def test_ledger_rejects_mutations_after_promotion(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config())
    source = ApprovedSource(source_id="doc-1", source_sha256=SHA)
    ledger.record_source(source)
    ledger.complete_stage(Stage.REGISTERED)
    ledger.record_acquisition(AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", SHA, 10))
    ledger.complete_stage(Stage.ACQUIRED)
    ledger.record_extraction(ExtractionRecord("doc-1", "2026-09-07T00:00:00Z", "synthetic", SHA, 10))
    ledger.complete_stage(Stage.PARSED)
    ledger.record_disposition(
        source_id="doc-1", disposition=SourceDisposition.INDEXED,
        reviewer_id="reviewer-a", reason_code="APPROVED",
    )
    for stage in (
        Stage.REVIEWED, Stage.CHUNKED, Stage.EMBEDDED, Stage.INDEX_VALIDATED,
        Stage.EVALUATED, Stage.PROMOTION_READY, Stage.PROMOTED,
    ):
        if stage.value in {"CHUNKED", "EMBEDDED", "INDEX_VALIDATED", "EVALUATED", "PROMOTION_READY", "PROMOTED"}:
            ledger.record_stage_artifact(
                stage, artifact_kind=StageArtifactKind.for_stage(stage), artifact_sha256=SHA
            )
        ledger.complete_stage(stage)

    with pytest.raises(CorpusV2LedgerError, match="C2_RELEASE_IMMUTABLE"):
        ledger.record_source(ApprovedSource(source_id="doc-2", source_sha256=SHA))
    with pytest.raises(CorpusV2LedgerError, match="C2_RELEASE_IMMUTABLE"):
        ledger.record_acquisition(AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", SHA, 10))
    with pytest.raises(CorpusV2LedgerError, match="C2_RELEASE_IMMUTABLE"):
        ledger.record_extraction(ExtractionRecord("doc-1", "2026-09-07T00:00:00Z", "synthetic", SHA, 10))
    with pytest.raises(CorpusV2LedgerError, match="C2_RELEASE_IMMUTABLE"):
        ledger.record_disposition(
            source_id="doc-1", disposition=SourceDisposition.INDEXED,
            reviewer_id="reviewer-a", reason_code="APPROVED",
        )


def test_ledger_requires_durable_artifacts_for_chunked_through_promoted_stages(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config())
    ledger.record_source(ApprovedSource(source_id="doc-1", source_sha256=SHA))
    ledger.complete_stage(Stage.REGISTERED)
    ledger.record_acquisition(AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", SHA, 10))
    ledger.complete_stage(Stage.ACQUIRED)
    ledger.record_extraction(ExtractionRecord("doc-1", "2026-09-07T00:00:00Z", "synthetic", SHA, 10))
    ledger.complete_stage(Stage.PARSED)
    ledger.record_disposition(
        source_id="doc-1", disposition=SourceDisposition.INDEXED,
        reviewer_id="reviewer-a", reason_code="APPROVED",
    )
    ledger.complete_stage(Stage.REVIEWED)
    with pytest.raises(CorpusV2LedgerError, match="C2_STAGE_ARTIFACT_REQUIRED"):
        ledger.complete_stage(Stage.CHUNKED)

    ledger.record_stage_artifact(
        Stage.CHUNKED, artifact_kind=StageArtifactKind.CHUNK_MANIFEST, artifact_sha256=SHA
    )
    ledger.complete_stage(Stage.CHUNKED)


def test_terminal_disposition_accepts_matching_mechanical_extraction_outcome(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config())
    ledger.record_source(ApprovedSource(source_id="doc-1", source_sha256=SHA))
    ledger.complete_stage(Stage.REGISTERED)
    ledger.record_acquisition(AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", SHA, 10))
    ledger.complete_stage(Stage.ACQUIRED)
    ledger.record_extraction(
        ExtractionRecord(
            "doc-1", "2026-09-07T00:00:00Z", "synthetic", None, 0, ExtractionOutcome.EMPTY
        )
    )
    ledger.complete_stage(Stage.PARSED)
    ledger.record_disposition(
        source_id="doc-1", disposition=SourceDisposition.EMPTY,
        reviewer_id="reviewer-a", reason_code="NO_EXTRACTED_TEXT",
    )
    ledger.complete_stage(Stage.REVIEWED)


def test_terminal_extraction_outcome_cannot_be_marked_indexed(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config())
    ledger.record_source(ApprovedSource(source_id="doc-1", source_sha256=SHA))
    ledger.complete_stage(Stage.REGISTERED)
    ledger.record_acquisition(AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", SHA, 10))
    ledger.complete_stage(Stage.ACQUIRED)
    ledger.record_extraction(
        ExtractionRecord("doc-1", "2026-09-07T00:00:00Z", "synthetic", None, 0, ExtractionOutcome.EMPTY)
    )
    ledger.complete_stage(Stage.PARSED)
    with pytest.raises(CorpusV2LedgerError, match="C2_DISPOSITION_PREREQUISITE"):
        ledger.record_disposition(
            source_id="doc-1", disposition=SourceDisposition.INDEXED,
            reviewer_id="reviewer-a", reason_code="APPROVED",
        )


def test_ledger_rejects_secret_or_wrong_stage_artifact_kind(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config())
    ledger.record_source(ApprovedSource(source_id="doc-1", source_sha256=SHA))
    ledger.complete_stage(Stage.REGISTERED)
    ledger.record_acquisition(AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", SHA, 10))
    ledger.complete_stage(Stage.ACQUIRED)
    ledger.record_extraction(ExtractionRecord("doc-1", "2026-09-07T00:00:00Z", "synthetic", SHA, 10))
    ledger.complete_stage(Stage.PARSED)
    ledger.record_disposition(
        source_id="doc-1", disposition=SourceDisposition.INDEXED,
        reviewer_id="reviewer-a", reason_code="APPROVED",
    )
    ledger.complete_stage(Stage.REVIEWED)
    with pytest.raises(CorpusV2LedgerError, match="C2_STAGE_ARTIFACT_INVALID"):
        ledger.record_stage_artifact(Stage.CHUNKED, artifact_kind="sk-proj-opaque123", artifact_sha256=SHA)  # type: ignore[arg-type]
    with pytest.raises(CorpusV2LedgerError, match="C2_STAGE_ARTIFACT_INVALID"):
        ledger.record_stage_artifact(
            Stage.CHUNKED, artifact_kind=StageArtifactKind.EMBEDDING_MANIFEST, artifact_sha256=SHA
        )
    assert all("sk-proj-opaque123" not in str(event.payload) for event in ledger.event_history())


def test_ledger_open_rejects_partial_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "partial.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE release (release_id TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(CorpusV2LedgerError, match="C2_LEDGER_INVALID"):
        CorpusV2Ledger.open(database_path)
