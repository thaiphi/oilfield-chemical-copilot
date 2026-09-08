from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from oilfield_chemical_copilot.corpus_v2.models import (
    ExtractionOutcome,
    ExtractionRecord,
    SourceDisposition,
)
from oilfield_chemical_copilot.corpus_v2.processing import (
    CorpusV2ProcessingError,
    ReviewDecision,
    UnsupportedSnapshotError,
    build_v2_chunk,
    ensure_unique_v2_chunk_ids,
    extract_snapshot,
    record_extraction_candidate,
    review_gate_status,
)
from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk


SHA = "a" * 64


def extraction(
    source_id: str,
    outcome: ExtractionOutcome = ExtractionOutcome.SUCCESS,
) -> ExtractionRecord:
    return ExtractionRecord(
        source_id=source_id,
        extracted_at="2026-09-07T00:00:00Z",
        extractor="parser-v1",
        text_sha256=SHA if outcome is ExtractionOutcome.SUCCESS else None,
        character_count=8 if outcome is ExtractionOutcome.SUCCESS else 0,
        outcome=outcome,
    )


def parsed_chunk(*, text: str = "evidence", location: str = "page:1") -> LoadedChunk:
    return LoadedChunk(
        text=text,
        metadata=ChunkMetadata(
            source_file="synthetic.pdf",
            source_path="/synthetic.pdf",
            topic="synthetic",
            parser_type="synthetic",
            page_or_sheet=location,
            chunk_index=0,
            chunk_id="legacy:path:0",
        ),
    )


def test_v2_chunk_id_binds_release_source_hash_location_and_text() -> None:
    first = build_v2_chunk(
        release_id="corpus-v2-test",
        source_id="doc-1",
        source_byte_sha256=SHA,
        parser_policy_version="parser-v1",
        chunk_policy_version="chunk-v1",
        location="page:1",
        ordinal=0,
        text="evidence",
        sealed_source_ids={"doc-1"},
    )
    changed_source = build_v2_chunk(
        release_id="corpus-v2-test",
        source_id="doc-2",
        source_byte_sha256=SHA,
        parser_policy_version="parser-v1",
        chunk_policy_version="chunk-v1",
        location="page:1",
        ordinal=0,
        text="evidence",
        sealed_source_ids={"doc-1", "doc-2"},
    )
    changed_text = build_v2_chunk(
        release_id="corpus-v2-test",
        source_id="doc-1",
        source_byte_sha256=SHA,
        parser_policy_version="parser-v1",
        chunk_policy_version="chunk-v1",
        location="page:1",
        ordinal=0,
        text="different evidence",
        sealed_source_ids={"doc-1"},
    )

    assert first.metadata.chunk_id != changed_source.metadata.chunk_id
    assert first.metadata.chunk_id != changed_text.metadata.chunk_id
    assert len(first.metadata.chunk_id) == 64
    assert first.metadata.extra["source_id"] == "doc-1"
    assert first.metadata.extra["source_byte_sha256"] == SHA
    assert first.metadata.source_path == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("release_id", "corpus-v2-other"),
        ("source_byte_sha256", "b" * 64),
        ("parser_policy_version", "parser-v2"),
        ("chunk_policy_version", "chunk-v2"),
        ("location", "page:2"),
        ("ordinal", 1),
    ],
)
def test_every_provenance_bound_field_changes_v2_chunk_id(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "release_id": "corpus-v2-test",
        "source_id": "doc-1",
        "source_byte_sha256": SHA,
        "parser_policy_version": "parser-v1",
        "chunk_policy_version": "chunk-v1",
        "location": "page:1",
        "ordinal": 0,
        "text": "evidence",
        "sealed_source_ids": {"doc-1"},
    }
    baseline = build_v2_chunk(**arguments)  # type: ignore[arg-type]
    arguments[field] = value
    changed = build_v2_chunk(**arguments)  # type: ignore[arg-type]
    assert changed.metadata.chunk_id != baseline.metadata.chunk_id


@pytest.mark.parametrize("location", ["page:0", "page:one", "sheet:../../secret", "C:/private"])
def test_v2_chunk_rejects_malformed_location(location: str) -> None:
    with pytest.raises(CorpusV2ProcessingError, match="C2_LOCATION_INVALID"):
        build_v2_chunk(
            release_id="corpus-v2-test",
            source_id="doc-1",
            source_byte_sha256=SHA,
            parser_policy_version="parser-v1",
            chunk_policy_version="chunk-v1",
            location=location,
            ordinal=0,
            text="evidence",
            sealed_source_ids={"doc-1"},
        )


def test_v2_chunk_rejects_unknown_source_and_legacy_path_id() -> None:
    with pytest.raises(CorpusV2ProcessingError, match="C2_SOURCE_UNKNOWN"):
        build_v2_chunk(
            release_id="corpus-v2-test",
            source_id="doc-2",
            source_byte_sha256=SHA,
            parser_policy_version="parser-v1",
            chunk_policy_version="chunk-v1",
            location="page:1",
            ordinal=0,
            text="evidence",
            sealed_source_ids={"doc-1"},
        )
    with pytest.raises(CorpusV2ProcessingError, match="C2_CHUNK_ID_INVALID"):
        build_v2_chunk(
            release_id="corpus-v2-test",
            source_id="doc-1",
            source_byte_sha256=SHA,
            parser_policy_version="parser-v1",
            chunk_policy_version="chunk-v1",
            location="page:1",
            ordinal=0,
            text="evidence",
            sealed_source_ids={"doc-1"},
            chunk_id="C:/legacy/path.pdf:page:1:0",
        )


def test_extraction_binds_snapshot_hash_and_maps_empty(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.blob"
    snapshot.write_bytes(b"approved bytes")
    candidate = extract_snapshot(
        release_id="corpus-v2-test",
        source_id="doc-1",
        snapshot_path=snapshot,
        expected_byte_sha256=hashlib.sha256(b"approved bytes").hexdigest(),
        parser_policy_version="parser-v1",
        parser=lambda _: [],
        sealed_source_ids={"doc-1"},
    )
    assert candidate.record.outcome is ExtractionOutcome.EMPTY
    assert candidate.record.text_sha256 is None
    assert candidate.locations == ()


def test_extraction_rejects_changed_snapshot_bytes(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.blob"
    snapshot.write_bytes(b"approved bytes")
    with pytest.raises(CorpusV2ProcessingError, match="C2_SNAPSHOT_HASH_MISMATCH"):
        extract_snapshot(
            release_id="corpus-v2-test",
            source_id="doc-1",
            snapshot_path=snapshot,
            expected_byte_sha256=SHA,
            parser_policy_version="parser-v1",
            parser=lambda _: [parsed_chunk()],
            sealed_source_ids={"doc-1"},
        )


@pytest.mark.parametrize(
    ("failure", "expected_outcome"),
    [
        (UnsupportedSnapshotError(), ExtractionOutcome.UNSUPPORTED),
        (RuntimeError("private parser detail"), ExtractionOutcome.FAILED),
    ],
)
def test_extraction_maps_only_explicit_unsupported_and_safe_failure(
    tmp_path: Path, failure: Exception, expected_outcome: ExtractionOutcome
) -> None:
    snapshot = tmp_path / "snapshot.blob"
    snapshot.write_bytes(b"approved bytes")
    candidate = extract_snapshot(
        release_id="corpus-v2-test",
        source_id="doc-1",
        snapshot_path=snapshot,
        expected_byte_sha256=hashlib.sha256(b"approved bytes").hexdigest(),
        parser_policy_version="parser-v1",
        parser=lambda _: (_ for _ in ()).throw(failure),
        sealed_source_ids={"doc-1"},
    )
    assert candidate.record.outcome is expected_outcome
    assert candidate.record.text_sha256 is None


def test_extraction_maps_a_generic_value_error_to_failed_not_unsupported(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.blob"
    snapshot.write_bytes(b"approved bytes")
    candidate = extract_snapshot(
        release_id="corpus-v2-test",
        source_id="doc-1",
        snapshot_path=snapshot,
        expected_byte_sha256=hashlib.sha256(b"approved bytes").hexdigest(),
        parser_policy_version="parser-v1",
        parser=lambda _: (_ for _ in ()).throw(ValueError("corrupt private input")),
        sealed_source_ids={"doc-1"},
    )
    assert candidate.record.outcome is ExtractionOutcome.FAILED


def test_record_candidate_passes_only_fixed_record_to_ledger(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.blob"
    snapshot.write_bytes(b"approved bytes")
    candidate = extract_snapshot(
        release_id="corpus-v2-test",
        source_id="doc-1",
        snapshot_path=snapshot,
        expected_byte_sha256=hashlib.sha256(b"approved bytes").hexdigest(),
        parser_policy_version="parser-v1",
        parser=lambda _: [parsed_chunk(text="private extracted evidence")],
        sealed_source_ids={"doc-1"},
    )

    class RecordingLedger:
        received: ExtractionRecord | None = None

        def record_extraction(self, received: ExtractionRecord) -> None:
            self.received = received

    ledger = RecordingLedger()
    record_extraction_candidate(ledger, candidate)
    assert ledger.received == candidate.record
    assert "private extracted evidence" not in repr(ledger.received)


def test_record_candidate_does_not_bypass_ledger_stage_order(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.blob"
    snapshot.write_bytes(b"approved bytes")
    candidate = extract_snapshot(
        release_id="corpus-v2-test",
        source_id="doc-1",
        snapshot_path=snapshot,
        expected_byte_sha256=hashlib.sha256(b"approved bytes").hexdigest(),
        parser_policy_version="parser-v1",
        parser=lambda _: [],
        sealed_source_ids={"doc-1"},
    )

    class StageLockedLedger:
        def record_extraction(self, received: ExtractionRecord) -> None:
            raise RuntimeError("C2_STAGE_PREREQUISITE")

    with pytest.raises(RuntimeError, match="C2_STAGE_PREREQUISITE"):
        record_extraction_candidate(StageLockedLedger(), candidate)


def test_extraction_rejects_duplicate_v2_chunk_ids(tmp_path: Path) -> None:
    chunk = build_v2_chunk(
        release_id="corpus-v2-test",
        source_id="doc-1",
        source_byte_sha256=SHA,
        parser_policy_version="parser-v1",
        chunk_policy_version="chunk-v1",
        location="page:1",
        ordinal=0,
        text="evidence",
        sealed_source_ids={"doc-1"},
    )
    with pytest.raises(CorpusV2ProcessingError, match="C2_CHUNK_DUPLICATE"):
        ensure_unique_v2_chunk_ids([chunk, chunk])


def test_review_rejects_invalid_reviewer_and_duplicate_disposition() -> None:
    with pytest.raises(CorpusV2ProcessingError, match="C2_REVIEWER_INVALID"):
        review_gate_status(
            [extraction("doc-1")],
            [ReviewDecision("doc-1", SourceDisposition.EMPTY, "bad reviewer", "NO_TEXT")],
            sealed_source_ids={"doc-1"},
            critical_source_ids=set(),
            usable_chunk_source_ids=set(),
        )
    with pytest.raises(CorpusV2ProcessingError, match="C2_DISPOSITION_DUPLICATE"):
        review_gate_status(
            [extraction("doc-1", ExtractionOutcome.EMPTY)],
            [
                ReviewDecision("doc-1", SourceDisposition.EMPTY, "reviewer-a", "NO_TEXT"),
                ReviewDecision("doc-1", SourceDisposition.EMPTY, "reviewer-b", "NO_TEXT"),
            ],
            sealed_source_ids={"doc-1"},
            critical_source_ids=set(),
            usable_chunk_source_ids=set(),
        )


def test_review_allows_ordinary_terminal_outcome_after_explicit_decision() -> None:
    assert (
        review_gate_status(
            [extraction("doc-1", ExtractionOutcome.EMPTY)],
            [ReviewDecision("doc-1", SourceDisposition.EMPTY, "reviewer-a", "NO_TEXT")],
            sealed_source_ids={"doc-1"},
            critical_source_ids=set(),
            usable_chunk_source_ids=set(),
        )
        == "READY"
    )


@pytest.mark.parametrize(
    ("extraction_outcome", "disposition"),
    [
        (ExtractionOutcome.NON_TEXT, SourceDisposition.NON_TEXT),
        (ExtractionOutcome.EMPTY, SourceDisposition.EMPTY),
        (ExtractionOutcome.UNSUPPORTED, SourceDisposition.UNSUPPORTED),
        (ExtractionOutcome.FAILED, SourceDisposition.EXTRACTION_FAILED),
    ],
)
def test_every_critical_failure_outcome_blocks_review_completion(
    extraction_outcome: ExtractionOutcome, disposition: SourceDisposition
) -> None:
    assert (
        review_gate_status(
            [extraction("doc-1", extraction_outcome)],
            [ReviewDecision("doc-1", disposition, "reviewer-a", "REVIEWED")],
            sealed_source_ids={"doc-1"},
            critical_source_ids={"doc-1"},
            usable_chunk_source_ids=set(),
        )
        == "BLOCKED"
    )


@pytest.mark.parametrize(
    "outcome",
    [ExtractionOutcome.NON_TEXT, ExtractionOutcome.EMPTY, ExtractionOutcome.UNSUPPORTED, ExtractionOutcome.FAILED],
)
def test_critical_terminal_extraction_blocks_even_when_label_claims_indexed(
    outcome: ExtractionOutcome,
) -> None:
    with pytest.raises(CorpusV2ProcessingError, match="C2_DISPOSITION_EVIDENCE_MISMATCH"):
        review_gate_status(
            [extraction("doc-1", outcome)],
            [ReviewDecision("doc-1", SourceDisposition.INDEXED, "reviewer-a", "APPROVED")],
            sealed_source_ids={"doc-1"},
            critical_source_ids={"doc-1"},
            usable_chunk_source_ids={"doc-1"},
        )


def test_review_rejects_indexed_label_without_successful_usable_evidence() -> None:
    with pytest.raises(CorpusV2ProcessingError, match="C2_DISPOSITION_EVIDENCE_MISMATCH"):
        review_gate_status(
            [extraction("doc-1")],
            [ReviewDecision("doc-1", SourceDisposition.INDEXED, "reviewer-a", "APPROVED")],
            sealed_source_ids={"doc-1"},
            critical_source_ids=set(),
            usable_chunk_source_ids=set(),
        )


def test_review_requires_exact_authoritative_extraction_parity() -> None:
    with pytest.raises(CorpusV2ProcessingError, match="C2_REVIEW_INCOMPLETE"):
        review_gate_status(
            [extraction("doc-1")],
            [
                ReviewDecision("doc-1", SourceDisposition.INDEXED, "reviewer-a", "APPROVED"),
                ReviewDecision("doc-2", SourceDisposition.INDEXED, "reviewer-a", "APPROVED"),
            ],
            sealed_source_ids={"doc-1", "doc-2"},
            critical_source_ids=set(),
            usable_chunk_source_ids={"doc-1", "doc-2"},
        )


def test_review_rejects_mismatched_terminal_label() -> None:
    with pytest.raises(CorpusV2ProcessingError, match="C2_DISPOSITION_EVIDENCE_MISMATCH"):
        review_gate_status(
            [extraction("doc-1", ExtractionOutcome.EMPTY)],
            [ReviewDecision("doc-1", SourceDisposition.UNSUPPORTED, "reviewer-a", "REVIEWED")],
            sealed_source_ids={"doc-1"},
            critical_source_ids=set(),
            usable_chunk_source_ids=set(),
        )


def test_review_validates_duplicate_representative_against_extraction_evidence() -> None:
    with pytest.raises(CorpusV2ProcessingError, match="C2_DUPLICATE_REPRESENTATIVE_INVALID"):
        review_gate_status(
            [extraction("doc-1"), extraction("doc-2")],
            [
                ReviewDecision("doc-1", SourceDisposition.DUPLICATE, "reviewer-a", "DUPLICATE", "doc-2"),
                ReviewDecision("doc-2", SourceDisposition.INDEXED, "reviewer-a", "APPROVED"),
            ],
            sealed_source_ids={"doc-1", "doc-2"},
            critical_source_ids=set(),
            usable_chunk_source_ids=set(),
        )
