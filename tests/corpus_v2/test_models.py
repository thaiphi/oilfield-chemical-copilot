from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from oilfield_chemical_copilot.corpus_v2.models import (
    AcquisitionRecord,
    ApprovedSource,
    CorpusV2Chunk,
    CorpusV2ContractError,
    EmbeddingRecord,
    ExtractionRecord,
    ReleaseBinding,
    ReleaseConfig,
    SourceDisposition,
)


SHA = "a" * 64


def valid_release_config(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "release_id": "corpus-v2-2026-09-07",
        "release_root": "C:/private/corpus-v2",
        "candidate_database_name": "oilfield_copilot_v2_candidate",
        "configured_database_name": "oilfield_copilot_v1",
        "legacy_database_names": ["oilfield_copilot"],
        "expected_source_count": 385,
        "source_register_sha256": SHA,
        "critical_source_register_sha256": SHA,
    }
    payload.update(overrides)
    return payload


def test_release_config_accepts_distinct_candidate_and_rejects_legacy_database() -> None:
    assert ReleaseConfig.from_mapping(valid_release_config()).candidate_database_name == (
        "oilfield_copilot_v2_candidate"
    )
    payload = valid_release_config(candidate_database_name="oilfield_copilot_v1")
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig.from_mapping(payload)

    payload = valid_release_config(candidate_database_name="oilfield_copilot")
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig.from_mapping(payload)

    payload = valid_release_config(unexpected=True)
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig.from_mapping(payload)


def test_release_config_requires_exact_non_boolean_source_count() -> None:
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig.from_mapping(valid_release_config(expected_source_count=True))


@pytest.mark.parametrize("release_id", ("sk-proj-secret", "C:/private/release", "corpus-v2-"))
def test_release_config_rejects_nonpublic_release_identifier(release_id: str) -> None:
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig.from_mapping(valid_release_config(release_id=release_id))


def test_release_config_requires_a_critical_register_digest() -> None:
    payload = valid_release_config()
    del payload["critical_source_register_sha256"]
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig.from_mapping(payload)


def test_release_config_direct_constructor_enforces_release_isolation() -> None:
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig(
            release_id="corpus-v2-test",
            release_root=Path("C:/private/corpus-v2"),
            candidate_database_name="oilfield_copilot_v1",
            configured_database_name="oilfield_copilot_v1",
            legacy_database_names=("oilfield_copilot",),
            expected_source_count=385,
            source_register_sha256=SHA,
            critical_source_register_sha256=SHA,
        )
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig(
            release_id="corpus-v2-test",
            release_root=Path("C:/private/corpus-v2"),
            candidate_database_name="oilfield_copilot_v2_candidate",
            configured_database_name="oilfield_copilot_v1",
            legacy_database_names=("oilfield_copilot",),
            expected_source_count=1.5,
            source_register_sha256=SHA,
            critical_source_register_sha256=SHA,
        )


def test_public_records_are_frozen_and_validate_sha256_values() -> None:
    source = ApprovedSource(source_id="doc-1", source_sha256=SHA)
    with pytest.raises(FrozenInstanceError):
        source.source_id = "doc-2"  # type: ignore[misc]

    records = (
        AcquisitionRecord(source_id="doc-1", acquired_at="2026-09-07T00:00:00Z", content_sha256=SHA, byte_count=0),
        ExtractionRecord(source_id="doc-1", extracted_at="2026-09-07T00:00:00Z", extractor="pypdf", text_sha256=SHA, character_count=0),
        CorpusV2Chunk(chunk_id="doc-1/chunk-0", source_id="doc-1", ordinal=0, text_sha256=SHA, character_count=0),
        EmbeddingRecord(chunk_id="doc-1/chunk-0", embedding_model="local-model", embedding_sha256=SHA, vector_dimensions=384),
        ReleaseBinding(release_id="corpus-v2-2026-09-07", source_register_sha256=SHA, index_manifest_sha256=SHA, chunk_count=0),
    )
    assert records[0].byte_count == 0
    with pytest.raises(CorpusV2ContractError, match="C2_SHA256_INVALID"):
        ApprovedSource(source_id="doc-1", source_sha256="A" * 64)


def test_record_contracts_accept_documented_model_names_and_offset_timestamps() -> None:
    acquisition = AcquisitionRecord(
        source_id="doc-1", acquired_at="2026-09-07T00:00:00+00:00", content_sha256=SHA, byte_count=0
    )
    embedding = EmbeddingRecord(
        chunk_id="doc-1/chunk-0", embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_sha256=SHA, vector_dimensions=384,
    )
    assert acquisition.acquired_at.endswith("+00:00")
    assert embedding.embedding_model.startswith("sentence-transformers/")
    with pytest.raises(CorpusV2ContractError, match="C2_EMBEDDING_MODEL_INVALID"):
        EmbeddingRecord(
            chunk_id="doc-1/chunk-0", embedding_model="sentence-transformers/sk-proj-opaque123",
            embedding_sha256=SHA, vector_dimensions=384,
        )


def test_source_disposition_has_only_the_approved_values() -> None:
    assert {member.value for member in SourceDisposition} == {
        "INDEXED", "DUPLICATE", "NON_TEXT", "EMPTY", "UNSUPPORTED",
        "EXTRACTION_FAILED", "INTENTIONALLY_EXCLUDED", "UNRESOLVED",
    }


@pytest.mark.parametrize("source_id", [
    "drive:1sk-proj-opaque123456789", "drive:1abcdefghij1234567890",
    "doc-0", "doc-01", "doc-1/private", "doc-１", None,
])
def test_public_record_constructors_reject_non_pseudonym_source_ids(source_id: str) -> None:
    factories = (
        lambda: ApprovedSource(source_id, SHA),
        lambda: AcquisitionRecord(source_id, "2026-09-07T00:00:00Z", SHA, 0),
        lambda: ExtractionRecord(source_id, "2026-09-07T00:00:00Z", "pypdf", SHA, 0),
        lambda: CorpusV2Chunk("doc-1/chunk-0", source_id, 0, SHA, 0),
    )
    for factory in factories:
        with pytest.raises(CorpusV2ContractError, match="C2_SOURCE_ID_INVALID"):
            factory()


@pytest.mark.parametrize("chunk_id", [
    "drive:1sk-proj-opaque123456789:0", "doc-1:0", "doc-0/chunk-0",
    "doc-1/chunk-01", "doc-1/chunk--1", None,
])
def test_public_record_constructors_reject_non_pseudonym_chunk_ids(chunk_id: str) -> None:
    with pytest.raises(CorpusV2ContractError, match="C2_CHUNK_ID_INVALID"):
        CorpusV2Chunk(chunk_id, "doc-1", 0, SHA, 0)
    with pytest.raises(CorpusV2ContractError, match="C2_CHUNK_ID_INVALID"):
        EmbeddingRecord(chunk_id, "local-model", SHA, 384)


@pytest.mark.parametrize(("source_id", "ordinal"), [("doc-2", 0), ("doc-1", 1)])
def test_chunk_constructor_rejects_provenance_mismatch(source_id: str, ordinal: int) -> None:
    with pytest.raises(CorpusV2ContractError, match="C2_CHUNK_PROVENANCE_INVALID"):
        CorpusV2Chunk("doc-1/chunk-0", source_id, ordinal, SHA, 10)


def test_chunk_constructor_accepts_matching_provenance() -> None:
    chunk = CorpusV2Chunk("doc-385/chunk-12", "doc-385", 12, SHA, 10)
    assert chunk.chunk_id == f"{chunk.source_id}/chunk-{chunk.ordinal}"
