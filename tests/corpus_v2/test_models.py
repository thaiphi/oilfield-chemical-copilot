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
        )


def test_public_records_are_frozen_and_validate_sha256_values() -> None:
    source = ApprovedSource(source_id="doc-1", source_sha256=SHA)
    with pytest.raises(FrozenInstanceError):
        source.source_id = "doc-2"  # type: ignore[misc]

    records = (
        AcquisitionRecord(source_id="doc-1", acquired_at="2026-09-07T00:00:00Z", content_sha256=SHA, byte_count=0),
        ExtractionRecord(source_id="doc-1", extracted_at="2026-09-07T00:00:00Z", extractor="pypdf", text_sha256=SHA, character_count=0),
        CorpusV2Chunk(chunk_id="doc-1:0", source_id="doc-1", ordinal=0, text_sha256=SHA, character_count=0),
        EmbeddingRecord(chunk_id="doc-1:0", embedding_model="local-model", embedding_sha256=SHA, vector_dimensions=384),
        ReleaseBinding(release_id="corpus-v2-2026-09-07", source_register_sha256=SHA, index_manifest_sha256=SHA, chunk_count=0),
    )
    assert records[0].byte_count == 0
    with pytest.raises(CorpusV2ContractError, match="C2_SHA256_INVALID"):
        ApprovedSource(source_id="doc-1", source_sha256="A" * 64)


def test_source_disposition_has_only_the_approved_values() -> None:
    assert {member.value for member in SourceDisposition} == {
        "INDEXED", "DUPLICATE", "NON_TEXT", "EMPTY", "UNSUPPORTED",
        "EXTRACTION_FAILED", "INTENTIONALLY_EXCLUDED", "UNRESOLVED",
    }
