from __future__ import annotations

import json

import pytest

from oilfield_chemical_copilot.corpus_v2.canonical import (
    CorpusV2CanonicalError,
    canonical_jsonl,
    sha256_canonical_jsonl,
)


def test_canonical_jsonl_is_stable_and_newline_terminated() -> None:
    assert canonical_jsonl([{"b": 2, "a": 1}]) == b'{"a":1,"b":2}\n'
    assert sha256_canonical_jsonl([{"a": 1, "b": 2}]) == (
        "e8d38819d39f705646bfb643368eca78f7db476c16471dbc33b941b27326410d"
    )


@pytest.mark.parametrize(
    "record",
    [
        {"source_id": "doc-1", "source_sha256": "a" * 64},
        {
            "source_id": "doc-1", "acquired_at": "2026-09-07T00:00:00Z",
            "content_sha256": "a" * 64, "byte_count": 10,
        },
        {
            "source_id": "doc-1", "acquired_at": "2026-09-07T00:00:00+00:00",
            "content_sha256": "a" * 64, "byte_count": 10,
        },
        {
            "source_id": "doc-1", "extracted_at": "2026-09-07T00:00:00Z",
            "extractor": "pypdf", "text_sha256": "a" * 64, "character_count": 10,
            "outcome": "SUCCESS",
        },
        {
            "source_id": "doc-1", "extracted_at": "2026-09-07T00:00:00Z",
            "extractor": "pypdf", "text_sha256": None, "character_count": 0,
            "outcome": "EMPTY",
        },
        {
            "chunk_id": "doc-1:0", "embedding_model": "local-model",
            "embedding_sha256": "a" * 64, "vector_dimensions": 384,
        },
        {
            "chunk_id": "doc-1:0", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_sha256": "a" * 64, "vector_dimensions": 384,
        },
    ],
)
def test_canonical_jsonl_round_trips_valid_public_task_record_mappings(
    record: dict[str, object],
) -> None:
    assert json.loads(canonical_jsonl([record]).decode("utf-8")) == record


@pytest.mark.parametrize(
    "record",
    [
        {"score": 1.5},
        {"source_text": "private"},
        {"input_path": "C:/private/document.pdf"},
        {"api_key": "secret"},
        {"body": "private"},
        {"uri": "https://private.example/document"},
        {"password": "secret"},
        {"authorization": "Bearer secret"},
        {"source_id": "C:/private/document.pdf", "source_sha256": "a" * 64},
        {"source_id": "Bearer secret", "source_sha256": "a" * 64},
        {"source_id": "raw private content", "source_sha256": "a" * 64},
        {"source_id": "sk-proj-opaque123", "source_sha256": "a" * 64},
        {"source_id": "drive:sk-proj-opaque123456789", "source_sha256": "a" * 64},
        {
            "chunk_id": "doc-1:0", "embedding_model": "password=secret",
            "embedding_sha256": "a" * 64, "vector_dimensions": 384,
        },
        {
            "chunk_id": "doc-1:0", "embedding_model": "sentence-transformers/sk-proj-opaque123",
            "embedding_sha256": "a" * 64, "vector_dimensions": 384,
        },
    ],
)
def test_canonical_jsonl_rejects_unsafe_public_aggregate_fields(record: dict[str, object]) -> None:
    with pytest.raises(CorpusV2CanonicalError, match="C2_CANONICAL_INVALID"):
        canonical_jsonl([record])
