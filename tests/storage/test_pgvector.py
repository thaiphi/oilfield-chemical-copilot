from __future__ import annotations

from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.storage.pgvector import (
    SEARCH_SQL,
    PgVectorStore,
    chunk_record,
    validate_embedding_dimensions,
)


def _chunk(chunk_id: str = "chunk-1") -> LoadedChunk:
    return LoadedChunk(
        text="Iron sulfide deposits can restrict production.",
        metadata=ChunkMetadata(
            source_file="docs/iron_sulfide_overview.md",
            source_path="C:/sample/docs/iron_sulfide_overview.md",
            topic="iron_sulfide",
            parser_type="text",
            page_or_sheet="document",
            chunk_index=0,
            chunk_id=chunk_id,
        ),
    )


def test_chunk_record_preserves_current_chunk_metadata() -> None:
    record = chunk_record(_chunk(), embedding=[0.1, 0.2, 0.3], embedding_model="fake-3")

    assert record["chunk_id"] == "chunk-1"
    assert record["source_file"] == "docs/iron_sulfide_overview.md"
    assert record["source_path"] == "C:/sample/docs/iron_sulfide_overview.md"
    assert record["topic"] == "iron_sulfide"
    assert record["parser_type"] == "text"
    assert record["page_sheet"] == "document"
    assert record["page_or_sheet"] == "document"
    assert record["chunk_index"] == 0
    assert record["content"] == "Iron sulfide deposits can restrict production."
    assert record["embedding"] == [0.1, 0.2, 0.3]
    assert record["embedding_model"] == "fake-3"


def test_vector_search_filters_by_embedding_model() -> None:
    assert "embedding_model = %(embedding_model)s" in SEARCH_SQL


def test_validate_embedding_dimensions_rejects_mismatches() -> None:
    validate_embedding_dimensions([[0.1, 0.2]], expected_dimension=2)

    try:
        validate_embedding_dimensions([[0.1]], expected_dimension=2)
    except ValueError as error:
        assert "Embedding dimension mismatch" in str(error)
    else:
        raise AssertionError("dimension mismatch should fail")


def test_pgvector_store_builds_retrieval_hits_from_rows() -> None:
    row = {
        "chunk_id": "chunk-1",
        "content": "Scale water analysis text",
        "source_file": "docs/scale.md",
        "source_path": "C:/sample/docs/scale.md",
        "topic": "scale",
        "parser_type": "text",
        "page_or_sheet": "document",
        "chunk_index": 2,
        "metadata": {"quality": "sample"},
        "score": 0.87,
    }

    hit = PgVectorStore.row_to_hit(row, retrieval_method="vector")

    assert hit == RetrievalHit(
        chunk_id="chunk-1",
        text="Scale water analysis text",
        score=0.87,
        retrieval_method="vector",
        source_file="docs/scale.md",
        source_path="C:/sample/docs/scale.md",
        topic="scale",
        parser_type="text",
        page_or_sheet="document",
        chunk_index=2,
        metadata={"quality": "sample"},
    )