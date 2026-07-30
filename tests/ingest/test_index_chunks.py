from __future__ import annotations

import json
from pathlib import Path

from ingestion.index_chunks import (
    _build_argument_parser,
    index_chunks,
    load_chunks_jsonl,
    validate_chunk_embeddings,
)
from oilfield_chemical_copilot.retrieval.embeddings import DeterministicEmbeddingProvider


class FakeStore:
    def __init__(self) -> None:
        self.upserts = []

    def upsert_chunks(self, chunks, embeddings, *, embedding_model: str) -> int:
        self.upserts.append((chunks, embeddings, embedding_model))
        return len(chunks)


def _write_chunks(path: Path, count: int = 2) -> None:
    rows = []
    for index in range(count):
        topic = "iron_sulfide" if index % 2 == 0 else "scale"
        rows.append(
            {
                "text": f"Sample chunk {index} for {topic}",
                "metadata": {
                    "chunk_id": f"chunk-{index}",
                    "source_file": f"docs/{topic}.md",
                    "source_path": f"C:/sample/docs/{topic}.md",
                    "topic": topic,
                    "parser_type": "text",
                    "page_or_sheet": "document",
                    "chunk_index": index,
                    "extra": {},
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_load_chunks_jsonl_round_trips_loaded_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path)

    chunks = load_chunks_jsonl(path)

    assert [chunk.metadata.chunk_id for chunk in chunks] == ["chunk-0", "chunk-1"]
    assert chunks[0].text == "Sample chunk 0 for iron_sulfide"


def test_index_chunks_embeds_and_upserts_in_batches(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path)
    store = FakeStore()

    count = index_chunks(
        chunks_path=path,
        store=store,
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        batch_size=1,
    )

    assert count == 2
    assert len(store.upserts) == 2
    assert store.upserts[0][2] == "deterministic-hash-8"
    assert len(store.upserts[0][1][0]) == 8


def test_dry_run_validation_embeds_every_batch(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path, count=5)
    chunks = load_chunks_jsonl(path)

    embedded_count = validate_chunk_embeddings(
        chunks=chunks,
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        batch_size=2,
    )

    assert embedded_count == 5


def test_argument_parser_uses_embedding_environment_defaults(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    parser = _build_argument_parser()
    args = parser.parse_args(["--input", "chunks.jsonl", "--dry-run"])

    assert args.embedding_provider == "sentence-transformers"
    assert args.embedding_dimension == 384
    assert args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"