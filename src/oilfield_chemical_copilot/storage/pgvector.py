from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Sequence

import psycopg
from pgvector import Vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector

from oilfield_chemical_copilot.ingest.models import LoadedChunk
from oilfield_chemical_copilot.retrieval.models import RetrievalHit

UPSERT_SQL = """
insert into chunks (
    chunk_id,
    source_file,
    source_path,
    topic,
    parser_type,
    page_sheet,
    page_or_sheet,
    chunk_index,
    content,
    embedding,
    embedding_model,
    metadata,
    updated_at
)
values (
    %(chunk_id)s,
    %(source_file)s,
    %(source_path)s,
    %(topic)s,
    %(parser_type)s,
    %(page_sheet)s,
    %(page_or_sheet)s,
    %(chunk_index)s,
    %(content)s,
    %(embedding)s,
    %(embedding_model)s,
    %(metadata)s,
    now()
)
on conflict (chunk_id) do update set
    source_file = excluded.source_file,
    source_path = excluded.source_path,
    topic = excluded.topic,
    parser_type = excluded.parser_type,
    page_sheet = excluded.page_sheet,
    page_or_sheet = excluded.page_or_sheet,
    chunk_index = excluded.chunk_index,
    content = excluded.content,
    embedding = excluded.embedding,
    embedding_model = excluded.embedding_model,
    metadata = excluded.metadata,
    updated_at = now()
"""

SEARCH_SQL = """
select
    chunk_id,
    content,
    source_file,
    source_path,
    topic,
    parser_type,
    page_sheet,
    page_or_sheet,
    chunk_index,
    metadata,
    1 - (embedding <=> %(embedding)s) as score
from chunks
where embedding is not null
  and (%(topic)s::text is null or topic = %(topic)s::text)
  and embedding_model = %(embedding_model)s
order by embedding <=> %(embedding)s
limit %(limit)s
"""

LIST_SQL = """
select
    chunk_id,
    content,
    source_file,
    source_path,
    topic,
    parser_type,
    page_sheet,
    page_or_sheet,
    chunk_index,
    metadata,
    0.0 as score
from chunks
order by source_file, page_or_sheet, chunk_index, chunk_id
"""


def validate_embedding_dimensions(embeddings: Sequence[Sequence[float]], *, expected_dimension: int) -> None:
    for index, embedding in enumerate(embeddings):
        if len(embedding) != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch at index {index}: "
                f"expected {expected_dimension}, got {len(embedding)}"
            )


def chunk_record(
    chunk: LoadedChunk,
    *,
    embedding: Sequence[float],
    embedding_model: str,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk.metadata.chunk_id,
        "source_file": chunk.metadata.source_file,
        "source_path": chunk.metadata.source_path,
        "topic": chunk.metadata.topic,
        "parser_type": chunk.metadata.parser_type,
        "page_sheet": chunk.metadata.page_or_sheet,
        "page_or_sheet": chunk.metadata.page_or_sheet,
        "chunk_index": chunk.metadata.chunk_index,
        "content": chunk.text,
        "embedding": list(embedding),
        "embedding_model": embedding_model,
        "metadata": Jsonb(asdict(chunk.metadata).get("extra", {})),
    }


class PgVectorStore:
    def __init__(self, database_url: str | None = None, *, embedding_dimension: int = 384) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for PGVector storage")
        self.embedding_dimension = embedding_dimension

    def upsert_chunks(
        self,
        chunks: Sequence[LoadedChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embedding_model: str,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        validate_embedding_dimensions(embeddings, expected_dimension=self.embedding_dimension)
        records = [
            chunk_record(chunk, embedding=embedding, embedding_model=embedding_model)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        if not records:
            return 0
        with psycopg.connect(self.database_url) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.executemany(UPSERT_SQL, records)
            connection.commit()
        return len(records)

    def list_chunks(self) -> list[RetrievalHit]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute(LIST_SQL)
                return [self.row_to_hit(row, retrieval_method="stored") for row in cursor.fetchall()]

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        topic: str | None = None,
        embedding_model: str,
    ) -> list[RetrievalHit]:
        if limit < 1:
            return []
        validate_embedding_dimensions([query_embedding], expected_dimension=self.embedding_dimension)
        params = {
            "embedding": Vector(list(query_embedding)),
            "limit": limit,
            "topic": topic,
            "embedding_model": embedding_model,
        }
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(SEARCH_SQL, params)
                return [self.row_to_hit(row, retrieval_method="vector") for row in cursor.fetchall()]

    @staticmethod
    def row_to_hit(row: dict[str, Any], *, retrieval_method: str) -> RetrievalHit:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return RetrievalHit(
            chunk_id=str(row["chunk_id"]),
            text=str(row["content"]),
            score=float(row.get("score") or 0.0),
            retrieval_method=retrieval_method,
            source_file=str(row["source_file"]),
            source_path=str(row["source_path"]),
            topic=str(row["topic"]),
            parser_type=str(row["parser_type"]),
            page_or_sheet=str(row["page_or_sheet"]),
            chunk_index=int(row["chunk_index"]),
            metadata=dict(metadata),
        )


def load_chunks(chunks: list[LoadedChunk]) -> int:
    """Legacy scaffold compatibility: count chunks without writing embeddings."""
    return len(chunks)

