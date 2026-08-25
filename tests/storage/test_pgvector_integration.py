from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk
from oilfield_chemical_copilot.retrieval.embeddings import DeterministicEmbeddingProvider
from oilfield_chemical_copilot.storage.pgvector import PgVectorStore

pytestmark = pytest.mark.integration

DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/oilfield_copilot_test",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _database_name(database_url: str) -> str:
    name = urlsplit(database_url).path.lstrip("/")
    if not name:
        raise ValueError("TEST_DATABASE_URL must include a database name")
    return name


def _validate_disposable_database(database_url: str) -> str:
    database_name = _database_name(database_url)
    if not database_name.endswith("_test"):
        raise ValueError("PGVector integration tests require a disposable database ending in '_test'")
    return database_name


def _admin_database_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/postgres", parsed.query, parsed.fragment))


def _ensure_test_database() -> None:
    database_name = _validate_disposable_database(DATABASE_URL)
    with psycopg.connect(_admin_database_url(DATABASE_URL), autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("select 1 from pg_database where datname = %s", (database_name,))
            if cursor.fetchone() is None:
                cursor.execute(sql.SQL("create database {}").format(sql.Identifier(database_name)))


def _connect_with_retry():
    _ensure_test_database()
    last_error: Exception | None = None
    for _ in range(30):
        try:
            return psycopg.connect(DATABASE_URL)
        except psycopg.OperationalError as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError(f"Could not connect to test PGVector database: {last_error}")


def _migration_sql(filename: str) -> str:
    return (PROJECT_ROOT / "db" / "migrations" / filename).read_text(encoding="utf-8-sig")


def _run_migrations() -> None:
    with _connect_with_retry() as connection:
        with connection.cursor() as cursor:
            cursor.execute("create extension if not exists vector")
            cursor.execute("create extension if not exists pgcrypto")
            cursor.execute(_migration_sql("0001_oilfield_chemical_copilot_schema.sql"))
            cursor.execute("truncate table chunks cascade")
            cursor.execute(_migration_sql("0002_milestone_3_retrieval.sql"))
            cursor.execute("truncate table chunks cascade")
        connection.commit()


def _chunk(chunk_id: str, topic: str, text: str) -> LoadedChunk:
    return LoadedChunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            source_file=f"docs/{topic}.md",
            source_path=f"C:/sample/docs/{topic}.md",
            topic=topic,
            parser_type="text",
            page_or_sheet="document",
            chunk_index=0,
            extra={"sample": "true"},
        ),
    )


@pytest.mark.skipif(
    os.getenv("RUN_PGVECTOR_INTEGRATION") != "1",
    reason="set RUN_PGVECTOR_INTEGRATION=1 to run against Docker PGVector",
)
def test_pgvector_store_upserts_and_searches_live_database() -> None:
    _run_migrations()
    provider = DeterministicEmbeddingProvider(dimension=384)
    chunks = [
        _chunk("scale-1", "scale", "Scale water analysis chloride sulfate saturation"),
        _chunk("iron-1", "iron_sulfide", "Iron sulfide solids restrict production tubing"),
    ]
    embeddings = provider.embed_documents([chunk.text for chunk in chunks])
    store = PgVectorStore(DATABASE_URL, embedding_dimension=provider.dimension)

    assert store.upsert_chunks(chunks, embeddings, embedding_model=provider.model_name) == 2

    hits = store.search(provider.embed_query(chunks[0].text), limit=2, embedding_model=provider.model_name)
    assert hits[0].chunk_id == "scale-1"
    assert hits[0].retrieval_method == "vector"
    assert hits[0].metadata == {"sample": "true"}

    topic_hits = store.search(
        provider.embed_query(chunks[0].text),
        limit=5,
        topic="iron_sulfide",
        embedding_model=provider.model_name,
    )
    assert [hit.chunk_id for hit in topic_hits] == ["iron-1"]

    mixed_model_hits = store.search(
        provider.embed_query(chunks[0].text),
        limit=5,
        embedding_model="other-384",
    )
    assert mixed_model_hits == []

    stored = store.list_chunks()
    assert {hit.chunk_id for hit in stored} == {"scale-1", "iron-1"}


def test_integration_database_guard_rejects_non_test_database() -> None:
    with pytest.raises(ValueError, match="disposable database"):
        _validate_disposable_database("postgresql://postgres:postgres@localhost:5432/oilfield_copilot")