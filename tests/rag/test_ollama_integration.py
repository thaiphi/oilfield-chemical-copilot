from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from ingestion.ingest import generate_chunks
from oilfield_chemical_copilot.ingest.models import LoadedChunk
from oilfield_chemical_copilot.rag.generator_factory import build_answer_generator
from oilfield_chemical_copilot.rag.service import BasicRagService
from oilfield_chemical_copilot.retrieval.embeddings import build_embedding_provider
from oilfield_chemical_copilot.retrieval.pipeline import BasicRetrievalPipeline, RetrievalSettings
from oilfield_chemical_copilot.storage.pgvector import PgVectorStore

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_ollama_indexes_sample_chunks_and_returns_cited_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.getenv("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("set RUN_OLLAMA_INTEGRATION=1 to run against local Ollama")

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    database_url = _test_database_url()
    _ensure_test_database(database_url)
    _run_migrations(database_url)
    chunks = _test_chunks(tmp_path)
    chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)
    provider = build_embedding_provider()
    store = PgVectorStore(database_url, embedding_dimension=provider.dimension)

    try:
        assert store.upsert_chunks(
            chunks,
            provider.embed_documents([chunk.text for chunk in chunks]),
            embedding_model=provider.model_name,
        ) == len(chunks)
        service = BasicRagService.from_settings(
            retriever=BasicRetrievalPipeline(
                store=store,
                embedding_provider=provider,
                settings=RetrievalSettings(top_k=5, min_score=0.2, max_context_chars=4000),
            ),
            generator=build_answer_generator(),
            settings=RetrievalSettings(top_k=5, min_score=0.2, max_context_chars=4000),
        )

        answer = service.answer("How should I assess scale risk from produced water analysis?")

        assert answer.weak_evidence is False
        assert answer.sources
        assert "Evidence from retrieved sources:" in answer.text
    finally:
        _delete_test_rows(database_url, chunk_ids)


def _test_database_url() -> str:
    database_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/oilfield_copilot_test",
    )
    database_name = urlsplit(database_url).path.lstrip("/")
    if not database_name.endswith("_test"):
        raise ValueError("TEST_DATABASE_URL must use a disposable database ending in '_test'")
    return database_url


def _ensure_test_database(database_url: str) -> None:
    database_name = urlsplit(database_url).path.lstrip("/")
    parsed = urlsplit(database_url)
    admin_url = urlunsplit((parsed.scheme, parsed.netloc, "/postgres", parsed.query, parsed.fragment))
    with psycopg.connect(admin_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("select 1 from pg_database where datname = %s", (database_name,))
            if cursor.fetchone() is None:
                cursor.execute(sql.SQL("create database {}").format(sql.Identifier(database_name)))


def _run_migrations(database_url: str) -> None:
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(_migration_sql("0001_oilfield_chemical_copilot_schema.sql"))
            cursor.execute(
                "select exists ("
                "select 1 from information_schema.columns "
                "where table_name = 'chunks' and column_name = 'embedding_model'"
                ")"
            )
            if not cursor.fetchone()[0]:
                cursor.execute(_migration_sql("0002_milestone_3_retrieval.sql"))
        connection.commit()


def _migration_sql(filename: str) -> str:
    return (PROJECT_ROOT / "db" / "migrations" / filename).read_text(encoding="utf-8-sig")


def _test_chunks(tmp_path: Path) -> list[LoadedChunk]:
    test_prefix = f"ollama-integration-{uuid.uuid4().hex}"
    return [
        replace(chunk, metadata=replace(chunk.metadata, chunk_id=f"{test_prefix}-{chunk.metadata.chunk_id}"))
        for chunk in generate_chunks("data/sample", output_dir=tmp_path, max_files=20)
    ]


def _delete_test_rows(database_url: str, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("delete from chunks where chunk_id = any(%s)", (chunk_ids,))
        connection.commit()
