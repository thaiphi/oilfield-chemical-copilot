from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from pathlib import Path

import ingestion.index_chunks as index_chunks_module
from oilfield_chemical_copilot.ollama import OllamaClientError

from ingestion.index_chunks import (
    _build_argument_parser,
    index_chunks,
    load_chunks_jsonl,
    validate_chunk_embeddings,
)
from oilfield_chemical_copilot.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    OllamaEmbeddingProvider,
)


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
    assert store.upserts[0][2] == "deterministic-token-hash-8"
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
    assert args.embedding_model is None

def test_argument_parser_uses_ollama_embedding_defaults(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "granite-embedding:latest")

    parser = _build_argument_parser()
    args = parser.parse_args(["--input", "chunks.jsonl", "--dry-run"])

    assert args.embedding_provider == "ollama"
    assert args.embedding_dimension == 384
    assert args.embedding_model is None

def test_main_passes_explicit_ollama_configuration_to_factory(monkeypatch) -> None:
    captured_settings = []

    def fake_build_embedding_provider(settings):
        captured_settings.append(settings)
        return SimpleNamespace(model_name=settings.ollama_embedding_model)

    monkeypatch.setenv("EMBEDDING_PROVIDER", "deterministic")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setattr(index_chunks_module, "build_embedding_provider", fake_build_embedding_provider)
    monkeypatch.setattr(index_chunks_module, "load_chunks_jsonl", lambda _path: [])
    monkeypatch.setattr(index_chunks_module, "validate_chunk_embeddings", lambda **_kwargs: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "index_chunks.py",
            "--input",
            "chunks.jsonl",
            "--dry-run",
            "--embedding-provider",
            "ollama",
            "--embedding-model",
            "custom-embedding:latest",
        ],
    )

    assert index_chunks_module.main() == 0
    assert captured_settings[0].provider == "ollama"
    assert captured_settings[0].ollama_embedding_model == "custom-embedding:latest"
    assert captured_settings[0].ollama_base_url == "http://ollama.internal:11434"


def test_main_converts_ollama_client_error_to_safe_parser_error(monkeypatch, capsys) -> None:
    def raise_ollama_error(_settings):
        raise OllamaClientError("response body: secret")

    monkeypatch.setattr(index_chunks_module, "build_embedding_provider", raise_ollama_error)
    monkeypatch.setattr(sys, "argv", ["index_chunks.py", "--input", "chunks.jsonl"])

    with pytest.raises(SystemExit) as error:
        index_chunks_module.main()

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "Ollama request failed" in captured.err
    assert "response body: secret" not in captured.err

def test_main_uses_ollama_default_model_after_provider_override(monkeypatch) -> None:
    captured_providers = []

    monkeypatch.setenv("EMBEDDING_PROVIDER", "deterministic")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "default-ollama-model:latest")
    monkeypatch.setattr(index_chunks_module, "load_chunks_jsonl", lambda _path: [])
    monkeypatch.setattr(
        index_chunks_module,
        "validate_chunk_embeddings",
        lambda *, embedding_provider, **_kwargs: captured_providers.append(embedding_provider) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "index_chunks.py",
            "--input",
            "chunks.jsonl",
            "--dry-run",
            "--embedding-provider",
            "ollama",
        ],
    )

    assert index_chunks_module.main() == 0
    assert isinstance(captured_providers[0], OllamaEmbeddingProvider)
    assert captured_providers[0].model_name == "default-ollama-model:latest"
    assert captured_providers[0]._client._base_url == "http://ollama.internal:11434"

def test_argument_parser_defaults_to_ollama_when_environment_is_absent(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)

    parser = _build_argument_parser()
    args = parser.parse_args(["--input", "chunks.jsonl", "--dry-run"])

    assert args.embedding_provider == "ollama"
    assert args.embedding_dimension == 384


def test_example_and_rendered_compose_default_to_ollama() -> None:
    import subprocess

    project_root = Path(__file__).resolve().parents[2]
    example = project_root / ".env.example"

    assert "EMBEDDING_PROVIDER=ollama" in example.read_text(encoding="utf-8")
    rendered = subprocess.run(
        ["docker", "compose", "--env-file", str(example), "config"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "EMBEDDING_PROVIDER: ollama" in rendered.stdout
