from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk
from oilfield_chemical_copilot.retrieval.embeddings import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_EMBEDDING_MODEL,
    DEFAULT_SENTENCE_TRANSFORMERS_DIMENSION,
    DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
    EmbeddingProvider,
    EmbeddingSettings,
    build_embedding_provider,
)
from oilfield_chemical_copilot.ollama import OllamaClientError
from oilfield_chemical_copilot.storage.pgvector import PgVectorStore


def load_chunks_jsonl(path: str | Path) -> list[LoadedChunk]:
    chunks_path = Path(path).expanduser().resolve()
    if not chunks_path.exists():
        raise ValueError(f"Chunks file does not exist: {chunks_path}")
    chunks: list[LoadedChunk] = []
    with chunks_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                chunks.append(_chunk_from_payload(payload))
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid chunk JSONL row {line_number}: {error}") from error
    return chunks


def index_chunks(
    *,
    chunks_path: str | Path,
    store,
    embedding_provider: EmbeddingProvider,
    batch_size: int = 32,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    chunks = load_chunks_jsonl(chunks_path)
    total = 0
    for batch in _batches(chunks, batch_size):
        embeddings = embedding_provider.embed_documents([chunk.text for chunk in batch])
        if len(embeddings) != len(batch):
            raise ValueError("embedding provider returned the wrong number of vectors")
        total += store.upsert_chunks(
            batch,
            embeddings,
            embedding_model=embedding_provider.model_name,
        )
    return total


def validate_chunk_embeddings(
    *,
    chunks: list[LoadedChunk],
    embedding_provider: EmbeddingProvider,
    batch_size: int,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    embedded_count = 0
    for batch in _batches(chunks, batch_size):
        embeddings = embedding_provider.embed_documents([chunk.text for chunk in batch])
        if len(embeddings) != len(batch):
            raise ValueError("embedding provider returned the wrong number of vectors")
        embedded_count += len(embeddings)
    return embedded_count


def _batches(chunks: list[LoadedChunk], batch_size: int) -> list[list[LoadedChunk]]:
    return [chunks[start : start + batch_size] for start in range(0, len(chunks), batch_size)]


def _chunk_from_payload(payload: dict[str, object]) -> LoadedChunk:
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an object")
    return LoadedChunk(
        text=str(payload["text"]),
        metadata=ChunkMetadata(
            chunk_id=str(metadata["chunk_id"]),
            source_file=str(metadata["source_file"]),
            source_path=str(metadata["source_path"]),
            topic=str(metadata["topic"]),
            parser_type=str(metadata["parser_type"]),
            page_or_sheet=str(metadata["page_or_sheet"]),
            chunk_index=int(metadata["chunk_index"]),
            extra=dict(metadata.get("extra", {})),
        ),
    )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed chunks.jsonl and upsert rows into PGVector.")
    parser.add_argument("--input", required=True, help="Path to chunks.jsonl.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="PostgreSQL URL.")
    parser.add_argument(
        "--embedding-provider",
        choices=("deterministic", "sentence-transformers", "ollama"),
        default=os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model name for the selected embedding provider.",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=_env_int("EMBEDDING_DIMENSION", DEFAULT_SENTENCE_TRANSFORMERS_DIMENSION),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Read chunks and embed without database writes.")
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    embedding_model = args.embedding_model
    if embedding_model is None:
        embedding_model = (
            os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL)
            if args.embedding_provider == "ollama"
            else os.getenv("SENTENCE_TRANSFORMERS_MODEL", DEFAULT_SENTENCE_TRANSFORMERS_MODEL)
        )
    try:
        provider = build_embedding_provider(
            EmbeddingSettings(
                provider=args.embedding_provider,
                dimension=args.embedding_dimension,
                sentence_transformers_model=embedding_model,
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
                ollama_embedding_model=embedding_model,
            )
        )
        chunks = load_chunks_jsonl(args.input)
        if args.dry_run:
            embedded_count = validate_chunk_embeddings(
                chunks=chunks,
                embedding_provider=provider,
                batch_size=args.batch_size,
            )
            print(f"Loaded {len(chunks)} chunk(s).")
            print(f"Embedded {embedded_count} chunk(s) for dry run with {provider.model_name}.")
            return 0
        store = PgVectorStore(args.database_url, embedding_dimension=provider.dimension)
        count = index_chunks(
            chunks_path=args.input,
            store=store,
            embedding_provider=provider,
            batch_size=args.batch_size,
        )
    except OllamaClientError:
        parser.error("Ollama request failed")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Indexed {count} chunk(s) with {provider.model_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())