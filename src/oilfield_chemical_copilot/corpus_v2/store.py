"""Pure, fail-closed contracts for an isolated Corpus V2 PGVector release.

This module intentionally has no database driver dependency.  Operational database
connections are a separately authorized concern; these contracts prepare and verify
only typed rows through an injected in-memory/read-only boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable, Protocol, Sequence, TypeVar
from urllib.parse import unquote, urlparse

from .models import (
    CorpusV2Chunk,
    ReleaseBinding,
    ReleaseConfig,
    is_valid_embedding_model,
    is_valid_public_chunk_id,
    is_valid_public_source_id,
)
from .processing import validate_location


class CorpusV2StoreError(ValueError):
    """Raised when a candidate-store contract is invalid."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
VECTOR_DIMENSIONS = 384


def _require_sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CorpusV2StoreError("C2_INDEX_METADATA_INVALID")
    return value


def _require_dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != VECTOR_DIMENSIONS:
        raise CorpusV2StoreError("C2_INDEX_METADATA_INVALID")
    return value


def _require_finite_vector(vector: object) -> tuple[float, ...]:
    if not isinstance(vector, tuple) or len(vector) != VECTOR_DIMENSIONS:
        raise CorpusV2StoreError("C2_EMBEDDING_INVALID")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise CorpusV2StoreError("C2_EMBEDDING_INVALID")
    return vector


def vector_sha256(vector: tuple[float, ...]) -> str:
    """Create the sealed identity for one validated candidate embedding vector."""
    finite_vector = _require_finite_vector(vector)
    payload = json.dumps(finite_vector, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def database_name_from_url(database_url: object) -> str:
    """Extract one PostgreSQL database name without making a connection."""
    if not isinstance(database_url, str) or not database_url:
        raise CorpusV2StoreError("C2_DATABASE_TARGET_INVALID")
    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgresql", "postgres"} or not parsed.hostname:
        raise CorpusV2StoreError("C2_DATABASE_TARGET_INVALID")
    database_name = unquote(parsed.path.lstrip("/"))
    if not database_name or "/" in database_name:
        raise CorpusV2StoreError("C2_DATABASE_TARGET_INVALID")
    return database_name


@dataclass(frozen=True)
class V2ChunkManifestEntry:
    """The public, fixed metadata expected for exactly one candidate V2 chunk."""

    chunk: CorpusV2Chunk
    source_sha256: str
    manifest_sha256: str
    location: str
    embedding_model: str
    vector_dimensions: int
    content: str
    expected_embedding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, CorpusV2Chunk):
            raise CorpusV2StoreError("C2_INDEX_METADATA_INVALID")
        if not is_valid_public_chunk_id(self.chunk.chunk_id) or not is_valid_public_source_id(self.chunk.source_id):
            raise CorpusV2StoreError("C2_INDEX_METADATA_INVALID")
        _require_sha256(self.source_sha256)
        _require_sha256(self.manifest_sha256)
        try:
            validate_location(self.location)
        except Exception as error:
            raise CorpusV2StoreError("C2_INDEX_METADATA_INVALID") from error
        if not is_valid_embedding_model(self.embedding_model):
            raise CorpusV2StoreError("C2_INDEX_METADATA_INVALID")
        _require_dimension(self.vector_dimensions)
        if not isinstance(self.content, str) or not self.content:
            raise CorpusV2StoreError("C2_INDEX_METADATA_INVALID")
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.chunk.text_sha256:
            raise CorpusV2StoreError("C2_INDEX_METADATA_INVALID")
        _require_sha256(self.expected_embedding_sha256)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source_id(self) -> str:
        return self.chunk.source_id


@dataclass(frozen=True)
class V2Embedding:
    chunk_id: str
    embedding_model: str
    embedding_sha256: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not is_valid_public_chunk_id(self.chunk_id) or not is_valid_embedding_model(self.embedding_model):
            raise CorpusV2StoreError("C2_EMBEDDING_INVALID")
        _require_sha256(self.embedding_sha256)
        if vector_sha256(self.vector) != self.embedding_sha256:
            raise CorpusV2StoreError("C2_EMBEDDING_INVALID")


@dataclass(frozen=True)
class V2StoredChunk:
    chunk_id: str
    release_id: str
    source_id: str
    source_sha256: str
    text_sha256: str
    manifest_sha256: str
    location: str
    embedding_model: str
    vector_dimensions: int
    embedding_sha256: str
    content: str
    vector: tuple[float, ...]


class V2IndexReader(Protocol):
    def read_rows(self) -> Sequence[V2StoredChunk]: ...


class CorpusV2Store:
    """Candidate-only pure store facade with a fake row buffer for contract tests.

    It never opens a database connection and deliberately has no legacy-store alias.
    """

    def __init__(self, database_url: str, legacy_database_url: str, release_config: ReleaseConfig) -> None:
        candidate_name = database_name_from_url(database_url)
        legacy_name = database_name_from_url(legacy_database_url)
        forbidden = {release_config.configured_database_name, *release_config.legacy_database_names, legacy_name}
        if candidate_name != release_config.candidate_database_name or candidate_name in forbidden:
            raise CorpusV2StoreError("C2_DATABASE_TARGET_INVALID")
        self._release_config = release_config
        self._rows: list[V2StoredChunk] = []

    def prepare_rows(
        self,
        manifest: Iterable[V2ChunkManifestEntry],
        embeddings: Iterable[V2Embedding],
    ) -> tuple[V2StoredChunk, ...]:
        entries = tuple(manifest)
        embedded = tuple(embeddings)
        expected = _index_by_chunk_id(entries, "C2_INDEX_EXACT_SET_MISMATCH")
        observed = _index_by_chunk_id(embedded, "C2_INDEX_EXACT_SET_MISMATCH")
        if set(expected) != set(observed):
            raise CorpusV2StoreError("C2_INDEX_EXACT_SET_MISMATCH")
        rows: list[V2StoredChunk] = []
        for chunk_id, entry in expected.items():
            embedding = observed[chunk_id]
            if embedding.embedding_model != entry.embedding_model or entry.vector_dimensions != VECTOR_DIMENSIONS:
                raise CorpusV2StoreError("C2_EMBEDDING_INVALID")
            try:
                _require_finite_vector(embedding.vector)
            except CorpusV2StoreError as error:
                raise CorpusV2StoreError("C2_EMBEDDING_INVALID") from error
            if (
                vector_sha256(embedding.vector) != embedding.embedding_sha256
                or embedding.embedding_sha256 != entry.expected_embedding_sha256
            ):
                raise CorpusV2StoreError("C2_EMBEDDING_INVALID")
            rows.append(
                V2StoredChunk(
                    chunk_id=chunk_id,
                    release_id=self._release_config.release_id,
                    source_id=entry.source_id,
                    source_sha256=entry.source_sha256,
                    text_sha256=entry.chunk.text_sha256,
                    manifest_sha256=entry.manifest_sha256,
                    location=entry.location,
                    embedding_model=entry.embedding_model,
                    vector_dimensions=entry.vector_dimensions,
                    embedding_sha256=embedding.embedding_sha256,
                    content=entry.content,
                    vector=embedding.vector,
                )
            )
        return tuple(rows)

    # Test-only injected boundary: no database I/O exists in this implementation.
    def replace_fake_rows(self, rows: Iterable[V2StoredChunk]) -> None:
        self._rows = list(rows)

    def insert_fake_row(self, row: V2StoredChunk) -> None:
        self._rows.append(row)

    def read_rows(self) -> Sequence[V2StoredChunk]:
        return tuple(self._rows)


def validate_v2_index(
    reader: V2IndexReader,
    release_binding: ReleaseBinding,
    expected_manifest: Iterable[V2ChunkManifestEntry],
) -> None:
    """Require the read-only candidate rows to equal the sealed manifest exactly."""
    expected = _index_by_chunk_id(tuple(expected_manifest), "C2_INDEX_EXACT_SET_MISMATCH")
    if len(expected) != release_binding.chunk_count:
        raise CorpusV2StoreError("C2_INDEX_EXACT_SET_MISMATCH")
    for entry in expected.values():
        if entry.manifest_sha256 != release_binding.index_manifest_sha256:
            raise CorpusV2StoreError("C2_INDEX_EXACT_SET_MISMATCH")
    rows = tuple(reader.read_rows())
    actual = _index_by_chunk_id(rows, "C2_INDEX_EXACT_SET_MISMATCH")
    if set(actual) != set(expected):
        raise CorpusV2StoreError("C2_INDEX_EXACT_SET_MISMATCH")
    for chunk_id, entry in expected.items():
        row = actual[chunk_id]
        if (
            row.release_id != release_binding.release_id
            or row.source_id != entry.source_id
            or row.source_sha256 != entry.source_sha256
            or row.text_sha256 != entry.chunk.text_sha256
            or hashlib.sha256(row.content.encode("utf-8")).hexdigest() != entry.chunk.text_sha256
            or row.manifest_sha256 != release_binding.index_manifest_sha256
            or row.location != entry.location
            or row.embedding_model != entry.embedding_model
            or row.vector_dimensions != entry.vector_dimensions
            or row.vector_dimensions != VECTOR_DIMENSIONS
            or row.embedding_sha256 != vector_sha256(row.vector)
            or row.embedding_sha256 != entry.expected_embedding_sha256
        ):
            raise CorpusV2StoreError("C2_INDEX_EXACT_SET_MISMATCH")


T = TypeVar("T")


def _index_by_chunk_id(records: Iterable[T], error_code: str) -> dict[str, T]:
    indexed: dict[str, T] = {}
    for record in records:
        chunk_id = getattr(record, "chunk_id", None)
        if not isinstance(chunk_id, str) or chunk_id in indexed:
            raise CorpusV2StoreError(error_code)
        indexed[chunk_id] = record
    return indexed
