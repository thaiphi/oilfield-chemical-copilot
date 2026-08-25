"""Fail-closed metadata preflight for future private retrieval experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psycopg
from psycopg.rows import dict_row

from oilfield_chemical_copilot.evaluation.private_retrieval import PRIVATE_RETRIEVAL_ROOT


class E1IndexPreflightError(RuntimeError):
    """Raised when the approved private evaluation index cannot be verified."""


@dataclass(frozen=True)
class IndexFingerprint:
    """Metadata-only identity for one approved evaluation index."""

    chunk_count: int
    distinct_source_count: int
    embedding_models: tuple[str, ...]
    embedding_dimensions: tuple[int, ...]
    inventory_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "distinct_source_count": self.distinct_source_count,
            "embedding_models": list(self.embedding_models),
            "embedding_dimensions": list(self.embedding_dimensions),
            "inventory_sha256": self.inventory_sha256,
        }


def _fail_closed() -> None:
    raise E1IndexPreflightError("E1_INDEX_PREFLIGHT_FAILED")


def _require_private_contract_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PRIVATE_RETRIEVAL_ROOT.resolve())
    except ValueError:
        _fail_closed()
    return resolved


def capture_index_fingerprint(database_url: str) -> IndexFingerprint:
    """Read index identity data without loading fixtures or initializing any model."""
    if not database_url.strip():
        _fail_closed()
    try:
        with psycopg.connect(
            database_url,
            options="-c default_transaction_read_only=on",
            row_factory=dict_row,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select
                        count(*) as chunk_count,
                        count(distinct source_file) as distinct_source_count,
                        array_agg(distinct embedding_model order by embedding_model)
                            filter (where embedding is not null) as embedding_models,
                        array_agg(distinct vector_dims(embedding) order by vector_dims(embedding))
                            filter (where embedding is not null) as embedding_dimensions
                    from chunks
                    """
                )
                summary = cursor.fetchone()
                cursor.execute(
                    """
                    select chunk_id, source_file, parser_type, page_or_sheet, embedding_model
                    from chunks
                    order by chunk_id, source_file, parser_type, page_or_sheet, embedding_model
                    """
                )
                inventory_rows = cursor.fetchall()
    except Exception:
        _fail_closed()
    if summary is None:
        _fail_closed()
    models = tuple(str(value) for value in (summary["embedding_models"] or ()))
    dimensions = tuple(int(value) for value in (summary["embedding_dimensions"] or ()))
    if not models or not dimensions:
        _fail_closed()
    digest = hashlib.sha256()
    for row in inventory_rows:
        digest.update(
            json.dumps(
                [
                    str(row["chunk_id"]),
                    str(row["source_file"]),
                    str(row["parser_type"]),
                    str(row["page_or_sheet"]),
                    str(row["embedding_model"]),
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return IndexFingerprint(
        chunk_count=int(summary["chunk_count"]),
        distinct_source_count=int(summary["distinct_source_count"]),
        embedding_models=models,
        embedding_dimensions=dimensions,
        inventory_sha256=digest.hexdigest(),
    )


def write_index_contract(fingerprint: IndexFingerprint, destination: Path) -> None:
    """Write an explicitly captured contract only under the private evaluation root."""
    _require_private_contract_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(fingerprint.to_mapping(), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_index_contract(path: Path) -> IndexFingerprint:
    _require_private_contract_path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if set(record) != {
            "chunk_count",
            "distinct_source_count",
            "embedding_models",
            "embedding_dimensions",
            "inventory_sha256",
        }:
            _fail_closed()
        chunk_count = record["chunk_count"]
        source_count = record["distinct_source_count"]
        models = record["embedding_models"]
        dimensions = record["embedding_dimensions"]
        digest = record["inventory_sha256"]
        if (
            type(chunk_count) is not int
            or type(source_count) is not int
            or not isinstance(models, list)
            or any(not isinstance(model, str) or not model for model in models)
            or not isinstance(dimensions, list)
            or any(type(dimension) is not int or dimension < 1 for dimension in dimensions)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            _fail_closed()
        return IndexFingerprint(
            chunk_count=chunk_count,
            distinct_source_count=source_count,
            embedding_models=tuple(models),
            embedding_dimensions=tuple(dimensions),
            inventory_sha256=digest,
        )
    except E1IndexPreflightError:
        raise
    except Exception:
        _fail_closed()


def verify_e1_index_contract(
    *,
    database_url: str,
    contract_path: Path,
    fingerprint_loader: Callable[[str], IndexFingerprint] = capture_index_fingerprint,
    on_verified: Callable[[], None] | None = None,
) -> IndexFingerprint:
    """Verify the exact approved index before any E1 fixture or model initialization."""
    expected = _read_index_contract(contract_path)
    try:
        actual = fingerprint_loader(database_url)
    except Exception:
        _fail_closed()
    if actual != expected:
        _fail_closed()
    if on_verified is not None:
        on_verified()
    return actual
