"""Canonical serialization for public, aggregate-only Corpus V2 records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
import re
from typing import Any

from .models import is_valid_embedding_model, is_valid_timestamp


class CorpusV2CanonicalError(ValueError):
    """Raised when a record is not safe for public canonicalization."""


_RECORD_SCHEMAS = {
    frozenset({"source_id", "source_sha256"}): {"source_id": "source_id", "source_sha256": "sha256"},
    frozenset({"source_id", "acquired_at", "content_sha256", "byte_count"}): {
        "source_id": "source_id", "acquired_at": "timestamp", "content_sha256": "sha256", "byte_count": "count",
    },
    frozenset({"source_id", "extracted_at", "extractor", "text_sha256", "character_count"}): {
        "source_id": "source_id", "extracted_at": "timestamp", "extractor": "extractor",
        "text_sha256": "sha256", "character_count": "count",
    },
    frozenset({"source_id", "extracted_at", "extractor", "text_sha256", "character_count", "outcome"}): {
        "source_id": "source_id", "extracted_at": "timestamp", "extractor": "extractor",
        "text_sha256": "optional_sha256", "character_count": "count", "outcome": "outcome",
    },
    frozenset({"chunk_id", "source_id", "ordinal", "text_sha256", "character_count"}): {
        "chunk_id": "chunk_id", "source_id": "source_id", "ordinal": "count",
        "text_sha256": "sha256", "character_count": "count",
    },
    frozenset({"chunk_id", "embedding_model", "embedding_sha256", "vector_dimensions"}): {
        "chunk_id": "chunk_id", "embedding_model": "embedding_model", "embedding_sha256": "sha256",
        "vector_dimensions": "positive_count",
    },
    frozenset({"release_id", "source_register_sha256", "index_manifest_sha256", "chunk_count"}): {
        "release_id": "release_id", "source_register_sha256": "sha256",
        "index_manifest_sha256": "sha256", "chunk_count": "count",
    },
}
_SOURCE_ID = re.compile(r"^(?:doc-\d+|drive:1[A-Za-z0-9_-]{19,127})$")
_CHUNK_ID = re.compile(r"^(?:doc-\d+|drive:1[A-Za-z0-9_-]{19,127}):\d+$")
_RELEASE_ID = re.compile(r"^corpus-v2-\d{4}-\d{2}-\d{2}$")
_EXTRACTOR = re.compile(r"^(?:pypdf|python-docx|synthetic)$")


def _validate(record: Mapping[str, Any]) -> None:
    schema = _RECORD_SCHEMAS.get(frozenset(record))
    if schema is None:
        if any(not isinstance(key, str) or not isinstance(value, (int, bool, type(None))) for key, value in record.items()):
            raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        return
    for field_name, field_type in schema.items():
        value = record[field_name]
        if field_type in {"count", "positive_count"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
            if field_type == "positive_count" and value == 0:
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        elif field_type in {"sha256", "optional_sha256"}:
            if field_type == "optional_sha256" and value is None:
                continue
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        elif field_type == "source_id":
            if not isinstance(value, str) or not _SOURCE_ID.fullmatch(value):
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        elif field_type == "chunk_id":
            if not isinstance(value, str) or not _CHUNK_ID.fullmatch(value):
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        elif field_type == "release_id":
            if not isinstance(value, str) or not _RELEASE_ID.fullmatch(value):
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        elif field_type == "extractor":
            if not isinstance(value, str) or not _EXTRACTOR.fullmatch(value):
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        elif field_type == "embedding_model":
            if not is_valid_embedding_model(value):
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        elif field_type == "timestamp":
            if not is_valid_timestamp(value):
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        elif field_type == "outcome" and value not in {"SUCCESS", "NON_TEXT", "EMPTY", "UNSUPPORTED", "FAILED"}:
            raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
    if "outcome" in schema:
        if record["outcome"] == "SUCCESS" and record["text_sha256"] is None:
            raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        if record["outcome"] != "SUCCESS" and (record["text_sha256"] is not None or record["character_count"] != 0):
            raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")


def canonical_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Return newline-terminated UTF-8 JSONL with deterministic key ordering."""
    rows: list[bytes] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        _validate(record)
        rows.append(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return b"".join(row + b"\n" for row in rows)


def sha256_canonical_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_jsonl(records)).hexdigest()
