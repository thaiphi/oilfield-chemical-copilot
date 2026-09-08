"""Canonical serialization for public, aggregate-only Corpus V2 records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class CorpusV2CanonicalError(ValueError):
    """Raised when a record is not safe for public canonicalization."""


_STRING_FIELDS = {
    "release_id", "source_id", "chunk_id", "stage", "disposition", "reason_code",
    "reviewer_id", "status", "model", "sha256", "source_sha256", "content_sha256",
    "text_sha256", "embedding_sha256", "index_manifest_sha256",
}


def _validate(record: Mapping[str, Any]) -> None:
    for field_name, value in record.items():
        if not isinstance(field_name, str) or isinstance(value, (float, Path, list, tuple, Mapping)):
            raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        if isinstance(value, str) and field_name not in _STRING_FIELDS:
            raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
        if value is not None and not isinstance(value, (str, int, bool)):
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
