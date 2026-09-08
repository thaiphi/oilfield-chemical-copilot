"""Canonical serialization for public, aggregate-only Corpus V2 records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class CorpusV2CanonicalError(ValueError):
    """Raised when a record is not safe for public canonicalization."""


_FORBIDDEN_FIELD_FRAGMENTS = ("path", "text", "content", "credential", "secret", "token", "key")


def _validate(value: Any, *, field_name: str | None = None) -> None:
    if field_name and any(fragment in field_name.lower() for fragment in _FORBIDDEN_FIELD_FRAGMENTS):
        raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
    if isinstance(value, (float, Path)):
        raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise CorpusV2CanonicalError("C2_CANONICAL_INVALID")
            _validate(nested, field_name=key)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate(nested)
    elif value is not None and not isinstance(value, (str, int, bool)):
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
