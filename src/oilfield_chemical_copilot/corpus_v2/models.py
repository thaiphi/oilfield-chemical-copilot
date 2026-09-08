"""Fail-closed, public metadata contracts for the Corpus V2 release."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class CorpusV2ContractError(ValueError):
    """Raised when a public Corpus V2 contract is malformed."""


class SourceDisposition(str, Enum):
    INDEXED = "INDEXED"
    DUPLICATE = "DUPLICATE"
    NON_TEXT = "NON_TEXT"
    EMPTY = "EMPTY"
    UNSUPPORTED = "UNSUPPORTED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    INTENTIONALLY_EXCLUDED = "INTENTIONALLY_EXCLUDED"
    UNRESOLVED = "UNRESOLVED"


class Stage(str, Enum):
    REGISTERED = "REGISTERED"
    ACQUIRED = "ACQUIRED"
    PARSED = "PARSED"
    REVIEWED = "REVIEWED"
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    INDEX_VALIDATED = "INDEX_VALIDATED"
    EVALUATED = "EVALUATED"
    PROMOTION_READY = "PROMOTION_READY"
    PROMOTED = "PROMOTED"


def _invalid_config() -> None:
    raise CorpusV2ContractError("C2_CONFIG_INVALID")


def _non_empty(value: object, *, code: str = "C2_CONTRACT_INVALID") -> str:
    if not isinstance(value, str) or not value:
        raise CorpusV2ContractError(code)
    return value


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusV2ContractError("C2_COUNT_INVALID")
    return value


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CorpusV2ContractError("C2_SHA256_INVALID")
    return value


@dataclass(frozen=True)
class ReleaseConfig:
    release_id: str
    release_root: Path
    candidate_database_name: str
    configured_database_name: str
    legacy_database_names: tuple[str, ...]
    expected_source_count: int
    source_register_sha256: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReleaseConfig":
        expected = {
            "release_id", "release_root", "candidate_database_name",
            "configured_database_name", "legacy_database_names", "expected_source_count",
            "source_register_sha256",
        }
        if set(payload) != expected:
            _invalid_config()
        try:
            release_id = _non_empty(payload["release_id"], code="C2_CONFIG_INVALID")
            release_root_value = _non_empty(payload["release_root"], code="C2_CONFIG_INVALID")
            candidate = _non_empty(payload["candidate_database_name"], code="C2_CONFIG_INVALID")
            configured = _non_empty(payload["configured_database_name"], code="C2_CONFIG_INVALID")
            legacy_value = payload["legacy_database_names"]
            if isinstance(legacy_value, (str, bytes)) or not isinstance(legacy_value, list):
                _invalid_config()
            legacy = tuple(_non_empty(name, code="C2_CONFIG_INVALID") for name in legacy_value)
            count = payload["expected_source_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                _invalid_config()
            source_register_sha256 = _sha256(payload["source_register_sha256"])
        except (KeyError, TypeError):
            _invalid_config()
        if candidate != configured or candidate in legacy:
            _invalid_config()
        return cls(
            release_id=release_id,
            release_root=Path(release_root_value),
            candidate_database_name=candidate,
            configured_database_name=configured,
            legacy_database_names=legacy,
            expected_source_count=count,
            source_register_sha256=source_register_sha256,
        )


@dataclass(frozen=True)
class ApprovedSource:
    source_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        _non_empty(self.source_id)
        _sha256(self.source_sha256)


@dataclass(frozen=True)
class AcquisitionRecord:
    source_id: str
    acquired_at: str
    content_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        _non_empty(self.source_id)
        _non_empty(self.acquired_at)
        _sha256(self.content_sha256)
        _count(self.byte_count)


@dataclass(frozen=True)
class ExtractionRecord:
    source_id: str
    extracted_at: str
    extractor: str
    text_sha256: str
    character_count: int

    def __post_init__(self) -> None:
        _non_empty(self.source_id)
        _non_empty(self.extracted_at)
        _non_empty(self.extractor)
        _sha256(self.text_sha256)
        _count(self.character_count)


@dataclass(frozen=True)
class CorpusV2Chunk:
    chunk_id: str
    source_id: str
    ordinal: int
    text_sha256: str
    character_count: int

    def __post_init__(self) -> None:
        _non_empty(self.chunk_id)
        _non_empty(self.source_id)
        _count(self.ordinal)
        _sha256(self.text_sha256)
        _count(self.character_count)


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    embedding_model: str
    embedding_sha256: str
    vector_dimensions: int

    def __post_init__(self) -> None:
        _non_empty(self.chunk_id)
        _non_empty(self.embedding_model)
        _sha256(self.embedding_sha256)
        if _count(self.vector_dimensions) <= 0:
            raise CorpusV2ContractError("C2_COUNT_INVALID")


@dataclass(frozen=True)
class ReleaseBinding:
    release_id: str
    source_register_sha256: str
    index_manifest_sha256: str
    chunk_count: int

    def __post_init__(self) -> None:
        _non_empty(self.release_id)
        _sha256(self.source_register_sha256)
        _sha256(self.index_manifest_sha256)
        _count(self.chunk_count)
