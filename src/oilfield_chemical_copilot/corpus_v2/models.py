"""Fail-closed, public metadata contracts for the Corpus V2 release."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import re
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


class ExtractionOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    NON_TEXT = "NON_TEXT"
    EMPTY = "EMPTY"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"


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


class StageArtifactKind(str, Enum):
    CHUNK_MANIFEST = "CHUNK_MANIFEST"
    EMBEDDING_MANIFEST = "EMBEDDING_MANIFEST"
    INDEX_VALIDATION_REPORT = "INDEX_VALIDATION_REPORT"
    EVALUATION_REPORT = "EVALUATION_REPORT"
    PROMOTION_APPROVAL = "PROMOTION_APPROVAL"
    PROMOTION_RECEIPT = "PROMOTION_RECEIPT"

    @classmethod
    def for_stage(cls, stage: Stage) -> "StageArtifactKind":
        mapping = {
            Stage.CHUNKED: cls.CHUNK_MANIFEST,
            Stage.EMBEDDED: cls.EMBEDDING_MANIFEST,
            Stage.INDEX_VALIDATED: cls.INDEX_VALIDATION_REPORT,
            Stage.EVALUATED: cls.EVALUATION_REPORT,
            Stage.PROMOTION_READY: cls.PROMOTION_APPROVAL,
            Stage.PROMOTED: cls.PROMOTION_RECEIPT,
        }
        try:
            return mapping[stage]
        except KeyError as error:
            raise CorpusV2ContractError("C2_STAGE_ARTIFACT_INVALID") from error


_EMBEDDING_MODEL = re.compile(r"^(?:local-model|sentence-transformers/[A-Za-z0-9][A-Za-z0-9._-]{0,127})$")


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


def is_valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not (value.endswith("Z") or re.search(r"[+-]\d{2}:\d{2}$", value)):
        return False
    try:
        timestamp = value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(timestamp).tzinfo is not None
    except ValueError:
        return False


def is_valid_embedding_model(value: object) -> bool:
    return isinstance(value, str) and _EMBEDDING_MODEL.fullmatch(value) is not None


@dataclass(frozen=True)
class ReleaseConfig:
    release_id: str
    release_root: Path
    candidate_database_name: str
    configured_database_name: str
    legacy_database_names: tuple[str, ...]
    expected_source_count: int
    source_register_sha256: str

    def __post_init__(self) -> None:
        try:
            _non_empty(self.release_id, code="C2_CONFIG_INVALID")
            if not isinstance(self.release_root, Path):
                _invalid_config()
            candidate = _non_empty(self.candidate_database_name, code="C2_CONFIG_INVALID")
            configured = _non_empty(self.configured_database_name, code="C2_CONFIG_INVALID")
            if not isinstance(self.legacy_database_names, tuple):
                _invalid_config()
            legacy = tuple(
                _non_empty(name, code="C2_CONFIG_INVALID") for name in self.legacy_database_names
            )
            if (
                isinstance(self.expected_source_count, bool)
                or not isinstance(self.expected_source_count, int)
                or self.expected_source_count <= 0
            ):
                _invalid_config()
            _sha256(self.source_register_sha256)
        except (CorpusV2ContractError, TypeError):
            _invalid_config()
        if candidate == configured or candidate in legacy:
            _invalid_config()

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
        if not is_valid_timestamp(self.acquired_at):
            raise CorpusV2ContractError("C2_TIMESTAMP_INVALID")
        _sha256(self.content_sha256)
        _count(self.byte_count)


@dataclass(frozen=True)
class ExtractionRecord:
    source_id: str
    extracted_at: str
    extractor: str
    text_sha256: str | None
    character_count: int
    outcome: ExtractionOutcome = ExtractionOutcome.SUCCESS

    def __post_init__(self) -> None:
        _non_empty(self.source_id)
        if not is_valid_timestamp(self.extracted_at):
            raise CorpusV2ContractError("C2_TIMESTAMP_INVALID")
        _non_empty(self.extractor)
        _count(self.character_count)
        if not isinstance(self.outcome, ExtractionOutcome):
            raise CorpusV2ContractError("C2_EXTRACTION_INVALID")
        if self.outcome is ExtractionOutcome.SUCCESS:
            _sha256(self.text_sha256)
        elif self.text_sha256 is not None or self.character_count != 0:
            raise CorpusV2ContractError("C2_EXTRACTION_INVALID")


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
        if not is_valid_embedding_model(self.embedding_model):
            raise CorpusV2ContractError("C2_EMBEDDING_MODEL_INVALID")
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
