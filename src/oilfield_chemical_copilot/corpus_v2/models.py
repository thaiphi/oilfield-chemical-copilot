"""Fail-closed, public metadata contracts for the Corpus V2 release."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
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
    ACQUISITION_MANIFEST = "ACQUISITION_MANIFEST"
    CHUNK_MANIFEST = "CHUNK_MANIFEST"
    EMBEDDING_MANIFEST = "EMBEDDING_MANIFEST"
    INDEX_VALIDATION_REPORT = "INDEX_VALIDATION_REPORT"
    EVALUATION_REPORT = "EVALUATION_REPORT"
    PROMOTION_APPROVAL = "PROMOTION_APPROVAL"
    PROMOTION_RECEIPT = "PROMOTION_RECEIPT"

    @classmethod
    def for_stage(cls, stage: Stage) -> "StageArtifactKind":
        mapping = {
            Stage.ACQUIRED: cls.ACQUISITION_MANIFEST,
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


_EMBEDDING_MODELS = frozenset({"local-model", "sentence-transformers/all-MiniLM-L6-v2"})
_PUBLIC_SOURCE_ID = re.compile(r"doc-[1-9][0-9]*")
_PUBLIC_CHUNK_ID = re.compile(r"doc-[1-9][0-9]*/chunk-(?:0|[1-9][0-9]*)")
_PUBLIC_RELEASE_ID = re.compile(r"corpus-v2-[a-z0-9]+(?:-[a-z0-9]+)*")


def is_valid_public_source_id(value: object) -> bool:
    """Accept a stable document pseudonym, never an upstream source identifier."""
    return isinstance(value, str) and _PUBLIC_SOURCE_ID.fullmatch(value) is not None


def is_valid_public_chunk_id(value: object) -> bool:
    return isinstance(value, str) and (
        _PUBLIC_CHUNK_ID.fullmatch(value) is not None
        or re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def is_valid_chunk_provenance(chunk_id: object, source_id: object, ordinal: object) -> bool:
    """Require the chunk pseudonym to encode this exact source and ordinal."""
    return (
        is_valid_public_chunk_id(chunk_id)
        and is_valid_public_source_id(source_id)
        and isinstance(ordinal, int)
        and not isinstance(ordinal, bool)
        and ordinal >= 0
        and chunk_id == f"{source_id}/chunk-{ordinal}"
    )


def _public_source_id(value: object) -> None:
    if not is_valid_public_source_id(value):
        raise CorpusV2ContractError("C2_SOURCE_ID_INVALID")


def _public_chunk_id(value: object) -> None:
    if not is_valid_public_chunk_id(value):
        raise CorpusV2ContractError("C2_CHUNK_ID_INVALID")


def _invalid_config() -> None:
    raise CorpusV2ContractError("C2_CONFIG_INVALID")


def is_valid_public_release_id(value: object) -> bool:
    """Accept a release label suitable for public aggregates and marker binding."""
    return isinstance(value, str) and _PUBLIC_RELEASE_ID.fullmatch(value) is not None


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
    return isinstance(value, str) and value in _EMBEDDING_MODELS


@dataclass(frozen=True)
class ReleaseConfig:
    release_id: str
    release_root: Path
    candidate_database_name: str
    configured_database_name: str
    legacy_database_names: tuple[str, ...]
    expected_source_count: int
    source_register_sha256: str
    critical_source_register_sha256: str

    def __post_init__(self) -> None:
        try:
            if not is_valid_public_release_id(self.release_id):
                _invalid_config()
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
            _sha256(self.critical_source_register_sha256)
        except (CorpusV2ContractError, TypeError):
            _invalid_config()
        if candidate == configured or candidate in legacy:
            _invalid_config()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReleaseConfig":
        expected = {
            "release_id", "release_root", "candidate_database_name",
            "configured_database_name", "legacy_database_names", "expected_source_count",
            "source_register_sha256", "critical_source_register_sha256",
        }
        if set(payload) != expected:
            _invalid_config()
        try:
            release_id = _non_empty(payload["release_id"], code="C2_CONFIG_INVALID")
            if not is_valid_public_release_id(release_id):
                _invalid_config()
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
            critical_source_register_sha256 = _sha256(payload["critical_source_register_sha256"])
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
            critical_source_register_sha256=critical_source_register_sha256,
        )


@dataclass(frozen=True)
class ApprovedSource:
    source_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        _public_source_id(self.source_id)
        _sha256(self.source_sha256)


@dataclass(frozen=True)
class AcquisitionRecord:
    source_id: str
    acquired_at: str
    content_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        _public_source_id(self.source_id)
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
        _public_source_id(self.source_id)
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
class ChunkProvenance:
    """Private provenance required to authenticate a processing SHA256 identity."""

    release_id: str
    source_byte_sha256: str
    parser_policy_version: str
    chunk_policy_version: str
    location: str

    def __post_init__(self) -> None:
        from .processing import CorpusV2ProcessingError, validate_location

        if not is_valid_public_release_id(self.release_id):
            raise CorpusV2ContractError("C2_CHUNK_PROVENANCE_INVALID")
        _sha256(self.source_byte_sha256)
        for policy in (self.parser_policy_version, self.chunk_policy_version):
            if not isinstance(policy, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", policy) is None:
                raise CorpusV2ContractError("C2_CHUNK_PROVENANCE_INVALID")
        try:
            validate_location(self.location)
        except CorpusV2ProcessingError:
            raise CorpusV2ContractError("C2_CHUNK_PROVENANCE_INVALID") from None

    def chunk_id(self, source_id: str, ordinal: int, text_sha256: str) -> str:
        identity = {
            "release_id": self.release_id,
            "source_byte_sha256": self.source_byte_sha256,
            "parser_policy_version": self.parser_policy_version,
            "chunk_policy_version": self.chunk_policy_version,
            "location": self.location,
            "source_id": source_id,
            "ordinal": ordinal,
            "text_sha256": text_sha256,
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class CorpusV2Chunk:
    chunk_id: str
    source_id: str
    ordinal: int
    text_sha256: str
    character_count: int
    provenance: ChunkProvenance | None = None

    def __post_init__(self) -> None:
        _public_chunk_id(self.chunk_id)
        _public_source_id(self.source_id)
        _count(self.ordinal)
        _sha256(self.text_sha256)
        _count(self.character_count)
        if self.provenance is None:
            if not is_valid_chunk_provenance(self.chunk_id, self.source_id, self.ordinal):
                raise CorpusV2ContractError("C2_CHUNK_PROVENANCE_INVALID")
        elif (
            not isinstance(self.provenance, ChunkProvenance)
            or self.chunk_id != self.provenance.chunk_id(self.source_id, self.ordinal, self.text_sha256)
        ):
            raise CorpusV2ContractError("C2_CHUNK_PROVENANCE_INVALID")

    @classmethod
    def from_loaded_chunk(cls, chunk: Any, *, release_id: str) -> "CorpusV2Chunk":
        """Authenticate processing output without replacing its SHA256 identity."""
        from oilfield_chemical_copilot.ingest.models import LoadedChunk

        if not isinstance(chunk, LoadedChunk):
            raise CorpusV2ContractError("C2_CHUNK_PROVENANCE_INVALID")
        metadata = chunk.metadata
        try:
            provenance = ChunkProvenance(
                release_id, metadata.extra["source_byte_sha256"],
                metadata.extra["parser_policy_version"], metadata.extra["chunk_policy_version"],
                metadata.page_or_sheet,
            )
            digest = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            if metadata.extra["text_sha256"] != digest or metadata.source_file != metadata.extra["source_id"]:
                raise CorpusV2ContractError("C2_CHUNK_PROVENANCE_INVALID")
            return cls(metadata.chunk_id, metadata.extra["source_id"], metadata.chunk_index,
                       digest, len(chunk.text), provenance)
        except (KeyError, TypeError, AttributeError):
            raise CorpusV2ContractError("C2_CHUNK_PROVENANCE_INVALID") from None


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    embedding_model: str
    embedding_sha256: str
    vector_dimensions: int

    def __post_init__(self) -> None:
        _public_chunk_id(self.chunk_id)
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
        if not is_valid_public_release_id(self.release_id):
            raise CorpusV2ContractError("C2_CONTRACT_INVALID")
        _sha256(self.source_register_sha256)
        _sha256(self.index_manifest_sha256)
        _count(self.chunk_count)
