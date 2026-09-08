"""Pure, provenance-bound Corpus V2 extraction and chunking contracts.

This module deliberately accepts an injected parser.  It never discovers sources,
opens a Drive client, writes extracted text, or uses a production database.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk

from .models import (
    ExtractionOutcome,
    ExtractionRecord,
    SourceDisposition,
    is_valid_public_release_id,
    is_valid_public_source_id,
)


class CorpusV2ProcessingError(RuntimeError):
    """Raised when a V2 extraction, review, or chunk contract is violated."""


class UnsupportedSnapshotError(ValueError):
    """Raised by an injected parser only when its format is mechanically unsupported."""


class ExtractionLedger(Protocol):
    """The one narrow ledger action used by the processing controller."""

    def record_extraction(self, extraction: ExtractionRecord) -> None: ...


Parser = Callable[[Path], Sequence[LoadedChunk]]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POLICY_VERSION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REVIEWER_ID = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_SHEET_LOCATION = re.compile(r"^sheet:[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")
_PAGE_LOCATION = re.compile(r"^page:[1-9][0-9]*$")
_SIMPLE_LOCATION = frozenset({"document", "csv"})


@dataclass(frozen=True)
class ExtractionCandidate:
    """Private-run extraction facts, excluding raw source text from durable metadata."""

    record: ExtractionRecord
    source_byte_sha256: str
    parser_policy_version: str
    parser_types: tuple[str, ...]
    locations: tuple[str, ...]
    chunks: tuple[LoadedChunk, ...]


@dataclass(frozen=True)
class ReviewDecision:
    """One explicit human disposition; a duplicate must name its representative."""

    source_id: str
    disposition: SourceDisposition
    reviewer_id: str
    reason_code: str
    representative_source_id: str | None = None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, code: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CorpusV2ProcessingError(code)
    return value


def _require_policy_version(value: object) -> str:
    if not isinstance(value, str) or _POLICY_VERSION.fullmatch(value) is None:
        raise CorpusV2ProcessingError("C2_POLICY_INVALID")
    return value


def validate_location(location: object) -> str:
    """Validate a portable, non-path page/sheet/document location."""
    if not isinstance(location, str) or (
        location not in _SIMPLE_LOCATION
        and _PAGE_LOCATION.fullmatch(location) is None
        and _SHEET_LOCATION.fullmatch(location) is None
    ):
        raise CorpusV2ProcessingError("C2_LOCATION_INVALID")
    return location


def _content_digest(chunks: Iterable[LoadedChunk]) -> str:
    payload = [
        {
            "location": chunk.metadata.page_or_sheet,
            "text_sha256": _sha256_bytes(chunk.text.encode("utf-8")),
        }
        for chunk in chunks
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_existing_document(snapshot_path: Path) -> Sequence[LoadedChunk]:
    """Adapt the existing parser's unsupported-format signal to the V2 contract.

    Calling this function is an operational action performed only after later
    authorization; keeping it here provides no discovery or implicit processing.
    """
    from oilfield_chemical_copilot.ingest.parsers import parse_document

    try:
        return parse_document(snapshot_path, source_root=None, topic="corpus-v2")
    except ValueError as error:
        raise UnsupportedSnapshotError from error


def build_v2_chunk(
    *,
    release_id: str,
    source_id: str,
    source_byte_sha256: str,
    parser_policy_version: str,
    chunk_policy_version: str,
    location: str,
    ordinal: int,
    text: str,
    sealed_source_ids: Iterable[str],
    chunk_id: str | None = None,
    parser_type: str = "v2-parser",
) -> LoadedChunk:
    """Build one V2 chunk whose identifier binds every provenance-bearing field."""
    if not is_valid_public_release_id(release_id):
        raise CorpusV2ProcessingError("C2_RELEASE_INVALID")
    if not is_valid_public_source_id(source_id) or source_id not in set(sealed_source_ids):
        raise CorpusV2ProcessingError("C2_SOURCE_UNKNOWN")
    _require_sha256(source_byte_sha256, code="C2_SNAPSHOT_HASH_INVALID")
    _require_policy_version(parser_policy_version)
    _require_policy_version(chunk_policy_version)
    validate_location(location)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise CorpusV2ProcessingError("C2_CHUNK_ORDINAL_INVALID")
    if not isinstance(text, str) or not text.strip():
        raise CorpusV2ProcessingError("C2_CHUNK_TEXT_INVALID")
    if not isinstance(parser_type, str) or not _POLICY_VERSION.fullmatch(parser_type):
        raise CorpusV2ProcessingError("C2_PARSER_INVALID")

    text_sha256 = _sha256_bytes(text.encode("utf-8"))
    identity = {
        "chunk_policy_version": chunk_policy_version,
        "location": location,
        "ordinal": ordinal,
        "parser_policy_version": parser_policy_version,
        "release_id": release_id,
        "source_byte_sha256": source_byte_sha256,
        "source_id": source_id,
        "text_sha256": text_sha256,
    }
    expected_chunk_id = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if chunk_id is not None and chunk_id != expected_chunk_id:
        raise CorpusV2ProcessingError("C2_CHUNK_ID_INVALID")
    return LoadedChunk(
        text=text,
        metadata=ChunkMetadata(
            source_file=source_id,
            source_path="",
            topic="corpus-v2",
            parser_type=parser_type,
            page_or_sheet=location,
            chunk_index=ordinal,
            chunk_id=expected_chunk_id,
            extra={
                "chunk_policy_version": chunk_policy_version,
                "parser_policy_version": parser_policy_version,
                "source_byte_sha256": source_byte_sha256,
                "source_id": source_id,
                "text_sha256": text_sha256,
            },
        ),
    )


def ensure_unique_v2_chunk_ids(chunks: Iterable[LoadedChunk]) -> None:
    """Reject a manifest which repeats a provenance-bound V2 chunk identity."""
    identifiers = [chunk.metadata.chunk_id for chunk in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise CorpusV2ProcessingError("C2_CHUNK_DUPLICATE")


def extract_snapshot(
    *,
    release_id: str,
    source_id: str,
    snapshot_path: Path,
    expected_byte_sha256: str,
    parser_policy_version: str,
    parser: Parser,
    sealed_source_ids: Iterable[str],
    chunk_policy_version: str = "chunk-v1",
    extracted_at: str | None = None,
) -> ExtractionCandidate:
    """Parse a known private snapshot after validating its recorded byte identity."""
    sealed_ids = frozenset(sealed_source_ids)
    if not is_valid_public_release_id(release_id):
        raise CorpusV2ProcessingError("C2_RELEASE_INVALID")
    if not is_valid_public_source_id(source_id) or source_id not in sealed_ids:
        raise CorpusV2ProcessingError("C2_SOURCE_UNKNOWN")
    _require_sha256(expected_byte_sha256, code="C2_SNAPSHOT_HASH_INVALID")
    _require_policy_version(parser_policy_version)
    _require_policy_version(chunk_policy_version)
    if not isinstance(snapshot_path, Path) or not snapshot_path.is_file():
        raise CorpusV2ProcessingError("C2_SNAPSHOT_INVALID")
    try:
        observed_byte_sha256 = _sha256_bytes(snapshot_path.read_bytes())
    except OSError:
        raise CorpusV2ProcessingError("C2_SNAPSHOT_INVALID") from None
    if observed_byte_sha256 != expected_byte_sha256:
        raise CorpusV2ProcessingError("C2_SNAPSHOT_HASH_MISMATCH")

    try:
        parsed = tuple(parser(snapshot_path))
    except UnsupportedSnapshotError:
        return _terminal_candidate(
            source_id, observed_byte_sha256, parser_policy_version, ExtractionOutcome.UNSUPPORTED, extracted_at
        )
    except Exception:
        return _terminal_candidate(
            source_id, observed_byte_sha256, parser_policy_version, ExtractionOutcome.FAILED, extracted_at
        )
    if not parsed:
        return _terminal_candidate(
            source_id, observed_byte_sha256, parser_policy_version, ExtractionOutcome.EMPTY, extracted_at
        )

    chunks: list[LoadedChunk] = []
    for ordinal, parsed_chunk in enumerate(parsed):
        if not isinstance(parsed_chunk, LoadedChunk):
            raise CorpusV2ProcessingError("C2_PARSER_INVALID")
        chunks.append(
            build_v2_chunk(
                release_id=release_id,
                source_id=source_id,
                source_byte_sha256=observed_byte_sha256,
                parser_policy_version=parser_policy_version,
                chunk_policy_version=chunk_policy_version,
                location=validate_location(parsed_chunk.metadata.page_or_sheet),
                ordinal=ordinal,
                text=parsed_chunk.text,
                sealed_source_ids=sealed_ids,
                parser_type=parsed_chunk.metadata.parser_type,
            )
        )
    ensure_unique_v2_chunk_ids(chunks)
    locations = tuple(chunk.metadata.page_or_sheet for chunk in chunks)
    parser_types = tuple(sorted({chunk.metadata.parser_type for chunk in chunks}))
    record = ExtractionRecord(
        source_id=source_id,
        extracted_at=extracted_at or _timestamp(),
        extractor=parser_policy_version,
        text_sha256=_content_digest(chunks),
        character_count=sum(len(chunk.text) for chunk in chunks),
        outcome=ExtractionOutcome.SUCCESS,
    )
    return ExtractionCandidate(
        record=record,
        source_byte_sha256=observed_byte_sha256,
        parser_policy_version=parser_policy_version,
        parser_types=parser_types,
        locations=locations,
        chunks=tuple(chunks),
    )


def _terminal_candidate(
    source_id: str,
    source_byte_sha256: str,
    parser_policy_version: str,
    outcome: ExtractionOutcome,
    extracted_at: str | None,
) -> ExtractionCandidate:
    return ExtractionCandidate(
        record=ExtractionRecord(
            source_id=source_id,
            extracted_at=extracted_at or _timestamp(),
            extractor=parser_policy_version,
            text_sha256=None,
            character_count=0,
            outcome=outcome,
        ),
        source_byte_sha256=source_byte_sha256,
        parser_policy_version=parser_policy_version,
        parser_types=(),
        locations=(),
        chunks=(),
    )


def record_extraction_candidate(ledger: ExtractionLedger, candidate: ExtractionCandidate) -> None:
    """Persist only the fixed extraction record; never its text or parser exception."""
    ledger.record_extraction(candidate.record)


def review_gate_status(
    extractions: Iterable[ExtractionRecord],
    decisions: Iterable[ReviewDecision],
    *,
    sealed_source_ids: Iterable[str],
    critical_source_ids: Iterable[str],
    usable_chunk_source_ids: Iterable[str],
) -> str:
    """Return READY or BLOCKED after evidence-bound final review validation."""
    sealed_ids = frozenset(sealed_source_ids)
    critical_ids = frozenset(critical_source_ids)
    usable_ids = frozenset(usable_chunk_source_ids)
    if not critical_ids.issubset(sealed_ids):
        raise CorpusV2ProcessingError("C2_SOURCE_UNKNOWN")
    if not usable_ids.issubset(sealed_ids):
        raise CorpusV2ProcessingError("C2_SOURCE_UNKNOWN")
    extractions_by_source: dict[str, ExtractionRecord] = {}
    for extraction in extractions:
        if not isinstance(extraction, ExtractionRecord) or extraction.source_id not in sealed_ids:
            raise CorpusV2ProcessingError("C2_SOURCE_UNKNOWN")
        if extraction.source_id in extractions_by_source:
            raise CorpusV2ProcessingError("C2_EXTRACTION_DUPLICATE")
        extractions_by_source[extraction.source_id] = extraction
    if set(extractions_by_source) != sealed_ids:
        raise CorpusV2ProcessingError("C2_REVIEW_INCOMPLETE")
    decisions_by_source: dict[str, ReviewDecision] = {}
    for decision in decisions:
        _validate_review_decision(decision, sealed_ids)
        if decision.source_id in decisions_by_source:
            raise CorpusV2ProcessingError("C2_DISPOSITION_DUPLICATE")
        decisions_by_source[decision.source_id] = decision
    if set(decisions_by_source) != sealed_ids:
        raise CorpusV2ProcessingError("C2_REVIEW_INCOMPLETE")

    for decision in decisions_by_source.values():
        _validate_disposition_evidence(
            decision,
            extraction=extractions_by_source[decision.source_id],
            usable_chunk_source_ids=usable_ids,
        )
        if decision.disposition is SourceDisposition.DUPLICATE:
            representative = decisions_by_source.get(decision.representative_source_id)
            if (
                representative is None
                or representative.disposition is not SourceDisposition.INDEXED
                or representative.source_id not in usable_ids
            ):
                raise CorpusV2ProcessingError("C2_DUPLICATE_REPRESENTATIVE_INVALID")
    for critical_id in critical_ids:
        extraction = extractions_by_source[critical_id]
        decision = decisions_by_source[critical_id]
        if extraction.outcome is not ExtractionOutcome.SUCCESS:
            return "BLOCKED"
        if decision.disposition is SourceDisposition.INDEXED:
            usable = critical_id in usable_ids
        elif decision.disposition is SourceDisposition.DUPLICATE:
            representative_id = decision.representative_source_id
            representative = decisions_by_source.get(representative_id)
            representative_extraction = extractions_by_source.get(representative_id)
            usable = (
                representative is not None
                and representative.disposition is SourceDisposition.INDEXED
                and representative_extraction is not None
                and representative_extraction.outcome is ExtractionOutcome.SUCCESS
                and representative_id in usable_ids
            )
        else:
            usable = False
        if not usable:
            return "BLOCKED"
    return "READY"


def _validate_disposition_evidence(
    decision: ReviewDecision,
    *,
    extraction: ExtractionRecord,
    usable_chunk_source_ids: frozenset[str],
) -> None:
    terminal_dispositions = {
        ExtractionOutcome.NON_TEXT: SourceDisposition.NON_TEXT,
        ExtractionOutcome.EMPTY: SourceDisposition.EMPTY,
        ExtractionOutcome.UNSUPPORTED: SourceDisposition.UNSUPPORTED,
        ExtractionOutcome.FAILED: SourceDisposition.EXTRACTION_FAILED,
    }
    if extraction.outcome in terminal_dispositions:
        if decision.disposition is not terminal_dispositions[extraction.outcome]:
            raise CorpusV2ProcessingError("C2_DISPOSITION_EVIDENCE_MISMATCH")
        return
    if extraction.outcome is not ExtractionOutcome.SUCCESS:
        raise CorpusV2ProcessingError("C2_DISPOSITION_EVIDENCE_MISMATCH")
    if decision.disposition is SourceDisposition.INDEXED:
        if (
            extraction.text_sha256 is None
            or extraction.character_count <= 0
            or decision.source_id not in usable_chunk_source_ids
        ):
            raise CorpusV2ProcessingError("C2_DISPOSITION_EVIDENCE_MISMATCH")
    elif decision.disposition not in {SourceDisposition.DUPLICATE, SourceDisposition.INTENTIONALLY_EXCLUDED}:
        raise CorpusV2ProcessingError("C2_DISPOSITION_EVIDENCE_MISMATCH")


def _validate_review_decision(decision: object, sealed_source_ids: frozenset[str]) -> None:
    if not isinstance(decision, ReviewDecision) or decision.source_id not in sealed_source_ids:
        raise CorpusV2ProcessingError("C2_SOURCE_UNKNOWN")
    if not isinstance(decision.disposition, SourceDisposition) or decision.disposition is SourceDisposition.UNRESOLVED:
        raise CorpusV2ProcessingError("C2_DISPOSITION_INVALID")
    if not isinstance(decision.reviewer_id, str) or _REVIEWER_ID.fullmatch(decision.reviewer_id) is None:
        raise CorpusV2ProcessingError("C2_REVIEWER_INVALID")
    if not isinstance(decision.reason_code, str) or _REASON_CODE.fullmatch(decision.reason_code) is None:
        raise CorpusV2ProcessingError("C2_DISPOSITION_INVALID")
    if decision.disposition is SourceDisposition.DUPLICATE:
        if (
            decision.representative_source_id is None
            or decision.representative_source_id == decision.source_id
            or decision.representative_source_id not in sealed_source_ids
        ):
            raise CorpusV2ProcessingError("C2_DUPLICATE_REPRESENTATIVE_INVALID")
    elif decision.representative_source_id is not None:
        raise CorpusV2ProcessingError("C2_DISPOSITION_INVALID")
