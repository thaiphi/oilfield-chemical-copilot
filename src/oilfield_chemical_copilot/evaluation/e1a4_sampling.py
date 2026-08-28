"""Strict metadata-only sampling contracts for E1a-4."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Literal

from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
    E1A3SourceMetadata,
    build_sampling_slots,
    private_sampling_payload_digest,
)


Topic = Literal["iron_sulfide", "scale", "corrosion", "paraffin"]
SourceRole = Literal["foundational", "supporting"]
_TOPICS: frozenset[str] = frozenset(("iron_sulfide", "scale", "corrosion", "paraffin"))
_SOURCE_ROLES: frozenset[str] = frozenset(("foundational", "supporting"))
_ALLOCATION_FIELDS = frozenset(
    (
        "slot_id",
        "topic",
        "source_role",
        "question_form",
        "evidence_depth",
        "replicate",
        "source_id",
        "parser_type",
        "locator",
    )
)


class E1A4SamplingError(ValueError):
    """Raised with a safe E1a-4 sampling-contract error code."""


def _fail(code: str) -> None:
    raise E1A4SamplingError(code)


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        _fail("E1A4_MAPPING_SOURCE_INVALID")
    return value


def _topic(value: object) -> Topic:
    if not isinstance(value, str) or value not in _TOPICS:
        _fail("E1A4_MAPPING_SOURCE_INVALID")
    return value  # type: ignore[return-value]


def _source_role(value: object) -> SourceRole:
    if not isinstance(value, str) or value not in _SOURCE_ROLES:
        _fail("E1A4_MAPPING_SOURCE_INVALID")
    return value  # type: ignore[return-value]


def _private_path(path: Path, *, private_root: Path) -> Path:
    root = private_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail("E1A4_PRIVATE_PATH_REQUIRED")
    return resolved


@dataclass(frozen=True)
class E1A4MappedSource:
    source_id: str
    topic: Topic
    source_role: SourceRole
    parser_type: str
    locators: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> E1A4MappedSource:
        if not isinstance(value, dict) or set(value) != {
            "source_id",
            "topic",
            "source_role",
            "parser_type",
            "locators",
        }:
            _fail("E1A4_MAPPING_SOURCE_INVALID")
        locators = value["locators"]
        if not isinstance(locators, list):
            _fail("E1A4_MAPPING_SOURCE_INVALID")
        parsed_locators = tuple(_required_text(item) for item in locators)
        if not parsed_locators or parsed_locators != tuple(sorted(parsed_locators)) or len(
            parsed_locators
        ) != len(set(parsed_locators)):
            _fail("E1A4_MAPPING_SOURCE_INVALID")
        return cls(
            source_id=_required_text(value["source_id"]),
            topic=_topic(value["topic"]),
            source_role=_source_role(value["source_role"]),
            parser_type=_required_text(value["parser_type"]),
            locators=parsed_locators,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "topic": self.topic,
            "source_role": self.source_role,
            "parser_type": self.parser_type,
            "locators": list(self.locators),
        }


@dataclass(frozen=True)
class E1A4PriorAllocation:
    payload_sha256: str
    slot_count: int
    locator_keys: frozenset[str]


def validate_mapping_sources(values: Iterable[object]) -> tuple[E1A4MappedSource, ...]:
    """Validate, preserve, and deterministically order mapped source records."""
    sources = tuple(E1A4MappedSource.from_mapping(value) for value in values)
    locator_keys = [
        f"{source.source_id}:{locator}" for source in sources for locator in source.locators
    ]
    if len(locator_keys) != len(set(locator_keys)):
        _fail("E1A4_MAPPING_SOURCE_DUPLICATE_LOCATOR")
    return tuple(
        sorted(
            sources,
            key=lambda item: (
                item.source_id,
                item.topic,
                item.source_role,
                item.parser_type,
                item.locators,
            ),
        )
    )


def mapping_sources_as_sampling_metadata(
    sources: Iterable[E1A4MappedSource],
) -> tuple[E1A3SourceMetadata, ...]:
    """Translate each mapped source one-for-one into E1a-3 allocator metadata."""
    return tuple(
        E1A3SourceMetadata(
            source_id=source.source_id,
            topic=source.topic,
            source_role=source.source_role,
            parser_type=source.parser_type,
            locators=source.locators,
            eligibility_status="eligible",
        )
        for source in sources
    )


def _load_payload(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E1A4_PRIOR_ALLOCATION_INVALID")
    if not isinstance(value, dict) or set(value) != {"schema_version", "allocations"}:
        _fail("E1A4_PRIOR_ALLOCATION_INVALID")
    schema_version = value["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        _fail("E1A4_PRIOR_ALLOCATION_INVALID")
    if not isinstance(value["allocations"], list):
        _fail("E1A4_PRIOR_ALLOCATION_INVALID")
    return value


def load_e1a3_prior_allocation(
    *, payload_path: Path, manifest_path: Path, private_root: Path
) -> E1A4PriorAllocation:
    """Load a canonically sealed exact E1a-3 96-slot allocation."""
    payload_file = _private_path(payload_path, private_root=private_root)
    manifest_file = _private_path(manifest_path, private_root=private_root)
    payload = _load_payload(payload_file)
    digest = private_sampling_payload_digest(payload)
    try:
        payload_bytes = payload_file.read_bytes()
        manifest = manifest_file.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        _fail("E1A4_PRIOR_ALLOCATION_MANIFEST_INVALID")
    if payload_bytes != (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    ) or manifest != f"{digest}\n":
        _fail("E1A4_PRIOR_ALLOCATION_MANIFEST_INVALID")

    allocations = payload["allocations"]
    if not isinstance(allocations, list) or len(allocations) != 96:
        _fail("E1A4_PRIOR_ALLOCATION_INVALID")
    expected_slots = {
        (
            slot.slot_id,
            slot.topic,
            slot.source_role,
            slot.question_form,
            slot.evidence_depth,
            slot.replicate,
        )
        for slot in build_sampling_slots()
    }
    slot_identities: set[tuple[str, str, str, str, str, int]] = set()
    locator_keys: set[str] = set()
    for allocation in allocations:
        if not isinstance(allocation, dict) or set(allocation) != _ALLOCATION_FIELDS:
            _fail("E1A4_PRIOR_ALLOCATION_INVALID")
        replicate = allocation["replicate"]
        if not isinstance(replicate, int) or isinstance(replicate, bool):
            _fail("E1A4_PRIOR_ALLOCATION_INVALID")
        fields = tuple(
            allocation[name]
            for name in (
                "slot_id",
                "topic",
                "source_role",
                "question_form",
                "evidence_depth",
            )
        )
        if any(not isinstance(field, str) or not field for field in fields):
            _fail("E1A4_PRIOR_ALLOCATION_INVALID")
        identity = (*fields, replicate)
        if identity not in expected_slots:
            _fail("E1A4_PRIOR_ALLOCATION_INVALID")
        source_id = allocation["source_id"]
        parser_type = allocation["parser_type"]
        locator = allocation["locator"]
        if any(
            not isinstance(field, str) or not field
            for field in (source_id, parser_type, locator)
        ):
            _fail("E1A4_PRIOR_ALLOCATION_INVALID")
        slot_identities.add(identity)
        locator_keys.add(f"{source_id}:{locator}")
    if slot_identities != expected_slots or len(locator_keys) != 96:
        _fail("E1A4_PRIOR_ALLOCATION_INVALID")
    return E1A4PriorAllocation(
        payload_sha256=digest,
        slot_count=96,
        locator_keys=frozenset(locator_keys),
    )
