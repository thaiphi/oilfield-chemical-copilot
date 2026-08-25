"""Private, metadata-only sampling contracts for E1a-3 Gate 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Literal, Mapping, Sequence

from oilfield_chemical_copilot.evaluation.private_retrieval import PRIVATE_RETRIEVAL_ROOT


Topic = Literal["iron_sulfide", "scale", "corrosion", "paraffin"]
SourceRole = Literal["foundational", "supporting"]
QuestionForm = Literal["definition_mechanism", "diagnostic_interpretive", "operational_procedural"]
EvidenceDepth = Literal["single_claim", "multi_claim"]

TOPICS: tuple[Topic, ...] = ("iron_sulfide", "scale", "corrosion", "paraffin")
SOURCE_ROLES: tuple[SourceRole, ...] = ("foundational", "supporting")
QUESTION_FORMS: tuple[QuestionForm, ...] = (
    "definition_mechanism",
    "diagnostic_interpretive",
    "operational_procedural",
)
EVIDENCE_DEPTHS: tuple[EvidenceDepth, ...] = ("single_claim", "multi_claim")


class E1A3SamplingError(ValueError):
    """Raised with a safe E1a-3 sampling-contract error code."""


def _fail(code: str) -> None:
    raise E1A3SamplingError(code)


def _private_root(root: Path) -> Path:
    resolved = root.resolve()
    try:
        resolved.relative_to(PRIVATE_RETRIEVAL_ROOT.resolve())
    except ValueError:
        _fail("E1A3_PRIVATE_ROOT_INVALID")
    return resolved


def _private_path(path: Path, *, private_root: Path) -> Path:
    root = _private_root(private_root)
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail("E1A3_PRIVATE_PATH_REQUIRED")
    return resolved


@dataclass(frozen=True)
class E1A3SourceMetadata:
    """Metadata-only source availability; no source text or question content."""

    source_id: str
    source_role: SourceRole
    topic: Topic
    parser_type: str
    locators: tuple[str, ...]
    eligibility_status: Literal["eligible"]

    def __post_init__(self) -> None:
        if (
            not self.source_id
            or self.source_role not in SOURCE_ROLES
            or self.topic not in TOPICS
            or not self.parser_type
            or not self.locators
            or self.eligibility_status != "eligible"
            or any(not locator for locator in self.locators)
            or len(self.locators) != len(set(self.locators))
        ):
            _fail("E1A3_SOURCE_REGISTER_INVALID")

    def to_mapping(self) -> dict[str, object]:
        return {**asdict(self), "locators": list(self.locators)}


@dataclass(frozen=True)
class E1A3SamplingSlot:
    """One pre-registered population slot before question authoring."""

    slot_id: str
    topic: Topic
    source_role: SourceRole
    question_form: QuestionForm
    evidence_depth: EvidenceDepth
    replicate: int

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class E1A3SlotAllocation:
    """A metadata-only source and locator assignment for one sampling slot."""

    slot_id: str
    topic: Topic
    source_role: SourceRole
    question_form: QuestionForm
    evidence_depth: EvidenceDepth
    replicate: int
    source_id: str
    parser_type: str
    locator: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def build_sampling_slots() -> tuple[E1A3SamplingSlot, ...]:
    """Build the immutable 4 x 2 x 3 x 2 x 2 population grid."""
    slots = []
    for topic in TOPICS:
        for source_role in SOURCE_ROLES:
            for question_form in QUESTION_FORMS:
                for evidence_depth in EVIDENCE_DEPTHS:
                    for replicate in (1, 2):
                        slots.append(
                            E1A3SamplingSlot(
                                slot_id=(
                                    f"{topic}:{source_role}:{question_form}:{evidence_depth}:{replicate}"
                                ),
                                topic=topic,
                                source_role=source_role,
                                question_form=question_form,
                                evidence_depth=evidence_depth,
                                replicate=replicate,
                            )
                        )
    result = tuple(slots)
    _validate_sampling_slots(result)
    return result


def _slot_identity(slot: E1A3SamplingSlot) -> tuple[str, Topic, SourceRole, QuestionForm, EvidenceDepth, int]:
    return (
        slot.slot_id,
        slot.topic,
        slot.source_role,
        slot.question_form,
        slot.evidence_depth,
        slot.replicate,
    )


def _expected_slot_identities() -> frozenset[tuple[str, Topic, SourceRole, QuestionForm, EvidenceDepth, int]]:
    return frozenset(
        (
            f"{topic}:{source_role}:{question_form}:{evidence_depth}:{replicate}",
            topic,
            source_role,
            question_form,
            evidence_depth,
            replicate,
        )
        for topic in TOPICS
        for source_role in SOURCE_ROLES
        for question_form in QUESTION_FORMS
        for evidence_depth in EVIDENCE_DEPTHS
        for replicate in (1, 2)
    )


def _validate_sampling_slots(slots: Sequence[E1A3SamplingSlot]) -> None:
    if len(slots) != 96 or {_slot_identity(slot) for slot in slots} != _expected_slot_identities():
        _fail("E1A3_SLOT_GRID_INVALID")


def allocate_sampling_slots(
    *, slots: Sequence[E1A3SamplingSlot], sources: Iterable[E1A3SourceMetadata]
) -> tuple[E1A3SlotAllocation, ...]:
    """Assign only matching metadata sources, reusing a source only at a new locator."""
    source_list = tuple(sources)
    _validate_sampling_slots(slots)
    if not source_list:
        _fail("E1A3_SOURCE_REGISTER_INVALID")
    assigned_count: dict[str, int] = {}
    used_locators: set[tuple[str, str]] = set()
    allocations: list[E1A3SlotAllocation] = []
    for slot in sorted(slots, key=lambda item: item.slot_id):
        candidates = [
            (source, locator)
            for source in source_list
            if source.topic == slot.topic and source.source_role == slot.source_role
            for locator in source.locators
            if (source.source_id, locator) not in used_locators
        ]
        if not candidates:
            _fail("E1A3_ALLOCATION_UNAVAILABLE")
        source, locator = min(
            candidates,
            key=lambda item: (assigned_count.get(item[0].source_id, 0), item[0].source_id, item[1]),
        )
        used_locators.add((source.source_id, locator))
        assigned_count[source.source_id] = assigned_count.get(source.source_id, 0) + 1
        allocations.append(
            E1A3SlotAllocation(
                slot_id=slot.slot_id,
                topic=slot.topic,
                source_role=slot.source_role,
                question_form=slot.question_form,
                evidence_depth=slot.evidence_depth,
                replicate=slot.replicate,
                source_id=source.source_id,
                parser_type=source.parser_type,
                locator=locator,
            )
        )
    return tuple(allocations)


def build_source_metadata_from_index_rows(
    *,
    rows: Iterable[Mapping[str, object]],
    foundational_locators: Mapping[str, Mapping[Topic, Sequence[str]]],
    excluded_supporting_sources: frozenset[str] = frozenset(),
) -> tuple[E1A3SourceMetadata, ...]:
    """Build source metadata using explicit topic-scoped foundational locators."""
    by_source: dict[str, dict[str, set[str]]] = {}
    parser_by_source: dict[str, set[str]] = {}
    for row in rows:
        source_id = str(row["source_file"])
        topic = str(row["topic"])
        locator = str(row["page_or_sheet"])
        by_source.setdefault(source_id, {}).setdefault(topic, set()).add(locator)
        parser_by_source.setdefault(source_id, set()).add(str(row["parser_type"]))

    records: list[E1A3SourceMetadata] = []
    for source_id in sorted(by_source):
        if source_id in excluded_supporting_sources:
            continue
        parser_types = parser_by_source[source_id]
        if len(parser_types) != 1:
            continue
        parser_type = next(iter(parser_types))
        all_locators = {locator for locators in by_source[source_id].values() for locator in locators}
        if source_id in foundational_locators:
            for topic, configured_locators in foundational_locators[source_id].items():
                locators = tuple(sorted(configured_locators))
                if not locators or len(locators) != len(set(locators)) or not set(locators).issubset(all_locators):
                    _fail("E1A3_FOUNDATIONAL_LOCATOR_INVALID")
                records.append(
                    E1A3SourceMetadata(
                        source_id=source_id,
                        source_role="foundational",
                        topic=topic,
                        parser_type=parser_type,
                        locators=locators,
                        eligibility_status="eligible",
                    )
                )
            continue
        for topic in TOPICS:
            locators = tuple(sorted(by_source[source_id].get(topic, set())))
            if locators:
                records.append(
                    E1A3SourceMetadata(
                        source_id=source_id,
                        source_role="supporting",
                        topic=topic,
                        parser_type=parser_type,
                        locators=locators,
                        eligibility_status="eligible",
                    )
                )
    return tuple(records)


def write_private_sampling_json(
    payload: dict[str, object], *, destination: Path, private_root: Path
) -> None:
    """Write a one-time metadata artifact only below the controller private root."""
    resolved = _private_path(destination, private_root=private_root)
    if resolved.exists():
        _fail("E1A3_ARTIFACT_ALREADY_EXISTS")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def private_sampling_payload_digest(payload: dict[str, object]) -> str:
    """Return the digest for the canonical bytes used by all sealed sampling artifacts."""
    return sha256((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")).hexdigest()


def seal_private_sampling_json(
    payload: dict[str, object], *, sealed_path: Path, digest_path: Path, private_root: Path
) -> str:
    """Seal canonical metadata-only sampling artifacts with a SHA-256 manifest."""
    sealed = _private_path(sealed_path, private_root=private_root)
    digest = _private_path(digest_path, private_root=private_root)
    if sealed.exists() or digest.exists():
        _fail("E1A3_ARTIFACT_ALREADY_EXISTS")
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    sealed.parent.mkdir(parents=True, exist_ok=True)
    digest.parent.mkdir(parents=True, exist_ok=True)
    sealed.write_bytes(content)
    digest.write_text(sha256(content).hexdigest() + "\n", encoding="ascii")
    return sha256(content).hexdigest()


def seal_private_sampling_artifact_set(
    *,
    artifacts: Sequence[tuple[dict[str, object], Path, Path]],
    private_root: Path,
) -> tuple[str, ...]:
    """Publish a related artifact set only after all destinations validate."""
    prepared: list[tuple[bytes, Path, bytes, Path]] = []
    destinations: set[Path] = set()
    for payload, sealed_path, digest_path in artifacts:
        sealed = _private_path(sealed_path, private_root=private_root)
        digest = _private_path(digest_path, private_root=private_root)
        if sealed == digest or sealed in destinations or digest in destinations or sealed.exists() or digest.exists():
            _fail("E1A3_ARTIFACT_ALREADY_EXISTS")
        destinations.update((sealed, digest))
        content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        prepared.append((content, sealed, (sha256(content).hexdigest() + "\n").encode("ascii"), digest))

    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for content, sealed, digest_content, digest in prepared:
            for output, content_bytes in ((sealed, content), (digest, digest_content)):
                output.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as temporary:
                    temporary.write(content_bytes)
                    temporary_path = Path(temporary.name)
                staged.append((temporary_path, output))
        for temporary_path, output in staged:
            replace(temporary_path, output)
            published.append(output)
    except OSError as error:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
        for output in published:
            output.unlink(missing_ok=True)
        raise E1A3SamplingError("E1A3_ARTIFACT_WRITE_FAILED") from error

    return tuple(sha256(content).hexdigest() for content, _, _, _ in prepared)
