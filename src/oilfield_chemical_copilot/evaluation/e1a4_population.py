"""Fresh E1a-4 population contracts, without private artifact I/O."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Sequence

from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
    E1A3SamplingError,
    E1A3SlotAllocation,
    build_sampling_slots,
    private_sampling_payload_digest,
    seal_private_sampling_artifact_set,
)
from oilfield_chemical_copilot.evaluation.private_retrieval import PRIVATE_RETRIEVAL_ROOT

TOPICS = ("iron_sulfide", "scale", "corrosion", "paraffin")
SOURCE_ROLES = ("foundational", "supporting")
QUESTION_FORMS = (
    "definition_mechanism",
    "diagnostic_interpretive",
    "operational_procedural",
)
EVIDENCE_DEPTHS = ("single_claim", "multi_claim")
E1A4_POPULATION_SIZE = 96
E1A4_PRIVATE_ROOT = PRIVATE_RETRIEVAL_ROOT / "e1a4"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CASE_FIELDS = {
    "question_id",
    "question",
    "topic",
    "expected_source",
    "expected_locator",
    "expected_source_role",
    "question_form",
    "evidence_depth",
}


class E1A4PopulationError(ValueError):
    """Raised with a safe E1a-4 population-contract error code."""


def _fail(code: str) -> None:
    raise E1A4PopulationError(code)


def _text(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def normalize_question(question: object) -> str:
    """Normalize only for private novelty and duplicate checks."""
    return re.sub(r"\s+", " ", _text(question, code="E1A4_POPULATION_QUESTION_INVALID")).casefold()


@dataclass(frozen=True)
class E1A4PopulationCase:
    """One answerable user question with private canonical-source lineage."""

    question_id: str
    question: str
    topic: str
    expected_source: str
    expected_locator: str
    expected_source_role: str
    question_form: str
    evidence_depth: str

    @classmethod
    def from_mapping(cls, record: object) -> E1A4PopulationCase:
        if not isinstance(record, dict) or set(record) != _CASE_FIELDS:
            _fail("E1A4_POPULATION_CASE_INVALID")
        return cls(
            **{field: _text(record[field], code="E1A4_POPULATION_CASE_INVALID") for field in _CASE_FIELDS}
        )

    def __post_init__(self) -> None:
        for value in (
            self.question_id,
            self.question,
            self.topic,
            self.expected_source,
            self.expected_locator,
            self.expected_source_role,
            self.question_form,
            self.evidence_depth,
        ):
            _text(value, code="E1A4_POPULATION_CASE_INVALID")

    @property
    def canonical_locator_key(self) -> str:
        source = _text(self.expected_source, code="E1A4_POPULATION_LOCATOR_INVALID")
        locator = _text(self.expected_locator, code="E1A4_POPULATION_LOCATOR_INVALID")
        return f"{source}:{locator}"

    def to_mapping(self) -> dict[str, str]:
        return {key: _text(value, code="E1A4_POPULATION_CASE_INVALID") for key, value in asdict(self).items()}


@dataclass(frozen=True)
class E1A4CanonicalClaim:
    """A material answer obligation supported by the assigned canonical locator."""

    claim_id: str
    claim: str

    @classmethod
    def from_mapping(cls, record: object) -> E1A4CanonicalClaim:
        if not isinstance(record, dict) or set(record) != {"claim_id", "claim"}:
            _fail("E1A4_CANONICAL_CLAIMS_INVALID")
        return cls(
            claim_id=_text(record["claim_id"], code="E1A4_CANONICAL_CLAIMS_INVALID"),
            claim=_text(record["claim"], code="E1A4_CANONICAL_CLAIMS_INVALID"),
        )

    def __post_init__(self) -> None:
        _text(self.claim_id, code="E1A4_CANONICAL_CLAIMS_INVALID")
        _text(self.claim, code="E1A4_CANONICAL_CLAIMS_INVALID")

    def to_mapping(self) -> dict[str, str]:
        return {
            "claim_id": _text(self.claim_id, code="E1A4_CANONICAL_CLAIMS_INVALID"),
            "claim": _text(self.claim, code="E1A4_CANONICAL_CLAIMS_INVALID"),
        }


@dataclass(frozen=True)
class E1A4ClaimFixture:
    """All evaluator-only material claims for one fresh E1a-4 question."""

    question_id: str
    claims: tuple[E1A4CanonicalClaim, ...]

    @classmethod
    def from_mapping(cls, record: object) -> E1A4ClaimFixture:
        if (
            not isinstance(record, dict)
            or set(record) != {"question_id", "claims"}
            or not isinstance(record["claims"], list)
        ):
            _fail("E1A4_CANONICAL_CLAIMS_INVALID")
        return cls(
            question_id=_text(record["question_id"], code="E1A4_CANONICAL_CLAIMS_INVALID"),
            claims=tuple(E1A4CanonicalClaim.from_mapping(item) for item in record["claims"]),
        )

    def __post_init__(self) -> None:
        _text(self.question_id, code="E1A4_CANONICAL_CLAIMS_INVALID")
        if not self.claims or len({claim.claim_id for claim in self.claims}) != len(self.claims):
            _fail("E1A4_CANONICAL_CLAIMS_INVALID")

    def to_mapping(self) -> dict[str, object]:
        return {
            "question_id": _text(self.question_id, code="E1A4_CANONICAL_CLAIMS_INVALID"),
            "claims": [claim.to_mapping() for claim in self.claims],
        }


def _expected_grid() -> Counter[tuple[str, str, str, str]]:
    return Counter(
        (topic, source_role, question_form, evidence_depth)
        for topic in TOPICS
        for source_role in SOURCE_ROLES
        for question_form in QUESTION_FORMS
        for evidence_depth in EVIDENCE_DEPTHS
        for _ in range(2)
    )


def has_exact_population_grid(cases: Sequence[E1A4PopulationCase]) -> bool:
    """Return whether cases satisfy the fixed, pre-registered 96-slot grid."""
    observed = Counter(
        (case.topic, case.expected_source_role, case.question_form, case.evidence_depth)
        for case in cases
    )
    return len(cases) == E1A4_POPULATION_SIZE and observed == _expected_grid()


def _validate_grid(cases: Sequence[E1A4PopulationCase]) -> None:
    if not has_exact_population_grid(cases):
        _fail("E1A4_POPULATION_GRID_INVALID")


def _allocation_grid_is_exact(allocations: Sequence[E1A3SlotAllocation]) -> bool:
    if any(type(item.replicate) is not int for item in allocations):
        return False
    expected = {
        (slot.slot_id, slot.topic, slot.source_role, slot.question_form, slot.evidence_depth, slot.replicate)
        for slot in build_sampling_slots()
    }
    observed = {
        (
            item.slot_id,
            item.topic,
            item.source_role,
            item.question_form,
            item.evidence_depth,
            item.replicate,
        )
        for item in allocations
    }
    return len(allocations) == E1A4_POPULATION_SIZE and len(observed) == len(allocations) and observed == expected


def require_population_matches_allocations(
    *, cases: Sequence[E1A4PopulationCase], allocations: Sequence[E1A3SlotAllocation]
) -> None:
    """Bind every authored case to exactly one pre-sealed sampling allocation."""
    if not _allocation_grid_is_exact(allocations) or any(
        not isinstance(item.parser_type, str) or not item.parser_type.strip() for item in allocations
    ):
        _fail("E1A4_ALLOCATION_GRID_INVALID")
    try:
        allocated = Counter(
            (
                item.topic,
                item.source_role,
                item.question_form,
                item.evidence_depth,
                _text(item.source_id, code="E1A4_ALLOCATION_GRID_INVALID"),
                _text(item.locator, code="E1A4_ALLOCATION_GRID_INVALID"),
            )
            for item in allocations
        )
        authored = Counter(
            (
                case.topic,
                case.expected_source_role,
                case.question_form,
                case.evidence_depth,
                _text(case.expected_source, code="E1A4_POPULATION_LINEAGE_INVALID"),
                _text(case.expected_locator, code="E1A4_POPULATION_LINEAGE_INVALID"),
            )
            for case in cases
        )
    except AttributeError as error:
        raise E1A4PopulationError("E1A4_ALLOCATION_GRID_INVALID") from error
    if authored != allocated:
        _fail("E1A4_POPULATION_LINEAGE_INVALID")


def validate_population_and_claims(
    *,
    cases: Sequence[E1A4PopulationCase],
    claims: Sequence[E1A4ClaimFixture],
    earlier_questions: Sequence[str],
    e1a3_locators: Sequence[str],
) -> None:
    """Validate freshness and canonical-claim limits before any model work."""
    _validate_grid(cases)
    case_by_id = {
        _text(case.question_id, code="E1A4_POPULATION_CASE_INVALID"): case for case in cases
    }
    if len(case_by_id) != len(cases):
        _fail("E1A4_POPULATION_CASE_INVALID")

    normalized_questions = tuple(normalize_question(case.question) for case in cases)
    if len(set(normalized_questions)) != len(normalized_questions):
        _fail("E1A4_POPULATION_QUESTION_DUPLICATE")
    prior = {normalize_question(question) for question in earlier_questions}
    if set(normalized_questions).intersection(prior):
        _fail("E1A4_POPULATION_NOT_UNSEEN")

    locator_keys = tuple(case.canonical_locator_key for case in cases)
    if len(set(locator_keys)) != len(locator_keys):
        _fail("E1A4_POPULATION_LOCATOR_DUPLICATE")
    known_e1a3_locators = {_text(locator, code="E1A4_POPULATION_LOCATOR_INVALID") for locator in e1a3_locators}
    if set(locator_keys).intersection(known_e1a3_locators):
        _fail("E1A4_POPULATION_LOCATOR_REUSED")

    fixture_by_id = {
        _text(fixture.question_id, code="E1A4_CANONICAL_CLAIMS_INVALID"): fixture
        for fixture in claims
    }
    if len(fixture_by_id) != len(claims) or set(fixture_by_id) != set(case_by_id):
        _fail("E1A4_CANONICAL_CLAIMS_QUESTION_SET_INVALID")
    claim_ids = [
        _text(claim.claim_id, code="E1A4_CANONICAL_CLAIMS_INVALID")
        for fixture in claims
        for claim in fixture.claims
    ]
    if len(claim_ids) != len(set(claim_ids)):
        _fail("E1A4_CANONICAL_CLAIMS_INVALID")

    for case in cases:
        question_id = _text(case.question_id, code="E1A4_POPULATION_CASE_INVALID")
        count = len(fixture_by_id[question_id].claims)
        if count > 3 or (case.evidence_depth == "single_claim" and count != 1) or (
            case.evidence_depth == "multi_claim" and count < 2
        ):
            _fail("E1A4_CANONICAL_CLAIMS_DEPTH_INVALID")


def build_population_and_claim_payloads(
    *,
    cases: Sequence[E1A4PopulationCase],
    claims: Sequence[E1A4ClaimFixture],
    allocations: Sequence[E1A3SlotAllocation],
    allocation_sha256: str,
    earlier_questions: Sequence[str],
    e1a3_locators: Sequence[str],
) -> tuple[dict[str, object], str, dict[str, object], str]:
    """Build deterministic, provenance-bound payloads after full validation."""
    if not isinstance(allocation_sha256, str) or _SHA256_PATTERN.fullmatch(allocation_sha256) is None:
        _fail("E1A4_ALLOCATION_MANIFEST_INVALID")
    validate_population_and_claims(
        cases=cases,
        claims=claims,
        earlier_questions=earlier_questions,
        e1a3_locators=e1a3_locators,
    )
    require_population_matches_allocations(cases=cases, allocations=allocations)
    population: dict[str, object] = {
        "schema_version": 1,
        "allocation_sha256": allocation_sha256,
        "cases": [case.to_mapping() for case in sorted(cases, key=lambda item: item.question_id)],
    }
    population_digest = private_sampling_payload_digest(population)
    canonical_claims: dict[str, object] = {
        "schema_version": 1,
        "population_sha256": population_digest,
        "fixtures": [fixture.to_mapping() for fixture in sorted(claims, key=lambda item: item.question_id)],
    }
    claims_digest = private_sampling_payload_digest(canonical_claims)
    return population, population_digest, canonical_claims, claims_digest


def seal_population_and_claims(
    *,
    cases: Sequence[E1A4PopulationCase],
    claims: Sequence[E1A4ClaimFixture],
    allocations: Sequence[E1A3SlotAllocation],
    allocation_sha256: str,
    earlier_questions: Sequence[str],
    e1a3_locators: Sequence[str],
    population_path: Path,
    population_digest_path: Path,
    claims_path: Path,
    claims_digest_path: Path,
    private_root: Path = E1A4_PRIVATE_ROOT,
) -> tuple[str, str]:
    """Validate and publish the population and claims as one private artifact set."""
    population, population_digest, canonical_claims, claims_digest = build_population_and_claim_payloads(
        cases=cases,
        claims=claims,
        allocations=allocations,
        allocation_sha256=allocation_sha256,
        earlier_questions=earlier_questions,
        e1a3_locators=e1a3_locators,
    )
    try:
        sealed_digests = seal_private_sampling_artifact_set(
            artifacts=(
                (population, population_path, population_digest_path),
                (canonical_claims, claims_path, claims_digest_path),
            ),
            private_root=private_root,
        )
    except E1A3SamplingError as error:
        raise E1A4PopulationError("E1A4_POPULATION_SEAL_FAILED") from error
    if sealed_digests != (population_digest, claims_digest):
        _fail("E1A4_POPULATION_SEAL_FAILED")
    return population_digest, claims_digest
