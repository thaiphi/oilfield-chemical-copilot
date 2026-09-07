"""Contracts for a requirements-aware, abstention-first evidence gate.

This module has no model or retrieval dependency. A future adapter may ask a
model for requirement support, but the controller owns validation, evidence
state aggregation, and the answer-boundary decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from oilfield_chemical_copilot.evaluation.evidence_state import (
    DeliveredEvidence,
    EvidenceState,
)

SupportStatus = Literal["SUPPORTED", "UNSUPPORTED", "UNCLEAR"]
AnswerBoundary = Literal[
    "FULL_ANSWER_PERMITTED",
    "SUPPORTED_ONLY_RESPONSE_REQUIRED",
    "ABSTENTION_REQUIRED",
]
_SUPPORT_STATUSES: tuple[SupportStatus, ...] = (
    "SUPPORTED",
    "UNSUPPORTED",
    "UNCLEAR",
)
_REQUIREMENT_IDS = ("r1", "r2", "r3")


class RequirementsEvidenceGateError(ValueError):
    """Raised with a safe requirements-aware evidence-gate error code."""


def _fail(code: str) -> None:
    raise RequirementsEvidenceGateError(code)


def _require_text(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


@dataclass(frozen=True)
class Requirement:
    """One atomic user-request obligation frozen before evidence is loaded."""

    requirement_id: str
    requirement: str

    def __post_init__(self) -> None:
        if (
            self.requirement_id not in _REQUIREMENT_IDS
            or not isinstance(self.requirement, str)
            or not self.requirement.strip()
            or len(self.requirement) > 240
        ):
            _fail("E1A4_REQUIREMENT_FIXTURE_INVALID")


@dataclass(frozen=True)
class RequirementFixture:
    """Question-only requirements in their frozen controller-owned order."""

    question_id: str
    requirements: tuple[Requirement, ...]

    def __post_init__(self) -> None:
        _require_text(self.question_id, code="E1A4_REQUIREMENT_FIXTURE_INVALID")
        expected = _REQUIREMENT_IDS[: len(self.requirements)]
        observed = tuple(item.requirement_id for item in self.requirements)
        if not 1 <= len(self.requirements) <= 3 or observed != expected:
            _fail("E1A4_REQUIREMENT_FIXTURE_INVALID")


@dataclass(frozen=True)
class RequirementSupport:
    """One support judgment and its exact delivered-passage provenance."""

    requirement_id: str
    status: SupportStatus
    supporting_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.requirement_id not in _REQUIREMENT_IDS or self.status not in _SUPPORT_STATUSES:
            _fail("E1A4_SUPPORT_RESULT_INVALID")
        if any(type(rank) is not int or rank < 1 for rank in self.supporting_ranks):
            _fail("E1A4_SUPPORT_RESULT_INVALID")


@dataclass(frozen=True)
class RequirementSupportResult:
    """Model-proposed support only; the controller determines the final state."""

    question_id: str
    requirement_support: tuple[RequirementSupport, ...]

    def __post_init__(self) -> None:
        _require_text(self.question_id, code="E1A4_SUPPORT_RESULT_INVALID")


class RequirementExtractor(Protocol):
    """Model-swappable question-only requirement extraction boundary."""

    def extract(self, *, question_id: str, question: str) -> RequirementFixture: ...


class RequirementSupportJudge(Protocol):
    """Model-swappable support judgment boundary over frozen delivered evidence."""

    def judge(
        self,
        *,
        question_id: str,
        question: str,
        requirements: RequirementFixture,
        evidence: Sequence[DeliveredEvidence],
    ) -> RequirementSupportResult: ...


def validate_support_result(
    *,
    fixture: RequirementFixture,
    evidence: Sequence[DeliveredEvidence],
    result: RequirementSupportResult,
) -> None:
    """Reject output that adds requirements or cites non-delivered evidence."""
    delivered_ranks = tuple(item.rank for item in evidence)
    if (
        len(set(delivered_ranks)) != len(delivered_ranks)
        or result.question_id != fixture.question_id
        or tuple(item.requirement_id for item in result.requirement_support)
        != tuple(item.requirement_id for item in fixture.requirements)
    ):
        _fail("E1A4_SUPPORT_RESULT_INVALID")

    allowed_ranks = set(delivered_ranks)
    for item in result.requirement_support:
        ranks = item.supporting_ranks
        if len(set(ranks)) != len(ranks) or tuple(sorted(ranks)) != ranks:
            _fail("E1A4_SUPPORT_RESULT_INVALID")
        if item.status == "SUPPORTED":
            if not ranks or not set(ranks).issubset(allowed_ranks):
                _fail("E1A4_SUPPORT_RESULT_INVALID")
        elif ranks:
            _fail("E1A4_SUPPORT_RESULT_INVALID")


def derive_evidence_state(
    *,
    fixture: RequirementFixture,
    evidence: Sequence[DeliveredEvidence],
    result: RequirementSupportResult,
) -> EvidenceState:
    """Validate support provenance before applying the fixed state rule."""
    validate_support_result(fixture=fixture, evidence=evidence, result=result)
    statuses = tuple(item.status for item in result.requirement_support)
    if all(status == "SUPPORTED" for status in statuses):
        return "SUFFICIENT"
    if any(status == "SUPPORTED" for status in statuses):
        return "PARTIALLY_SUFFICIENT"
    return "INSUFFICIENT"


def derive_answer_boundary(state: EvidenceState) -> AnswerBoundary:
    """Return the allowed response boundary without generating any answer."""
    if state == "SUFFICIENT":
        return "FULL_ANSWER_PERMITTED"
    if state == "PARTIALLY_SUFFICIENT":
        return "SUPPORTED_ONLY_RESPONSE_REQUIRED"
    if state == "INSUFFICIENT":
        return "ABSTENTION_REQUIRED"
    _fail("E1A4_EVIDENCE_STATE_INVALID")
