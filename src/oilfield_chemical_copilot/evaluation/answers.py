"""Privacy-safe deterministic checks for public answer evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_CASE_FIELDS = {
    "question_id",
    "question",
    "allowed_evidence_ids",
    "evidence_sufficient",
    "expect_citations",
    "expect_abstention",
}
Status = Literal["pass", "fail"]


@dataclass(frozen=True)
class AnswerEvaluationCase:
    question_id: str
    question: str
    allowed_evidence_ids: tuple[str, ...]
    evidence_sufficient: bool
    expect_citations: bool
    expect_abstention: bool


@dataclass(frozen=True)
class DeterministicAnswerResult:
    question_id: str
    citation_status: Status
    abstention_status: Status


def load_answer_evaluation_cases(path: Path) -> list[AnswerEvaluationCase]:
    """Load public evaluation cases, including synthetic public questions."""
    cases: list[AnswerEvaluationCase] = []
    question_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("dataset records must be valid JSON objects") from error
        if not isinstance(record, dict) or set(record) != _CASE_FIELDS:
            raise ValueError("dataset records must have exactly the expected fields")
        question_id = record["question_id"]
        question = record["question"]
        allowed_evidence_ids = record["allowed_evidence_ids"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question_id must not be blank")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be blank")
        if question_id in question_ids:
            raise ValueError(f"duplicate question_id: {question_id}")
        if (
            not isinstance(allowed_evidence_ids, list)
            or any(not isinstance(value, str) or not value.strip() for value in allowed_evidence_ids)
        ):
            raise ValueError("evidence IDs must not be blank")
        outcomes = (
            record["evidence_sufficient"],
            record["expect_citations"],
            record["expect_abstention"],
        )
        if not all(isinstance(value, bool) for value in outcomes):
            raise ValueError("expected outcomes must be boolean")
        question_ids.add(question_id)
        cases.append(
            AnswerEvaluationCase(
                question_id=question_id,
                question=question,
                allowed_evidence_ids=tuple(allowed_evidence_ids),
                evidence_sufficient=record["evidence_sufficient"],
                expect_citations=record["expect_citations"],
                expect_abstention=record["expect_abstention"],
            )
        )
    return cases


def evaluate_answer(
    case: AnswerEvaluationCase, *, cited_evidence_ids: tuple[str, ...], abstained: bool
) -> DeterministicAnswerResult:
    """Return only safe statuses for citation and abstention expectations."""
    citations_are_valid = bool(cited_evidence_ids) and set(cited_evidence_ids) <= set(
        case.allowed_evidence_ids
    )
    citation_status: Status = "pass" if (
        citations_are_valid if case.expect_citations else not cited_evidence_ids
    ) else "fail"
    abstention_status: Status = "pass" if abstained == case.expect_abstention else "fail"
    return DeterministicAnswerResult(case.question_id, citation_status, abstention_status)
