"""Local-only, ID-only records for controlled public citation diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision
from oilfield_chemical_copilot.evaluation.answers import AnswerEvaluationCase
from oilfield_chemical_copilot.evaluation.live_rag import LiveAnswerCapture

_MODES = frozenset(("vector", "hybrid"))
_ACTIONS = frozenset(("allow", "abstain"))
_CATEGORIES = frozenset(
    (
        "general_review",
        "site_specific_determination",
        "field_ready_prescription",
        "complete_input_substitution",
    )
)
_GENERATION_OUTCOMES = frozenset(("not_called", "succeeded", "failed"))


@dataclass(frozen=True)
class LocalCitationDiagnostic:
    question_id: str
    policy_action: str
    policy_category: str
    allowed_evidence_ids: tuple[str, ...]
    retrieved_evidence_ids: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...]
    abstained: bool
    generation_outcome: str


def _invalid() -> ValueError:
    return ValueError("invalid local citation diagnostics")


def _valid_ids(value: object) -> bool:
    return (
        type(value) is tuple
        and all(type(item) is str and bool(item) for item in value)
        and len(set(value)) == len(value)
    )


def _validate(record: object) -> LocalCitationDiagnostic:
    if not isinstance(record, LocalCitationDiagnostic):
        raise _invalid()
    if (
        type(record.question_id) is not str
        or not record.question_id
        or record.policy_action not in _ACTIONS
        or record.policy_category not in _CATEGORIES
        or not all(
            _valid_ids(value)
            for value in (
                record.allowed_evidence_ids,
                record.retrieved_evidence_ids,
                record.cited_evidence_ids,
            )
        )
        or type(record.abstained) is not bool
        or record.generation_outcome not in _GENERATION_OUTCOMES
    ):
        raise _invalid()
    return record


def local_citation_diagnostic(
    case: AnswerEvaluationCase,
    decision: AbstentionPolicyDecision,
    capture: LiveAnswerCapture,
) -> LocalCitationDiagnostic:
    if (
        not isinstance(case, AnswerEvaluationCase)
        or not isinstance(decision, AbstentionPolicyDecision)
        or not isinstance(capture, LiveAnswerCapture)
        or capture.answer.question_id != case.question_id
    ):
        raise _invalid()
    return _validate(
        LocalCitationDiagnostic(
            question_id=case.question_id,
            policy_action=decision.action,
            policy_category=decision.category,
            allowed_evidence_ids=case.allowed_evidence_ids,
            retrieved_evidence_ids=capture.retrieved_evidence_ids,
            cited_evidence_ids=capture.answer.cited_evidence_ids,
            abstained=capture.answer.abstained,
            generation_outcome=capture.generation_outcome,
        )
    )


def write_local_citation_diagnostics(
    records_by_mode: Mapping[str, list[LocalCitationDiagnostic]], destination: Path
) -> None:
    if not isinstance(destination, Path) or set(records_by_mode) != _MODES:
        raise _invalid()
    records = {
        mode: [_serialize(_validate(record)) for record in records_by_mode[mode]]
        for mode in sorted(_MODES)
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"schema_version": 1, "modes": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _serialize(record: LocalCitationDiagnostic) -> dict[str, object]:
    return {
        "question_id": record.question_id,
        "policy_action": record.policy_action,
        "policy_category": record.policy_category,
        "allowed_evidence_ids": list(record.allowed_evidence_ids),
        "retrieved_evidence_ids": list(record.retrieved_evidence_ids),
        "cited_evidence_ids": list(record.cited_evidence_ids),
        "abstained": record.abstained,
        "generation_outcome": record.generation_outcome,
    }
