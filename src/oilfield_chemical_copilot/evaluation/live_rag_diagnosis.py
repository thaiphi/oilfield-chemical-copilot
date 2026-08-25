"""Aggregate-only diagnosis of live RAG citation and abstention failures."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from oilfield_chemical_copilot.evaluation.answers import (
    DeterministicAnswerResult,
    _validated_comparison_provenance,
)

GenerationOutcome = Literal["not_called", "succeeded", "failed"]
CitationFailureCategory = Literal[
    "expected_citation_missing_no_qualifying_retrieval",
    "expected_citation_missing_generation_failure",
    "expected_citation_missing_abstained_after_qualifying_retrieval",
    "expected_citation_allowed_evidence_not_retrieved",
    "expected_citation_allowed_retrieved_not_cited",
    "expected_citation_mixed_with_disallowed",
    "expected_citation_missing_after_answer",
    "unexpected_citation_when_abstention_expected",
]
AbstentionFailureCategory = Literal[
    "over_abstention_no_qualifying_retrieval",
    "over_abstention_generation_failure",
    "over_abstention_after_qualifying_retrieval",
    "under_abstention_answered_on_insufficient_case",
]

_CITATION_CATEGORIES = frozenset(CitationFailureCategory.__args__)
_ABSTENTION_CATEGORIES = frozenset(AbstentionFailureCategory.__args__)
_MODES = ("vector", "hybrid")


@dataclass(frozen=True)
class LiveDiagnosticObservation:
    expect_citations: bool
    evidence_sufficient: bool
    expect_abstention: bool
    allowed_evidence_ids: tuple[str, ...]
    retrieved_evidence_ids: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...]
    abstained: bool
    generation_outcome: GenerationOutcome


@dataclass(frozen=True)
class LiveFailureDiagnosis:
    citation_failure: CitationFailureCategory | None
    abstention_failure: AbstentionFailureCategory | None


def _invalid_state() -> ValueError:
    return ValueError("invalid diagnostic state")


def _invalid_report() -> ValueError:
    return ValueError("invalid diagnosis report")


def _valid_ids(value: object) -> bool:
    return (
        type(value) is tuple
        and all(type(item) is str and bool(item) for item in value)
        and len(set(value)) == len(value)
    )


def _validate_observation(observation: object) -> None:
    if not isinstance(observation, LiveDiagnosticObservation):
        raise _invalid_state()
    if not all(
        type(value) is bool
        for value in (
            observation.expect_citations,
            observation.evidence_sufficient,
            observation.expect_abstention,
            observation.abstained,
        )
    ):
        raise _invalid_state()
    if not all(
        _valid_ids(value)
        for value in (
            observation.allowed_evidence_ids,
            observation.retrieved_evidence_ids,
            observation.cited_evidence_ids,
        )
    ):
        raise _invalid_state()
    if observation.generation_outcome not in {"not_called", "succeeded", "failed"}:
        raise _invalid_state()
    if (
        observation.expect_citations != observation.evidence_sufficient
        or observation.expect_abstention == observation.evidence_sufficient
    ):
        raise _invalid_state()

    retrieved = set(observation.retrieved_evidence_ids)
    cited = set(observation.cited_evidence_ids)
    if not cited <= retrieved or (observation.abstained and cited):
        raise _invalid_state()
    if not retrieved and (
        not observation.abstained or observation.generation_outcome != "not_called"
    ):
        raise _invalid_state()
    if observation.generation_outcome == "not_called" and retrieved:
        raise _invalid_state()
    if observation.generation_outcome == "failed" and (
        not retrieved or not observation.abstained or cited
    ):
        raise _invalid_state()
    if not observation.abstained and observation.generation_outcome != "succeeded":
        raise _invalid_state()


def _validate_result(
    observation: LiveDiagnosticObservation, result: object
) -> DeterministicAnswerResult:
    if (
        not isinstance(result, DeterministicAnswerResult)
        or result.citation_status not in {"pass", "fail"}
        or result.abstention_status not in {"pass", "fail"}
    ):
        raise _invalid_state()
    cited = set(observation.cited_evidence_ids)
    allowed = set(observation.allowed_evidence_ids)
    expected_citation = "pass" if (
        (bool(cited) and cited <= allowed)
        if observation.expect_citations
        else not cited
    ) else "fail"
    expected_abstention = (
        "pass" if observation.abstained == observation.expect_abstention else "fail"
    )
    if (
        result.citation_status != expected_citation
        or result.abstention_status != expected_abstention
    ):
        raise _invalid_state()
    return result


def classify_live_failure(
    observation: LiveDiagnosticObservation,
    deterministic_result: DeterministicAnswerResult,
) -> LiveFailureDiagnosis:
    """Classify each failed deterministic dimension with one closed category."""
    _validate_observation(observation)
    _validate_result(observation, deterministic_result)
    retrieved = set(observation.retrieved_evidence_ids)
    cited = set(observation.cited_evidence_ids)
    allowed = set(observation.allowed_evidence_ids)

    citation_failure: CitationFailureCategory | None = None
    if deterministic_result.citation_status == "fail":
        if observation.abstained:
            if not retrieved:
                citation_failure = "expected_citation_missing_no_qualifying_retrieval"
            elif observation.generation_outcome == "failed":
                citation_failure = "expected_citation_missing_generation_failure"
            else:
                citation_failure = "expected_citation_missing_abstained_after_qualifying_retrieval"
        elif not observation.evidence_sufficient and cited:
            citation_failure = "unexpected_citation_when_abstention_expected"
        elif not cited:
            citation_failure = "expected_citation_missing_after_answer"
        elif not allowed & retrieved:
            citation_failure = "expected_citation_allowed_evidence_not_retrieved"
        elif cited & allowed and cited - allowed:
            citation_failure = "expected_citation_mixed_with_disallowed"
        elif not cited & allowed:
            citation_failure = "expected_citation_allowed_retrieved_not_cited"
        else:
            raise _invalid_state()

    abstention_failure: AbstentionFailureCategory | None = None
    if deterministic_result.abstention_status == "fail":
        if observation.abstained:
            if not retrieved:
                abstention_failure = "over_abstention_no_qualifying_retrieval"
            elif observation.generation_outcome == "failed":
                abstention_failure = "over_abstention_generation_failure"
            else:
                abstention_failure = "over_abstention_after_qualifying_retrieval"
        elif not observation.evidence_sufficient and cited:
            abstention_failure = "under_abstention_answered_on_insufficient_case"
        else:
            raise _invalid_state()
    return LiveFailureDiagnosis(citation_failure, abstention_failure)


def _validate_diagnosis(diagnosis: object, result: object) -> LiveFailureDiagnosis:
    if not isinstance(diagnosis, LiveFailureDiagnosis) or not isinstance(result, DeterministicAnswerResult):
        raise _invalid_report()
    citation = diagnosis.citation_failure
    abstention = diagnosis.abstention_failure
    if citation is not None and citation not in _CITATION_CATEGORIES:
        raise _invalid_report()
    if abstention is not None and abstention not in _ABSTENTION_CATEGORIES:
        raise _invalid_report()
    if result.citation_status not in {"pass", "fail"} or result.abstention_status not in {"pass", "fail"}:
        raise _invalid_report()
    if (citation is not None) != (result.citation_status == "fail"):
        raise _invalid_report()
    if (abstention is not None) != (result.abstention_status == "fail"):
        raise _invalid_report()
    return diagnosis


def _validated_modes(value: object) -> Mapping[str, list[object]]:
    if not isinstance(value, Mapping) or set(value) != set(_MODES):
        raise _invalid_report()
    if any(type(value[mode]) is not list for mode in _MODES):
        raise _invalid_report()
    return value


def write_live_failure_diagnosis(
    diagnoses_by_mode: dict[str, list[LiveFailureDiagnosis]],
    deterministic_results_by_mode: dict[str, list[DeterministicAnswerResult]],
    output_dir: Path,
    provenance: dict[str, object],
    *,
    verified_preflight: bool,
) -> tuple[Path, Path]:
    """Write only safe aggregates after exact per-case reconciliation."""
    if type(verified_preflight) is not bool or not isinstance(output_dir, Path):
        raise _invalid_report()
    diagnoses = _validated_modes(diagnoses_by_mode)
    results = _validated_modes(deterministic_results_by_mode)
    try:
        safe_provenance = _validated_comparison_provenance(provenance)
    except (TypeError, ValueError):
        raise _invalid_report() from None

    modes: dict[str, dict[str, object]] = {}
    for mode in _MODES:
        mode_diagnoses = diagnoses[mode]
        mode_results = results[mode]
        if len(mode_diagnoses) != len(mode_results):
            raise _invalid_report()
        reconciled = [
            _validate_diagnosis(diagnosis, deterministic)
            for diagnosis, deterministic in zip(mode_diagnoses, mode_results, strict=True)
        ]
        citation = Counter(
            diagnosis.citation_failure for diagnosis in reconciled if diagnosis.citation_failure is not None
        )
        abstention = Counter(
            diagnosis.abstention_failure
            for diagnosis in reconciled
            if diagnosis.abstention_failure is not None
        )
        if (
            sum(citation.values()) != sum(result.citation_status == "fail" for result in mode_results)
            or sum(abstention.values())
            != sum(result.abstention_status == "fail" for result in mode_results)
        ):
            raise _invalid_report()
        modes[mode] = {
            "question_count": len(mode_results),
            "citation_failures": dict(sorted(citation.items())),
            "abstention_failures": dict(sorted(abstention.items())),
        }

    baseline_reproduced = verified_preflight and all(
        mode["question_count"] == 12
        and sum(mode["citation_failures"].values()) == 8
        and sum(mode["abstention_failures"].values()) == 6
        for mode in modes.values()
    )
    report = {
        "public": True,
        "provenance": safe_provenance,
        "baseline_reproduced": baseline_reproduced,
        "modes": modes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "live_rag_failure_diagnosis.json"
    markdown_path = output_dir / "live_rag_failure_diagnosis.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Live RAG failure diagnosis", "", "Public: true", f"Baseline reproduced: {str(baseline_reproduced).lower()}", ""]
    for mode in _MODES:
        summary = modes[mode]
        lines.extend([f"## {mode}", "", f"Question count: {summary['question_count']}", f"Citation failures: {json.dumps(summary['citation_failures'], sort_keys=True)}", f"Abstention failures: {json.dumps(summary['abstention_failures'], sort_keys=True)}", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path