"""Privacy-safe deterministic checks and report summaries for public answer evaluation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from oilfield_chemical_copilot.evaluation.judge import AnswerJudge, JudgeResult

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


@dataclass(frozen=True)
class GeneratedAnswer:
    """Runtime-only answer material supplied to the evaluator."""

    question_id: str
    answer: str
    evidence: str
    cited_evidence_ids: tuple[str, ...]
    abstained: bool


@dataclass(frozen=True)
class AnswerEvaluationResult:
    """Report-safe result that excludes answer and evidence text."""

    question_id: str
    deterministic: DeterministicAnswerResult
    judge: JudgeResult


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


def evaluate_cases(
    cases: list[AnswerEvaluationCase],
    answers: list[GeneratedAnswer],
    judge: AnswerJudge,
) -> list[AnswerEvaluationResult]:
    answers_by_id = {answer.question_id: answer for answer in answers}
    if len(answers_by_id) != len(answers):
        raise ValueError("generated answers must not contain duplicate question IDs")
    expected_ids = {case.question_id for case in cases}
    if set(answers_by_id) != expected_ids:
        raise ValueError("generated answer IDs must exactly match evaluation case IDs")

    results: list[AnswerEvaluationResult] = []
    for case in cases:
        answer = answers_by_id[case.question_id]
        results.append(
            AnswerEvaluationResult(
                question_id=case.question_id,
                deterministic=evaluate_answer(
                    case,
                    cited_evidence_ids=answer.cited_evidence_ids,
                    abstained=answer.abstained,
                ),
                judge=judge.judge(case, answer=answer.answer, evidence=answer.evidence),
            )
        )
    return results


def write_report(results: list[AnswerEvaluationResult], output_dir: Path) -> tuple[Path, Path]:
    """Write the existing single-fixture report, including safe case identifiers."""
    report = _report_summary(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "answer_eval.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_dir / "answer_eval.md"
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def write_mode_comparison_report(
    results_by_mode: Mapping[str, list[AnswerEvaluationResult]],
    output_dir: Path,
    provenance: Mapping[str, object],
) -> tuple[Path, Path]:
    """Write public aggregate-only reports for vector and hybrid answer evaluation."""
    if set(results_by_mode) != {"vector", "hybrid"}:
        raise ValueError("comparison results must contain exactly vector and hybrid modes")
    report = {
        "public": True,
        "provenance": _validated_comparison_provenance(provenance),
        "modes": {mode: _comparison_summary(results_by_mode[mode]) for mode in ("vector", "hybrid")},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "answer_eval_comparison.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_dir / "answer_eval_comparison.md"
    markdown_path.write_text(_comparison_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


_SAFE_COMPARISON_PROVENANCE_FIELDS = {
    "dataset_sha256",
    "corpus_sha256",
    "embedding_provider",
    "generation_provider",
    "judge_provider",
    "generation_model_sha256",
    "judge_model_sha256",
    "retrieval_mode_settings",
    "retrieval_settings",
    "temperature",
    "topic_filter",
}
_SAFE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PROVIDER_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _validated_comparison_provenance(provenance: Mapping[str, object]) -> dict[str, object]:
    if set(provenance) != _SAFE_COMPARISON_PROVENANCE_FIELDS:
        raise ValueError("comparison provenance must use the approved public schema")
    hash_fields = (
        "dataset_sha256",
        "corpus_sha256",
        "generation_model_sha256",
        "judge_model_sha256",
    )
    provider_fields = ("embedding_provider", "generation_provider", "judge_provider")
    if any(
        type(provenance[field]) is not str or not _SAFE_SHA256.fullmatch(provenance[field])
        for field in hash_fields
    ) or any(
        type(provenance[field]) is not str or not _SAFE_PROVIDER_LABEL.fullmatch(provenance[field])
        for field in provider_fields
    ):
        raise ValueError("comparison provenance contains unsafe identities")
    if (
        provenance["retrieval_mode_settings"] != {"vector": "vector", "hybrid": "hybrid"}
        or type(provenance["temperature"]) is not int
        or provenance["temperature"] != 0
        or provenance["topic_filter"] != "none"
        or not _valid_retrieval_settings(provenance["retrieval_settings"])
    ):
        raise ValueError("comparison provenance must retain fixed live evaluation settings")
    return {
        **{field: provenance[field] for field in hash_fields + provider_fields},
        "retrieval_mode_settings": {"vector": "vector", "hybrid": "hybrid"},
        "retrieval_settings": dict(provenance["retrieval_settings"]),
        "temperature": 0,
        "topic_filter": "none",
    }

_RETRIEVAL_INTEGER_FIELDS = {"top_k", "max_context_chars", "hybrid_candidate_limit", "hybrid_rrf_k"}
_RETRIEVAL_SCORE_FIELDS = {"min_score", "hybrid_min_rrf_score"}
_RETRIEVAL_SETTINGS_FIELDS = _RETRIEVAL_INTEGER_FIELDS | _RETRIEVAL_SCORE_FIELDS


def _valid_retrieval_settings(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != _RETRIEVAL_SETTINGS_FIELDS:
        return False
    integers = (value[field] for field in _RETRIEVAL_INTEGER_FIELDS)
    scores = (value[field] for field in _RETRIEVAL_SCORE_FIELDS)
    return all(type(item) is int and 1 <= item <= 1_000_000 for item in integers) and all(
        type(item) in {int, float} and 0 <= float(item) <= 1 and math.isfinite(float(item))
        for item in scores
    )

def _report_summary(results: list[AnswerEvaluationResult]) -> dict[str, object]:
    ordered_results = sorted(results, key=lambda result: result.question_id)
    return {"cases": [result.question_id for result in ordered_results]} | _aggregate_summary(ordered_results)


def _comparison_summary(results: list[AnswerEvaluationResult]) -> dict[str, object]:
    return {"question_count": len(results)} | _aggregate_summary(results, include_models=False, include_providers=False)


def _aggregate_summary(
    results: list[AnswerEvaluationResult], *,
    include_models: bool = True,
    include_providers: bool = True,
) -> dict[str, object]:
    from oilfield_chemical_copilot.evaluation.judge import JudgeAvailable

    citation_counts = Counter(result.deterministic.citation_status for result in results)
    abstention_counts = Counter(result.deterministic.abstention_status for result in results)
    available = [result.judge for result in results if isinstance(result.judge, JudgeAvailable)]
    judge_status = Counter(result.judge.status for result in results)
    score_fields = (
        "groundedness",
        "relevance",
        "limitation_awareness",
        "operational_certainty",
    )
    scores = {
        field: mean(getattr(result.scores, field) for result in available) for field in score_fields
    } if available else {}
    judge_summary: dict[str, object] = {
        "status": dict(sorted(judge_status.items())),

        "scores": scores,
    }
    if include_providers:
        judge_summary["providers"] = sorted({result.judge.provider for result in results})
    if include_models:
        judge_summary["models"] = sorted({result.judge.model for result in results})
    return {
        "deterministic": {
            "citations": dict(sorted(citation_counts.items())),
            "abstention": dict(sorted(abstention_counts.items())),
        },
        "judge": judge_summary,
    }


def _markdown_report(report: dict[str, object]) -> str:
    deterministic = report["deterministic"]
    judge = report["judge"]
    assert isinstance(deterministic, dict)
    assert isinstance(judge, dict)
    return "\n".join(
        [
            "# Grounded answer evaluation",
            "",
            f"Cases: {len(report['cases'])}",
            "",
            "| deterministic check | pass | fail |",
            "| --- | ---: | ---: |",
            "| citations | {pass_count} | {fail_count} |".format(
                pass_count=deterministic["citations"].get("pass", 0),
                fail_count=deterministic["citations"].get("fail", 0),
            ),
            "| abstention | {pass_count} | {fail_count} |".format(
                pass_count=deterministic["abstention"].get("pass", 0),
                fail_count=deterministic["abstention"].get("fail", 0),
            ),
            "",
            "## Judge",
            "",
            f"Available: {judge['status'].get('available', 0)}",
            f"Unavailable: {judge['status'].get('unavailable', 0)}",
            "",
        ]
    )


def _comparison_markdown_report(report: dict[str, object]) -> str:
    modes = report["modes"]
    assert isinstance(modes, dict)
    lines = [
        "# Public grounded answer evaluation comparison",
        "",
        "Public: true",
        f"Provenance: {json.dumps(report['provenance'], sort_keys=True)}",
        "",
    ]
    for mode in ("vector", "hybrid"):
        summary = modes[mode]
        assert isinstance(summary, dict)
        deterministic = summary["deterministic"]
        judge = summary["judge"]
        assert isinstance(deterministic, dict)
        assert isinstance(judge, dict)
        lines.extend(
            [
                f"## {mode}",
                "",
                f"Questions: {summary['question_count']}",
                f"Citation checks: {json.dumps(deterministic['citations'], sort_keys=True)}",
                f"Abstention checks: {json.dumps(deterministic['abstention'], sort_keys=True)}",
                f"Judge status: {json.dumps(judge['status'], sort_keys=True)}",


                f"Aggregate rubric scores: {json.dumps(judge['scores'], sort_keys=True)}",
                "",
            ]
        )
    return "\n".join(lines)