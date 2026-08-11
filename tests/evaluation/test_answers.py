import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest

from oilfield_chemical_copilot.evaluation.answers import (
    AnswerEvaluationCase,
    DeterministicAnswerResult,
    evaluate_answer,
    load_answer_evaluation_cases,
    write_mode_comparison_report,
)


def _case(
    *, evidence_sufficient: bool = True, expect_citations: bool = True
) -> AnswerEvaluationCase:
    return AnswerEvaluationCase(
        question="What public indicators should be reviewed for scale risk?",
        question_id="scale-01",
        allowed_evidence_ids=("public-scale",),
        evidence_sufficient=evidence_sufficient,
        expect_citations=expect_citations,
        expect_abstention=not evidence_sufficient,
    )


def test_valid_citation_for_sufficient_evidence_passes() -> None:
    result = evaluate_answer(_case(), cited_evidence_ids=("public-scale",), abstained=False)

    assert result == DeterministicAnswerResult(
        question_id="scale-01",
        citation_status="pass",
        abstention_status="pass",
    )


def test_absent_citations_fail_when_citations_are_expected() -> None:
    result = evaluate_answer(_case(), cited_evidence_ids=(), abstained=False)

    assert result.citation_status == "fail"
    assert result.abstention_status == "pass"


def test_citation_outside_allowed_public_evidence_fails() -> None:
    result = evaluate_answer(_case(), cited_evidence_ids=("private-source",), abstained=False)

    assert result.citation_status == "fail"
    assert result.abstention_status == "pass"


@pytest.mark.parametrize(
    ("cited_evidence_ids", "citation_status"),
    [
        ((), "pass"),
        (("public-scale",), "fail"),
        (("private-source",), "fail"),
        (("public-scale", "private-source"), "fail"),
    ],
)
def test_when_citations_are_not_expected_only_absence_passes(
    cited_evidence_ids: tuple[str, ...], citation_status: str
) -> None:
    result = evaluate_answer(
        _case(expect_citations=False), cited_evidence_ids=cited_evidence_ids, abstained=False
    )

    assert result.citation_status == citation_status

def test_insufficient_evidence_requires_abstention() -> None:
    result = evaluate_answer(
        _case(evidence_sufficient=False), cited_evidence_ids=("public-scale",), abstained=False
    )

    assert result.citation_status == "pass"
    assert result.abstention_status == "fail"


def test_insufficient_evidence_abstention_passes() -> None:
    result = evaluate_answer(
        _case(evidence_sufficient=False), cited_evidence_ids=("public-scale",), abstained=True
    )

    assert result.citation_status == "pass"
    assert result.abstention_status == "pass"


def test_loader_returns_exactly_twelve_public_cases() -> None:
    cases = load_answer_evaluation_cases(Path("eval/public_answer_evaluation.jsonl"))

    assert len(cases) == 12
    assert {case.question_id for case in cases} == {
        "corrosion-01",
        "corrosion-02",
        "dosage-01",
        "dosage-02",
        "iron-01",
        "iron-02",
        "paraffin-01",
        "paraffin-02",
        "scale-01",
        "scale-02",
        "water-01",
        "water-02",
    }
    assert all(case.question.strip() for case in cases)
    assert all("private" not in case.question.lower() for case in cases)

def test_public_fixture_uses_reviewed_sufficient_evidence_allowlists() -> None:
    cases_by_id = {
        case.question_id: case for case in load_answer_evaluation_cases(Path("eval/public_answer_evaluation.jsonl"))
    }
    expected_allowlist = (
        "docs:scale_water_analysis_overview.md:document:0:d38047e6931f",
        "docs:water_analysis_interpretation.md:document:0:a36132aa7fbd",
        "water_analysis_examples:sample_water_analysis.csv:csv:0:234ac6fabd87",
    )

    assert cases_by_id["scale-01"].allowed_evidence_ids == expected_allowlist
    assert cases_by_id["water-01"].allowed_evidence_ids == expected_allowlist
    assert all(
        not case.expect_citations and case.expect_abstention
        for case in cases_by_id.values()
        if not case.evidence_sufficient
    )


@pytest.mark.parametrize(
    "records",
    [
        [
            {
                "question_id": "scale-01",
                "question": "What public indicators should be reviewed for scale risk?",
                "allowed_evidence_ids": ["public-scale"],
                "evidence_sufficient": True,
                "expect_citations": True,
                "expect_abstention": False,
            },
            {
                "question_id": "scale-01",
                "question": "What public indicators should be reviewed for scale risk?",
                "allowed_evidence_ids": ["public-scale"],
                "evidence_sufficient": True,
                "expect_citations": True,
                "expect_abstention": False,
            },
        ],
        [
            {
                "question_id": " ",
                "question": "What public indicators should be reviewed for scale risk?",
                "allowed_evidence_ids": ["public-scale"],
                "evidence_sufficient": True,
                "expect_citations": True,
                "expect_abstention": False,
            }
        ],
    ],
)
def test_loader_rejects_blank_and_duplicate_case_ids(
    tmp_path: Path, records: list[dict[str, object]]
) -> None:
    path = tmp_path / "answer-cases.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="blank|duplicate"):
        load_answer_evaluation_cases(path)


def test_report_safe_models_do_not_retain_answer_or_source_path_text() -> None:
    result = evaluate_answer(_case(), cited_evidence_ids=("public-scale",), abstained=False)

    retained_fields = {field.name for field in fields(AnswerEvaluationCase)} | {
        field.name for field in fields(DeterministicAnswerResult)
    }
    assert retained_fields == {
        "question_id",
        "question",
        "allowed_evidence_ids",
        "evidence_sufficient",
        "expect_citations",
        "expect_abstention",
        "citation_status",
        "abstention_status",
    }
    assert "answer text that must not be retained" not in repr(result)
    assert "C:/private/source.md" not in repr(result)


def test_mode_comparison_report_accepts_only_safe_provenance_schema(tmp_path: Path) -> None:
    provenance = {
        "dataset_sha256": hashlib.sha256(b"public dataset").hexdigest(),
        "corpus_sha256": hashlib.sha256(b"public corpus identifiers").hexdigest(),
        "embedding_provider": "ollama",
        "generation_provider": "ollama",
        "judge_provider": "ollama",
        "generation_model_sha256": hashlib.sha256(b"granite4.1:8b").hexdigest(),
        "judge_model_sha256": hashlib.sha256(b"granite4.1:8b").hexdigest(),
        "retrieval_mode_settings": {"vector": "vector", "hybrid": "hybrid"},
        "retrieval_settings": {
            "top_k": 5,
            "min_score": 0.2,
            "max_context_chars": 4000,
            "hybrid_candidate_limit": 10,
            "hybrid_rrf_k": 60,
            "hybrid_min_rrf_score": 0.015,
        },
        "temperature": 0,
        "topic_filter": "none",
    }

    json_path, _ = write_mode_comparison_report(
        {"vector": [], "hybrid": []}, tmp_path, provenance
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["provenance"] == provenance


def test_mode_comparison_report_rejects_incomplete_or_unsafe_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance"):
        write_mode_comparison_report(
            {"vector": [], "hybrid": []},
            tmp_path,
            {"dataset_sha256": "SOURCE-ID-SENTINEL"},
        )
@pytest.mark.parametrize(
    ("field", "value"),
    (("top_k", True), ("min_score", float("inf")), ("hybrid_min_rrf_score", 1.1)),
)
def test_mode_comparison_report_rejects_non_numeric_or_unbounded_retrieval_settings(
    tmp_path: Path, field: str, value: object
) -> None:
    provenance = {
        "dataset_sha256": hashlib.sha256(b"public dataset").hexdigest(),
        "corpus_sha256": hashlib.sha256(b"public corpus identifiers").hexdigest(),
        "embedding_provider": "ollama",
        "generation_provider": "ollama",
        "judge_provider": "ollama",
        "generation_model_sha256": hashlib.sha256(b"granite4.1:8b").hexdigest(),
        "judge_model_sha256": hashlib.sha256(b"granite4.1:8b").hexdigest(),
        "retrieval_mode_settings": {"vector": "vector", "hybrid": "hybrid"},
        "retrieval_settings": {
            "top_k": 5,
            "min_score": 0.2,
            "max_context_chars": 4000,
            "hybrid_candidate_limit": 10,
            "hybrid_rrf_k": 60,
            "hybrid_min_rrf_score": 0.015,
        },
        "temperature": 0,
        "topic_filter": "none",
    }
    provenance["retrieval_settings"][field] = value

    with pytest.raises(ValueError, match="provenance"):
        write_mode_comparison_report({"vector": [], "hybrid": []}, tmp_path, provenance)