from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from oilfield_chemical_copilot.evaluation.semantic_grounding import (
    Approval,
    EvaluationError,
    ReviewDecision,
    SemanticCase,
    evaluate_case,
    evaluate_once,
    preflight,
    seal_cases,
    validate_cases,
    _validate_aggregate_payload,
    verify_no_prior_overlap,
)


_CATEGORIES = (
    "exact_value",
    "range_bound",
    "unit",
    "qualifier_condition",
    "conflicting_evidence",
    "no_established_threshold",
)


def _case(category: str, index: int, expected_outcome: str) -> SemanticCase:
    return SemanticCase(
        case_id=f"{category}-{index:02d}",
        category=category,
        question=f"Synthetic review question {category} {index}.",
        excerpts=(f"Synthetic evidence {category} {index}.",),
        answer=f"Synthetic evidence {category} {index}.",
        expected_outcome=expected_outcome,
        failure_class="grounded_claim" if expected_outcome == "allow" else "semantic_mismatch",
        author_id="author-v4",
    )


def _valid_cases() -> tuple[SemanticCase, ...]:
    return tuple(
        _case(category, offset + slot, "allow" if slot < 3 else "fallback")
        for offset, category in enumerate(_CATEGORIES, start=1)
        for slot in range(6)
    )


def _reviews(cases: tuple[SemanticCase, ...]) -> tuple[ReviewDecision, ...]:
    return tuple(ReviewDecision(case.case_id, "reviewer-v4", "approved") for case in cases)


def _write_fixture(path: Path, cases: tuple[SemanticCase, ...]) -> None:
    path.write_text(
        "\n".join(json.dumps(case.to_mapping(), sort_keys=True) for case in cases) + "\n",
        encoding="utf-8",
    )


def test_valid_fixture_has_six_balanced_categories_and_both_outcomes() -> None:
    summary = validate_cases(_valid_cases(), _reviews(_valid_cases()))

    assert summary.case_count == 36
    assert summary.category_counts == {category: 6 for category in _CATEGORIES}
    assert summary.outcome_counts == {"allow": 18, "fallback": 18}


@pytest.mark.parametrize(
    ("cases", "code"),
    [
        (_valid_cases()[:-1], "CASE_COUNT_INVALID"),
        (_valid_cases()[:-1] + (_valid_cases()[0],), "DUPLICATE_CASE_ID"),
        (
            tuple(
                replace(case, category="unknown")
                if index < 6
                else case
                for index, case in enumerate(_valid_cases())
            ),
            "CATEGORY_INVALID",
        ),
            (
                tuple(
                    replace(case, expected_outcome="allow")
                    if index < 6
                    else replace(case, expected_outcome="fallback")
                    if index < 9
                    else case
                    for index, case in enumerate(_valid_cases())
                ),
            "CATEGORY_OUTCOME_BALANCE_INVALID",
        ),
    ],
)
def test_invalid_fixture_is_rejected_without_exposing_case_content(
    cases: tuple[SemanticCase, ...], code: str
) -> None:
    with pytest.raises(EvaluationError, match=f"^{code}$"):
        validate_cases(cases, _reviews(cases))


def test_sealing_is_canonical_and_detects_prior_question_overlap(tmp_path: Path) -> None:
    cases = _valid_cases()
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.jsonl"
    sealed_path = tmp_path / "sealed.jsonl"
    digest_path = tmp_path / "sealed.sha256"
    _write_fixture(draft_path, cases)
    review_path.write_text(
        "\n".join(
            json.dumps({"case_id": item.case_id, "reviewer_id": item.reviewer_id, "verdict": item.verdict})
            for item in _reviews(cases)
        )
        + "\n",
        encoding="utf-8",
    )

    summary = seal_cases(draft_path, review_path, sealed_path, digest_path)

    assert summary.case_count == 36
    assert digest_path.read_text(encoding="ascii").strip().isalnum()
    assert len(digest_path.read_text(encoding="ascii").strip()) == 64
    previous_path = tmp_path / "previous.jsonl"
    previous_path.write_text(json.dumps({"question": cases[0].question}) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="^PRIOR_CASE_OVERLAP$"):
        verify_no_prior_overlap(cases, (previous_path,))


def test_evaluate_case_reflects_the_production_formatter_unit_rejection() -> None:
    case = SemanticCase(
        case_id="unit-negative",
        category="unit",
        question="Synthetic unit review.",
        excerpts=("The laboratory result was 500 mg/L under the stated method.",),
        answer="The laboratory result was 500 ppm under the stated method.",
        expected_outcome="fallback",
        failure_class="unsupported_unit_conversion",
        author_id="author-v4",
    )

    observation = evaluate_case(case)

    assert observation.expected_outcome == "fallback"
    assert observation.observed_outcome == "fallback"
    assert observation.failure_class is None


def test_evaluate_once_consumes_approval_and_writes_aggregate_only_report(tmp_path: Path) -> None:
    cases = _valid_cases()
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.jsonl"
    private_root = tmp_path / ".private"
    sealed_path = private_root / "sealed.jsonl"
    digest_path = private_root / "sealed.sha256"
    approval_path = private_root / "approval.json"
    state_path = private_root / "state.json"
    private_result_path = private_root / "diagnostics.json"
    report_path = tmp_path / "aggregate.json"
    _write_fixture(draft_path, cases)
    review_path.write_text(
        "\n".join(
            json.dumps({"case_id": item.case_id, "reviewer_id": item.reviewer_id, "verdict": item.verdict})
            for item in _reviews(cases)
        )
        + "\n",
        encoding="utf-8",
    )
    seal_cases(draft_path, review_path, sealed_path, digest_path)
    approval = Approval.for_current_artifacts(sealed_path, Path("src/oilfield_chemical_copilot/rag/formatter.py"))
    approval_path.write_text(json.dumps(approval.to_mapping(), sort_keys=True) + "\n", encoding="utf-8")
    preflight_summary = preflight(
        sealed_path,
        digest_path,
        approval_path,
        state_path,
        Path("src/oilfield_chemical_copilot/rag/formatter.py"),
        cases,
        (),
        private_root,
        private_result_path,
        report_path,
    )

    summary = evaluate_once(
        sealed_path,
        digest_path,
        approval_path,
        state_path,
        private_result_path,
        report_path,
        cases=cases,
        prior_paths=(),
        private_root=private_root,
    )

    assert summary.case_count == 36
    assert preflight_summary.seal_valid is True
    assert state_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {"categories", "counts", "failure_classes", "gates", "status"}
    assert report["counts"]["pass_rate_percent"] == 50.0
    assert "question" not in report_path.read_text(encoding="utf-8").lower()
    with pytest.raises(EvaluationError, match="^ATTEMPT_UNAVAILABLE$"):
        evaluate_once(
            sealed_path,
            digest_path,
            approval_path,
            state_path,
            private_result_path,
            report_path,
            cases=cases,
            prior_paths=(),
            private_root=private_root,
        )


def test_preflight_rejects_mismatched_approval_before_scoring(tmp_path: Path) -> None:
    cases = _valid_cases()
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.jsonl"
    private_root = tmp_path / ".private"
    sealed_path = private_root / "sealed.jsonl"
    digest_path = private_root / "sealed.sha256"
    approval_path = private_root / "approval.json"
    _write_fixture(draft_path, cases)
    review_path.write_text(
        "\n".join(
            json.dumps({"case_id": item.case_id, "reviewer_id": item.reviewer_id, "verdict": item.verdict})
            for item in _reviews(cases)
        )
        + "\n",
        encoding="utf-8",
    )
    seal_cases(draft_path, review_path, sealed_path, digest_path)
    approval = Approval.for_current_artifacts(sealed_path, Path("src/oilfield_chemical_copilot/rag/formatter.py"))
    approval_path.write_text(
        json.dumps({**approval.to_mapping(), "formatter_sha256": "0" * 64}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="^APPROVAL_DIGEST_MISMATCH$"):
        preflight(
            sealed_path,
            digest_path,
            approval_path,
            private_root / "state.json",
            Path("src/oilfield_chemical_copilot/rag/formatter.py"),
            cases,
            (),
            private_root,
            private_root / "diagnostics.json",
            tmp_path / "aggregate.json",
        )


def test_evaluate_once_rejects_private_diagnostics_outside_private_root(tmp_path: Path) -> None:
    cases = _valid_cases()
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.jsonl"
    private_root = tmp_path / ".private"
    sealed_path = private_root / "sealed.jsonl"
    digest_path = private_root / "sealed.sha256"
    approval_path = private_root / "approval.json"
    state_path = private_root / "state.json"
    _write_fixture(draft_path, cases)
    review_path.write_text(
        "\n".join(
            json.dumps({"case_id": item.case_id, "reviewer_id": item.reviewer_id, "verdict": item.verdict})
            for item in _reviews(cases)
        )
        + "\n",
        encoding="utf-8",
    )
    seal_cases(draft_path, review_path, sealed_path, digest_path)
    approval = Approval.for_current_artifacts(sealed_path, Path("src/oilfield_chemical_copilot/rag/formatter.py"))
    approval_path.write_text(json.dumps(approval.to_mapping(), sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="^PRIVATE_PATH_REQUIRED$"):
        evaluate_once(
            sealed_path,
            digest_path,
            approval_path,
            state_path,
            tmp_path / "diagnostics.json",
            tmp_path / "aggregate.json",
            private_root=private_root,
            cases=cases,
            prior_paths=(),
        )


def test_sealing_rejects_sensitive_failure_class_name_before_one_shot_consumption(tmp_path: Path) -> None:
    cases = _valid_cases()
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.jsonl"
    private_root = tmp_path / ".private"
    sealed_path = private_root / "sealed.jsonl"
    digest_path = private_root / "sealed.sha256"
    state_path = private_root / "state.json"
    report_path = tmp_path / "aggregate.json"
    unsafe_cases = tuple(replace(case, failure_class="question") for case in cases)
    _write_fixture(draft_path, unsafe_cases)
    review_path.write_text(
        "\n".join(
            json.dumps({"case_id": item.case_id, "reviewer_id": item.reviewer_id, "verdict": item.verdict})
            for item in _reviews(unsafe_cases)
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="^FAILURE_CLASS_PRIVACY_INVALID$"):
        seal_cases(draft_path, review_path, sealed_path, digest_path)

    assert not state_path.exists()
    assert not report_path.exists()


@pytest.mark.parametrize(
    "payload",
    (
        {"counts": {"source": 1}},
        {"counts": {"label": "private path"}},
    ),
)
def test_aggregate_payload_rejects_nested_sensitive_content(payload: object) -> None:
    with pytest.raises(EvaluationError, match="^AGGREGATE_REPORT_PRIVACY_VIOLATION$"):
        _validate_aggregate_payload(payload)


def test_evaluate_once_rejects_prior_overlap_before_consuming_state(tmp_path: Path) -> None:
    cases = _valid_cases()
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.jsonl"
    private_root = tmp_path / ".private"
    sealed_path = private_root / "sealed.jsonl"
    digest_path = private_root / "sealed.sha256"
    approval_path = private_root / "approval.json"
    state_path = private_root / "state.json"
    private_result_path = private_root / "diagnostics.json"
    report_path = tmp_path / "aggregate.json"
    prior_path = tmp_path / "prior.jsonl"
    _write_fixture(draft_path, cases)
    review_path.write_text(
        "\n".join(
            json.dumps({"case_id": item.case_id, "reviewer_id": item.reviewer_id, "verdict": item.verdict})
            for item in _reviews(cases)
        )
        + "\n",
        encoding="utf-8",
    )
    seal_cases(draft_path, review_path, sealed_path, digest_path)
    approval = Approval.for_current_artifacts(sealed_path, Path("src/oilfield_chemical_copilot/rag/formatter.py"))
    approval_path.write_text(json.dumps(approval.to_mapping(), sort_keys=True) + "\n", encoding="utf-8")
    prior_path.write_text(json.dumps({"question": cases[0].question}) + "\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="^PRIOR_CASE_OVERLAP$"):
        evaluate_once(
            sealed_path,
            digest_path,
            approval_path,
            state_path,
            private_result_path,
            report_path,
            cases=cases,
            prior_paths=(prior_path,),
            private_root=private_root,
        )

    assert not state_path.exists()


def test_v5_approval_scope_is_accepted_only_for_a_v5_run(tmp_path: Path) -> None:
    sealed_path = tmp_path / "sealed.jsonl"
    formatter_path = Path("src/oilfield_chemical_copilot/rag/formatter.py")
    sealed_path.write_text("sealed\n", encoding="utf-8")

    approval = Approval.for_current_artifacts(
        sealed_path,
        formatter_path,
        scope="semantic-grounding-v5",
    )

    assert approval.scope == "semantic-grounding-v5"


def test_v6_approval_scope_requires_matching_preflight_scope(tmp_path: Path) -> None:
    cases = _valid_cases()
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.jsonl"
    private_root = tmp_path / ".private"
    sealed_path = private_root / "sealed.jsonl"
    digest_path = private_root / "sealed.sha256"
    approval_path = private_root / "approval.json"
    state_path = private_root / "state.json"
    private_result_path = private_root / "diagnostics.json"
    report_path = tmp_path / "aggregate.json"
    formatter_path = Path("src/oilfield_chemical_copilot/rag/formatter.py")
    _write_fixture(draft_path, cases)
    review_path.write_text(
        "\n".join(
            json.dumps({"case_id": item.case_id, "reviewer_id": item.reviewer_id, "verdict": item.verdict})
            for item in _reviews(cases)
        )
        + "\n",
        encoding="utf-8",
    )
    seal_cases(draft_path, review_path, sealed_path, digest_path)
    approval = Approval.for_current_artifacts(sealed_path, formatter_path, scope="semantic-grounding-v6")
    approval_path.write_text(json.dumps(approval.to_mapping(), sort_keys=True) + "\n", encoding="utf-8")

    summary = preflight(
        sealed_path,
        digest_path,
        approval_path,
        state_path,
        formatter_path,
        cases,
        (),
        private_root,
        private_result_path,
        report_path,
        expected_scope="semantic-grounding-v6",
    )

    assert summary.attempt_available is True
    with pytest.raises(EvaluationError, match="^APPROVAL_SCOPE_MISMATCH$"):
        preflight(
            sealed_path,
            digest_path,
            approval_path,
            state_path,
            formatter_path,
            cases,
            (),
            private_root,
            private_result_path,
            report_path,
            expected_scope="semantic-grounding-v5",
        )
    assert not state_path.exists()
