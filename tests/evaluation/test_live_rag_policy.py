from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from oilfield_chemical_copilot.evaluation.abstention_policy import (
    AbstentionPolicyDecision,
    classify_claim_scope,
)
from oilfield_chemical_copilot.evaluation.answers import (
    AnswerEvaluationCase,
    DeterministicAnswerResult,
    GeneratedAnswer,
    evaluate_answer,
)
from oilfield_chemical_copilot.evaluation.live_rag import LiveAnswerCapture
from oilfield_chemical_copilot.evaluation.live_rag_policy import (
    POLICY_NAME,
    POLICY_VERSION,
    PolicyCounterfactualScore,
    PolicyDeterministicMetrics,
    score_policy_counterfactual,
    write_live_rag_policy_investigation,
)


def case(index: int, *, expects_abstention: bool) -> AnswerEvaluationCase:
    question = (
        f"Which public indicators should frame an initial review {index}?"
        if index < 6
        else f"Can the public material determine the root cause at a named asset {index}?"
    )
    return AnswerEvaluationCase(
        question_id=f"case-{index}",
        question=question,
        allowed_evidence_ids=(f"allowed-{index}",),
        evidence_sufficient=not expects_abstention,
        expect_citations=not expects_abstention,
        expect_abstention=expects_abstention,
    )


def capture(index: int, *, cited_ids: tuple[str, ...], abstained: bool = False) -> LiveAnswerCapture:
    return LiveAnswerCapture(
        GeneratedAnswer(
            question_id=f"case-{index}",
            answer=f"answer-{index}",
            evidence=f"evidence-{index}",
            cited_evidence_ids=cited_ids,
            abstained=abstained,
        ),
        retrieved_evidence_ids=cited_ids,
        generation_outcome="succeeded",
    )


def control(case_value: AnswerEvaluationCase, capture_value: LiveAnswerCapture) -> DeterministicAnswerResult:
    return evaluate_answer(
        case_value,
        cited_evidence_ids=capture_value.answer.cited_evidence_ids,
        abstained=capture_value.answer.abstained,
    )


def provenance() -> dict[str, object]:
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    return {
        "dataset_sha256": digest("dataset"),
        "corpus_sha256": digest("corpus"),
        "embedding_provider": "ollama",
        "generation_provider": "ollama",
        "judge_provider": "ollama",
        "generation_model_sha256": digest("generation"),
        "judge_model_sha256": digest("judge"),
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


def score(index: int, *, abstain: bool, cited_ids: tuple[str, ...] | None = None) -> PolicyCounterfactualScore:
    case_value = case(index, expects_abstention=abstain)
    capture_value = capture(index, cited_ids=cited_ids if cited_ids is not None else (f"allowed-{index}",))
    decision = classify_claim_scope(case_value.question)
    return score_policy_counterfactual(decision, case_value, capture_value, control(case_value, capture_value))


def baseline_scores() -> list[PolicyCounterfactualScore]:
    scores = [score(index, abstain=False) for index in range(6)]
    scores += [score(index, abstain=True) for index in range(6, 12)]
    scores[0] = score(0, abstain=False, cited_ids=("other-0",))
    scores[1] = score(1, abstain=False, cited_ids=("allowed-1", "other-1"))
    return scores


def test_abstain_policy_scores_no_citations_without_calling_runtime_dependencies() -> None:
    case_value = case(6, expects_abstention=True)
    capture_value = capture(6, cited_ids=("allowed-6",))

    score_value = score_policy_counterfactual(
        AbstentionPolicyDecision("abstain", "site_specific_determination"),
        case_value,
        capture_value,
        control(case_value, capture_value),
    )

    assert score_value.shadow == PolicyDeterministicMetrics("pass", "pass")
    assert score_value.control == PolicyDeterministicMetrics("fail", "fail")


def test_allow_policy_reuses_the_unmodified_capture() -> None:
    case_value = case(0, expects_abstention=False)
    capture_value = capture(0, cited_ids=("other-0",))

    score_value = score_policy_counterfactual(
        AbstentionPolicyDecision("allow", "general_review"),
        case_value,
        capture_value,
        control(case_value, capture_value),
    )

    assert score_value.shadow == score_value.control == PolicyDeterministicMetrics("fail", "pass")


@pytest.mark.parametrize("changed", ["case", "capture", "control"])
def test_scorer_rejects_swapped_or_unreconciled_case_pairings_without_runtime_values(changed: str) -> None:
    case_value = case(0, expects_abstention=False)
    capture_value = capture(0, cited_ids=("allowed-0",))
    control_value = control(case_value, capture_value)
    if changed == "case":
        case_value = replace(case_value, question_id="CASE-ID-SENTINEL")
    elif changed == "capture":
        capture_value = replace(capture_value, answer=replace(capture_value.answer, question_id="CASE-ID-SENTINEL"))
    else:
        control_value = replace(control_value, citation_status="fail")

    with pytest.raises(ValueError) as error:
        score_policy_counterfactual(
            AbstentionPolicyDecision("allow", "general_review"), case_value, capture_value, control_value
        )
    assert str(error.value) == "invalid policy counterfactual"
    assert "CASE-ID-SENTINEL" not in str(error.value)


def test_writer_emits_the_expected_safe_12_case_aggregate(tmp_path: Path) -> None:
    scores = baseline_scores()
    expected_actions = ["allow"] * 6 + ["abstain"] * 6
    assert [score_value.decision.action for score_value in scores] == expected_actions
    assert [
        score_value.decision
        for score_value in scores
    ] == [
        classify_claim_scope(case(index, expects_abstention=index >= 6).question)
        for index in range(12)
    ]
    json_path, markdown_path = write_live_rag_policy_investigation(
        {"vector": scores, "hybrid": scores}, tmp_path, provenance(), verified_preflight=True
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report == {
        "public": True,
        "provenance": provenance(),
        "verified_preflight": True,
        "baseline_reproduced": True,
        "policy": {"name": POLICY_NAME, "version": POLICY_VERSION},
        "modes": {
            mode: {
                "question_count": 12,
                "decision_categories": {"general_review": 6, "site_specific_determination": 6},
                "control": {"citation": {"fail": 8, "pass": 4}, "abstention": {"fail": 6, "pass": 6}},
                "shadow": {"citation": {"fail": 2, "pass": 10}, "abstention": {"pass": 12}},
            }
            for mode in ("vector", "hybrid")
        },
    }
    text = markdown_path.read_text(encoding="utf-8")
    assert "case-" not in text
    assert "answer-" not in text
    assert "allowed-" not in text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda scores: scores[:-1],
        lambda scores: [*scores[:6], score(6, abstain=False), *scores[7:]],
    ],
)
def test_writer_rejects_malformed_counts(tmp_path: Path, mutate) -> None:
    scores = baseline_scores()
    with pytest.raises(ValueError, match="^invalid policy investigation report$"):
        write_live_rag_policy_investigation(
            {"vector": mutate(scores), "hybrid": mutate(scores)}, tmp_path, provenance(), verified_preflight=True
        )


def test_writer_marks_baseline_drift_without_interpreting_it(tmp_path: Path) -> None:
    scores = baseline_scores()
    drifted = [replace(scores[0], control=PolicyDeterministicMetrics("pass", "pass")), *scores[1:]]
    json_path, _ = write_live_rag_policy_investigation(
        {"vector": drifted, "hybrid": drifted}, tmp_path, provenance(), verified_preflight=True
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["baseline_reproduced"] is False


def test_writer_rejects_over_abstention_missed_abstention_and_mode_decision_mismatch(tmp_path: Path) -> None:
    scores = baseline_scores()
    over_abstention = [score(0, abstain=True), *scores[1:]]
    missed_abstention = [*scores[:6], score(6, abstain=False), *scores[7:]]
    for vector, hybrid in ((over_abstention, scores), (missed_abstention, scores), (scores, [*scores[:5], score(5, abstain=True), *scores[6:]])):
        with pytest.raises(ValueError, match="^invalid policy investigation report$"):
            write_live_rag_policy_investigation(
                {"vector": vector, "hybrid": hybrid}, tmp_path, provenance(), verified_preflight=True
            )


def test_writer_rejects_same_category_scores_swapped_between_modes(tmp_path: Path) -> None:
    scores = baseline_scores()
    swapped = [scores[1], scores[0], *scores[2:]]

    with pytest.raises(ValueError, match="^invalid policy investigation report$"):
        write_live_rag_policy_investigation(
            {"vector": scores, "hybrid": swapped}, tmp_path, provenance(), verified_preflight=True
        )


def test_scorer_rejects_a_decision_from_a_different_question() -> None:
    case_value = case(6, expects_abstention=True)
    capture_value = capture(6, cited_ids=("allowed-6",))

    with pytest.raises(ValueError, match="^invalid policy counterfactual$"):
        score_policy_counterfactual(
            classify_claim_scope(case(0, expects_abstention=False).question),
            case_value,
            capture_value,
            control(case_value, capture_value),
        )


def test_writer_rejects_unknown_categories_unsafe_provenance_and_non_boolean_preflight(tmp_path: Path) -> None:
    scores = baseline_scores()
    unknown = [replace(scores[0], decision=AbstentionPolicyDecision("abstain", "unknown")), *scores[1:]]  # type: ignore[arg-type]
    unsafe = provenance()
    unsafe["embedding_provider"] = "PROVIDER-MODEL-SENTINEL"
    for report_provenance, preflight, report_scores in ((provenance(), "true", scores), (unsafe, True, scores), (provenance(), True, unknown)):
        with pytest.raises(ValueError) as error:
            write_live_rag_policy_investigation(
                {"vector": report_scores, "hybrid": report_scores}, tmp_path, report_provenance, verified_preflight=preflight
            )
        assert str(error.value) == "invalid policy investigation report"
        assert "PROVIDER-MODEL-SENTINEL" not in str(error.value)


def test_reports_models_and_errors_never_contain_runtime_sentinels(tmp_path: Path) -> None:
    sentinels = (
        "QUESTION-SENTINEL",
        "ANSWER-SENTINEL",
        "EVIDENCE-SENTINEL",
        "CASE-ID-SENTINEL",
        "CHUNK-ID-SENTINEL",
        "FILE-PATH-SENTINEL",
        "URL-SENTINEL",
        "CREDENTIAL-SENTINEL",
        "RAW-ERROR-SENTINEL",
    )
    case_value = replace(case(0, expects_abstention=False), question=sentinels[0], question_id=sentinels[3], allowed_evidence_ids=(sentinels[4],))
    capture_value = LiveAnswerCapture(
        GeneratedAnswer(sentinels[3], sentinels[1], sentinels[2], (sentinels[4], "other"), False),
        (sentinels[4], sentinels[5], sentinels[6], sentinels[7], sentinels[8]),
        "succeeded",
    )
    control_value = control(case_value, capture_value)
    policy_score = score_policy_counterfactual(classify_claim_scope(case_value.question), case_value, capture_value, control_value)
    scores = [policy_score, *baseline_scores()[1:]]
    json_path, markdown_path = write_live_rag_policy_investigation({"vector": scores, "hybrid": scores}, tmp_path, provenance(), verified_preflight=False)
    written = json_path.read_text(encoding="utf-8") + markdown_path.read_text(encoding="utf-8") + repr(policy_score)
    assert all(sentinel not in written for sentinel in sentinels)
    unsafe_provenance = provenance()
    unsafe_provenance["embedding_provider"] = "|".join(sentinels)
    with pytest.raises(ValueError) as error:
        write_live_rag_policy_investigation(
            {"vector": scores, "hybrid": scores}, tmp_path, unsafe_provenance, verified_preflight=False
        )
    assert all(sentinel not in str(error.value) for sentinel in sentinels)
    invalid_capture = replace(capture_value, answer=replace(capture_value.answer, question_id=sentinels[8]))
    with pytest.raises(ValueError) as error:
        score_policy_counterfactual(classify_claim_scope(case_value.question), case_value, invalid_capture, control_value)
    assert all(sentinel not in str(error.value) for sentinel in sentinels)
