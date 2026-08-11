from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from oilfield_chemical_copilot.evaluation.answers import DeterministicAnswerResult
from oilfield_chemical_copilot.evaluation.live_rag_diagnosis import (
    LiveDiagnosticObservation,
    LiveFailureDiagnosis,
    classify_live_failure,
    write_live_failure_diagnosis,
)


def observation(**changes: object) -> LiveDiagnosticObservation:
    values: dict[str, object] = {
        "expect_citations": True,
        "evidence_sufficient": True,
        "expect_abstention": False,
        "allowed_evidence_ids": ("allowed",),
        "retrieved_evidence_ids": ("allowed",),
        "cited_evidence_ids": ("allowed",),
        "abstained": False,
        "generation_outcome": "succeeded",
    }
    values.update(changes)
    return LiveDiagnosticObservation(**values)


def result(*, citation: str = "pass", abstention: str = "pass") -> DeterministicAnswerResult:
    return DeterministicAnswerResult("runtime-case-id", citation, abstention)  # type: ignore[arg-type]


def result_for_observation(runtime: LiveDiagnosticObservation) -> DeterministicAnswerResult:
    cited = set(runtime.cited_evidence_ids)
    allowed = set(runtime.allowed_evidence_ids)
    citation_passes = (
        bool(cited) and cited <= allowed
        if runtime.expect_citations
        else not cited
    )
    return result(
        citation="pass" if citation_passes else "fail",
        abstention="pass" if runtime.abstained == runtime.expect_abstention else "fail",
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
        "retrieval_settings": {"top_k": 5, "min_score": 0.2, "max_context_chars": 4000, "hybrid_candidate_limit": 10, "hybrid_rrf_k": 60, "hybrid_min_rrf_score": 0.015},
        "temperature": 0,
        "topic_filter": "none",
    }


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"retrieved_evidence_ids": (), "cited_evidence_ids": (), "abstained": True, "generation_outcome": "not_called"}, "expected_citation_missing_no_qualifying_retrieval"),
        ({"cited_evidence_ids": (), "abstained": True, "generation_outcome": "failed"}, "expected_citation_missing_generation_failure"),
        ({"cited_evidence_ids": (), "abstained": True}, "expected_citation_missing_abstained_after_qualifying_retrieval"),
        ({"retrieved_evidence_ids": ("other",), "cited_evidence_ids": ("other",)}, "expected_citation_allowed_evidence_not_retrieved"),
        ({"retrieved_evidence_ids": ("allowed", "other"), "cited_evidence_ids": ("other",)}, "expected_citation_allowed_retrieved_not_cited"),
        ({"retrieved_evidence_ids": ("allowed", "other"), "cited_evidence_ids": ("allowed", "other")}, "expected_citation_mixed_with_disallowed"),
        ({"cited_evidence_ids": ()}, "expected_citation_missing_after_answer"),
        ({"expect_citations": False, "evidence_sufficient": False, "expect_abstention": True, "cited_evidence_ids": ("allowed",)}, "unexpected_citation_when_abstention_expected"),
    ],
)
def test_classifies_each_citation_failure(changes: dict[str, object], expected: str) -> None:
    runtime = observation(**changes)
    assert classify_live_failure(runtime, result_for_observation(runtime)).citation_failure == expected


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"retrieved_evidence_ids": (), "cited_evidence_ids": (), "abstained": True, "generation_outcome": "not_called"}, "over_abstention_no_qualifying_retrieval"),
        ({"cited_evidence_ids": (), "abstained": True, "generation_outcome": "failed"}, "over_abstention_generation_failure"),
        ({"cited_evidence_ids": (), "abstained": True}, "over_abstention_after_qualifying_retrieval"),
        ({"expect_citations": False, "evidence_sufficient": False, "expect_abstention": True, "cited_evidence_ids": ("allowed",)}, "under_abstention_answered_on_insufficient_case"),
    ],
)
def test_classifies_each_abstention_failure(changes: dict[str, object], expected: str) -> None:
    runtime = observation(**changes)
    assert classify_live_failure(runtime, result_for_observation(runtime)).abstention_failure == expected


def test_passing_checks_produce_no_categories() -> None:
    assert classify_live_failure(observation(), result()) == LiveFailureDiagnosis(None, None)


@pytest.mark.parametrize("changes", [
    {"cited_evidence_ids": ("not-retrieved",)},
    {"retrieved_evidence_ids": (), "cited_evidence_ids": (), "abstained": False},
    {"generation_outcome": "failed", "abstained": False},
    {"generation_outcome": "not_called"},
    {"expect_citations": "true"},
    {"retrieved_evidence_ids": ["allowed"]},
])
def test_classifier_fails_closed_without_runtime_values(changes: dict[str, object]) -> None:
    sentinel = "QUESTION-SENTINEL-DO-NOT-LEAK"
    with pytest.raises(ValueError) as error:
        classify_live_failure(replace(observation(**changes), allowed_evidence_ids=(sentinel,)), result())
    assert str(error.value) == "invalid diagnostic state"
    assert sentinel not in str(error.value)


def test_classifier_rejects_deterministic_disagreement() -> None:
    with pytest.raises(ValueError, match="^invalid diagnostic state$"):
        classify_live_failure(observation(), result(citation="fail"))


def test_writer_rejects_closed_allowlist_and_per_case_disagreements(tmp_path: Path) -> None:
    cases = (
        (LiveFailureDiagnosis("unknown", None), result(citation="fail")),
        (LiveFailureDiagnosis("over_abstention_generation_failure", None), result(citation="fail")),
        (LiveFailureDiagnosis(None, None), result(citation="fail")),
    )
    for diagnosis, deterministic in cases:
        with pytest.raises(ValueError, match="^invalid diagnosis report$"):
            write_live_failure_diagnosis({"vector": [diagnosis], "hybrid": [diagnosis]}, {"vector": [deterministic], "hybrid": [deterministic]}, tmp_path, provenance(), verified_preflight=False)


def test_writer_rejects_unequal_lengths_and_aggregate_mismatch(tmp_path: Path) -> None:
    diagnosis = LiveFailureDiagnosis("expected_citation_missing_after_answer", None)
    with pytest.raises(ValueError, match="^invalid diagnosis report$"):
        write_live_failure_diagnosis({"vector": [diagnosis], "hybrid": []}, {"vector": [result(citation="fail"), result(citation="fail")], "hybrid": []}, tmp_path, provenance(), verified_preflight=False)
    with pytest.raises(ValueError, match="^invalid diagnosis report$"):
        write_live_failure_diagnosis({"vector": [diagnosis, diagnosis], "hybrid": [diagnosis, diagnosis]}, {"vector": [result(citation="fail"), result()], "hybrid": [result(citation="fail"), result()]}, tmp_path, provenance(), verified_preflight=False)


def baseline_inputs() -> tuple[dict[str, list[LiveFailureDiagnosis]], dict[str, list[DeterministicAnswerResult]]]:
    diagnoses = [LiveFailureDiagnosis("expected_citation_missing_after_answer", None) for _ in range(2)]
    diagnoses += [LiveFailureDiagnosis("expected_citation_missing_after_answer", "over_abstention_after_qualifying_retrieval") for _ in range(6)]
    diagnoses += [LiveFailureDiagnosis(None, None) for _ in range(4)]
    results = [result(citation="fail") for _ in range(2)] + [result(citation="fail", abstention="fail") for _ in range(6)] + [result() for _ in range(4)]
    return ({"vector": diagnoses, "hybrid": diagnoses}, {"vector": results, "hybrid": results})


def test_writer_requires_explicit_verified_preflight_for_baseline(tmp_path: Path) -> None:
    diagnoses, deterministic = baseline_inputs()
    json_path, _ = write_live_failure_diagnosis(diagnoses, deterministic, tmp_path, provenance(), verified_preflight=False)
    assert json.loads(json_path.read_text())["baseline_reproduced"] is False
    json_path, _ = write_live_failure_diagnosis(diagnoses, deterministic, tmp_path, provenance(), verified_preflight=True)
    assert json.loads(json_path.read_text())["baseline_reproduced"] is True


def test_writer_never_serializes_injected_runtime_sentinels(tmp_path: Path) -> None:
    sentinels = ("QUESTION-SENTINEL", "ANSWER-SENTINEL", "EVIDENCE-SENTINEL", "CASE-ID-SENTINEL", "CHUNK-ID-SENTINEL", "FILE-PATH-SENTINEL", "URL-SENTINEL", "CREDENTIAL-SENTINEL", "PROVIDER-MODEL-SENTINEL", "RAW-ERROR-SENTINEL")
    diagnosis = LiveFailureDiagnosis("expected_citation_missing_after_answer", None)
    deterministic = DeterministicAnswerResult(sentinels[3], "fail", "pass")
    json_path, markdown_path = write_live_failure_diagnosis({"vector": [diagnosis], "hybrid": [diagnosis]}, {"vector": [deterministic], "hybrid": [deterministic]}, tmp_path, provenance(), verified_preflight=False)
    written = json_path.read_text() + markdown_path.read_text()
    assert all(sentinel not in written for sentinel in sentinels)
    with pytest.raises(ValueError) as error:
        classify_live_failure(observation(allowed_evidence_ids=(sentinels[2],), retrieved_evidence_ids=("other",), cited_evidence_ids=(sentinels[4],)), result(citation="fail"))
    assert all(sentinel not in str(error.value) for sentinel in sentinels)
@pytest.mark.parametrize(
    ("runtime", "expected_statuses"),
    [
        (observation(), ("pass", "pass")),
        (observation(cited_evidence_ids=()), ("fail", "pass")),
        (observation(retrieved_evidence_ids=(), cited_evidence_ids=(), abstained=True, generation_outcome="not_called"), ("fail", "fail")),
        (observation(expect_citations=False, evidence_sufficient=False, expect_abstention=True), ("fail", "fail")),
        (observation(expect_citations=False, evidence_sufficient=False, expect_abstention=True, retrieved_evidence_ids=(), cited_evidence_ids=(), abstained=True, generation_outcome="not_called"), ("pass", "pass")),
    ],
)
@pytest.mark.parametrize("citation_status", ["pass", "fail"])
@pytest.mark.parametrize("abstention_status", ["pass", "fail"])
def test_classifier_rejects_every_contradictory_deterministic_pairing(
    runtime: LiveDiagnosticObservation,
    expected_statuses: tuple[str, str],
    citation_status: str,
    abstention_status: str,
) -> None:
    deterministic = result(citation=citation_status, abstention=abstention_status)
    if (citation_status, abstention_status) == expected_statuses:
        classify_live_failure(runtime, deterministic)
    else:
        with pytest.raises(ValueError, match="^invalid diagnostic state$"):
            classify_live_failure(runtime, deterministic)


@pytest.mark.parametrize(
    "changes",
    [
        {"evidence_sufficient": False},
        {"expect_citations": False},
        {"evidence_sufficient": False, "expect_abstention": False},
    ],
)
def test_classifier_rejects_contradictory_expectation_booleans(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="^invalid diagnostic state$"):
        classify_live_failure(observation(**changes), result())


def test_sentinels_are_injected_through_runtime_and_writer_error_routes(tmp_path: Path) -> None:
    sentinels = {
        "question": "QUESTION-SENTINEL",
        "answer": "ANSWER-SENTINEL",
        "evidence": "EVIDENCE-SENTINEL",
        "case_id": "CASE-ID-SENTINEL",
        "chunk_id": "CHUNK-ID-SENTINEL",
        "filename_path": "FILE-PATH-SENTINEL",
        "url": "URL-SENTINEL",
        "credential": "CREDENTIAL-SENTINEL",
        "provider_model": "PROVIDER-MODEL-SENTINEL",
        "raw_error": "RAW-ERROR-SENTINEL",
    }
    runtime = observation(
        allowed_evidence_ids=(sentinels["evidence"],),
        retrieved_evidence_ids=(sentinels["chunk_id"],),
        cited_evidence_ids=(sentinels["chunk_id"],),
    )
    deterministic = DeterministicAnswerResult(
        "|".join((sentinels["question"], sentinels["answer"], sentinels["case_id"])),
        "fail",
        "pass",
    )
    diagnosis = classify_live_failure(runtime, result_for_observation(runtime))
    deterministic = DeterministicAnswerResult(deterministic.question_id, "fail", "pass")
    json_path, markdown_path = write_live_failure_diagnosis(
        {"vector": [diagnosis], "hybrid": [diagnosis]},
        {"vector": [deterministic], "hybrid": [deterministic]},
        tmp_path,
        provenance(),
        verified_preflight=False,
    )
    report_text = json_path.read_text() + markdown_path.read_text()
    assert all(sentinel not in report_text for sentinel in sentinels.values())

    for unsafe_key in ("embedding_provider", "generation_model_sha256"):
        unsafe_provenance = provenance()
        unsafe_provenance[unsafe_key] = "|".join(
            (
                sentinels["filename_path"],
                sentinels["url"],
                sentinels["credential"],
                sentinels["provider_model"],
                sentinels["raw_error"],
            )
        )
        with pytest.raises(ValueError) as error:
            write_live_failure_diagnosis(
                {"vector": [diagnosis], "hybrid": [diagnosis]},
                {"vector": [deterministic], "hybrid": [deterministic]},
                tmp_path,
                unsafe_provenance,
                verified_preflight=False,
            )
        assert str(error.value) == "invalid diagnosis report"
        assert all(sentinel not in str(error.value) for sentinel in sentinels.values())