import json
import hashlib
from pathlib import Path

import pytest

from eval.answer_eval import GeneratedAnswer, evaluate_cases, write_report
from oilfield_chemical_copilot.evaluation.answers import AnswerEvaluationCase
from oilfield_chemical_copilot.evaluation.judge import OllamaJudgeProvider
from oilfield_chemical_copilot.evaluation.judge import AnswerJudge, JudgeScores, JudgeUnavailable


class FakeProvider:
    provider_name = "fake"
    model_name = "safe-test-model"

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _case() -> AnswerEvaluationCase:
    return AnswerEvaluationCase(
        "case-01", "What public evidence supports the recommended scale check?",
        ("evidence-01",), True, True, False,
    )


def _answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        question_id="case-01",
        answer="ANSWER-SENTINEL",
        evidence="EVIDENCE-SENTINEL",
        cited_evidence_ids=("evidence-01",),
        abstained=False,
    )


def test_judge_returns_strictly_validated_scores_from_fake_provider() -> None:
    provider = FakeProvider(
        '{"groundedness":5,"relevance":4,"limitation_awareness":3,"operational_certainty":2}'
    )

    result = AnswerJudge(provider=provider).judge(_case(), answer="ANSWER-SENTINEL", evidence="EVIDENCE-SENTINEL")

    assert result.question_id == "case-01"
    assert result.provider == "fake"
    assert result.model == f"sha256:{hashlib.sha256(b'safe-test-model').hexdigest()}"
    assert result.scores == JudgeScores(5, 4, 3, 2)
    assert provider.calls
    system_prompt, user_prompt = provider.calls[0]
    assert "groundedness" in system_prompt
    assert "1=" in system_prompt
    assert "5=" in system_prompt
    assert "What public evidence supports the recommended scale check?" in user_prompt


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"groundedness":5,"relevance":4,"limitation_awareness":3}',
        '{"groundedness":5,"relevance":4,"limitation_awareness":3,"operational_certainty":2,"extra":1}',
        '{"groundedness":5,"relevance":4,"limitation_awareness":3,"operational_certainty":0}',
        '{"groundedness":true,"relevance":4,"limitation_awareness":3,"operational_certainty":2}',
        '{"groundedness":5.0,"relevance":4,"limitation_awareness":3,"operational_certainty":2}',
        '{"groundedness":5,"groundedness":4,"relevance":4,"limitation_awareness":3,"operational_certainty":2}',
    ],
)
def test_judge_marks_malformed_or_out_of_range_rubric_unavailable(payload: str) -> None:
    result = AnswerJudge(provider=FakeProvider(payload)).judge(
        _case(), answer="ANSWER-SENTINEL", evidence="EVIDENCE-SENTINEL"
    )

    assert isinstance(result, JudgeUnavailable)
    assert result.question_id == "case-01"
    assert result.status == "unavailable"


def test_judge_marks_transport_failures_unavailable_without_error_detail() -> None:
    result = AnswerJudge(provider=FakeProvider(RuntimeError("PRIVATE-ERROR-SENTINEL"))).judge(
        _case(), answer="ANSWER-SENTINEL", evidence="EVIDENCE-SENTINEL"
    )

    assert result == JudgeUnavailable(
        "case-01", "fake", f"sha256:{hashlib.sha256(b'safe-test-model').hexdigest()}"
    )
    assert "PRIVATE-ERROR-SENTINEL" not in repr(result)


def test_aggregate_reports_exclude_runtime_text_and_paths(tmp_path: Path) -> None:
    results = evaluate_cases(
        [_case()],
        [_answer()],
        AnswerJudge(
            provider=FakeProvider(
                '{"groundedness":5,"relevance":4,"limitation_awareness":3,"operational_certainty":2}'
            )
        ),
    )

    json_path, markdown_path = write_report(results, tmp_path)

    serialized = json_path.read_text(encoding="utf-8") + markdown_path.read_text(encoding="utf-8")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report == {
        "cases": ["case-01"],
        "deterministic": {"abstention": {"pass": 1}, "citations": {"pass": 1}},
        "judge": {
            "models": [f"sha256:{hashlib.sha256(b'safe-test-model').hexdigest()}"],
            "providers": ["fake"],
            "scores": {
                "groundedness": 5.0,
                "limitation_awareness": 3.0,
                "operational_certainty": 2.0,
                "relevance": 4.0,
            },
            "status": {"available": 1},
        },
    }
    for sentinel in (
        "ANSWER-SENTINEL",
        "EVIDENCE-SENTINEL",
        "evidence-01",
        "C:/private",
        "safe-test-model",
    ):
        assert sentinel not in serialized


def test_configuration_failure_becomes_safe_unavailable_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWER_EVAL_JUDGE_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANSWER_EVAL_OPENAI_MODEL", "configured-model-sentinel")

    result = AnswerJudge().judge(_case(), answer="ANSWER-SENTINEL", evidence="EVIDENCE-SENTINEL")

    assert result == JudgeUnavailable(
        "case-01", "openai", f"sha256:{hashlib.sha256(b'configured-model-sentinel').hexdigest()}"
    )


def test_report_lists_safe_identities_for_unavailable_results(tmp_path: Path) -> None:
    result = AnswerJudge(provider=FakeProvider(RuntimeError("down"))).judge(
        _case(), answer="ANSWER-SENTINEL", evidence="EVIDENCE-SENTINEL"
    )

    json_path, _ = write_report(
        evaluate_cases([_case()], [_answer()], AnswerJudge(provider=FakeProvider(RuntimeError("down")))),
        tmp_path,
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(result, JudgeUnavailable)
    assert report["judge"]["providers"] == ["fake"]

    assert report["judge"]["models"] == [result.model]

def test_ollama_judge_requests_zero_temperature() -> None:
    calls: dict[str, object] = {}

    class RecordingClient:
        def chat(self, **kwargs: object) -> str:
            calls.update(kwargs)
            return "{}"

    provider = OllamaJudgeProvider(model="safe-test-model", client=RecordingClient())

    provider.judge(system_prompt="system", user_prompt="user")
    assert calls["generation_options"] == {"temperature": 0}


def test_judge_bounds_unsafe_provider_identity() -> None:
    provider = FakeProvider('{"groundedness":5,"relevance":4,"limitation_awareness":3,"operational_certainty":2}')
    provider.provider_name = "PRIVATE PROVIDER SENTINEL"

    result = AnswerJudge(provider=provider).judge(
        _case(), answer="ANSWER-SENTINEL", evidence="EVIDENCE-SENTINEL"
    )

    assert result.provider == "unknown"
