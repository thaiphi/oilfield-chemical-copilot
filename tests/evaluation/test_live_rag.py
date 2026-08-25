import json
from dataclasses import fields

import pytest

from oilfield_chemical_copilot.evaluation.answers import AnswerEvaluationCase, AnswerEvaluationResult
from oilfield_chemical_copilot.evaluation.live_rag import (
    RecordingAnswerGenerator,
    RecordingRetriever,
    build_live_ollama_generator,
    capture_live_answer,
)
from oilfield_chemical_copilot.rag.ollama_client import LazyOllamaAnswerClient
from oilfield_chemical_copilot.rag.models import RagAnswer, RagDraft, RagGenerationError, SourceEvidence


class Delegate:
    def __init__(self, draft: RagDraft) -> None:
        self.draft = draft

    def generate(self, **_kwargs: object) -> RagDraft:
        return self.draft


class Service:
    def __init__(self, answer: RagAnswer, recorder: RecordingAnswerGenerator | None = None) -> None:
        self.answer_value = answer
        self.recorder = recorder
        self.questions: list[str] = []

    def answer(self, question: str) -> RagAnswer:
        self.questions.append(question)
        if self.recorder is not None:
            self.recorder.generate()
        return self.answer_value


def _case() -> AnswerEvaluationCase:
    return AnswerEvaluationCase("case", "Public question", ("chunk",), True, True, False)


def _draft() -> RagDraft:
    return RagDraft("DRAFT-ANSWER", "WHY", ["Source 1"], ["one", "two", "three"], "LIMIT")


def _source() -> SourceEvidence:
    return SourceEvidence("Source 1", "chunk", "public.md", "document", "scale", "EVIDENCE", 1.0)


def test_capture_uses_recorded_draft_and_records_success_outcome() -> None:
    recorder = RecordingAnswerGenerator(Delegate(_draft()))
    service = Service(RagAnswer("FORMATTED-ANSWER", [_source()]), recorder)

    captured = capture_live_answer(_case(), service, recorder)

    assert service.questions == ["Public question"]
    assert "DRAFT-ANSWER" in captured.answer.answer
    assert "FORMATTED-ANSWER" not in captured.answer.answer
    assert captured.answer.cited_evidence_ids == ("chunk",)
    assert captured.answer.evidence == "EVIDENCE"
    assert captured.answer.abstained is False
    assert captured.generation_outcome == "succeeded"


def test_capture_without_generation_records_not_called_outcome() -> None:
    recorder = RecordingAnswerGenerator(Delegate(_draft()))
    recorder.last_draft = _draft()

    captured = capture_live_answer(_case(), Service(RagAnswer("FALLBACK", [], weak_evidence=True)), recorder)

    assert recorder.last_draft is None
    assert captured.answer.cited_evidence_ids == ()
    assert captured.answer.evidence == ""
    assert captured.answer.abstained is True
    assert captured.generation_outcome == "not_called"


def test_recording_retriever_keeps_only_distinct_current_call_ids() -> None:
    class DelegateRetriever:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def retrieve(self, question: str, topic: str | None = None) -> list[SourceEvidence]:
            self.calls.append((question, topic))
            if question == "q1":
                return [_source()]
            return [SourceEvidence("Second", "second", "private.md", "path", "topic", "TEXT", 1.0)]

    delegate = DelegateRetriever()
    recorder = RecordingRetriever(delegate)

    recorder.retrieve("q1")
    assert recorder.retrieved_evidence_ids == ("chunk",)
    recorder.retrieve("q2", topic="topic")

    assert recorder.retrieved_evidence_ids == ("second",)
    assert delegate.calls == [("q1", None), ("q2", "topic")]
    assert set(vars(recorder)) == {"delegate", "retrieved_evidence_ids"}
    assert "TEXT" not in repr(recorder.retrieved_evidence_ids)
    assert "private.md" not in repr(recorder.retrieved_evidence_ids)
    assert "path" not in repr(recorder.retrieved_evidence_ids)


def test_generator_rag_generation_error_records_failed_without_retaining_exception() -> None:
    class FailingDelegate:
        def generate(self, **_kwargs: object) -> RagDraft:
            raise RagGenerationError("private failure")

    recorder = RecordingAnswerGenerator(FailingDelegate())
    recorder.reset()

    with pytest.raises(RagGenerationError, match="private failure"):
        recorder.generate()

    assert recorder.generation_outcome == "failed"
    assert recorder.last_draft is None
    assert set(vars(recorder)) == {"delegate", "last_draft", "generation_outcome"}
    assert "private failure" not in repr(vars(recorder))


def test_report_safe_result_has_no_runtime_answer_fields() -> None:
    assert {field.name for field in fields(AnswerEvaluationResult)} == {"question_id", "deterministic", "judge"}


def test_capture_includes_every_structured_draft_field() -> None:
    recorder = RecordingAnswerGenerator(Delegate(_draft()))
    captured = capture_live_answer(_case(), Service(RagAnswer("FORMATTED", [_source()]), recorder), recorder)

    for value in ("DRAFT-ANSWER", "WHY", "one", "two", "three", "LIMIT"):
        assert value in captured.answer.answer


def test_capture_after_rag_generation_failure_abstains_without_stale_data() -> None:
    class FailingDelegate:
        def generate(self, **_kwargs: object) -> RagDraft:
            raise RagGenerationError("private failure")

    class SafeFailureService(Service):
        def answer(self, question: str) -> RagAnswer:
            self.questions.append(question)
            try:
                self.recorder.generate()
            except RagGenerationError:
                return self.answer_value
            raise AssertionError("expected generation failure")

    recorder = RecordingAnswerGenerator(FailingDelegate())
    recorder.last_draft = _draft()
    service = SafeFailureService(RagAnswer("FALLBACK", [], weak_evidence=True), recorder)

    captured = capture_live_answer(_case(), service, recorder)

    assert recorder.last_draft is None
    assert captured.answer.cited_evidence_ids == ()
    assert captured.answer.evidence == ""
    assert captured.answer.abstained is True
    assert captured.generation_outcome == "failed"
    assert "private failure" not in repr(vars(recorder))


def test_live_generator_forwards_exact_zero_temperature_to_real_ollama_adapter() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat(self, **kwargs: object) -> str:
            self.calls.append(kwargs)
            return json.dumps(
                {
                    "answer": "answer",
                    "why_this_matters": "why",
                    "cited_source_ids": ["chunk"],
                    "recommended_next_checks": ["one", "two", "three"],
                    "limitations": "limits",
                }
            )

    generator = build_live_ollama_generator()
    assert isinstance(generator.delegate, LazyOllamaAnswerClient)
    client = RecordingClient()
    generator.delegate.client = client

    generator.generate(
        system_prompt="system",
        user_prompt="user",
        allowed_source_ids={"chunk"},
    )

    assert client.calls[0]["generation_options"] == {"temperature": 0}