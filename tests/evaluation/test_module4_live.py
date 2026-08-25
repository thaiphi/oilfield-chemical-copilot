from __future__ import annotations

from dataclasses import dataclass

import pytest

from oilfield_chemical_copilot.evaluation.live_rag import (
    RecordingAnswerGenerator,
    RecordingRetriever,
)
from oilfield_chemical_copilot.evaluation.module4_contract import Module4Case
from oilfield_chemical_copilot.evaluation.module4_live import (
    ModeRuntime,
    Module4RuntimeError,
    evaluate_module4_modes,
)
from oilfield_chemical_copilot.evaluation.module4_reports import ModeSummary
from oilfield_chemical_copilot.rag.models import RagAnswer, RagDraft, RagGenerationError, SourceEvidence


def _source(chunk_id: str, source_id: str) -> SourceEvidence:
    return SourceEvidence(source_id, chunk_id, "public.md", "document", "scale", "evidence", 1.0)


class _Delegate:
    def generate(self, **_kwargs: object) -> RagDraft:
        return RagDraft("answer", "why", ["Source 1"], ["one", "two", "three"], "limits")


class _RetrieverDelegate:
    def retrieve(self, _question: str, topic: str | None = None) -> list[SourceEvidence]:
        assert topic == "scale"
        return [_source("wrong", "Wrong"), _source("expected", "Source 1")]


class _Service:
    def __init__(self, retriever: RecordingRetriever, generator: RecordingAnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> RagAnswer:
        sources = self.retriever.retrieve(question, topic="scale")
        self.generator.generate()
        return RagAnswer(
            "formatted",
            [source for source in sources if source.chunk_id == "expected"],
        )


def _runtime(_mode: str) -> ModeRuntime:
    retriever = RecordingRetriever(_RetrieverDelegate())
    generator = RecordingAnswerGenerator(_Delegate())
    return ModeRuntime(_Service(retriever, generator), generator)


def test_evaluate_module4_modes_uses_real_rank_and_deterministic_checks() -> None:
    cases = (Module4Case("case-01", "question", "scale", ("expected",), True, False, True),)

    results = evaluate_module4_modes(cases, build_service=_runtime)

    assert results == {
        "vector": ModeSummary(1, 1.0, 0.5, 1, 0, 1, 0),
        "hybrid": ModeSummary(1, 1.0, 0.5, 1, 0, 1, 0),
    }
    assert "question" not in repr(results)
    assert "evidence" not in repr(results)


def test_abstention_case_is_excluded_from_retrieval_metric_denominator() -> None:
    cases = (
        Module4Case("case-01", "question", "scale", ("expected",), True, False, True),
        Module4Case("case-02", "closed question", "scale", (), False, True, True),
    )

    def runtime(mode: str) -> ModeRuntime:
        if mode == "vector":
            return _runtime(mode)
        retriever = RecordingRetriever(_RetrieverDelegate())
        generator = RecordingAnswerGenerator(_Delegate())

        @dataclass
        class AbstainingService:
            retriever: RecordingRetriever

            def answer(self, _question: str) -> RagAnswer:
                return RagAnswer("fallback", [], weak_evidence=True)

        return ModeRuntime(AbstainingService(retriever), generator)

    results = evaluate_module4_modes(cases, build_service=runtime)

    assert results["vector"].retrieval_case_count == 1
    assert results["hybrid"].retrieval_case_count == 1


def test_generation_failure_becomes_sanitized_runtime_error() -> None:
    cases = (Module4Case("case-01", "question", "scale", ("expected",), True, False, True),)

    class FailingDelegate:
        def generate(self, **_kwargs: object) -> RagDraft:
            raise RagGenerationError("private generation failure")

    class SafeFailureService:
        def __init__(self, retriever: RecordingRetriever, generator: RecordingAnswerGenerator) -> None:
            self.retriever = retriever
            self.generator = generator

        def answer(self, question: str) -> RagAnswer:
            self.retriever.retrieve(question, topic="scale")
            try:
                self.generator.generate()
            except RagGenerationError:
                return RagAnswer("fallback", [], weak_evidence=True)
            raise AssertionError("expected failure")

    def failing_runtime(_mode: str) -> ModeRuntime:
        retriever = RecordingRetriever(_RetrieverDelegate())
        generator = RecordingAnswerGenerator(FailingDelegate())
        return ModeRuntime(SafeFailureService(retriever, generator), generator)

    with pytest.raises(Module4RuntimeError, match="^RUNTIME_UNAVAILABLE$") as error:
        evaluate_module4_modes(cases, build_service=failing_runtime)

    assert "private generation failure" not in str(error.value)
