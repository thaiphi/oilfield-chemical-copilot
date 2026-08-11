"""Runtime-only capture for live public RAG answer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from oilfield_chemical_copilot.evaluation.answers import AnswerEvaluationCase, GeneratedAnswer
from oilfield_chemical_copilot.rag.models import RagAnswer, RagDraft, RagGenerationError
from oilfield_chemical_copilot.rag.ollama_client import LazyOllamaAnswerClient

GenerationOutcome = Literal["not_called", "succeeded", "failed"]


@dataclass(frozen=True)
class LiveAnswerCapture:
    answer: GeneratedAnswer
    retrieved_evidence_ids: tuple[str, ...]
    generation_outcome: GenerationOutcome


class RecordingRetriever:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.retrieved_evidence_ids: tuple[str, ...] = ()

    def retrieve(self, question: str, topic: str | None = None):
        self.retrieved_evidence_ids = ()
        hits = self.delegate.retrieve(question, topic=topic)
        self.retrieved_evidence_ids = tuple(hit.chunk_id for hit in hits)
        return hits


class RecordingAnswerGenerator:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.last_draft: RagDraft | None = None
        self.generation_outcome: GenerationOutcome = "not_called"

    def reset(self) -> None:
        self.last_draft = None
        self.generation_outcome = "not_called"

    def generate(self, **kwargs: object) -> RagDraft:
        try:
            draft = self.delegate.generate(**kwargs)
        except RagGenerationError:
            self.generation_outcome = "failed"
            raise
        self.last_draft = draft
        self.generation_outcome = "succeeded"
        return draft


def build_live_ollama_generator() -> RecordingAnswerGenerator:
    return RecordingAnswerGenerator(LazyOllamaAnswerClient(generation_options={"temperature": 0}))


def capture_live_answer(
    case: AnswerEvaluationCase, service, recording_generator: RecordingAnswerGenerator
) -> LiveAnswerCapture:
    recording_generator.reset()
    rag_answer: RagAnswer = service.answer(case.question)
    draft = recording_generator.last_draft
    if rag_answer.weak_evidence or draft is None:
        answer = GeneratedAnswer(case.question_id, rag_answer.text, "", (), True)
    else:
        runtime_answer = "\n".join(
            [draft.answer, draft.why_this_matters, *draft.recommended_next_checks, draft.limitations]
        )
        answer = GeneratedAnswer(
            question_id=case.question_id,
            answer=runtime_answer,
            evidence="\n".join(source.excerpt for source in rag_answer.sources),
            cited_evidence_ids=tuple(source.chunk_id for source in rag_answer.sources),
            abstained=False,
        )
    retrieved_ids = getattr(getattr(service, "retriever", None), "retrieved_evidence_ids", ())
    return LiveAnswerCapture(answer, tuple(retrieved_ids), recording_generator.generation_outcome)