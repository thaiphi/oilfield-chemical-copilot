from __future__ import annotations

from oilfield_chemical_copilot.rag.models import FALLBACK_MESSAGE, RagDraft, RagGenerationError
from oilfield_chemical_copilot.rag.service import BasicRagService
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.pipeline import RetrievalSettings


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits

    def retrieve(self, question: str, topic: str | None = None):
        return self.hits


class FakeGenerator:
    def __init__(self, draft: RagDraft | None = None, error: Exception | None = None) -> None:
        self.draft = draft
        self.error = error
        self.calls = 0

    def generate(
        self, *, system_prompt: str, user_prompt: str, allowed_source_ids: set[str]
    ) -> RagDraft:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.draft is not None
        return self.draft


def _hit(score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk_id="scale-1",
        text="Scale evidence from water analysis.",
        score=score,
        retrieval_method="vector",
        source_file="docs/scale.md",
        source_path="C:/private/docs/scale.md",
        topic="scale",
        parser_type="text",
        page_or_sheet="document",
        chunk_index=0,
        metadata={},
    )


def _draft() -> RagDraft:
    return RagDraft(
        answer="Scale risk should be reviewed from the water analysis.",
        why_this_matters="Scale can restrict flow.",
        cited_source_ids=["Source 1"],
        recommended_next_checks=["Check calcium", "Check sulfate", "Check temperature"],
        limitations="Sample evidence only.",
    )


def test_service_returns_fallback_without_openai_call_when_evidence_is_weak() -> None:
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService(retriever=FakeRetriever([]), generator=generator, min_score=0.2)

    answer = service.answer("What should I do?")

    assert FALLBACK_MESSAGE in answer.text
    assert answer.sources == []
    assert generator.calls == 0


def test_service_formats_grounded_openai_answer_with_citations() -> None:
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService(retriever=FakeRetriever([_hit()]), generator=generator, min_score=0.2)

    answer = service.answer("How do I assess scale risk?")

    assert "Answer:\nScale risk" in answer.text
    assert "Source 1: docs/scale.md" in answer.text
    assert "C:/private" not in answer.text
    assert generator.calls == 1


def test_service_uses_safe_fallback_when_generation_fails() -> None:
    service = BasicRagService(
        retriever=FakeRetriever([_hit()]),
        generator=FakeGenerator(error=RagGenerationError("raw provider details")),
        min_score=0.2,
    )

    answer = service.answer("How do I assess scale risk?")

    assert FALLBACK_MESSAGE in answer.text
    assert "raw provider details" not in answer.text


def test_service_accepts_qualified_hybrid_rrf_score() -> None:
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService(
        retriever=FakeRetriever([_hit(score=2 / 61)]), generator=generator, min_score=0.015
    )

    answer = service.answer("How should I assess scale risk?")

    assert answer.weak_evidence is False
    assert generator.calls == 1


def test_service_from_settings_uses_hybrid_rrf_evidence_threshold() -> None:
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService.from_settings(
        retriever=FakeRetriever([_hit(score=2 / 61)]),
        generator=generator,
        settings=RetrievalSettings(
            retrieval_mode="hybrid", min_score=0.2, hybrid_min_rrf_score=0.015
        ),
    )

    answer = service.answer("How should I assess scale risk?")

    assert answer.weak_evidence is False
    assert generator.calls == 1
