from __future__ import annotations

import pytest

from oilfield_chemical_copilot.rag.models import FALLBACK_MESSAGE, RagDraft, RagGenerationError
from oilfield_chemical_copilot.rag.service import BasicRagService
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.pipeline import RetrievalSettings


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls = 0
        self.queries: list[tuple[str, str | None]] = []

    def retrieve(self, question: str, topic: str | None = None):
        self.calls += 1
        self.queries.append((question, topic))
        return self.hits


class FakeGenerator:
    def __init__(self, draft: RagDraft | None = None, error: Exception | None = None) -> None:
        self.draft = draft
        self.error = error
        self.calls = 0
        self.user_prompts: list[str] = []

    def generate(
        self, *, system_prompt: str, user_prompt: str, allowed_source_ids: set[str]
    ) -> RagDraft:
        self.calls += 1
        self.user_prompts.append(user_prompt)
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


@pytest.mark.parametrize(
    ("question", "expected_limitation"),
    [
        (
            "Can this determine the root cause at a named asset?",
            "site-specific determination",
        ),
        (
            "Can you prescribe a field-ready dosage?",
            "field-ready prescription",
        ),
        (
            "What exact chemical dose should I inject for Well A tomorrow?",
            "field-ready prescription",
        ),
        (
            "Can this replace a complete field analysis?",
            "replace a complete analysis",
        ),
    ],
)
def test_service_abstains_before_retrieval_or_generation_for_closed_claim_scopes(
    question: str, expected_limitation: str
) -> None:
    retriever = FakeRetriever([_hit()])
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService(retriever=retriever, generator=generator)

    answer = service.answer(question)

    assert expected_limitation in answer.text.lower()
    assert answer.sources == []
    assert answer.weak_evidence is True
    assert retriever.calls == 0
    assert generator.calls == 0


def test_service_evaluation_only_bypass_preserves_ungated_rag_baseline() -> None:
    retriever = FakeRetriever([_hit()])
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService(
        retriever=retriever,
        generator=generator,
        apply_claim_scope_policy=False,
    )

    service.answer("Can you prescribe a field-ready dosage?")

    assert retriever.calls == 1
    assert generator.calls == 1


def test_service_allows_general_review_through_unchanged_rag_path() -> None:
    retriever = FakeRetriever([_hit()])
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService(retriever=retriever, generator=generator)

    answer = service.answer("How should I assess scale risk from produced water analysis?")

    assert answer.weak_evidence is False
    assert retriever.calls == 1
    assert generator.calls == 1


def test_service_uses_retrieval_query_without_replacing_the_original_question() -> None:
    retriever = FakeRetriever([_hit()])
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService(retriever=retriever, generator=generator)

    service.answer(
        "How should I assess scale risk from produced water analysis?",
        retrieval_query="produced-water scale screening",
    )

    assert retriever.queries == [("produced-water scale screening", None)]
    assert "How should I assess scale risk from produced water analysis?" in generator.user_prompts[0]
    assert "produced-water scale screening" not in generator.user_prompts[0]


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
