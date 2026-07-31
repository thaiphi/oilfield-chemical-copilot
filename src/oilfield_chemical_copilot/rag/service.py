from __future__ import annotations

from oilfield_chemical_copilot.rag.formatter import format_answer, weak_evidence_answer
from oilfield_chemical_copilot.rag.models import RagAnswer, RagGenerationError
from oilfield_chemical_copilot.rag.prompt_builder import build_prompt
from oilfield_chemical_copilot.retrieval.pipeline import RetrievalSettings


class BasicRagService:
    def __init__(
        self,
        *,
        retriever,
        generator,
        min_score: float = 0.2,
        max_context_chars: int = 4000,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.min_score = min_score
        self.max_context_chars = max_context_chars

    @classmethod
    def from_settings(
        cls, *, retriever, generator, settings: RetrievalSettings
    ) -> "BasicRagService":
        return cls(
            retriever=retriever,
            generator=generator,
            min_score=settings.evidence_threshold,
            max_context_chars=settings.max_context_chars,
        )

    def answer(self, question: str, topic: str | None = None) -> RagAnswer:
        hits = self.retriever.retrieve(question, topic=topic)
        if not hits or max(hit.score for hit in hits) < self.min_score:
            return weak_evidence_answer(
                limitations="No retrieved chunks met the evidence threshold for this question."
            )
        prompt = build_prompt(
            question=question,
            hits=hits,
            max_context_chars=self.max_context_chars,
        )
        if not prompt.sources:
            return weak_evidence_answer(
                limitations="Retrieved chunks did not fit the context budget."
            )
        try:
            draft = self.generator.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                allowed_source_ids={source.source_id for source in prompt.sources},
            )
            return format_answer(draft, prompt.sources)
        except RagGenerationError:
            return weak_evidence_answer(
                limitations="Answer generation failed safely after retrieval. Check configuration and retry."
            )
