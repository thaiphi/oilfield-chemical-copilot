from __future__ import annotations

import math
from pathlib import Path

from ingestion.ingest import generate_chunks
from oilfield_chemical_copilot.ingest.models import LoadedChunk
from oilfield_chemical_copilot.rag.models import RagDraft
from oilfield_chemical_copilot.rag.service import BasicRagService
from oilfield_chemical_copilot.retrieval.embeddings import DeterministicEmbeddingProvider
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.pipeline import BasicRetrievalPipeline, RetrievalSettings


class InMemoryVectorStore:
    def __init__(
        self,
        *,
        chunks: list[LoadedChunk],
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_model = embedding_model

    def search(
        self,
        query_embedding,
        *,
        limit: int,
        topic: str | None,
        embedding_model: str,
    ) -> list[RetrievalHit]:
        assert embedding_model == self.embedding_model
        scored: list[tuple[float, LoadedChunk]] = []
        for chunk, embedding in zip(self.chunks, self.embeddings, strict=True):
            if topic is not None and chunk.metadata.topic != topic:
                continue
            scored.append((_cosine(query_embedding, embedding), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(
                chunk_id=chunk.metadata.chunk_id,
                text=chunk.text,
                score=score,
                retrieval_method="vector",
                source_file=chunk.metadata.source_file,
                source_path=chunk.metadata.source_path,
                topic=chunk.metadata.topic,
                parser_type=chunk.metadata.parser_type,
                page_or_sheet=chunk.metadata.page_or_sheet,
                chunk_index=chunk.metadata.chunk_index,
                metadata=chunk.metadata.extra,
            )
            for score, chunk in scored[:limit]
        ]


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls = 0
        self.last_user_prompt = ""
        self.last_allowed_source_ids: set[str] = set()

    def generate(self, *, system_prompt: str, user_prompt: str, allowed_source_ids: set[str]):
        self.calls += 1
        self.last_user_prompt = user_prompt
        self.last_allowed_source_ids = allowed_source_ids
        return RagDraft(
            answer="Review scale tendency from the produced-water analysis before changing treatment.",
            why_this_matters="Scale risk can restrict flow and distort chemical dosage decisions.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=[
                "Check calcium and bicarbonate trends.",
                "Confirm temperature and pressure at the problem location.",
                "Compare recent treatment rate changes against symptoms.",
            ],
            limitations="Based only on the public sample corpus.",
        )


def test_public_sample_question_reaches_generator_and_returns_citation(tmp_path: Path) -> None:
    chunks = generate_chunks("data/sample", output_dir=tmp_path, max_files=20)
    provider = DeterministicEmbeddingProvider(dimension=384)
    store = InMemoryVectorStore(
        chunks=chunks,
        embeddings=provider.embed_documents([chunk.text for chunk in chunks]),
        embedding_model=provider.model_name,
    )
    retriever = BasicRetrievalPipeline(
        store=store,
        embedding_provider=provider,
        settings=RetrievalSettings(top_k=5, min_score=0.2, max_context_chars=4000),
    )
    generator = RecordingGenerator()
    service = BasicRagService.from_settings(
        retriever=retriever,
        generator=generator,
        settings=RetrievalSettings(top_k=5, min_score=0.2, max_context_chars=4000),
    )

    answer = service.answer("How should I assess scale risk from produced water analysis?")

    assert generator.calls == 1
    assert answer.sources
    assert answer.weak_evidence is False
    assert "Source 1:" in answer.text
    assert "source_path" not in generator.last_user_prompt
    assert "C:/" not in answer.text
    assert "\\" not in answer.sources[0].source_file
    assert "Source 1" in generator.last_allowed_source_ids


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator

def test_public_sample_retrieval_clears_threshold_for_representative_questions(tmp_path: Path) -> None:
    chunks = generate_chunks("data/sample", output_dir=tmp_path, max_files=20)
    provider = DeterministicEmbeddingProvider(dimension=384)
    store = InMemoryVectorStore(
        chunks=chunks,
        embeddings=provider.embed_documents([chunk.text for chunk in chunks]),
        embedding_model=provider.model_name,
    )
    retriever = BasicRetrievalPipeline(
        store=store,
        embedding_provider=provider,
        settings=RetrievalSettings(top_k=5, min_score=0.2, max_context_chars=4000),
    )

    questions = [
        "How should I assess scale risk from produced water analysis?",
        "What causes corrosion in produced water?",
        "How do I troubleshoot paraffin deposition?",
        "How should water analysis guide chemical treatment?",
    ]

    for question in questions:
        hits = retriever.retrieve(question)
        assert hits, question
        assert hits[0].score >= 0.2