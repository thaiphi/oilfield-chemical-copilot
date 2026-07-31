from __future__ import annotations

from oilfield_chemical_copilot.retrieval.embeddings import DeterministicEmbeddingProvider
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.pipeline import BasicRetrievalPipeline, RetrievalSettings


class FakeStore:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls = []

    def search(self, query_embedding, *, limit: int, topic: str | None, embedding_model: str):
        self.calls.append((query_embedding, limit, topic, embedding_model))
        return self.hits[:limit]


def _hit(chunk_id: str, score: float, text: str = "Scale inhibitor evidence") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text=text,
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


def test_pipeline_uses_vector_store_with_matching_embedding_model() -> None:
    store = FakeStore([_hit("scale-1", 0.92)])
    provider = DeterministicEmbeddingProvider(dimension=8)
    pipeline = BasicRetrievalPipeline(
        store=store,
        embedding_provider=provider,
        settings=RetrievalSettings(top_k=3, min_score=0.2, max_context_chars=1000),
    )

    hits = pipeline.retrieve("scale tendency", topic="scale")

    assert [hit.chunk_id for hit in hits] == ["scale-1"]
    assert len(store.calls[0][0]) == 8
    assert store.calls[0][1:] == (3, "scale", "deterministic-token-hash-8")


def test_pipeline_filters_weak_hits_and_rejects_blank_questions() -> None:
    pipeline = BasicRetrievalPipeline(
        store=FakeStore([_hit("weak", 0.05), _hit("strong", 0.7)]),
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        settings=RetrievalSettings(top_k=5, min_score=0.2, max_context_chars=1000),
    )

    assert [hit.chunk_id for hit in pipeline.retrieve("iron sulfide")] == ["strong"]
    assert pipeline.retrieve("   ") == []


def test_pipeline_bounds_total_context_characters() -> None:
    pipeline = BasicRetrievalPipeline(
        store=FakeStore([
            _hit("one", 0.9, text="a" * 40),
            _hit("two", 0.8, text="b" * 40),
            _hit("three", 0.7, text="c" * 40),
        ]),
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        settings=RetrievalSettings(top_k=5, min_score=0.2, max_context_chars=70),
    )

    hits = pipeline.retrieve("bounded context")

    assert [hit.chunk_id for hit in hits] == ["one"]