from __future__ import annotations

import pytest

from oilfield_chemical_copilot.retrieval.embeddings import DeterministicEmbeddingProvider
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.pipeline import (
    BasicRetrievalPipeline,
    HybridRetrievalPipeline,
    RetrievalSettings,
    _fit_context_budget,
    build_retrieval_pipeline,
)


class FakeStore:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls = []

    def search(self, query_embedding, *, limit: int, topic: str | None, embedding_model: str):
        self.calls.append((query_embedding, limit, topic, embedding_model))
        return self.hits[:limit]


class FakeKeywordIndex:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls = []

    def search(self, query: str, *, limit: int, topic: str | None) -> list[RetrievalHit]:
        self.calls.append((query, limit, topic))
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
    pipeline = BasicRetrievalPipeline(
        store=store,
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
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
        store=FakeStore(
            [
                _hit("one", 0.9, text="a" * 40),
                _hit("two", 0.8, text="b" * 40),
                _hit("three", 0.7, text="c" * 40),
            ]
        ),
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        settings=RetrievalSettings(top_k=5, min_score=0.2, max_context_chars=70),
    )

    assert [hit.chunk_id for hit in pipeline.retrieve("bounded context")] == ["one"]


def test_context_budget_selects_the_highest_scoring_combination_that_fits() -> None:
    hits = [
        _hit("one", 0.031, text="a" * 1199),
        _hit("two", 0.016, text="b" * 1125),
        _hit("three", 0.016, text="c" * 1192),
        _hit("four", 0.016, text="d" * 712),
        _hit("five", 0.016, text="e" * 738),
    ]

    selected = _fit_context_budget(hits, max_context_chars=4000)

    assert [hit.chunk_id for hit in selected] == ["one", "two", "four", "five"]


def test_hybrid_pipeline_fuses_keyword_and_vector_candidates() -> None:
    settings = RetrievalSettings(
        retrieval_mode="hybrid",
        top_k=3,
        min_score=0.9,
        max_context_chars=1000,
        hybrid_candidate_limit=3,
        hybrid_rrf_k=60,
        hybrid_min_rrf_score=0.015,
    )
    store = FakeStore([_hit("shared", 0.2), _hit("semantic", 0.1)])
    keyword_index = FakeKeywordIndex([_hit("shared", 1.0), _hit("exact", 0.5)])
    pipeline = HybridRetrievalPipeline(
        store=store,
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        keyword_index=keyword_index,
        settings=settings,
    )

    hits = pipeline.retrieve("SCALE-X compatibility", topic="scale")

    assert [hit.chunk_id for hit in hits] == ["shared", "exact", "semantic"]
    assert all(hit.retrieval_method == "hybrid" for hit in hits)
    assert store.calls[0][1] == 3
    assert keyword_index.calls == [("SCALE-X compatibility", 3, "scale")]


def test_hybrid_pipeline_uses_rrf_threshold_and_context_budget_only() -> None:
    pipeline = HybridRetrievalPipeline(
        store=FakeStore([_hit("weak-vector", 0.01, text="a" * 40)]),
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        keyword_index=FakeKeywordIndex([_hit("keyword-only", 1.0, text="b" * 40)]),
        settings=RetrievalSettings(
            retrieval_mode="hybrid",
            top_k=2,
            min_score=0.9,
            max_context_chars=70,
            hybrid_candidate_limit=3,
            hybrid_rrf_k=60,
            hybrid_min_rrf_score=0.016,
        ),
    )

    assert [hit.chunk_id for hit in pipeline.retrieve("SCALE-X")] == ["keyword-only"]


def test_hybrid_pipeline_rejects_blank_questions_without_searching() -> None:
    store = FakeStore([_hit("semantic", 0.9)])
    keyword_index = FakeKeywordIndex([_hit("exact", 1.0)])
    pipeline = HybridRetrievalPipeline(
        store=store,
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        keyword_index=keyword_index,
    )

    assert pipeline.retrieve("   ") == []
    assert store.calls == []
    assert keyword_index.calls == []


def test_build_retrieval_pipeline_selects_mode_and_requires_keyword_index() -> None:
    store = FakeStore([])
    provider = DeterministicEmbeddingProvider(dimension=8)

    assert isinstance(
        build_retrieval_pipeline(
            store=store,
            embedding_provider=provider,
            settings=RetrievalSettings(retrieval_mode="vector"),
        ),
        BasicRetrievalPipeline,
    )
    hybrid_pipeline = build_retrieval_pipeline(
        store=store,
        embedding_provider=provider,
        settings=RetrievalSettings(retrieval_mode="hybrid"),
        keyword_index=FakeKeywordIndex([]),
    )

    assert isinstance(hybrid_pipeline, HybridRetrievalPipeline)
    with pytest.raises(ValueError, match="keyword_index is required for hybrid retrieval"):
        build_retrieval_pipeline(
            store=store,
            embedding_provider=provider,
            settings=RetrievalSettings(retrieval_mode="hybrid"),
        )


def test_retrieval_settings_from_env_uses_hybrid_defaults_and_threshold(monkeypatch) -> None:
    for name in (
        "RETRIEVAL_MODE",
        "RAG_TOP_K",
        "RAG_MIN_SCORE",
        "RAG_MAX_CONTEXT_CHARS",
        "HYBRID_CANDIDATE_LIMIT",
        "HYBRID_RRF_K",
        "HYBRID_MIN_RRF_SCORE",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RetrievalSettings.from_env()

    assert settings == RetrievalSettings(
        retrieval_mode="hybrid",
        top_k=5,
        min_score=0.2,
        max_context_chars=4000,
        hybrid_candidate_limit=10,
        hybrid_rrf_k=60,
        hybrid_min_rrf_score=0.015,
    )
    assert settings.evidence_threshold == 0.015
    assert RetrievalSettings(retrieval_mode="vector", min_score=0.3).evidence_threshold == 0.3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RETRIEVAL_MODE", "keyword"),
        ("RAG_TOP_K", "0"),
        ("RAG_MAX_CONTEXT_CHARS", "0"),
        ("HYBRID_CANDIDATE_LIMIT", "0"),
        ("HYBRID_RRF_K", "0"),
        ("RAG_MIN_SCORE", "-0.1"),
        ("HYBRID_MIN_RRF_SCORE", "-0.1"),
    ],
)
def test_retrieval_settings_from_env_rejects_invalid_values(
    monkeypatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        RetrievalSettings.from_env()


def test_hybrid_pipeline_filters_fused_scores_below_rrf_threshold() -> None:
    pipeline = HybridRetrievalPipeline(
        store=FakeStore([_hit("shared-a", 0.9), _hit("shared-b", 0.8), _hit("low", 0.7)]),
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        keyword_index=FakeKeywordIndex([_hit("shared-a", 1.0), _hit("shared-b", 0.9)]),
        settings=RetrievalSettings(
            retrieval_mode="hybrid",
            top_k=3,
            min_score=0.0,
            max_context_chars=1000,
            hybrid_candidate_limit=3,
            hybrid_rrf_k=60,
            hybrid_min_rrf_score=0.02,
        ),
    )

    hits = pipeline.retrieve("scale")

    assert [hit.chunk_id for hit in hits] == ["shared-a", "shared-b"]
    assert all(hit.score >= 0.02 for hit in hits)


def test_hybrid_pipeline_truncates_eligible_candidates_to_top_k() -> None:
    pipeline = HybridRetrievalPipeline(
        store=FakeStore([]),
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        keyword_index=FakeKeywordIndex(
            [_hit("one", 1.0), _hit("two", 0.9), _hit("three", 0.8), _hit("four", 0.7)]
        ),
        settings=RetrievalSettings(
            retrieval_mode="hybrid",
            top_k=2,
            min_score=0.9,
            max_context_chars=1000,
            hybrid_candidate_limit=4,
            hybrid_rrf_k=60,
            hybrid_min_rrf_score=0.015,
        ),
    )

    hits = pipeline.retrieve("scale")

    assert [hit.chunk_id for hit in hits] == ["one", "two"]
