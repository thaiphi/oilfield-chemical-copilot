from __future__ import annotations

import os
from dataclasses import dataclass

from oilfield_chemical_copilot.retrieval.embeddings import EmbeddingProvider
from oilfield_chemical_copilot.retrieval.hybrid import fuse_ranked_hits
from oilfield_chemical_copilot.retrieval.keyword import KeywordSearchIndex
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.vector import VectorRetriever, VectorStore


@dataclass(frozen=True)
class RetrievalSettings:
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    min_score: float = 0.2
    max_context_chars: int = 4000
    hybrid_candidate_limit: int = 10
    hybrid_rrf_k: int = 60
    hybrid_min_rrf_score: float = 0.015

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        settings = cls(
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid") or "hybrid",
            top_k=_env_int("RAG_TOP_K", 5),
            min_score=_env_float("RAG_MIN_SCORE", 0.2),
            max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 4000),
            hybrid_candidate_limit=_env_int("HYBRID_CANDIDATE_LIMIT", 10),
            hybrid_rrf_k=_env_int("HYBRID_RRF_K", 60),
            hybrid_min_rrf_score=_env_float("HYBRID_MIN_RRF_SCORE", 0.015),
        )
        if settings.retrieval_mode not in {"hybrid", "vector"}:
            raise ValueError("RETRIEVAL_MODE must be 'hybrid' or 'vector'")
        if settings.top_k < 1:
            raise ValueError("RAG_TOP_K must be at least 1")
        if settings.max_context_chars < 1:
            raise ValueError("RAG_MAX_CONTEXT_CHARS must be at least 1")
        if settings.hybrid_candidate_limit < 1:
            raise ValueError("HYBRID_CANDIDATE_LIMIT must be at least 1")
        if settings.hybrid_rrf_k < 1:
            raise ValueError("HYBRID_RRF_K must be at least 1")
        if settings.min_score < 0:
            raise ValueError("RAG_MIN_SCORE must be non-negative")
        if settings.hybrid_min_rrf_score < 0:
            raise ValueError("HYBRID_MIN_RRF_SCORE must be non-negative")
        return settings

    @property
    def evidence_threshold(self) -> float:
        return self.min_score if self.retrieval_mode == "vector" else self.hybrid_min_rrf_score


class BasicRetrievalPipeline:
    def __init__(
        self,
        *,
        store: VectorStore,
        embedding_provider: EmbeddingProvider,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self.vector_retriever = VectorRetriever(store=store, embedding_provider=embedding_provider)

    def retrieve(self, question: str, topic: str | None = None) -> list[RetrievalHit]:
        if not question.strip():
            return []
        hits = self.vector_retriever.search(question, limit=self.settings.top_k, topic=topic)
        qualifying = [hit for hit in hits if hit.score >= self.settings.min_score]
        return _fit_context_budget(qualifying, self.settings.max_context_chars)


class HybridRetrievalPipeline:
    def __init__(
        self,
        *,
        store: VectorStore,
        embedding_provider: EmbeddingProvider,
        keyword_index: KeywordSearchIndex,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self.vector_retriever = VectorRetriever(store=store, embedding_provider=embedding_provider)
        self.keyword_index = keyword_index

    def retrieve(self, question: str, topic: str | None = None) -> list[RetrievalHit]:
        if not question.strip():
            return []
        vector_hits = self.vector_retriever.search(
            question, limit=self.settings.hybrid_candidate_limit, topic=topic
        )
        keyword_hits = self.keyword_index.search(
            question, limit=self.settings.hybrid_candidate_limit, topic=topic
        )
        fused = fuse_ranked_hits(
            keyword_hits,
            vector_hits,
            rrf_k=self.settings.hybrid_rrf_k,
            limit=self.settings.top_k,
        )
        qualifying = [hit for hit in fused if hit.score >= self.settings.hybrid_min_rrf_score]
        return _fit_context_budget(qualifying, self.settings.max_context_chars)


def build_retrieval_pipeline(
    *,
    store: VectorStore,
    embedding_provider: EmbeddingProvider,
    settings: RetrievalSettings,
    keyword_index: KeywordSearchIndex | None = None,
) -> BasicRetrievalPipeline | HybridRetrievalPipeline:
    if settings.retrieval_mode == "vector":
        return BasicRetrievalPipeline(
            store=store,
            embedding_provider=embedding_provider,
            settings=settings,
        )
    if keyword_index is None:
        raise ValueError("keyword_index is required for hybrid retrieval")
    return HybridRetrievalPipeline(
        store=store,
        embedding_provider=embedding_provider,
        keyword_index=keyword_index,
        settings=settings,
    )


def _fit_context_budget(hits: list[RetrievalHit], max_context_chars: int) -> list[RetrievalHit]:
    if max_context_chars < 1:
        return []
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, hit in enumerate(hits):
        additions: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (score, selected_indices) in states.items():
            total = used + len(hit.text)
            if total > max_context_chars:
                continue
            candidate = (score + hit.score, (*selected_indices, index))
            current = additions.get(total)
            if current is None or candidate[0] > current[0]:
                additions[total] = candidate
        for total, candidate in additions.items():
            current = states.get(total)
            if current is None or candidate[0] > current[0]:
                states[total] = candidate
        states = _prune_dominated_context_states(states)

    _, selected_indices = max(
        states.values(), key=lambda state: (state[0], len(state[1]), -sum(state[1]))
    )
    return [hits[index] for index in selected_indices]


def _prune_dominated_context_states(
    states: dict[int, tuple[float, tuple[int, ...]]]
) -> dict[int, tuple[float, tuple[int, ...]]]:
    pruned: dict[int, tuple[float, tuple[int, ...]]] = {}
    best_score = -1.0
    for used in sorted(states):
        state = states[used]
        if state[0] > best_score:
            pruned[used] = state
            best_score = state[0]
    return pruned


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
