from __future__ import annotations

import os
from dataclasses import dataclass

from oilfield_chemical_copilot.retrieval.embeddings import EmbeddingProvider
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.vector import VectorRetriever, VectorStore


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 5
    min_score: float = 0.2
    max_context_chars: int = 4000

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            top_k=_env_int("RAG_TOP_K", 5),
            min_score=_env_float("RAG_MIN_SCORE", 0.2),
            max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 4000),
        )


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


def _fit_context_budget(hits: list[RetrievalHit], max_context_chars: int) -> list[RetrievalHit]:
    if max_context_chars < 1:
        return []
    selected: list[RetrievalHit] = []
    used = 0
    for hit in hits:
        if used + len(hit.text) > max_context_chars:
            break
        selected.append(hit)
        used += len(hit.text)
    return selected


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