from __future__ import annotations

from typing import Protocol

from oilfield_chemical_copilot.retrieval.embeddings import EmbeddingProvider
from oilfield_chemical_copilot.retrieval.models import RetrievalHit


class VectorStore(Protocol):
    def search(
        self,
        query_embedding: list[float],
        *,
        limit: int,
        topic: str | None = None,
        embedding_model: str,
    ) -> list[RetrievalHit]:
        ...


class VectorRetriever:
    def __init__(self, *, store: VectorStore, embedding_provider: EmbeddingProvider) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def search(self, query: str, limit: int = 5, topic: str | None = None) -> list[RetrievalHit]:
        if not query.strip() or limit < 1:
            return []
        query_embedding = self.embedding_provider.embed_query(query)
        return self.store.search(
            query_embedding,
            limit=limit,
            topic=topic,
            embedding_model=self.embedding_provider.model_name,
        )