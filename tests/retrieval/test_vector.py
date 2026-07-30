from __future__ import annotations

from oilfield_chemical_copilot.retrieval.embeddings import DeterministicEmbeddingProvider
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.vector import VectorRetriever


class FakeStore:
    def __init__(self) -> None:
        self.calls = []

    def search(
        self,
        query_embedding,
        *,
        limit: int,
        topic: str | None = None,
        embedding_model: str | None = None,
    ):
        self.calls.append((query_embedding, limit, topic, embedding_model))
        return [
            RetrievalHit(
                chunk_id="chunk-1",
                text="Scale water analysis",
                score=0.5,
                retrieval_method="vector",
                source_file="scale.md",
                source_path="C:/sample/scale.md",
                topic="scale",
                parser_type="text",
                page_or_sheet="document",
                chunk_index=0,
                metadata={},
            )
        ]


def test_vector_retriever_embeds_query_and_delegates_to_store() -> None:
    store = FakeStore()
    retriever = VectorRetriever(store=store, embedding_provider=DeterministicEmbeddingProvider(8))

    hits = retriever.search("scale water analysis", limit=3, topic="scale")

    assert hits[0].chunk_id == "chunk-1"
    assert len(store.calls[0][0]) == 8
    assert store.calls[0][1:] == (3, "scale", "deterministic-hash-8")