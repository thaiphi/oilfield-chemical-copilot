from __future__ import annotations

from dataclasses import asdict

import minsearch

from oilfield_chemical_copilot.ingest.models import LoadedChunk
from oilfield_chemical_copilot.retrieval.models import RetrievalHit


class KeywordSearchIndex:
    def __init__(self, documents: list[dict[str, object]]) -> None:
        self._documents = {str(document["chunk_id"]): document for document in documents}
        self._index = minsearch.Index(
            text_fields=["content", "topic", "source_file"],
            keyword_fields=["chunk_id", "topic", "source_file"],
        )
        if documents:
            self._index.fit(documents)

    @classmethod
    def from_chunks(cls, chunks: list[LoadedChunk]) -> "KeywordSearchIndex":
        return cls([_document_for_chunk(chunk) for chunk in chunks])

    @classmethod
    def from_hits(cls, hits: list[RetrievalHit]) -> "KeywordSearchIndex":
        return cls([_document_for_hit(hit) for hit in hits])

    def search(self, query: str, limit: int = 5, topic: str | None = None) -> list[RetrievalHit]:
        if not query.strip() or limit < 1 or not self._documents:
            return []
        filter_dict = {"topic": topic} if topic else None
        results = self._index.search(
            query=query,
            filter_dict=filter_dict,
            boost_dict={"content": 2.0, "topic": 1.5, "source_file": 0.5},
            num_results=limit,
        )
        return [_hit_from_document(result, score=1.0 / rank) for rank, result in enumerate(results, start=1)]


def _document_for_chunk(chunk: LoadedChunk) -> dict[str, object]:
    metadata = asdict(chunk.metadata)
    return {
        "chunk_id": chunk.metadata.chunk_id,
        "content": chunk.text,
        "source_file": chunk.metadata.source_file,
        "source_path": chunk.metadata.source_path,
        "topic": chunk.metadata.topic,
        "parser_type": chunk.metadata.parser_type,
        "page_or_sheet": chunk.metadata.page_or_sheet,
        "chunk_index": chunk.metadata.chunk_index,
        "metadata": metadata.get("extra", {}),
    }


def _document_for_hit(hit: RetrievalHit) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk_id,
        "content": hit.text,
        "source_file": hit.source_file,
        "source_path": hit.source_path,
        "topic": hit.topic,
        "parser_type": hit.parser_type,
        "page_or_sheet": hit.page_or_sheet,
        "chunk_index": hit.chunk_index,
        "metadata": dict(hit.metadata),
    }


def _hit_from_document(document: dict[str, object], *, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=str(document["chunk_id"]),
        text=str(document["content"]),
        score=score,
        retrieval_method="keyword",
        source_file=str(document["source_file"]),
        source_path=str(document["source_path"]),
        topic=str(document["topic"]),
        parser_type=str(document["parser_type"]),
        page_or_sheet=str(document["page_or_sheet"]),
        chunk_index=int(document["chunk_index"]),
        metadata=dict(document.get("metadata", {})),
    )

