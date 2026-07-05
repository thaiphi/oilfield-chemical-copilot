from __future__ import annotations

from oilfield_chemical_copilot.ingest.models import LoadedChunk


def hybrid_search(query: str, limit: int = 5) -> list[LoadedChunk]:
    """Placeholder for keyword plus vector retrieval."""
    # TODO: Combine minsearch keyword results with PGVector similarity results.
    _ = (query, limit)
    return []
