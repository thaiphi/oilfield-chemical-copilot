from __future__ import annotations

from oilfield_chemical_copilot.ingest.models import LoadedChunk


def load_chunks(chunks: list[LoadedChunk]) -> int:
    """Placeholder for loading chunks and embeddings into PostgreSQL + PGVector."""
    # TODO: Generate embeddings, upsert chunks, and store vectors with metadata.
    return len(chunks)
