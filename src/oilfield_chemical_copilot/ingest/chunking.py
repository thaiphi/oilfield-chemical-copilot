from __future__ import annotations

from pathlib import Path

from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk


def build_placeholder_chunk(source_file: Path, text: str, topic: str, page_or_sheet: str) -> LoadedChunk:
    """Create a single scaffold chunk with required metadata."""
    # TODO: Replace this with token-aware chunking and stable chunk IDs.
    chunk_id = f"{source_file.stem}:placeholder:0"
    return LoadedChunk(
        text=text,
        metadata=ChunkMetadata(
            source_file=str(source_file),
            topic=topic,
            page_or_sheet=page_or_sheet,
            chunk_id=chunk_id,
        ),
    )
