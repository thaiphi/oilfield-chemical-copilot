from __future__ import annotations

from pathlib import Path

from oilfield_chemical_copilot.ingest.chunking import build_placeholder_chunk
from oilfield_chemical_copilot.ingest.models import LoadedChunk


def parse_pdf(path: Path) -> list[LoadedChunk]:
    """Scaffold PDF parser."""
    # TODO: Extract page text with pypdf and preserve page numbers in metadata.
    return [build_placeholder_chunk(path, "TODO: extracted PDF text", "unclassified", "page:unknown")]
