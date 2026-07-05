from __future__ import annotations

from pathlib import Path

from oilfield_chemical_copilot.ingest.chunking import build_placeholder_chunk
from oilfield_chemical_copilot.ingest.models import LoadedChunk


def parse_docx(path: Path) -> list[LoadedChunk]:
    """Scaffold DOCX parser."""
    # TODO: Extract paragraphs and tables with python-docx.
    return [build_placeholder_chunk(path, "TODO: extracted DOCX text", "unclassified", "document")]
