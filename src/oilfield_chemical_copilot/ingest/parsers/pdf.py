from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from oilfield_chemical_copilot.ingest.chunking import chunk_text
from oilfield_chemical_copilot.ingest.models import LoadedChunk


def parse_pdf(path: Path, *, source_root: Path | None = None, topic: str = "unknown") -> list[LoadedChunk]:
    reader = PdfReader(path)
    chunks: list[LoadedChunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.extend(
            chunk_text(
                text=text,
                source_file=path,
                source_root=source_root,
                topic=topic,
                parser_type="pdf",
                page_or_sheet=f"page:{page_number}",
            )
        )
    return chunks
