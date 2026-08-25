"""Document parser implementations."""

from __future__ import annotations

from pathlib import Path

from oilfield_chemical_copilot.ingest.models import LoadedChunk
from oilfield_chemical_copilot.ingest.parsers.docx import parse_docx
from oilfield_chemical_copilot.ingest.parsers.pdf import parse_pdf
from oilfield_chemical_copilot.ingest.parsers.spreadsheet import parse_tabular
from oilfield_chemical_copilot.ingest.parsers.text import parse_text


def parse_document(
    path: Path,
    *,
    source_root: Path | None = None,
    topic: str = "unknown",
) -> list[LoadedChunk]:
    extension = path.suffix.lower()
    if extension in {".md", ".txt"}:
        return parse_text(path, source_root=source_root, topic=topic)
    if extension == ".docx":
        return parse_docx(path, source_root=source_root, topic=topic)
    if extension == ".pdf":
        return parse_pdf(path, source_root=source_root, topic=topic)
    if extension in {".csv", ".xlsx"}:
        return parse_tabular(path, source_root=source_root, topic=topic)
    raise ValueError(f"Unsupported parser extension: {extension}")


__all__ = ["parse_document", "parse_docx", "parse_pdf", "parse_tabular", "parse_text"]
