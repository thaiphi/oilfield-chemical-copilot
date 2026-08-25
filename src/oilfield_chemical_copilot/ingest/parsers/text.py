from __future__ import annotations

from pathlib import Path

from oilfield_chemical_copilot.ingest.chunking import chunk_text
from oilfield_chemical_copilot.ingest.models import LoadedChunk


def parse_text(path: Path, *, source_root: Path | None = None, topic: str = "unknown") -> list[LoadedChunk]:
    text = path.read_text(encoding="utf-8")
    return chunk_text(
        text=text,
        source_file=path,
        source_root=source_root,
        topic=topic,
        parser_type="text",
        page_or_sheet="document",
    )
