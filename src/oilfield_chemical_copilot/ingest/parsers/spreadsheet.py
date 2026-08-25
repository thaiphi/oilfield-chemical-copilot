from __future__ import annotations

from pathlib import Path

import pandas as pd

from oilfield_chemical_copilot.ingest.chunking import chunk_text
from oilfield_chemical_copilot.ingest.models import LoadedChunk


def parse_tabular(path: Path, *, source_root: Path | None = None, topic: str = "unknown") -> list[LoadedChunk]:
    extension = path.suffix.lower()
    if extension == ".csv":
        frame = pd.read_csv(path, dtype=str).fillna("")
        return _chunks_for_frame(
            frame,
            path=path,
            source_root=source_root,
            topic=topic,
            page_or_sheet="csv",
        )
    if extension == ".xlsx":
        sheets = pd.read_excel(path, dtype=str, sheet_name=None).items()
        chunks: list[LoadedChunk] = []
        for sheet_name, frame in sheets:
            chunks.extend(
                _chunks_for_frame(
                    frame.fillna(""),
                    path=path,
                    source_root=source_root,
                    topic=topic,
                    page_or_sheet=f"sheet:{sheet_name}",
                )
            )
        return chunks
    raise ValueError(f"Unsupported tabular extension: {extension}")


def _chunks_for_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    source_root: Path | None,
    topic: str,
    page_or_sheet: str,
) -> list[LoadedChunk]:
    text = frame.to_csv(index=False).strip()
    return chunk_text(
        text=text,
        source_file=path,
        source_root=source_root,
        topic=topic,
        parser_type="spreadsheet",
        page_or_sheet=page_or_sheet,
    )
