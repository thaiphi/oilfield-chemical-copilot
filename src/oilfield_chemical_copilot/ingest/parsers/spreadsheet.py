from __future__ import annotations

from pathlib import Path

from oilfield_chemical_copilot.ingest.chunking import build_placeholder_chunk
from oilfield_chemical_copilot.ingest.models import LoadedChunk


def parse_tabular(path: Path) -> list[LoadedChunk]:
    """Scaffold XLSX/CSV parser."""
    # TODO: Extract workbook sheets or CSV rows with pandas/openpyxl and preserve sheet names.
    return [build_placeholder_chunk(path, "TODO: extracted tabular text", "unclassified", "sheet:unknown")]
