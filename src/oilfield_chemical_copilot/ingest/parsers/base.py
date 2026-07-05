from __future__ import annotations

from pathlib import Path
from typing import Protocol

from oilfield_chemical_copilot.ingest.models import LoadedChunk


class DocumentParser(Protocol):
    """Parser protocol for future PDF, DOCX, XLSX, and CSV extraction."""

    def parse(self, path: Path) -> list[LoadedChunk]:
        """Parse a source document into loaded chunks."""
        ...
