"""Ingestion scaffolds for source discovery, parsing, and chunking."""

from oilfield_chemical_copilot.ingest.config import DataMode, data_root_for_mode
from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk
from oilfield_chemical_copilot.ingest.scanner import scan_sources

__all__ = [
    "ChunkMetadata",
    "DataMode",
    "LoadedChunk",
    "data_root_for_mode",
    "scan_sources",
]
