from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkMetadata:
    source_file: str
    source_path: str
    topic: str
    parser_type: str
    page_or_sheet: str
    chunk_index: int
    chunk_id: str
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedChunk:
    text: str
    metadata: ChunkMetadata
