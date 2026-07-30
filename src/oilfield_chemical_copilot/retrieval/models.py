from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    text: str
    score: float
    retrieval_method: str
    source_file: str
    source_path: str
    topic: str
    parser_type: str
    page_or_sheet: str
    chunk_index: int
    metadata: dict[str, object] = field(default_factory=dict)
