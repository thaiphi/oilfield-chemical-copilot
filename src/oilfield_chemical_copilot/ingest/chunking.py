from __future__ import annotations

import hashlib
from pathlib import Path

from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk


def chunk_text(
    *,
    text: str,
    source_file: Path,
    source_root: Path | None,
    topic: str,
    parser_type: str,
    page_or_sheet: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[LoadedChunk]:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip().lstrip("\ufeff")
    if not cleaned:
        return []
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")

    chunks = _split_text(cleaned, max_chars=max_chars, overlap=overlap)
    relative_source = _relative_source_file(source_file, source_root)
    absolute_source = str(source_file.resolve())
    loaded: list[LoadedChunk] = []
    for index, chunk in enumerate(chunks):
        chunk_id = _chunk_id(relative_source, parser_type, page_or_sheet, index, chunk)
        loaded.append(
            LoadedChunk(
                text=chunk,
                metadata=ChunkMetadata(
                    source_file=relative_source,
                    source_path=absolute_source,
                    topic=topic,
                    parser_type=parser_type,
                    page_or_sheet=page_or_sheet,
                    chunk_index=index,
                    chunk_id=chunk_id,
                ),
            )
        )
    return loaded


def build_placeholder_chunk(source_file: Path, text: str, topic: str, page_or_sheet: str) -> LoadedChunk:
    chunks = chunk_text(
        text=text,
        source_file=source_file,
        source_root=source_file.parent,
        topic=topic,
        parser_type="placeholder",
        page_or_sheet=page_or_sheet,
    )
    if chunks:
        return chunks[0]
    return LoadedChunk(
        text="",
        metadata=ChunkMetadata(
            source_file=source_file.name,
            source_path=str(source_file.resolve()),
            topic=topic,
            parser_type="placeholder",
            page_or_sheet=page_or_sheet,
            chunk_index=0,
            chunk_id=_chunk_id(source_file.name, "placeholder", page_or_sheet, 0, ""),
        ),
    )


def _split_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = _next_start(start=start, end=end, overlap=overlap)
    return chunks


def _next_start(*, start: int, end: int, overlap: int) -> int:
    return max(end - overlap, start + 1)


def _relative_source_file(source_file: Path, source_root: Path | None) -> str:
    resolved = source_file.resolve()
    if source_root is None:
        return source_file.name
    try:
        return resolved.relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return source_file.name


def _chunk_id(
    source_file: str,
    parser_type: str,
    page_or_sheet: str,
    chunk_index: int,
    text: str,
) -> str:
    digest = hashlib.sha1(
        f"{source_file}|{parser_type}|{page_or_sheet}|{chunk_index}|{text}".encode("utf-8")
    ).hexdigest()[:12]
    safe_source = source_file.replace("/", ":").replace("\\", ":")
    return f"{safe_source}:{page_or_sheet}:{chunk_index}:{digest}"
