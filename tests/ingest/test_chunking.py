from __future__ import annotations

from pathlib import Path

from oilfield_chemical_copilot.ingest.chunking import _next_start, chunk_text


def test_chunk_text_splits_text_with_stable_metadata(tmp_path: Path) -> None:
    source = tmp_path / "Scale" / "notes.md"
    source.parent.mkdir()
    source.write_text("", encoding="utf-8")
    text = "alpha beta gamma delta " * 80

    first = chunk_text(
        text=text,
        source_file=source,
        source_root=tmp_path,
        topic="scale",
        parser_type="text",
        page_or_sheet="document",
        max_chars=220,
        overlap=40,
    )
    second = chunk_text(
        text=text,
        source_file=source,
        source_root=tmp_path,
        topic="scale",
        parser_type="text",
        page_or_sheet="document",
        max_chars=220,
        overlap=40,
    )

    assert len(first) > 1
    assert [chunk.metadata.chunk_id for chunk in first] == [
        chunk.metadata.chunk_id for chunk in second
    ]
    assert first[0].metadata.source_file == "Scale/notes.md"
    assert first[0].metadata.source_path == str(source.resolve())
    assert first[0].metadata.topic == "scale"
    assert first[0].metadata.parser_type == "text"
    assert first[0].metadata.page_or_sheet == "document"
    assert first[0].metadata.chunk_index == 0
    assert all(chunk.text.strip() for chunk in first)


def test_chunk_text_high_overlap_still_makes_forward_progress(tmp_path: Path) -> None:
    source = tmp_path / "long.md"
    source.write_text("", encoding="utf-8")
    text = ("word " * 400).strip()

    chunks = chunk_text(
        text=text,
        source_file=source,
        source_root=tmp_path,
        topic="unknown",
        parser_type="text",
        page_or_sheet="document",
        max_chars=100,
        overlap=90,
    )

    assert len(chunks) > 1
    assert chunks[-1].text.endswith("word")

def test_next_start_advances_when_overlap_exceeds_adjusted_boundary() -> None:
    assert _next_start(start=0, end=60, overlap=90) == 1