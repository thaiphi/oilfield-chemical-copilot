from __future__ import annotations

import json
from pathlib import Path

from ingestion.ingest import generate_chunks


def test_generate_chunks_writes_jsonl_from_supported_sample_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "sample"
    docs_dir = data_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "scale_overview.md").write_text(
        "# Scale Overview\nScale deposits can restrict production.", encoding="utf-8"
    )
    (data_dir / "water.csv").write_text("well_id,chloride_mg_l\nwell-1,35000\n", encoding="utf-8")
    (data_dir / "ignore.zip").write_bytes(b"zip")
    output_dir = tmp_path / "processed"

    chunks = generate_chunks(data_dir=data_dir, output_dir=output_dir, max_files=20)

    output_path = output_dir / "chunks.jsonl"
    assert output_path.exists()
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(chunks) == 2
    assert {row["metadata"]["parser_type"] for row in rows} == {"text", "spreadsheet"}
    assert rows[0]["metadata"]["source_file"] == "docs/scale_overview.md"
    assert all("embedding" not in row for row in rows)


def test_generate_chunks_records_an_unreadable_document_and_continues(tmp_path: Path) -> None:
    data_dir = tmp_path / "handouts"
    data_dir.mkdir()
    (data_dir / "scale_note.md").write_text(
        "Scale deposits can restrict production.", encoding="utf-8"
    )
    (data_dir / "encrypted.pdf").write_bytes(b"not a readable PDF")
    output_dir = tmp_path / "processed"

    chunks = generate_chunks(data_dir=data_dir, output_dir=output_dir)

    report = json.loads((output_dir / "ingestion_report.json").read_text(encoding="utf-8"))
    assert len(chunks) == 1
    assert report["parsed_files"] == 1
    assert report["failed_files"] == 1
    assert report["empty_files"] == 0
    assert report["failures"][0]["source_file"] == "encrypted.pdf"
    assert report["failures"][0]["error_type"] == "PdfStreamError"
