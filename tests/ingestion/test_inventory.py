from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ingestion.inventory import (
    CSV_COLUMNS,
    generate_inventory,
    infer_ingest_priority,
    infer_parser_type,
    infer_topic,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("1.0 Iron Sulfide/THPS treatment.pdf", "iron_sulfide"),
        ("2.0 Scale/deposit guide.docx", "scale"),
        ("Water Analyses/chloride results.xlsx", "water_analysis"),
        ("Corrosion/NACE coupon failure.pdf", "corrosion"),
        ("Paraffin/wax treatment.md", "paraffin"),
        ("Asphaltene/SARA results.csv", "asphaltene"),
        ("Useful Charts/ppm conversions.pdf", "useful_charts"),
        ("Reference Papers/history.pdf", "reference_papers"),
        ("Well Workover/job completion.txt", "workover"),
        ("Chemical Dosage/calculator.xlsx", "dosage"),
        ("lab/water guide.pdf", "water_analysis"),
        ("lab/analysis result.csv", "water_analysis"),
        ("misc/ppm table.pdf", "useful_charts"),
        ("misc/readme.bin", "unknown"),
    ],
)
def test_infer_topic_matches_supported_taxonomy(path: str, expected: str) -> None:
    assert infer_topic(path) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("handout.pdf", "pdf"),
        ("procedure.DOCX", "docx"),
        ("analysis.xlsx", "spreadsheet"),
        ("legacy.xls", "spreadsheet"),
        ("readings.csv", "spreadsheet"),
        ("photo.jpg", "image"),
        ("scan.JPEG", "image"),
        ("diagram.png", "image"),
        ("notes.txt", "text"),
        ("overview.md", "text"),
        ("archive.zip", "unsupported"),
    ],
)
def test_infer_parser_type_from_extension(path: str, expected: str) -> None:
    assert infer_parser_type(path) == expected


def test_infer_ingest_priority_skips_temp_files_and_folders() -> None:
    assert infer_ingest_priority(Path("Scale/temp_notes.pdf"), "scale", "pdf") == "skip"
    assert infer_ingest_priority(Path("Scale/temp/intermediate.csv"), "scale", "spreadsheet") == "skip"
    assert infer_ingest_priority(Path("Scale/temp.pdf"), "scale", "pdf") == "skip"
    assert infer_ingest_priority(Path("Scale/tmp.csv"), "scale", "spreadsheet") == "skip"


def test_infer_topic_prefers_explicit_folder_over_generic_keywords() -> None:
    assert infer_topic("Reference Papers/professional study.pdf") == "reference_papers"
    assert infer_topic("Scale/lab analysis.pdf") == "scale"
    assert infer_topic("Chemical Dosage/ppm calculator.xlsx") == "dosage"
    assert infer_topic("Reference Papers/scale treatment study.pdf") == "reference_papers"
    assert infer_topic("Chemical Dosage/corrosion failure calculator.xlsx") == "dosage"


def test_infer_ingest_priority_keeps_temperature_and_template_files() -> None:
    assert infer_ingest_priority(Path("Water Analysis/temperature_log.csv"), "water_analysis", "spreadsheet") == "high"
    assert infer_ingest_priority(Path("Scale/template.xlsx"), "scale", "spreadsheet") == "high"


def test_generate_inventory_writes_csv_and_summary_for_nested_folder(tmp_path: Path) -> None:
    data_dir = tmp_path / "corpus"
    nested_dir = data_dir / "Water Analysis"
    nested_dir.mkdir(parents=True)
    (nested_dir / "sample.csv").write_text("chloride,tds\n1200,2500\n", encoding="utf-8")
    (data_dir / "field_photo.png").write_bytes(b"png")
    (data_dir / "archive.zip").write_bytes(b"zip")
    output_dir = tmp_path / "processed"

    rows = generate_inventory(data_dir=data_dir, output_dir=output_dir)

    assert len(rows) == 3
    assert tuple(rows[0]) == CSV_COLUMNS
    image_row = next(row for row in rows if row["file_name"] == "field_photo.png")
    assert image_row["parser_type"] == "image"
    assert image_row["needs_ocr"] is True
    unsupported_row = next(row for row in rows if row["file_name"] == "archive.zip")
    assert unsupported_row["ingest_priority"] == "skip"

    with (output_dir / "inventory.csv").open(newline="", encoding="utf-8") as csv_file:
        written_rows = list(csv.DictReader(csv_file))
    assert written_rows[0].keys() == dict.fromkeys(CSV_COLUMNS).keys()
    assert {row["relative_path"] for row in written_rows} == {
        "Water Analysis/sample.csv",
        "archive.zip",
        "field_photo.png",
    }

    summary = (output_dir / "inventory_summary.md").read_text(encoding="utf-8")
    assert f"`{data_dir.resolve()}`" in summary
    assert "Total file count: **3**" in summary
    assert "## Files Needing OCR" in summary
    assert "## Unsupported File Types" in summary


def test_generate_inventory_supports_absolute_external_root(tmp_path: Path) -> None:
    external_root = (tmp_path / "mounted-drive").resolve()
    source_file = external_root / "Reference Papers" / "paper.pdf"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"pdf")

    rows = generate_inventory(
        data_dir=external_root,
        output_dir=tmp_path / "processed",
        summary_only=True,
    )

    assert rows[0]["relative_path"] == "Reference Papers/paper.pdf"
    assert rows[0]["absolute_path"] == str(source_file.resolve())
    assert rows[0]["topic_folder"] == "reference_papers"


def test_generate_inventory_filters_include_extensions(tmp_path: Path) -> None:
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "guide.pdf").write_bytes(b"pdf")
    (data_dir / "analysis.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (data_dir / "notes.md").write_text("notes", encoding="utf-8")

    rows = generate_inventory(
        data_dir=data_dir,
        output_dir=tmp_path / "processed",
        include_extensions={"pdf", ".CSV"},
    )

    assert [row["file_name"] for row in rows] == ["analysis.csv", "guide.pdf"]


def test_generate_inventory_stops_at_max_files_in_sorted_order(tmp_path: Path) -> None:
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    for file_name in ("c.pdf", "a.pdf", "b.pdf"):
        (data_dir / file_name).write_bytes(file_name.encode())

    rows = generate_inventory(
        data_dir=data_dir,
        output_dir=tmp_path / "processed",
        max_files=2,
    )

    assert [row["file_name"] for row in rows] == ["a.pdf", "b.pdf"]
