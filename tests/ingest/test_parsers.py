from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document

from oilfield_chemical_copilot.ingest.parsers import parse_document


def _write_minimal_text_pdf(path: Path, text: str) -> None:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(bytes(content))


def test_parse_document_reads_markdown_text_csv_xlsx_docx_and_pdf(tmp_path: Path) -> None:
    markdown = tmp_path / "docs" / "scale_note.md"
    markdown.parent.mkdir()
    markdown.write_text("\ufeff# Scale Note\nScale tendency rises with incompatible waters.", encoding="utf-8")

    text = tmp_path / "docs" / "corrosion_note.txt"
    text.write_text("Corrosion review includes inhibitor residuals.", encoding="utf-8")

    csv_path = tmp_path / "water.csv"
    csv_path.write_text("well_id,chloride_mg_l\nwell-1,35000\n", encoding="utf-8")

    xlsx_path = tmp_path / "analysis.xlsx"
    pd.DataFrame([{"well_id": "well-2", "sulfate_mg_l": 1200}]).to_excel(
        xlsx_path, index=False, sheet_name="Water"
    )

    docx_path = tmp_path / "procedure.docx"
    document = Document()
    document.add_paragraph("Paraffin treatment review uses oil temperature and wax appearance.")
    document.save(docx_path)

    pdf_path = tmp_path / "handout.pdf"
    _write_minimal_text_pdf(pdf_path, "Iron sulfide field note")

    parsed = {
        path.suffix: parse_document(path, source_root=tmp_path, topic="unknown")
        for path in (markdown, text, csv_path, xlsx_path, docx_path, pdf_path)
    }

    assert "Scale tendency" in parsed[".md"][0].text
    assert not parsed[".md"][0].text.startswith("\ufeff")
    assert parsed[".md"][0].metadata.parser_type == "text"
    assert "inhibitor residuals" in parsed[".txt"][0].text
    assert "chloride_mg_l" in parsed[".csv"][0].text
    assert parsed[".csv"][0].metadata.page_or_sheet == "csv"
    assert "sulfate_mg_l" in parsed[".xlsx"][0].text
    assert parsed[".xlsx"][0].metadata.page_or_sheet == "sheet:Water"
    assert "Paraffin treatment" in parsed[".docx"][0].text
    assert "Iron sulfide" in parsed[".pdf"][0].text
    assert parsed[".pdf"][0].metadata.page_or_sheet == "page:1"


def test_parse_document_rejects_legacy_xls_until_reader_dependency_exists(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.xls"
    legacy.write_bytes(b"not a real workbook")

    try:
        parse_document(legacy, source_root=tmp_path, topic="unknown")
    except ValueError as error:
        assert "Unsupported parser extension" in str(error)
    else:
        raise AssertionError("legacy .xls should not be advertised as supported in Milestone 2")