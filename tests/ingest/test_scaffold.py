from pathlib import Path

from oilfield_chemical_copilot.ingest.config import data_root_for_mode
from oilfield_chemical_copilot.ingest.scanner import scan_sources


def test_data_root_for_sample_mode() -> None:
    assert data_root_for_mode("sample") == Path("data/sample")


def test_scan_sources_finds_supported_nested_files(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    supported = nested / "water.csv"
    unsupported = nested / "notes.txt"
    supported.write_text("chloride,hardness\n35000,2500\n", encoding="utf-8")
    unsupported.write_text("ignore", encoding="utf-8")

    assert scan_sources(tmp_path) == [supported]
