from pathlib import Path

from oilfield_chemical_copilot.ingest.config import data_root_for_mode
from oilfield_chemical_copilot.ingest.scanner import scan_sources


def test_data_root_for_sample_mode() -> None:
    assert data_root_for_mode("sample") == Path("data/sample")


def test_scan_sources_finds_supported_sample_file() -> None:
    sources = scan_sources(Path("data/sample"))

    assert Path("data/sample/sample_water_analysis.csv") in sources
    assert all(path.suffix.lower() in {".pdf", ".docx", ".xlsx", ".csv"} for path in sources)
