from pathlib import Path


REQUIRED_SAMPLE_DOCS = {
    "iron_sulfide_overview.md",
    "scale_water_analysis_overview.md",
    "corrosion_root_cause.md",
    "paraffin_asphaltene_overview.md",
    "chemical_dosage_examples.md",
    "water_analysis_interpretation.md",
}


def test_public_sample_dataset_has_safe_reproducible_layout() -> None:
    docs_dir = Path("data/sample/docs")
    water_dir = Path("data/sample/water_analysis_examples")

    assert docs_dir.is_dir()
    assert water_dir.is_dir()
    assert REQUIRED_SAMPLE_DOCS.issubset({path.name for path in docs_dir.glob("*.md")})
    assert (water_dir / "sample_water_analysis.csv").is_file()

    readme = Path("data/sample/README.md").read_text(encoding="utf-8").lower()
    assert "synthetic" in readme
    assert "no private" in readme
