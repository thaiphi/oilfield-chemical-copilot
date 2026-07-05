from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaterAnalysisResult:
    summary: str


def summarize_water_analysis(chloride_mg_l: float, hardness_mg_l_as_caco3: float) -> WaterAnalysisResult:
    """Summarize basic water-analysis signals for troubleshooting."""
    # TODO: Add scale indices, compatibility checks, and references.
    return WaterAnalysisResult(
        summary=(
            "Placeholder interpretation: "
            f"chloride={chloride_mg_l:.0f} mg/L, "
            f"hardness={hardness_mg_l_as_caco3:.0f} mg/L as CaCO3."
        )
    )
