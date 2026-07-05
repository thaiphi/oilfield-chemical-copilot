from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DosageResult:
    summary: str
    estimated_gallons: float


def calculate_dosage(volume_bbl: float, target_ppm: float, active_fraction: float) -> DosageResult:
    """Estimate product dosage for a treatment target."""
    # TODO: Validate unit assumptions with domain references before production use.
    active_fraction = max(active_fraction, 0.01)
    estimated_gallons = volume_bbl * target_ppm / active_fraction / 1_000_000
    return DosageResult(
        summary=(
            f"Placeholder estimate: {estimated_gallons:.2f} gal product for "
            f"{volume_bbl:.0f} bbl at {target_ppm:.0f} ppm active."
        ),
        estimated_gallons=estimated_gallons,
    )
