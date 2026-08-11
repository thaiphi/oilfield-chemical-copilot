from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real


CONTRACT = "chemical_dosage.product_ppm_water_basis"
VERSION = "v1"
LABEL = "General product-dose calculation - not a field-ready prescription"


@dataclass(frozen=True)
class DosageResult:
    water_bbl_per_day: float
    product_ppm: float
    product_gallons_per_day: float
    label: str
    audit_metadata: dict[str, str]


def calculate_dosage(water_bbl_per_day: float, product_ppm: float) -> DosageResult:
    """Calculate a general product dose from product ppm on a water basis."""
    _validate_input("water_bbl_per_day", water_bbl_per_day, minimum=0, exclusive=True)
    _validate_input("product_ppm", product_ppm, minimum=0, exclusive=False)
    water_bbl_per_day = float(water_bbl_per_day)
    product_ppm = float(product_ppm)
    product_gallons_per_day = product_ppm * water_bbl_per_day * 42 / 1_000_000
    return DosageResult(
        water_bbl_per_day=water_bbl_per_day,
        product_ppm=product_ppm,
        product_gallons_per_day=product_gallons_per_day,
        label=LABEL,
        audit_metadata={"contract": CONTRACT, "version": VERSION, "status": "calculated"},
    )


def _validate_input(name: str, value: object, *, minimum: float, exclusive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    if (exclusive and value <= minimum) or (not exclusive and value < minimum):
        comparator = "greater than" if exclusive else "at least"
        raise ValueError(f"{name} must be {comparator} {minimum}")
