from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from oilfield_chemical_copilot.rag.models import RagAnswer


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


def product_dosage_answer(water_bbl_per_day: float, product_ppm: float) -> RagAnswer:
    return product_dosage_answer_from_result(calculate_dosage(water_bbl_per_day, product_ppm))


def product_dosage_answer_from_result(result: DosageResult) -> RagAnswer:
    text = (
        f"Answer:\n{result.label}.\n\n"
        f"Calculation: {result.product_ppm:g} product ppm x {result.water_bbl_per_day:g} water bbl/day "
        f"= {result.product_gallons_per_day:g} gallons/day.\n\n"
        "Why this matters:\nA fixed unit contract makes the arithmetic reviewable without treating it as a treatment recommendation.\n\n"
        "Evidence from retrieved sources:\n- No retrieval was run; this is a deterministic product-ppm water-basis calculation.\n\n"
        "Recommended next checks:\n"
        "1. Confirm that the requested ppm is a product-ppm target.\n"
        "2. Confirm the current water rate in barrels per day.\n"
        "3. Obtain qualified engineering review before applying a field treatment.\n\n"
        "Limitations:\nThis general calculation does not establish a field-ready dose."
    )
    return RagAnswer(text=text, sources=[], weak_evidence=False)


def _validate_input(name: str, value: object, *, minimum: float, exclusive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    if (exclusive and value <= minimum) or (not exclusive and value < minimum):
        comparator = "greater than" if exclusive else "at least"
        raise ValueError(f"{name} must be {comparator} {minimum}")
