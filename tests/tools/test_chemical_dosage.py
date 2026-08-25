from __future__ import annotations

from inspect import signature

import pytest

from oilfield_chemical_copilot.tools.chemical_dosage import (
    CONTRACT,
    LABEL,
    VERSION,
    calculate_dosage,
    product_dosage_answer,
)


def test_calculate_dosage_uses_the_product_ppm_water_basis_contract() -> None:
    result = calculate_dosage(water_bbl_per_day=1_000, product_ppm=100)

    assert result.water_bbl_per_day == 1_000.0
    assert result.product_ppm == 100.0
    assert result.product_gallons_per_day == 4.2
    assert result.label == LABEL
    assert result.audit_metadata == {
        "contract": CONTRACT,
        "version": VERSION,
        "status": "calculated",
    }


def test_calculate_dosage_allows_zero_product_ppm() -> None:
    result = calculate_dosage(water_bbl_per_day=1_000, product_ppm=0)

    assert result.product_gallons_per_day == 0.0


@pytest.mark.parametrize(
    ("water_bbl_per_day", "product_ppm"),
    [
        (0, 100),
        (-1, 100),
        (float("nan"), 100),
        (float("inf"), 100),
        (1_000, -1),
        (1_000, float("nan")),
        (1_000, "100"),
        (True, 100),
    ],
)
def test_calculate_dosage_rejects_invalid_contract_inputs(
    water_bbl_per_day: object, product_ppm: object
) -> None:
    with pytest.raises(ValueError):
        calculate_dosage(water_bbl_per_day=water_bbl_per_day, product_ppm=product_ppm)


def test_calculate_dosage_does_not_expose_an_active_fraction_argument() -> None:
    assert tuple(signature(calculate_dosage).parameters) == ("water_bbl_per_day", "product_ppm")


def test_product_dosage_answer_keeps_the_calculation_deterministic_and_non_prescriptive() -> None:
    answer = product_dosage_answer(1_000, 100)

    assert answer.sources == []
    assert answer.weak_evidence is False
    assert "4.2 gallons/day" in answer.text
    assert LABEL in answer.text
    assert "field-ready dose" in answer.text
