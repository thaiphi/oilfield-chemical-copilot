from dataclasses import replace
from pathlib import Path
import json
import re

import pytest

from production_hardening.holdout_contract import (
    AuthoredCase,
    ContractValidationError,
    HoldoutContract,
    RequiredPair,
    ReviewDecision,
    seal_holdout,
    validate_for_sealing,
    verify_seal,
)


def contract() -> HoldoutContract:
    return HoldoutContract(
        strata=("S01", "S02", "S03"),
        cases_per_stratum=12,
        allow_per_stratum=6,
        abstain_per_stratum=6,
        required_pairs=(
            RequiredPair("allow", "general_review", 18),
            RequiredPair("abstain", "site_specific_determination", 6),
            RequiredPair("abstain", "field_ready_prescription", 6),
            RequiredPair("abstain", "complete_input_substitution", 6),
        ),
    )


def toy_matrix() -> tuple[tuple[AuthoredCase, ...], tuple[ReviewDecision, ...]]:
    cases: list[AuthoredCase] = []
    reviews: list[ReviewDecision] = []
    abstain_categories = (
        "site_specific_determination",
        "field_ready_prescription",
        "complete_input_substitution",
    )
    for stratum_index, stratum_id in enumerate(("S01", "S02", "S03"), start=1):
        for record_index in range(1, 13):
            case_id = f"T{(stratum_index - 1) * 12 + record_index:02d}"
            if record_index <= 6:
                action, category = "allow", "general_review"
            else:
                action, category = "abstain", abstain_categories[(record_index - 7 + stratum_index - 1) % 3]
            cases.append(AuthoredCase(case_id, stratum_id, "toy prompt", action, category, "author", True))
            reviews.append(ReviewDecision(case_id, "reviewer", "approved"))
    return tuple(cases), tuple(reviews)


def assert_code(callable_object, *args) -> None:
    with pytest.raises(ContractValidationError) as error:
        callable_object(*args)
    assert re.fullmatch(r"[A-Z_]+", str(error.value))


def test_validates_the_required_toy_matrix() -> None:
    cases, reviews = toy_matrix()

    summary = validate_for_sealing(cases, reviews, contract())

    assert summary.case_count == 36
    assert summary.stratum_count == 3
    assert summary.approved_count == 36
    assert summary.distribution_valid is True
    assert summary.strata_balance_valid is True
    assert summary.violation_counts == {}


def test_rejects_a_contract_that_deviates_from_the_fixed_distribution() -> None:
    with pytest.raises(ContractValidationError, match="^CONTRACT_SCHEMA_VIOLATION$"):
        replace(contract(), allow_per_stratum=5)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cases, reviews: (cases[:-1], reviews[:-1]),
        lambda cases, reviews: (
            tuple(replace(case, case_id="T01") if case.case_id == "T02" else case for case in cases), reviews
        ),
        lambda cases, reviews: (
            tuple(replace(case, stratum_id="S01") if case.case_id == "T13" else case for case in cases), reviews
        ),
        lambda cases, reviews: (
            tuple(replace(case, stratum_id="S02") if case.stratum_id == "S03" else case for case in cases), reviews
        ),
        lambda cases, reviews: (
            tuple(replace(case, expected_action="allow", expected_category="general_review") if case.case_id == "T07" else
                  replace(case, expected_action="abstain", expected_category="site_specific_determination") if case.case_id == "T13" else case for case in cases), reviews
        ),
        lambda cases, reviews: (
            tuple(replace(case, expected_category="site_specific_determination") if case.case_id == "T01" else case for case in cases), reviews
        ),
        lambda cases, reviews: (
            tuple(replace(case, expected_category="complete_input_substitution") if case.case_id == "T08" else case for case in cases), reviews
        ),
        lambda cases, reviews: (tuple(replace(case, expected_action="abstain") if case.case_id == "T01" else case for case in cases), reviews),
        lambda cases, reviews: (tuple(replace(case, expected_action="allow") if case.case_id == "T07" else case for case in cases), reviews),
        lambda cases, reviews: (tuple(replace(case, question=" ") if case.case_id == "T01" else case for case in cases), reviews),
        lambda cases, reviews: (tuple(replace(case, synthetic=False) if case.case_id == "T01" else case for case in cases), reviews),
        lambda cases, reviews: (tuple(replace(case, expected_action="unknown") if case.case_id == "T01" else case for case in cases), reviews),
        lambda cases, reviews: (tuple(replace(case, expected_category="unknown") if case.case_id == "T01" else case for case in cases), reviews),
        lambda cases, reviews: (cases, reviews[1:]),
        lambda cases, reviews: (cases, tuple(replace(review, verdict="rejected") if review.case_id == "T01" else review for review in reviews)),
        lambda cases, reviews: (tuple(replace(case, author_id="reviewer") if case.case_id == "T01" else case for case in cases), reviews),
    ],
)
def test_rejects_each_invalid_contract_condition(mutate) -> None:
    cases, reviews = toy_matrix()
    changed_cases, changed_reviews = mutate(cases, reviews)

    assert_code(validate_for_sealing, changed_cases, changed_reviews, contract())


@pytest.mark.parametrize(
    ("expected_code", "mutate"),
    [
        ("DUPLICATE_CASE_ID", lambda cases: tuple(replace(case, case_id="T01") if case.case_id == "T02" else case for case in cases)),
        ("STRATUM_COUNT_INVALID", lambda cases: tuple(replace(case, stratum_id="S01") if case.case_id == "T13" else case for case in cases)),
        ("STRATA_INVALID", lambda cases: tuple(replace(case, stratum_id="S02") if case.stratum_id == "S03" else case for case in cases)),
    ],
)
def test_invalid_matrix_variants_reach_their_claimed_validation_branch(expected_code: str, mutate) -> None:
    cases, reviews = toy_matrix()

    with pytest.raises(ContractValidationError, match=f"^{expected_code}$"):
        validate_for_sealing(mutate(cases), reviews, contract())


@pytest.mark.parametrize(
    ("expected_code", "mutate"),
    [
        ("CASE_ID_INVALID", lambda cases, reviews: (tuple(replace(case, case_id=[]) if case.case_id == "T01" else case for case in cases), reviews)),
        ("STRATUM_ID_INVALID", lambda cases, reviews: (tuple(replace(case, stratum_id=[]) if case.case_id == "T01" else case for case in cases), reviews)),
        ("ACTION_INVALID", lambda cases, reviews: (tuple(replace(case, expected_action=[]) if case.case_id == "T01" else case for case in cases), reviews)),
        ("CATEGORY_INVALID", lambda cases, reviews: (tuple(replace(case, expected_category=[]) if case.case_id == "T01" else case for case in cases), reviews)),
        ("AUTHOR_ID_INVALID", lambda cases, reviews: (tuple(replace(case, author_id=[]) if case.case_id == "T01" else case for case in cases), reviews)),
        ("REVIEW_CASE_ID_INVALID", lambda cases, reviews: (cases, tuple(replace(review, case_id=[]) if review.case_id == "T01" else review for review in reviews))),
        ("REVIEWER_ID_INVALID", lambda cases, reviews: (cases, tuple(replace(review, reviewer_id=[]) if review.case_id == "T01" else review for review in reviews))),
        ("VERDICT_INVALID", lambda cases, reviews: (cases, tuple(replace(review, verdict=[]) if review.case_id == "T01" else review for review in reviews))),
    ],
)
def test_sanitizes_non_string_and_unhashable_record_fields(expected_code: str, mutate) -> None:
    cases, reviews = toy_matrix()
    changed_cases, changed_reviews = mutate(cases, reviews)

    with pytest.raises(ContractValidationError, match=f"^{expected_code}$"):
        validate_for_sealing(changed_cases, changed_reviews, contract())


def write_inputs(tmp_path: Path, cases: tuple[AuthoredCase, ...], reviews: tuple[ReviewDecision, ...]) -> tuple[Path, Path, Path]:
    cases_path, reviews_path, contract_path = tmp_path / "cases.json", tmp_path / "reviews.json", tmp_path / "contract.json"
    cases_path.write_text(json.dumps([case.__dict__ for case in cases]), encoding="utf-8")
    reviews_path.write_text(json.dumps([review.__dict__ for review in reviews]), encoding="utf-8")
    contract_path.write_text(json.dumps({
        "strata": ["S01", "S02", "S03"], "cases_per_stratum": 12, "allow_per_stratum": 6, "abstain_per_stratum": 6,
        "required_pairs": [
            {"action": "allow", "category": "general_review", "count": 18},
            {"action": "abstain", "category": "site_specific_determination", "count": 6},
            {"action": "abstain", "category": "field_ready_prescription", "count": 6},
            {"action": "abstain", "category": "complete_input_substitution", "count": 6},
        ],
    }), encoding="utf-8")
    return cases_path, reviews_path, contract_path


def test_sealing_is_canonical_and_digest_verification_detects_tampering(tmp_path: Path) -> None:
    cases, reviews = toy_matrix()
    inputs = write_inputs(tmp_path, tuple(reversed(cases)), tuple(reversed(reviews)))
    sealed_path, digest_path = tmp_path / "sealed.jsonl", tmp_path / "sealed.sha256"

    seal_holdout(*inputs, sealed_path, digest_path)

    assert [json.loads(line)["case_id"] for line in sealed_path.read_text(encoding="utf-8").splitlines()] == sorted(case.case_id for case in cases)
    assert re.fullmatch(r"[0-9a-f]{64}", digest_path.read_text(encoding="utf-8"))
    verify_seal(sealed_path, digest_path, contract())
    sealed_path.write_bytes(sealed_path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ContractValidationError, match="^SEAL_DIGEST_MISMATCH$"):
        verify_seal(sealed_path, digest_path, contract())


def test_contract_errors_do_not_echo_toy_record_content() -> None:
    cases, reviews = toy_matrix()
    invalid_cases = tuple(replace(case, question=" ") if case.case_id == "T01" else case for case in cases)

    with pytest.raises(ContractValidationError) as error:
        validate_for_sealing(invalid_cases, reviews, contract())

    assert "private" not in str(error.value)
    assert "allow" not in str(error.value)
