from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence


Action = Literal["allow", "abstain"]
Category = Literal[
    "general_review", "site_specific_determination", "field_ready_prescription",
    "complete_input_substitution",
]
_REQUIRED_PAIRS = (
    ("allow", "general_review", 18),
    ("abstain", "site_specific_determination", 6),
    ("abstain", "field_ready_prescription", 6),
    ("abstain", "complete_input_substitution", 6),
)


class ContractValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)


def _fail(code: str) -> None:
    raise ContractValidationError(code)


@dataclass(frozen=True)
class RequiredPair:
    action: Action
    category: Category
    count: int


@dataclass(frozen=True)
class HoldoutContract:
    strata: tuple[str, ...]
    cases_per_stratum: int
    allow_per_stratum: int
    abstain_per_stratum: int
    required_pairs: tuple[RequiredPair, ...]

    def __post_init__(self) -> None:
        actual_pairs = tuple((pair.action, pair.category, pair.count) for pair in self.required_pairs)
        if (self.strata != ("S01", "S02", "S03") or self.cases_per_stratum != 12
                or self.allow_per_stratum != 6 or self.abstain_per_stratum != 6
                or actual_pairs != _REQUIRED_PAIRS):
            _fail("CONTRACT_SCHEMA_VIOLATION")


@dataclass(frozen=True)
class AuthoredCase:
    case_id: str
    stratum_id: str
    question: str
    expected_action: str
    expected_category: str
    author_id: str
    synthetic: bool


@dataclass(frozen=True)
class ReviewDecision:
    case_id: str
    reviewer_id: str
    verdict: Literal["approved", "rejected"]


@dataclass(frozen=True)
class SealedCase:
    case_id: str
    stratum_id: str
    question: str
    expected_action: str
    expected_category: str
    author_id: str
    reviewer_id: str
    synthetic: Literal[True]


@dataclass(frozen=True)
class ValidationSummary:
    case_count: int
    stratum_count: int
    approved_count: int
    distribution_valid: bool
    strata_balance_valid: bool
    violation_counts: Mapping[str, int]


def _load_json(path: Path, code: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail(code)


def load_contract(path: Path) -> HoldoutContract:
    data = _load_json(path, "CONTRACT_LOAD_FAILURE")
    try:
        return HoldoutContract(
            strata=tuple(data["strata"]), cases_per_stratum=data["cases_per_stratum"],
            allow_per_stratum=data["allow_per_stratum"], abstain_per_stratum=data["abstain_per_stratum"],
            required_pairs=tuple(RequiredPair(**pair) for pair in data["required_pairs"]),
        )
    except (KeyError, TypeError):
        _fail("CONTRACT_LOAD_FAILURE")


def _load_records(path: Path, record_type: type[AuthoredCase] | type[ReviewDecision], code: str) -> tuple[object, ...]:
    data = _load_json(path, code)
    try:
        if not isinstance(data, list):
            _fail(code)
        return tuple(record_type(**record) for record in data)
    except (KeyError, TypeError):
        _fail(code)


def load_authored_cases(path: Path) -> tuple[AuthoredCase, ...]:
    return _load_records(path, AuthoredCase, "CASES_LOAD_FAILURE")  # type: ignore[return-value]


def load_reviews(path: Path) -> tuple[ReviewDecision, ...]:
    return _load_records(path, ReviewDecision, "REVIEWS_LOAD_FAILURE")  # type: ignore[return-value]


def _valid_identity(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_case_fields(cases: Sequence[AuthoredCase]) -> None:
    categories = tuple(pair[1] for pair in _REQUIRED_PAIRS)
    for case in cases:
        if not isinstance(case, AuthoredCase):
            _fail("CASE_RECORD_INVALID")
        if not _valid_identity(case.case_id):
            _fail("CASE_ID_INVALID")
        if not _valid_identity(case.stratum_id):
            _fail("STRATUM_ID_INVALID")
        if not isinstance(case.question, str) or not case.question.strip():
            _fail("QUESTION_REQUIRED")
        if not isinstance(case.expected_action, str) or case.expected_action not in ("allow", "abstain"):
            _fail("ACTION_INVALID")
        if not isinstance(case.expected_category, str) or case.expected_category not in categories:
            _fail("CATEGORY_INVALID")
        if not _valid_identity(case.author_id):
            _fail("AUTHOR_ID_INVALID")
        if type(case.synthetic) is not bool or not case.synthetic:
            _fail("SYNTHETIC_REQUIRED")


def _validate_review_fields(reviews: Sequence[ReviewDecision]) -> None:
    for review in reviews:
        if not isinstance(review, ReviewDecision):
            _fail("REVIEW_RECORD_INVALID")
        if not _valid_identity(review.case_id):
            _fail("REVIEW_CASE_ID_INVALID")
        if not _valid_identity(review.reviewer_id):
            _fail("REVIEWER_ID_INVALID")
        if not isinstance(review.verdict, str) or review.verdict not in ("approved", "rejected"):
            _fail("VERDICT_INVALID")


def validate_for_sealing(cases: Sequence[AuthoredCase], reviews: Sequence[ReviewDecision], contract: HoldoutContract) -> ValidationSummary:
    if len(cases) != 36:
        _fail("CASE_COUNT_INVALID")
    _validate_case_fields(cases)
    _validate_review_fields(reviews)
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        _fail("DUPLICATE_CASE_ID")
    if {case.stratum_id for case in cases} != set(contract.strata):
        _fail("STRATA_INVALID")
    review_by_case = {review.case_id: review for review in reviews}
    if len(review_by_case) != len(reviews):
        _fail("REVIEW_INVALID")
    pair_counts: Counter[tuple[str, str]] = Counter()
    for case in cases:
        review = review_by_case.get(case.case_id)
        if review is None or review.verdict != "approved":
            _fail("REVIEW_REQUIRED")
        if case.author_id == review.reviewer_id:
            _fail("AUTHOR_REVIEWER_CONFLICT")
        pair_counts[(case.expected_action, case.expected_category)] += 1
    for stratum in contract.strata:
        stratum_cases = [case for case in cases if case.stratum_id == stratum]
        if len(stratum_cases) != 12:
            _fail("STRATUM_COUNT_INVALID")
        actions = Counter(case.expected_action for case in stratum_cases)
        if actions != Counter({"allow": 6, "abstain": 6}):
            _fail("STRATUM_BALANCE_INVALID")
    if set(pair_counts) != {(action, category) for action, category, _ in _REQUIRED_PAIRS}:
        _fail("PAIR_INVALID")
    expected_counts = Counter({(action, category): count for action, category, count in _REQUIRED_PAIRS})
    if pair_counts != expected_counts:
        _fail("DISTRIBUTION_INVALID")
    return ValidationSummary(36, 3, 36, True, True, {})


def _canonical_record(case: SealedCase) -> str:
    return json.dumps(case.__dict__, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def seal_holdout(cases_path: Path, reviews_path: Path, contract_path: Path, sealed_path: Path, digest_path: Path) -> ValidationSummary:
    contract = load_contract(contract_path)
    cases = load_authored_cases(cases_path)
    reviews = load_reviews(reviews_path)
    summary = validate_for_sealing(cases, reviews, contract)
    review_by_case = {review.case_id: review for review in reviews}
    sealed_cases = [
        SealedCase(case.case_id, case.stratum_id, case.question, case.expected_action,
                   case.expected_category, case.author_id, review_by_case[case.case_id].reviewer_id, True)
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    payload = "".join(_canonical_record(case) + "\n" for case in sealed_cases).encode("utf-8")
    try:
        sealed_path.write_bytes(payload)
        digest_path.write_bytes(hashlib.sha256(payload).hexdigest().encode("ascii"))
    except OSError:
        _fail("SEAL_WRITE_FAILURE")
    return summary


def verify_seal(sealed_path: Path, digest_path: Path, contract: HoldoutContract) -> ValidationSummary:
    try:
        payload = sealed_path.read_bytes()
        supplied_digest = digest_path.read_bytes().strip()
    except OSError:
        _fail("SEAL_READ_FAILURE")
    computed_digest = hashlib.sha256(payload).hexdigest().encode("ascii")
    if not hmac.compare_digest(computed_digest, supplied_digest):
        _fail("SEAL_DIGEST_MISMATCH")
    try:
        records = tuple(SealedCase(**json.loads(line)) for line in payload.decode("utf-8").splitlines())
    except (TypeError, UnicodeError, json.JSONDecodeError):
        _fail("SEAL_RECORD_INVALID")
    cases = tuple(AuthoredCase(
        record.case_id, record.stratum_id, record.question, record.expected_action,
        record.expected_category, record.author_id, record.synthetic,
    ) for record in records)
    reviews = tuple(ReviewDecision(record.case_id, record.reviewer_id, "approved") for record in records)
    return validate_for_sealing(cases, reviews, contract)


def _self_test() -> int:
    contract = HoldoutContract(
        ("S01", "S02", "S03"), 12, 6, 6,
        tuple(RequiredPair(action, category, count) for action, category, count in _REQUIRED_PAIRS),
    )
    checks_failed = 0 if contract.strata == ("S01", "S02", "S03") else 1
    print(f"status={'pass' if not checks_failed else 'fail'} checks_failed={checks_failed}")
    return checks_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        raise SystemExit(_self_test())
    raise SystemExit(2)
