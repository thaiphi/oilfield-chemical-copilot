"""Private, one-shot semantic-grounding evaluation for the production formatter."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Literal, Mapping, Sequence

from oilfield_chemical_copilot.rag.formatter import format_answer
from oilfield_chemical_copilot.rag.models import RagDraft, SourceEvidence


Category = Literal[
    "exact_value",
    "range_bound",
    "unit",
    "qualifier_condition",
    "conflicting_evidence",
    "no_established_threshold",
]
ExpectedOutcome = Literal["allow", "fallback"]
EvaluationScope = Literal["semantic-grounding-v4", "semantic-grounding-v5", "semantic-grounding-v6"]
_CATEGORIES: tuple[Category, ...] = (
    "exact_value",
    "range_bound",
    "unit",
    "qualifier_condition",
    "conflicting_evidence",
    "no_established_threshold",
)
_OUTCOMES: tuple[ExpectedOutcome, ...] = ("allow", "fallback")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{2,80}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_UNSAFE_REPORT_KEY_PARTS = (
    "answer",
    "credential",
    "error",
    "excerpt",
    "path",
    "question",
    "source",
    "url",
)


class EvaluationError(ValueError):
    """A sanitized semantic-grounding evaluator error."""


def _fail(code: str) -> None:
    raise EvaluationError(code)


@dataclass(frozen=True)
class SemanticCase:
    case_id: str
    category: str
    question: str
    excerpts: tuple[str, ...]
    answer: str
    expected_outcome: str
    failure_class: str
    author_id: str

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["excerpts"] = list(self.excerpts)
        return data


@dataclass(frozen=True)
class ReviewDecision:
    case_id: str
    reviewer_id: str
    verdict: Literal["approved", "rejected"]


@dataclass(frozen=True)
class ValidationSummary:
    case_count: int
    category_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]


@dataclass(frozen=True)
class CaseObservation:
    category: str
    expected_outcome: str
    observed_outcome: ExpectedOutcome
    failure_class: str | None


@dataclass(frozen=True)
class SemanticGroundingSummary:
    case_count: int
    pass_count: int
    grounded_cases_correctly_allowed: int
    unsupported_cases_correctly_rejected: int
    false_allows: int
    false_fallbacks: int
    category_counts: Mapping[str, Mapping[str, int]]
    failure_class_counts: Mapping[str, int]


@dataclass(frozen=True)
class Approval:
    scope: EvaluationScope
    approved: Literal[True]
    fixture_sha256: str
    formatter_sha256: str
    evaluator_sha256: str
    nonce: str

    @classmethod
    def for_current_artifacts(
        cls,
        sealed_path: Path,
        formatter_path: Path,
        *,
        scope: EvaluationScope = "semantic-grounding-v4",
    ) -> "Approval":
        return cls(
            scope=scope,
            approved=True,
            fixture_sha256=_sha256(sealed_path),
            formatter_sha256=_sha256(formatter_path),
            evaluator_sha256=_sha256(Path(__file__)),
            nonce=f"{scope}-approved",
        )

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightSummary:
    case_count: int
    seal_valid: bool
    no_prior_overlap: bool
    approval_digest_matches: bool
    attempt_available: bool


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        _fail("DIGEST_READ_FAILURE")


def _require_identifier(value: object, code: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(code)


def _require_text(value: object, code: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(code)


def _require_safe_failure_class(value: object) -> None:
    _require_identifier(value, "FAILURE_CLASS_INVALID")
    if _has_unsafe_report_fragment(value):
        _fail("FAILURE_CLASS_PRIVACY_INVALID")


def _has_unsafe_report_fragment(value: str) -> bool:
    return any(unsafe_part in value.casefold() for unsafe_part in _UNSAFE_REPORT_KEY_PARTS)


def _validate_case(case: SemanticCase) -> None:
    _require_identifier(case.case_id, "CASE_ID_INVALID")
    if case.category not in _CATEGORIES:
        _fail("CATEGORY_INVALID")
    _require_text(case.question, "QUESTION_REQUIRED")
    if not case.excerpts or any(not isinstance(item, str) or not item.strip() for item in case.excerpts):
        _fail("EXCERPTS_REQUIRED")
    _require_text(case.answer, "ANSWER_REQUIRED")
    if case.expected_outcome not in _OUTCOMES:
        _fail("EXPECTED_OUTCOME_INVALID")
    _require_safe_failure_class(case.failure_class)
    _require_identifier(case.author_id, "AUTHOR_ID_INVALID")


def _validate_reviews(cases: Sequence[SemanticCase], reviews: Sequence[ReviewDecision]) -> None:
    review_by_case: dict[str, ReviewDecision] = {}
    for review in reviews:
        _require_identifier(review.case_id, "REVIEW_CASE_ID_INVALID")
        _require_identifier(review.reviewer_id, "REVIEWER_ID_INVALID")
        if review.verdict not in ("approved", "rejected") or review.case_id in review_by_case:
            _fail("REVIEW_INVALID")
        review_by_case[review.case_id] = review
    if set(review_by_case) != {case.case_id for case in cases}:
        _fail("REVIEW_REQUIRED")
    for case in cases:
        review = review_by_case[case.case_id]
        if review.verdict != "approved":
            _fail("REVIEW_REQUIRED")
        if review.reviewer_id == case.author_id:
            _fail("AUTHOR_REVIEWER_CONFLICT")


def validate_cases(cases: Sequence[SemanticCase], reviews: Sequence[ReviewDecision]) -> ValidationSummary:
    if len(cases) != 36:
        _fail("CASE_COUNT_INVALID")
    for case in cases:
        _validate_case(case)
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        _fail("DUPLICATE_CASE_ID")
    _validate_reviews(cases, reviews)
    category_counts = Counter(case.category for case in cases)
    if set(category_counts) != set(_CATEGORIES) or any(count != 6 for count in category_counts.values()):
        _fail("CATEGORY_BALANCE_INVALID")
    outcome_counts = Counter(case.expected_outcome for case in cases)
    if outcome_counts != Counter({"allow": 18, "fallback": 18}):
        _fail("OUTCOME_BALANCE_INVALID")
    for category in _CATEGORIES:
        category_outcomes = Counter(case.expected_outcome for case in cases if case.category == category)
        if category_outcomes != Counter({"allow": 3, "fallback": 3}):
            _fail("CATEGORY_OUTCOME_BALANCE_INVALID")
    return ValidationSummary(36, dict(sorted(category_counts.items())), dict(sorted(outcome_counts.items())))


def _case_from_mapping(data: object) -> SemanticCase:
    if not isinstance(data, dict) or set(data) != {
        "answer", "author_id", "case_id", "category", "expected_outcome", "excerpts", "failure_class", "question"
    }:
        _fail("CASE_RECORD_INVALID")
    try:
        excerpts = data["excerpts"]
        if not isinstance(excerpts, list):
            _fail("CASE_RECORD_INVALID")
        return SemanticCase(
            case_id=data["case_id"],
            category=data["category"],
            question=data["question"],
            excerpts=tuple(excerpts),
            answer=data["answer"],
            expected_outcome=data["expected_outcome"],
            failure_class=data["failure_class"],
            author_id=data["author_id"],
        )
    except (KeyError, TypeError):
        _fail("CASE_RECORD_INVALID")


def _review_from_mapping(data: object) -> ReviewDecision:
    if not isinstance(data, dict) or set(data) != {"case_id", "reviewer_id", "verdict"}:
        _fail("REVIEW_RECORD_INVALID")
    try:
        return ReviewDecision(**data)
    except TypeError:
        _fail("REVIEW_RECORD_INVALID")


def _load_jsonl(path: Path, parser: object, code: str) -> tuple[object, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            _fail(code)
        records = tuple(parser(json.loads(line)) for line in lines)  # type: ignore[operator]
        return records
    except EvaluationError:
        raise
    except (OSError, TypeError, UnicodeError, json.JSONDecodeError):
        _fail(code)


def load_cases(path: Path) -> tuple[SemanticCase, ...]:
    return _load_jsonl(path, _case_from_mapping, "CASES_LOAD_FAILURE")  # type: ignore[return-value]


def load_reviews(path: Path) -> tuple[ReviewDecision, ...]:
    return _load_jsonl(path, _review_from_mapping, "REVIEWS_LOAD_FAILURE")  # type: ignore[return-value]


def _canonical_payload(cases: Sequence[SemanticCase], reviews: Sequence[ReviewDecision]) -> bytes:
    reviewer_by_case = {review.case_id: review.reviewer_id for review in reviews}
    records = []
    for case in sorted(cases, key=lambda item: item.case_id):
        record = case.to_mapping()
        record["reviewer_id"] = reviewer_by_case[case.case_id]
        records.append(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return ("\n".join(records) + "\n").encode("utf-8")


def seal_cases(
    cases_path: Path, reviews_path: Path, sealed_path: Path, digest_path: Path
) -> ValidationSummary:
    cases = load_cases(cases_path)
    reviews = load_reviews(reviews_path)
    summary = validate_cases(cases, reviews)
    try:
        sealed_path.parent.mkdir(parents=True, exist_ok=True)
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_payload(cases, reviews)
        sealed_path.write_bytes(payload)
        digest_path.write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")
    except OSError:
        _fail("SEAL_WRITE_FAILURE")
    return summary


def _load_sealed_cases(sealed_path: Path, digest_path: Path) -> tuple[SemanticCase, ...]:
    try:
        payload = sealed_path.read_bytes()
        supplied = digest_path.read_text(encoding="ascii").strip()
    except OSError:
        _fail("SEAL_READ_FAILURE")
    computed = hashlib.sha256(payload).hexdigest()
    if _DIGEST.fullmatch(supplied) is None or not hmac.compare_digest(computed, supplied):
        _fail("SEAL_DIGEST_MISMATCH")
    try:
        records = tuple(json.loads(line) for line in payload.decode("utf-8").splitlines())
        cases = tuple(_case_from_mapping({key: value for key, value in record.items() if key != "reviewer_id"}) for record in records)
        reviews = tuple(ReviewDecision(record["case_id"], record["reviewer_id"], "approved") for record in records)
        validate_cases(cases, reviews)
        return cases
    except EvaluationError:
        raise
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError):
        _fail("SEAL_RECORD_INVALID")


def _question_digest(question: str) -> str:
    normalized = " ".join(question.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def verify_no_prior_overlap(cases: Sequence[SemanticCase], prior_paths: Sequence[Path]) -> None:
    prior_digests: set[str] = set()
    try:
        for path in prior_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if isinstance(record, dict) and isinstance(record.get("question"), str):
                    prior_digests.add(_question_digest(record["question"]))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("PRIOR_CASE_LOAD_FAILURE")
    if any(_question_digest(case.question) in prior_digests for case in cases):
        _fail("PRIOR_CASE_OVERLAP")


def _sources_for(case: SemanticCase) -> list[SourceEvidence]:
    return [
        SourceEvidence(
            source_id=f"Evidence {index}",
            chunk_id=f"semantic-{case.case_id}-{index}",
            source_file=f"synthetic/{case.category}.md",
            page_or_sheet="synthetic",
            topic=case.category,
            excerpt=excerpt,
            score=0.9,
        )
        for index, excerpt in enumerate(case.excerpts, start=1)
    ]


def evaluate_case(case: SemanticCase) -> CaseObservation:
    sources = _sources_for(case)
    draft = RagDraft(
        answer=case.answer,
        why_this_matters="The evidence must preserve the technical claim's meaning.",
        cited_source_ids=[source.source_id for source in sources],
        recommended_next_checks=[
            "Review the cited method.",
            "Confirm the operating context.",
            "Obtain qualified engineering review before any field treatment change.",
        ],
        limitations="Synthetic semantic-grounding evaluation only.",
    )
    answer = format_answer(draft, sources, question=case.question)
    observed: ExpectedOutcome = "fallback" if answer.weak_evidence else "allow"
    return CaseObservation(
        category=case.category,
        expected_outcome=case.expected_outcome,
        observed_outcome=observed,
        failure_class=None if observed == case.expected_outcome else case.failure_class,
    )


def aggregate(observations: Sequence[CaseObservation]) -> SemanticGroundingSummary:
    if len(observations) != 36:
        _fail("OBSERVATION_COUNT_INVALID")
    category_counts: dict[str, dict[str, int]] = {
        category: {"total": 0, "pass": 0, "false_allow": 0, "false_fallback": 0}
        for category in _CATEGORIES
    }
    failure_classes: Counter[str] = Counter()
    correct_allow = correct_fallback = false_allows = false_fallbacks = 0
    for item in observations:
        if item.category not in category_counts or item.expected_outcome not in _OUTCOMES or item.observed_outcome not in _OUTCOMES:
            _fail("OBSERVATION_INVALID")
        category = category_counts[item.category]
        category["total"] += 1
        if item.expected_outcome == item.observed_outcome:
            category["pass"] += 1
            correct_allow += int(item.expected_outcome == "allow")
            correct_fallback += int(item.expected_outcome == "fallback")
            continue
        if item.expected_outcome == "fallback":
            category["false_allow"] += 1
            false_allows += 1
        else:
            category["false_fallback"] += 1
            false_fallbacks += 1
        if item.failure_class is None:
            _fail("FAILURE_CLASS_REQUIRED")
        failure_classes[item.failure_class] += 1
    return SemanticGroundingSummary(
        case_count=len(observations),
        pass_count=correct_allow + correct_fallback,
        grounded_cases_correctly_allowed=correct_allow,
        unsupported_cases_correctly_rejected=correct_fallback,
        false_allows=false_allows,
        false_fallbacks=false_fallbacks,
        category_counts=category_counts,
        failure_class_counts=dict(sorted(failure_classes.items())),
    )


def _load_approval(path: Path) -> Approval:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {
            "approved", "evaluator_sha256", "fixture_sha256", "formatter_sha256", "nonce", "scope"
        }:
            _fail("APPROVAL_INVALID")
        approval = Approval(**data)
    except EvaluationError:
        raise
    except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        _fail("APPROVAL_INVALID")
    if (
        approval.scope not in ("semantic-grounding-v4", "semantic-grounding-v5", "semantic-grounding-v6")
        or approval.approved is not True
        or not isinstance(approval.nonce, str)
        or not approval.nonce.strip()
        or any(_DIGEST.fullmatch(value) is None for value in (
            approval.fixture_sha256,
            approval.formatter_sha256,
            approval.evaluator_sha256,
        ))
    ):
        _fail("APPROVAL_INVALID")
    return approval


def _approval_matches(approval: Approval, sealed_path: Path, formatter_path: Path) -> bool:
    return all(
        hmac.compare_digest(actual, expected)
        for actual, expected in zip(
            (_sha256(sealed_path), _sha256(formatter_path), _sha256(Path(__file__))),
            (approval.fixture_sha256, approval.formatter_sha256, approval.evaluator_sha256),
        )
    )


def _require_under_private_root(path: Path, private_root: Path) -> None:
    try:
        path.resolve().relative_to(private_root.resolve())
    except ValueError:
        _fail("PRIVATE_PATH_REQUIRED")


def _validate_run_paths(
    private_root: Path,
    private_paths: Sequence[Path],
    report_path: Path,
) -> None:
    for path in private_paths:
        _require_under_private_root(path, private_root)
    try:
        report_path.resolve().relative_to(private_root.resolve())
    except ValueError:
        return
    _fail("PUBLIC_REPORT_REQUIRED")


def preflight(
    sealed_path: Path,
    digest_path: Path,
    approval_path: Path,
    state_path: Path,
    formatter_path: Path,
    cases: Sequence[SemanticCase],
    prior_paths: Sequence[Path],
    private_root: Path,
    private_result_path: Path,
    report_path: Path,
    *,
    expected_scope: EvaluationScope = "semantic-grounding-v4",
) -> PreflightSummary:
    _validate_run_paths(
        private_root,
        (sealed_path, digest_path, approval_path, state_path, private_result_path),
        report_path,
    )
    sealed_cases = _load_sealed_cases(sealed_path, digest_path)
    verify_no_prior_overlap(cases, prior_paths)
    if tuple(sorted(cases, key=lambda item: item.case_id)) != sealed_cases:
        _fail("SEALED_CASES_MISMATCH")
    approval = _load_approval(approval_path)
    if approval.scope != expected_scope:
        _fail("APPROVAL_SCOPE_MISMATCH")
    if not _approval_matches(approval, sealed_path, formatter_path):
        _fail("APPROVAL_DIGEST_MISMATCH")
    if state_path.exists():
        _fail("ATTEMPT_UNAVAILABLE")
    return PreflightSummary(
        case_count=len(sealed_cases),
        seal_valid=True,
        no_prior_overlap=True,
        approval_digest_matches=_approval_matches(approval, sealed_path, formatter_path),
        attempt_available=not state_path.exists(),
    )


def _consume(state_path: Path, approval: Approval) -> None:
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"state": "consumed", "nonce": approval.nonce}, sort_keys=True) + "\n")
    except FileExistsError:
        _fail("ATTEMPT_UNAVAILABLE")
    except OSError:
        _fail("LOCK_CREATE_FAILURE")


def _write_private_diagnostics(path: Path, observations: Sequence[CaseObservation]) -> None:
    payload = [asdict(item) for item in observations]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        _fail("PRIVATE_DIAGNOSTIC_WRITE_FAILURE")


def _validate_aggregate_payload(payload: object) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str) or _has_unsafe_report_fragment(key):
                _fail("AGGREGATE_REPORT_PRIVACY_VIOLATION")
            _validate_aggregate_payload(value)
        return
    if isinstance(payload, (list, tuple)):
        for value in payload:
            _validate_aggregate_payload(value)
        return
    if isinstance(payload, str):
        if _has_unsafe_report_fragment(payload):
            _fail("AGGREGATE_REPORT_PRIVACY_VIOLATION")
        return
    if isinstance(payload, (bool, int, float)):
        return
    _fail("AGGREGATE_REPORT_PRIVACY_VIOLATION")


def _write_aggregate_report(path: Path, summary: SemanticGroundingSummary) -> None:
    payload = {
        "status": "pass" if summary.pass_count == summary.case_count else "fail",
        "counts": {
            "case_count": summary.case_count,
            "pass_count": summary.pass_count,
            "pass_rate_percent": round(100 * summary.pass_count / summary.case_count, 1),
            "grounded_cases_correctly_allowed": summary.grounded_cases_correctly_allowed,
            "unsupported_cases_correctly_rejected": summary.unsupported_cases_correctly_rejected,
            "false_allows": summary.false_allows,
            "false_fallbacks": summary.false_fallbacks,
        },
        "categories": summary.category_counts,
        "failure_classes": summary.failure_class_counts,
        "gates": {
            "actual_formatter_called": True,
            "fixture_sealed": True,
            "approval_digest_matches": True,
            "one_shot_consumed": True,
        },
    }
    _validate_aggregate_payload(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        _fail("AGGREGATE_REPORT_WRITE_FAILURE")


def evaluate_once(
    sealed_path: Path,
    digest_path: Path,
    approval_path: Path,
    state_path: Path,
    private_result_path: Path,
    report_path: Path,
    formatter_path: Path = Path("src/oilfield_chemical_copilot/rag/formatter.py"),
    *,
    cases: Sequence[SemanticCase],
    prior_paths: Sequence[Path],
    private_root: Path = Path(".private"),
    expected_scope: EvaluationScope = "semantic-grounding-v4",
) -> SemanticGroundingSummary:
    preflight(
        sealed_path,
        digest_path,
        approval_path,
        state_path,
        formatter_path,
        cases,
        prior_paths,
        private_root,
        private_result_path,
        report_path,
        expected_scope=expected_scope,
    )
    cases = _load_sealed_cases(sealed_path, digest_path)
    approval = _load_approval(approval_path)
    if approval.scope != expected_scope:
        _fail("APPROVAL_SCOPE_MISMATCH")
    if not _approval_matches(approval, sealed_path, formatter_path):
        _fail("APPROVAL_DIGEST_MISMATCH")
    _consume(state_path, approval)
    observations = tuple(evaluate_case(case) for case in cases)
    summary = aggregate(observations)
    _write_private_diagnostics(private_result_path, observations)
    _write_aggregate_report(report_path, summary)
    return summary
