"""Sealed local-case contract for Module 4 evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_ROOT = PROJECT_ROOT / ".private" / "evaluation" / "module4_handouts"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{2,80}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_FIELDS = {
    "case_id",
    "question",
    "topic",
    "expected_chunk_ids",
    "expect_citations",
    "expect_abstention",
    "reviewed",
}


class Module4ContractError(ValueError):
    """A sanitized Module 4 contract error."""


@dataclass(frozen=True)
class Module4Case:
    case_id: str
    question: str
    topic: str
    expected_chunk_ids: tuple[str, ...]
    expect_citations: bool
    expect_abstention: bool
    reviewed: bool

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["expected_chunk_ids"] = list(self.expected_chunk_ids)
        return data


def _fail(code: str) -> None:
    raise Module4ContractError(code)


def _require_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        _fail("PRIVATE_BOUNDARY_VIOLATION")
    return resolved


def _require_identifier(value: object, code: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(code)
    return value


def _case_from_mapping(record: object) -> Module4Case:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        _fail("CASE_RECORD_INVALID")
    expected = record["expected_chunk_ids"]
    if not isinstance(expected, list) or any(
        not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in expected
    ):
        _fail("EXPECTED_CHUNK_IDS_INVALID")
    if len(set(expected)) != len(expected):
        _fail("EXPECTED_CHUNK_IDS_INVALID")
    question = record["question"]
    if not isinstance(question, str) or not question.strip():
        _fail("QUESTION_INVALID")
    expectation_values = (
        record["expect_citations"],
        record["expect_abstention"],
        record["reviewed"],
    )
    if any(type(value) is not bool for value in expectation_values):
        _fail("CASE_RECORD_INVALID")
    expect_citations, expect_abstention, reviewed = expectation_values
    if reviewed is not True:
        _fail("REVIEW_REQUIRED")
    if expect_abstention:
        if expected or expect_citations:
            _fail("CASE_EXPECTATION_INVALID")
    elif not expected or expect_citations is not True:
        _fail("CASE_EXPECTATION_INVALID")
    return Module4Case(
        _require_identifier(record["case_id"], "CASE_ID_INVALID"),
        question,
        _require_identifier(record["topic"], "TOPIC_INVALID"),
        tuple(expected),
        expect_citations,
        expect_abstention,
        reviewed,
    )


def _load_cases(path: Path) -> tuple[Module4Case, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        _fail("CASES_LOAD_FAILURE")
    if not lines:
        _fail("CASES_EMPTY")
    cases: list[Module4Case] = []
    for line in lines:
        try:
            cases.append(_case_from_mapping(json.loads(line)))
        except json.JSONDecodeError:
            _fail("CASE_RECORD_INVALID")
    if len({case.case_id for case in cases}) != len(cases):
        _fail("DUPLICATE_CASE_ID")
    return tuple(cases)


def _canonical_payload(cases: tuple[Module4Case, ...]) -> bytes:
    return b"".join(
        json.dumps(case.to_mapping(), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for case in cases
    )


def seal_cases(draft_path: Path, sealed_path: Path, digest_path: Path) -> str:
    draft = _require_under(draft_path, PRIVATE_ROOT / "dataset")
    sealed = _require_under(sealed_path, PRIVATE_ROOT / "sealed")
    digest = _require_under(digest_path, PRIVATE_ROOT / "sealed")
    cases = _load_cases(draft)
    payload = _canonical_payload(cases)
    dataset_sha256 = hashlib.sha256(payload).hexdigest()
    sealed.parent.mkdir(parents=True, exist_ok=True)
    sealed.write_bytes(payload)
    digest.write_text(dataset_sha256 + "\n", encoding="ascii")
    return dataset_sha256


def verify_seal(sealed_path: Path, digest_path: Path) -> tuple[Module4Case, ...]:
    sealed = _require_under(sealed_path, PRIVATE_ROOT / "sealed")
    digest = _require_under(digest_path, PRIVATE_ROOT / "sealed")
    try:
        expected_digest = digest.read_text(encoding="ascii").strip()
        payload = sealed.read_bytes()
    except OSError:
        _fail("SEAL_LOAD_FAILURE")
    if _SHA256.fullmatch(expected_digest) is None:
        _fail("SEAL_DIGEST_INVALID")
    actual_digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        _fail("SEAL_DIGEST_MISMATCH")
    cases = _load_cases(sealed)
    if not hmac.compare_digest(_canonical_payload(cases), payload):
        _fail("SEAL_CANONICALIZATION_MISMATCH")
    return cases


def consume_one_shot(state_path: Path, dataset_sha256: str) -> None:
    state = _require_under(state_path, PRIVATE_ROOT / "results")
    if _SHA256.fullmatch(dataset_sha256) is None:
        _fail("DATASET_DIGEST_INVALID")
    if state.exists():
        _fail("ATTEMPT_UNAVAILABLE")
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps({"dataset_sha256": dataset_sha256}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
