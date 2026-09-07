"""Private, policy-blind contracts for D2 claim-support depth review."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
from itertools import combinations
import json
from pathlib import Path
from typing import Sequence

from oilfield_chemical_copilot.evaluation.private_retrieval import PRIVATE_RETRIEVAL_ROOT
from oilfield_chemical_copilot.retrieval.models import RetrievalHit

class ClaimSupportDepthError(ValueError):
    """Raised with a sanitized D2 claim-support contract error."""


def _fail(code: str) -> None:
    raise ClaimSupportDepthError(code)


def _text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _private_path(path: Path) -> Path:
    try:
        path.resolve().relative_to(PRIVATE_RETRIEVAL_ROOT.resolve())
    except ValueError:
        _fail("PRIVATE_PATH_REQUIRED")
    return path


@dataclass(frozen=True)
class RequiredClaim:
    claim_id: str
    claim: str

    @classmethod
    def from_mapping(cls, record: object) -> "RequiredClaim":
        if not isinstance(record, dict) or set(record) != {"claim_id", "claim"}:
            _fail("CLAIM_FIXTURE_INVALID")
        return cls(
            claim_id=_text(record["claim_id"], "CLAIM_FIXTURE_INVALID"),
            claim=_text(record["claim"], "CLAIM_FIXTURE_INVALID"),
        )

    def to_mapping(self) -> dict[str, str]:
        return {"claim_id": self.claim_id, "claim": self.claim}


@dataclass(frozen=True)
class RequiredClaimFixture:
    question_id: str
    status: str
    required_claims: tuple[RequiredClaim, ...]

    @classmethod
    def from_mapping(cls, record: object) -> "RequiredClaimFixture":
        if not isinstance(record, dict) or set(record) != {"question_id", "status", "required_claims"}:
            _fail("CLAIM_FIXTURE_INVALID")
        status = _text(record["status"], "CLAIM_FIXTURE_INVALID")
        claims_raw = record["required_claims"]
        if status not in {"ready", "canonical_evidence_insufficient"} or not isinstance(claims_raw, list):
            _fail("CLAIM_FIXTURE_INVALID")
        claims = tuple(RequiredClaim.from_mapping(item) for item in claims_raw)
        if len({claim.claim_id for claim in claims}) != len(claims):
            _fail("CLAIM_FIXTURE_INVALID")
        if (status == "ready" and not claims) or (status == "canonical_evidence_insufficient" and claims):
            _fail("CLAIM_FIXTURE_INVALID")
        return cls(
            question_id=_text(record["question_id"], "CLAIM_FIXTURE_INVALID"),
            status=status,
            required_claims=claims,
        )

    def require_ready(self) -> "RequiredClaimFixture":
        if self.status != "ready":
            _fail("CLAIM_FIXTURE_NOT_READY")
        return self

    def to_mapping(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "status": self.status,
            "required_claims": [claim.to_mapping() for claim in self.required_claims],
        }


@dataclass(frozen=True)
class ClaimRankSupport:
    rank: int
    claim_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, record: object) -> "ClaimRankSupport":
        if not isinstance(record, dict) or set(record) != {"rank", "claim_ids"}:
            _fail("CLAIM_LABEL_INVALID")
        rank = record["rank"]
        claim_ids = record["claim_ids"]
        if type(rank) is not int or rank < 1 or not isinstance(claim_ids, list):
            _fail("CLAIM_LABEL_INVALID")
        values = tuple(_text(value, "CLAIM_LABEL_INVALID") for value in claim_ids)
        if len(set(values)) != len(values):
            _fail("CLAIM_LABEL_INVALID")
        return cls(rank=rank, claim_ids=values)


@dataclass(frozen=True)
class ClaimSupportLabel:
    question_id: str
    reviewer_id: str
    claim_support_by_rank: tuple[ClaimRankSupport, ...]

    @classmethod
    def from_mapping(cls, record: object) -> "ClaimSupportLabel":
        if not isinstance(record, dict) or set(record) != {
            "question_id",
            "reviewer_id",
            "claim_support_by_rank",
        }:
            _fail("CLAIM_LABEL_INVALID")
        entries = record["claim_support_by_rank"]
        if not isinstance(entries, list):
            _fail("CLAIM_LABEL_INVALID")
        supports = tuple(ClaimRankSupport.from_mapping(item) for item in entries)
        if len({item.rank for item in supports}) != len(supports):
            _fail("CLAIM_LABEL_INVALID")
        return cls(
            question_id=_text(record["question_id"], "CLAIM_LABEL_INVALID"),
            reviewer_id=_text(record["reviewer_id"], "CLAIM_LABEL_INVALID"),
            claim_support_by_rank=supports,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "reviewer_id": self.reviewer_id,
            "claim_support_by_rank": [
                {"rank": support.rank, "claim_ids": list(support.claim_ids)}
                for support in self.claim_support_by_rank
            ],
        }


@dataclass(frozen=True)
class ClaimSupportResult:
    first_sufficient_depth: int | None
    single_passage_sufficient: bool
    multiple_passages_required: bool
    unique_claim_contributor_ranks: tuple[int, ...]


@dataclass(frozen=True)
class ReviewPacketAnalysis:
    first_sufficient_depth: int | None
    single_passage_sufficient: bool
    multiple_passages_required: bool
    unique_claim_contributor_ranks: tuple[int, ...]
    minimum_claim_cover_ranks: tuple[int, ...] | None
    minimum_claim_cover_chars: int | None
    required_evidence_fits_context_budget: bool | None


@dataclass(frozen=True)
class MinimumClaimCover:
    ranks: tuple[int, ...] | None
    total_chars: int | None


def _canonical_bytes(fixtures: Sequence[RequiredClaimFixture]) -> bytes:
    return b"".join(
        json.dumps(fixture.to_mapping(), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for fixture in sorted(fixtures, key=lambda item: item.question_id)
    )


def load_required_claims(path: Path) -> tuple[RequiredClaimFixture, ...]:
    _private_path(path)
    try:
        records = tuple(
            RequiredClaimFixture.from_mapping(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("CLAIM_FIXTURE_LOAD_FAILURE")
    if not records or len({record.question_id for record in records}) != len(records):
        _fail("CLAIM_FIXTURE_INVALID")
    return records


def _validate_question_ids(
    fixtures: Sequence[RequiredClaimFixture], expected_question_ids: set[str]
) -> None:
    if {fixture.question_id for fixture in fixtures} != expected_question_ids:
        _fail("CLAIM_FIXTURE_QUESTION_SET_INVALID")


def seal_required_claims(
    *,
    draft_path: Path,
    sealed_path: Path,
    digest_path: Path,
    expected_question_ids: set[str],
) -> str:
    """Freeze required claims before policy-blind retrieval review begins."""
    fixtures = load_required_claims(draft_path)
    _validate_question_ids(fixtures, expected_question_ids)
    _private_path(sealed_path)
    _private_path(digest_path)
    payload = _canonical_bytes(fixtures)
    digest = sha256(payload).hexdigest()
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.write_bytes(payload)
    digest_path.write_text(digest + "\n", encoding="ascii")
    return digest


def load_sealed_required_claims(
    *,
    sealed_path: Path,
    digest_path: Path,
    expected_question_ids: set[str],
) -> tuple[RequiredClaimFixture, ...]:
    _private_path(sealed_path)
    _private_path(digest_path)
    try:
        payload = sealed_path.read_bytes()
        expected_digest = digest_path.read_text(encoding="ascii").strip()
    except OSError:
        _fail("CLAIM_FIXTURE_LOAD_FAILURE")
    if len(expected_digest) != 64 or not hmac.compare_digest(expected_digest, sha256(payload).hexdigest()):
        _fail("CLAIM_FIXTURE_DIGEST_MISMATCH")
    try:
        fixtures = tuple(
            RequiredClaimFixture.from_mapping(json.loads(line))
            for line in payload.decode("utf-8").splitlines()
        )
    except (UnicodeError, json.JSONDecodeError):
        _fail("CLAIM_FIXTURE_LOAD_FAILURE")
    if _canonical_bytes(fixtures) != payload:
        _fail("CLAIM_FIXTURE_CANONICALIZATION_MISMATCH")
    _validate_question_ids(fixtures, expected_question_ids)
    return fixtures


def _load_labels(path: Path) -> tuple[ClaimSupportLabel, ...]:
    _private_path(path)
    try:
        labels = tuple(
            ClaimSupportLabel.from_mapping(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("CLAIM_LABEL_LOAD_FAILURE")
    if not labels or len({label.question_id for label in labels}) != len(labels):
        _fail("CLAIM_LABEL_INVALID")
    return labels


def _canonical_label_bytes(labels: Sequence[ClaimSupportLabel]) -> bytes:
    return b"".join(
        json.dumps(label.to_mapping(), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for label in sorted(labels, key=lambda item: item.question_id)
    )


def _require_digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        _fail("D2_REVIEW_CONTRACT_INVALID")
    return value


def seal_policy_blind_labels(
    *,
    draft_path: Path,
    sealed_path: Path,
    digest_path: Path,
    contract_path: Path,
    expected_question_ids: set[str],
    claims_digest: str,
    packet_digest: str,
) -> str:
    """Freeze policy-blind labels and bind them to frozen reviewer inputs."""
    labels = _load_labels(draft_path)
    if {label.question_id for label in labels} != expected_question_ids:
        _fail("CLAIM_LABEL_QUESTION_SET_INVALID")
    _private_path(sealed_path)
    _private_path(digest_path)
    _private_path(contract_path)
    payload = _canonical_label_bytes(labels)
    digest = sha256(payload).hexdigest()
    contract = {
        "claims_digest": _require_digest(claims_digest),
        "packet_digest": _require_digest(packet_digest),
        "labels_digest": digest,
    }
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.write_bytes(payload)
    digest_path.write_text(digest + "\n", encoding="ascii")
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def load_sealed_policy_blind_labels(
    *,
    sealed_path: Path,
    digest_path: Path,
    contract_path: Path,
    expected_question_ids: set[str],
    claims_digest: str,
    packet_digest: str,
) -> tuple[ClaimSupportLabel, ...]:
    _private_path(sealed_path)
    _private_path(digest_path)
    _private_path(contract_path)
    try:
        payload = sealed_path.read_bytes()
        expected_digest = digest_path.read_text(encoding="ascii").strip()
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("CLAIM_LABEL_LOAD_FAILURE")
    actual = sha256(payload).hexdigest()
    if len(expected_digest) != 64 or not hmac.compare_digest(expected_digest, actual):
        _fail("CLAIM_LABEL_DIGEST_MISMATCH")
    expected_contract = {
        "claims_digest": _require_digest(claims_digest),
        "packet_digest": _require_digest(packet_digest),
        "labels_digest": actual,
    }
    if contract != expected_contract:
        _fail("D2_REVIEW_CONTRACT_MISMATCH")
    try:
        labels = tuple(
            ClaimSupportLabel.from_mapping(json.loads(line))
            for line in payload.decode("utf-8").splitlines()
        )
    except (UnicodeError, json.JSONDecodeError):
        _fail("CLAIM_LABEL_LOAD_FAILURE")
    if _canonical_label_bytes(labels) != payload or {label.question_id for label in labels} != expected_question_ids:
        _fail("CLAIM_LABEL_CANONICALIZATION_MISMATCH")
    return labels


def build_review_packet(
    *,
    question_id: str,
    question: str,
    fixture: RequiredClaimFixture,
    ranked_hits: Sequence[RetrievalHit],
) -> dict[str, object]:
    """Build reviewer input without delivery-policy, source, or score provenance."""
    fixture.require_ready()
    if question_id != fixture.question_id or not question.strip() or not ranked_hits:
        _fail("REVIEW_PACKET_INVALID")
    return {
        "question_id": question_id,
        "question": question.strip(),
        "required_claims": [claim.to_mapping() for claim in fixture.required_claims],
        "ranked_evidence": [
            {"rank": rank, "passage_text": hit.text}
            for rank, hit in enumerate(ranked_hits[:5], start=1)
        ],
    }


def analyze_claim_support(
    fixture: RequiredClaimFixture,
    label: ClaimSupportLabel,
    *,
    ranked_depth: int,
) -> ClaimSupportResult:
    """Derive prefix sufficiency only from a sealed claim fixture and blind labels."""
    fixture.require_ready()
    if label.question_id != fixture.question_id or ranked_depth < 1:
        _fail("CLAIM_LABEL_INVALID")
    required_ids = {claim.claim_id for claim in fixture.required_claims}
    by_rank: dict[int, set[str]] = {}
    for support in label.claim_support_by_rank:
        if support.rank > ranked_depth or not set(support.claim_ids).issubset(required_ids):
            _fail("CLAIM_LABEL_INVALID")
        by_rank[support.rank] = set(support.claim_ids)

    covered: set[str] = set()
    first_sufficient: int | None = None
    for rank in range(1, ranked_depth + 1):
        covered.update(by_rank.get(rank, set()))
        if covered == required_ids:
            first_sufficient = rank
            break
    single = any(claims == required_ids for claims in by_rank.values())
    contributor_ranks = tuple(
        rank
        for rank in sorted(by_rank)
        if any(
            claim_id not in set().union(
                *(claims for other_rank, claims in by_rank.items() if other_rank != rank)
            )
            for claim_id in by_rank[rank]
        )
    )
    return ClaimSupportResult(
        first_sufficient_depth=first_sufficient,
        single_passage_sufficient=single,
        multiple_passages_required=first_sufficient is not None and not single,
        unique_claim_contributor_ranks=contributor_ranks,
    )


def analyze_review_packet(
    *,
    fixture: RequiredClaimFixture,
    label: ClaimSupportLabel,
    packet: object,
    max_context_chars: int,
) -> ReviewPacketAnalysis:
    """Measure claim-support depth and capacity without selecting delivery context."""
    if max_context_chars < 1 or not isinstance(packet, dict):
        _fail("REVIEW_PACKET_INVALID")
    expected_claims = [claim.to_mapping() for claim in fixture.required_claims]
    ranked_evidence = packet.get("ranked_evidence")
    if (
        set(packet) != {"question_id", "question", "required_claims", "ranked_evidence"}
        or packet.get("question_id") != fixture.question_id
        or packet.get("required_claims") != expected_claims
        or not isinstance(ranked_evidence, list)
        or not ranked_evidence
    ):
        _fail("REVIEW_PACKET_INVALID")
    evidence_lengths: dict[int, int] = {}
    for expected_rank, item in enumerate(ranked_evidence, start=1):
        if (
            not isinstance(item, dict)
            or set(item) != {"rank", "passage_text"}
            or item.get("rank") != expected_rank
            or not isinstance(item.get("passage_text"), str)
            or not item["passage_text"]
        ):
            _fail("REVIEW_PACKET_INVALID")
        evidence_lengths[expected_rank] = len(item["passage_text"])
    support = analyze_claim_support(fixture, label, ranked_depth=len(ranked_evidence))
    if support.first_sufficient_depth is None:
        return ReviewPacketAnalysis(
            first_sufficient_depth=None,
            single_passage_sufficient=False,
            multiple_passages_required=False,
            unique_claim_contributor_ranks=support.unique_claim_contributor_ranks,
            minimum_claim_cover_ranks=None,
            minimum_claim_cover_chars=None,
            required_evidence_fits_context_budget=None,
        )

    required = {claim.claim_id for claim in fixture.required_claims}
    support_by_rank = {
        item.rank: set(item.claim_ids)
        for item in label.claim_support_by_rank
        if item.rank in evidence_lengths
    }
    covers: list[tuple[int, tuple[int, ...]]] = []
    ranks = sorted(evidence_lengths)
    for subset_size in range(1, len(ranks) + 1):
        for subset in combinations(ranks, subset_size):
            covered = set().union(*(support_by_rank.get(rank, set()) for rank in subset))
            if covered == required:
                covers.append((sum(evidence_lengths[rank] for rank in subset), subset))
        if covers:
            break
    if not covers:
        _fail("CLAIM_LABEL_INVALID")
    minimum_chars, minimum_ranks = min(covers, key=lambda item: (item[0], item[1]))
    return ReviewPacketAnalysis(
        first_sufficient_depth=support.first_sufficient_depth,
        single_passage_sufficient=support.single_passage_sufficient,
        multiple_passages_required=support.multiple_passages_required,
        unique_claim_contributor_ranks=support.unique_claim_contributor_ranks,
        minimum_claim_cover_ranks=minimum_ranks,
        minimum_claim_cover_chars=minimum_chars,
        required_evidence_fits_context_budget=minimum_chars <= max_context_chars,
    )


def delivered_claims_sufficient(
    fixture: RequiredClaimFixture,
    label: ClaimSupportLabel,
    *,
    retained_ranks: Sequence[int],
) -> bool:
    """Compare a delivery policy only after the reviewer labels have been sealed."""
    fixture.require_ready()
    if label.question_id != fixture.question_id or any(type(rank) is not int or rank < 1 for rank in retained_ranks):
        _fail("CLAIM_LABEL_INVALID")
    required = {claim.claim_id for claim in fixture.required_claims}
    covered = set().union(
        *(
            set(support.claim_ids)
            for support in label.claim_support_by_rank
            if support.rank in set(retained_ranks)
        )
    )
    if not covered.issubset(required):
        _fail("CLAIM_LABEL_INVALID")
    return covered == required


def minimum_claim_cover(
    fixture: RequiredClaimFixture,
    *,
    labels: Sequence[ClaimSupportLabel],
    passage_lengths: dict[int, int],
    candidate_depth: int,
) -> MinimumClaimCover:
    """Measure the oracle-smallest claim-covering subset; never use it as delivery."""
    fixture.require_ready()
    if candidate_depth < 1 or set(passage_lengths) != set(range(1, candidate_depth + 1)):
        _fail("CLAIM_COVER_INPUT_INVALID")
    if any(type(length) is not int or length < 1 for length in passage_lengths.values()):
        _fail("CLAIM_COVER_INPUT_INVALID")
    required = {claim.claim_id for claim in fixture.required_claims}
    support_by_rank: dict[int, set[str]] = {}
    for label in labels:
        if label.question_id != fixture.question_id:
            _fail("CLAIM_COVER_INPUT_INVALID")
        for entry in label.claim_support_by_rank:
            if not set(entry.claim_ids).issubset(required):
                _fail("CLAIM_COVER_INPUT_INVALID")
            if entry.rank > candidate_depth:
                continue
            support_by_rank.setdefault(entry.rank, set()).update(entry.claim_ids)
    candidates: list[tuple[int, tuple[int, ...]]] = []
    ranks = tuple(range(1, candidate_depth + 1))
    for subset_size in range(1, candidate_depth + 1):
        for subset in combinations(ranks, subset_size):
            covered = set().union(*(support_by_rank.get(rank, set()) for rank in subset))
            if covered == required:
                candidates.append((sum(passage_lengths[rank] for rank in subset), subset))
        if candidates:
            total_chars, selected = min(candidates, key=lambda item: (item[0], item[1]))
            return MinimumClaimCover(ranks=selected, total_chars=total_chars)
    return MinimumClaimCover(ranks=None, total_chars=None)
