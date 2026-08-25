"""Private, no-oracle retrieval evaluation contracts and experiment helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import hmac
import json
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

from oilfield_chemical_copilot.retrieval.hybrid import fuse_ranked_hits
from oilfield_chemical_copilot.retrieval.keyword import KeywordSearchIndex
from oilfield_chemical_copilot.retrieval.models import RetrievalHit
from oilfield_chemical_copilot.retrieval.pipeline import _fit_context_budget
from oilfield_chemical_copilot.retrieval.vector import VectorRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_RETRIEVAL_ROOT = PROJECT_ROOT / ".private" / "retrieval-evaluation" / "v1"
_CASE_FIELDS = {
    "question_id",
    "question",
    "question_class",
    "expected_source",
    "expected_topic",
    "expected_evidence_exists",
    "difficulty",
    "notes",
    "acceptable_sources",
    "expected_locator",
    "expected_source_role",
    "alias_or_terminology_case",
    "split",
    "author_id",
}
_REVIEW_FIELDS = {"question_id", "reviewer_id", "verdict"}
_QUESTION_CLASSES = {
    "fundamental_definition",
    "short_field_terminology",
    "paraphrased_question",
    "descriptive_context",
    "supporting_document",
    "no_answer_insufficient_evidence",
}
_SOURCE_ROLES = {"foundational", "supporting", "other"}
_SPLITS = {"development", "holdout"}
_DIFFICULTIES = {"easy", "moderate", "hard", "ambiguous"}
_BOOST_FIELDS = {"content", "topic", "source_file"}


class PrivateRetrievalError(ValueError):
    """Raised with a sanitized private retrieval evaluation error code."""


def _fail(code: str) -> None:
    raise PrivateRetrievalError(code)


def _require_private_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PRIVATE_RETRIEVAL_ROOT.resolve())
    except ValueError:
        _fail("PRIVATE_PATH_REQUIRED")
    return resolved


def _require_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code)
    return value.strip()


def _source_key(value: str) -> str:
    return value.replace("\\", "/").strip().casefold()


@dataclass(frozen=True)
class PrivateRetrievalCase:
    question_id: str
    question: str
    question_class: str
    expected_source: str | None
    expected_topic: str | None
    expected_evidence_exists: bool
    difficulty: str
    notes: str
    acceptable_sources: tuple[str, ...]
    expected_locator: str | None
    expected_source_role: str | None
    alias_or_terminology_case: bool
    split: str
    author_id: str

    @classmethod
    def from_mapping(cls, record: object) -> "PrivateRetrievalCase":
        if not isinstance(record, dict) or set(record) != _CASE_FIELDS:
            _fail("CASE_RECORD_INVALID")
        question_class = _require_text(record["question_class"], "QUESTION_CLASS_INVALID")
        split = _require_text(record["split"], "SPLIT_INVALID")
        difficulty = _require_text(record["difficulty"], "DIFFICULTY_INVALID")
        if question_class not in _QUESTION_CLASSES or split not in _SPLITS or difficulty not in _DIFFICULTIES:
            _fail("CASE_RECORD_INVALID")
        evidence_exists = record["expected_evidence_exists"]
        alias_case = record["alias_or_terminology_case"]
        if type(evidence_exists) is not bool or type(alias_case) is not bool:
            _fail("CASE_RECORD_INVALID")
        acceptable = record["acceptable_sources"]
        if not isinstance(acceptable, list) or any(
            not isinstance(value, str) or not value.strip() for value in acceptable
        ) or len(set(_source_key(value) for value in acceptable)) != len(acceptable):
            _fail("ACCEPTABLE_SOURCES_INVALID")
        expected_source = record["expected_source"]
        expected_topic = record["expected_topic"]
        locator = record["expected_locator"]
        source_role = record["expected_source_role"]
        optional_values = (expected_source, expected_topic, locator, source_role)
        if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in optional_values):
            _fail("CASE_RECORD_INVALID")
        if evidence_exists:
            if (
                question_class == "no_answer_insufficient_evidence"
                or not isinstance(expected_source, str)
                or not isinstance(expected_topic, str)
                or not isinstance(locator, str)
                or source_role not in _SOURCE_ROLES
                or not acceptable
                or _source_key(expected_source) not in {_source_key(value) for value in acceptable}
            ):
                _fail("CASE_EXPECTATION_INVALID")
        elif (
            question_class != "no_answer_insufficient_evidence"
            or expected_source is not None
            or expected_topic is not None
            or locator is not None
            or source_role is not None
            or acceptable
        ):
            _fail("CASE_EXPECTATION_INVALID")
        return cls(
            question_id=_require_text(record["question_id"], "QUESTION_ID_INVALID"),
            question=_require_text(record["question"], "QUESTION_INVALID"),
            question_class=question_class,
            expected_source=expected_source,
            expected_topic=expected_topic,
            expected_evidence_exists=evidence_exists,
            difficulty=difficulty,
            notes=_require_text(record["notes"], "NOTES_INVALID"),
            acceptable_sources=tuple(acceptable),
            expected_locator=locator,
            expected_source_role=source_role,
            alias_or_terminology_case=alias_case,
            split=split,
            author_id=_require_text(record["author_id"], "AUTHOR_ID_INVALID"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            **asdict(self),
            "acceptable_sources": list(self.acceptable_sources),
        }


@dataclass(frozen=True)
class ReviewDecision:
    question_id: str
    reviewer_id: str
    verdict: str

    @classmethod
    def from_mapping(cls, record: object) -> "ReviewDecision":
        if not isinstance(record, dict) or set(record) != _REVIEW_FIELDS:
            _fail("REVIEW_RECORD_INVALID")
        verdict = _require_text(record["verdict"], "REVIEW_RECORD_INVALID")
        if verdict not in {"approved", "rejected"}:
            _fail("REVIEW_RECORD_INVALID")
        return cls(
            question_id=_require_text(record["question_id"], "REVIEW_RECORD_INVALID"),
            reviewer_id=_require_text(record["reviewer_id"], "REVIEW_RECORD_INVALID"),
            verdict=verdict,
        )


def _load_jsonl(path: Path, parser: Callable[[object], object], error_code: str) -> tuple[object, ...]:
    _require_private_path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            _fail(error_code)
        return tuple(parser(json.loads(line)) for line in lines)
    except PrivateRetrievalError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        _fail(error_code)


def load_cases(path: Path) -> tuple[PrivateRetrievalCase, ...]:
    return _load_jsonl(path, PrivateRetrievalCase.from_mapping, "CASES_LOAD_FAILURE")  # type: ignore[return-value]


def load_reviews(path: Path) -> tuple[ReviewDecision, ...]:
    return _load_jsonl(path, ReviewDecision.from_mapping, "REVIEWS_LOAD_FAILURE")  # type: ignore[return-value]


def validate_fixture(cases: Sequence[PrivateRetrievalCase], reviews: Sequence[ReviewDecision]) -> None:
    if len(cases) != 40 or len({case.question_id for case in cases}) != len(cases):
        _fail("CASE_COUNT_INVALID")
    development = [case for case in cases if case.split == "development"]
    holdout = [case for case in cases if case.split == "holdout"]
    if len(development) != 28 or len(holdout) != 12:
        _fail("SPLIT_BALANCE_INVALID")
    if sum(case.expected_evidence_exists for case in cases) != 36:
        _fail("EVIDENCE_BALANCE_INVALID")
    if sum(not case.expected_evidence_exists for case in cases) != 4:
        _fail("EVIDENCE_BALANCE_INVALID")
    foundational = sum(case.expected_source_role == "foundational" for case in cases)
    supporting = sum(case.expected_source_role == "supporting" for case in cases)
    if foundational < 14 or supporting < 12:
        _fail("SOURCE_ROLE_BALANCE_INVALID")
    review_by_case: dict[str, ReviewDecision] = {}
    for review in reviews:
        if review.question_id in review_by_case:
            _fail("REVIEW_RECORD_INVALID")
        review_by_case[review.question_id] = review
    if set(review_by_case) != {case.question_id for case in cases}:
        _fail("REVIEW_REQUIRED")
    for case in cases:
        review = review_by_case[case.question_id]
        if review.verdict != "approved" or review.reviewer_id == case.author_id:
            _fail("REVIEW_REQUIRED")


def _canonical_bytes(cases: Iterable[PrivateRetrievalCase]) -> bytes:
    return b"".join(
        json.dumps(case.to_mapping(), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for case in sorted(cases, key=lambda case: case.question_id)
    )


def _write_sealed(path: Path, digest_path: Path, cases: Sequence[PrivateRetrievalCase]) -> str:
    _require_private_path(path)
    _require_private_path(digest_path)
    payload = _canonical_bytes(cases)
    digest = sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest_path.write_text(digest + "\n", encoding="ascii")
    return digest


def seal_fixture(
    *,
    draft_path: Path,
    review_path: Path,
    development_path: Path,
    holdout_path: Path,
    development_digest_path: Path,
    holdout_digest_path: Path,
) -> dict[str, str]:
    cases = load_cases(draft_path)
    reviews = load_reviews(review_path)
    validate_fixture(cases, reviews)
    return {
        "development": _write_sealed(
            development_path,
            development_digest_path,
            [case for case in cases if case.split == "development"],
        ),
        "holdout": _write_sealed(
            holdout_path,
            holdout_digest_path,
            [case for case in cases if case.split == "holdout"],
        ),
    }


def _load_sealed(path: Path, digest_path: Path) -> tuple[PrivateRetrievalCase, ...]:
    _require_private_path(path)
    _require_private_path(digest_path)
    try:
        payload = path.read_bytes()
        expected = digest_path.read_text(encoding="ascii").strip()
    except OSError:
        _fail("SEAL_LOAD_FAILURE")
    actual = sha256(payload).hexdigest()
    if len(expected) != 64 or not hmac.compare_digest(expected, actual):
        _fail("SEAL_DIGEST_MISMATCH")
    try:
        cases = tuple(PrivateRetrievalCase.from_mapping(json.loads(line)) for line in payload.decode("utf-8").splitlines())
    except (UnicodeError, json.JSONDecodeError):
        _fail("SEAL_LOAD_FAILURE")
    if _canonical_bytes(cases) != payload:
        _fail("SEAL_CANONICALIZATION_MISMATCH")
    return cases


def load_and_validate_fixture(
    *,
    development_path: Path,
    holdout_path: Path,
    development_digest_path: Path,
    holdout_digest_path: Path,
) -> tuple[tuple[PrivateRetrievalCase, ...], tuple[PrivateRetrievalCase, ...]]:
    development = _load_sealed(development_path, development_digest_path)
    holdout = _load_sealed(holdout_path, holdout_digest_path)
    if len(development) != 28 or len(holdout) != 12:
        _fail("SPLIT_BALANCE_INVALID")
    return development, holdout


def load_sealed_development_cases(*, development_path: Path, development_digest_path: Path) -> tuple[PrivateRetrievalCase, ...]:
    """Load only the development split; the sealed holdout is intentionally untouched."""
    development = _load_sealed(development_path, development_digest_path)
    if len(development) != 28 or any(case.split != "development" for case in development):
        _fail("SPLIT_BALANCE_INVALID")
    return development


@dataclass(frozen=True)
class RetrievalExperimentConfig:
    candidate_limit: int = 10
    final_top_k: int = 5
    rrf_k: int = 60
    vector_min_score: float = 0.2
    hybrid_min_rrf_score: float = 0.015
    max_context_chars: int = 4000
    keyword_boosts: Mapping[str, float] | None = None

    @classmethod
    def baseline(cls) -> "RetrievalExperimentConfig":
        return cls(keyword_boosts={"content": 2.0, "topic": 1.5, "source_file": 0.5})

    def validate(self) -> None:
        if self.candidate_limit < 1 or self.final_top_k < 1 or self.rrf_k < 1 or self.max_context_chars < 1:
            _fail("CONFIG_INVALID")
        if self.vector_min_score < 0 or self.hybrid_min_rrf_score < 0:
            _fail("CONFIG_INVALID")
        boosts = self.keyword_boosts
        if not isinstance(boosts, Mapping) or set(boosts) != _BOOST_FIELDS:
            _fail("CONFIG_INVALID")
        if any(type(value) not in {int, float} or float(value) < 0 for value in boosts.values()):
            _fail("CONFIG_INVALID")

    def safe_mapping(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_limit": self.candidate_limit,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
            "vector_min_score": self.vector_min_score,
            "hybrid_min_rrf_score": self.hybrid_min_rrf_score,
            "max_context_chars": self.max_context_chars,
            "keyword_boosts": dict(self.keyword_boosts or {}),
        }


@dataclass(frozen=True)
class RetrievalResponse:
    ranked_hits: tuple[RetrievalHit, ...]
    delivery_hits: tuple[RetrievalHit, ...]
    candidate_hits: tuple[RetrievalHit, ...]
    latency_adjustment_ms: float = 0.0


class CachedQueryEmbeddingProvider:
    """Cache query embeddings within one read-only experiment invocation."""

    def __init__(self, provider: object) -> None:
        self._provider = provider
        self.model_name = str(getattr(provider, "model_name"))
        self.dimension = int(getattr(provider, "dimension"))
        self._cache: dict[str, list[float]] = {}
        self._latency_ms: dict[str, float] = {}
        self.last_query_latency_ms = 0.0
        self.last_query_cache_hit = False

    def embed_query(self, question: str) -> list[float]:
        self.last_query_cache_hit = question in self._cache
        if not self.last_query_cache_hit:
            embed_query = getattr(self._provider, "embed_query")
            started = perf_counter()
            self._cache[question] = list(embed_query(question))
            self._latency_ms[question] = max(0.0, (perf_counter() - started) * 1000)
        self.last_query_latency_ms = self._latency_ms[question]
        return list(self._cache[question])


@dataclass(frozen=True)
class CaseObservation:
    question_id: str
    expected_evidence_exists: bool
    question_class: str
    expected_source_role: str | None
    expected_rank: int | None
    delivery_rank: int | None
    candidate_rank: int | None
    delivery_has_evidence: bool
    latency_ms: float


@dataclass(frozen=True)
class PrivateRetrievalRun:
    path: str
    configuration: RetrievalExperimentConfig
    mode_results: Mapping[str, tuple[CaseObservation, ...]]


Retriever = Callable[[str], RetrievalResponse | list[RetrievalHit]]


def _response(value: RetrievalResponse | list[RetrievalHit], delivery_limit: int) -> RetrievalResponse:
    if isinstance(value, RetrievalResponse):
        return value
    hits = tuple(value)
    return RetrievalResponse(hits, hits[:delivery_limit], hits)


def _rank_for(case: PrivateRetrievalCase, hits: Sequence[RetrievalHit]) -> int | None:
    if not case.expected_evidence_exists:
        return None
    accepted = {_source_key(value) for value in case.acceptable_sources}
    for rank, hit in enumerate(hits, start=1):
        if _source_key(hit.source_file) in accepted:
            return rank
    return None


def evaluate_no_oracle(
    *,
    cases: Sequence[PrivateRetrievalCase],
    retrievers: Mapping[str, Retriever],
    ranking_limit: int,
    delivery_limit: int,
    configuration: RetrievalExperimentConfig | None = None,
) -> PrivateRetrievalRun:
    if ranking_limit < 1 or delivery_limit < 1 or not retrievers:
        _fail("EVALUATOR_INPUT_INVALID")
    config = configuration or RetrievalExperimentConfig.baseline()
    config.validate()
    mode_results: dict[str, tuple[CaseObservation, ...]] = {}
    for mode, retrieve in retrievers.items():
        observations: list[CaseObservation] = []
        for case in cases:
            started = perf_counter()
            response = _response(retrieve(case.question), delivery_limit)
            elapsed = max(0.0, (perf_counter() - started) * 1000) + response.latency_adjustment_ms
            ranked = response.ranked_hits[:ranking_limit]
            delivery = response.delivery_hits[:delivery_limit]
            observations.append(
                CaseObservation(
                    question_id=case.question_id,
                    expected_evidence_exists=case.expected_evidence_exists,
                    question_class=case.question_class,
                    expected_source_role=case.expected_source_role,
                    expected_rank=_rank_for(case, ranked),
                    delivery_rank=_rank_for(case, delivery),
                    candidate_rank=_rank_for(case, response.candidate_hits),
                    delivery_has_evidence=bool(delivery),
                    latency_ms=elapsed,
                )
            )
        mode_results[mode] = tuple(observations)
    return PrivateRetrievalRun(
        path="REALISTIC_NO_ORACLE_RETRIEVAL",
        configuration=config,
        mode_results=mode_results,
    )


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]


def _mode_summary(observations: Sequence[CaseObservation]) -> dict[str, object]:
    answerable = [item for item in observations if item.expected_evidence_exists]
    no_answer = [item for item in observations if not item.expected_evidence_exists]
    ranks = [item.expected_rank for item in answerable]
    latencies = [item.latency_ms for item in observations]
    return {
        "answerable": {
            "case_count": len(answerable),
            "hit_rate_at_1": _rate([rank == 1 for rank in ranks]),
            "hit_rate_at_5": _rate([rank is not None and rank <= 5 for rank in ranks]),
            "hit_rate_at_10": _rate([rank is not None and rank <= 10 for rank in ranks]),
            "delivery_hit_rate": _rate([item.delivery_rank is not None for item in answerable]),
            "mrr_at_10": sum(1 / rank for rank in ranks if rank is not None and rank <= 10) / len(answerable) if answerable else 0.0,
            "candidate_to_delivery_loss_count": sum(
                item.candidate_rank is not None and item.delivery_rank is None for item in answerable
            ),
        },
        "no_answer": {
            "case_count": len(no_answer),
            "false_evidence_acceptance_count": sum(item.delivery_has_evidence for item in no_answer),
            "retrieval_abstention_correct_count": sum(not item.delivery_has_evidence for item in no_answer),
        },
        "latency_ms": {
            "median": median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
    }


def summarize_run(run: PrivateRetrievalRun) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluation_path": run.path,
        "configuration": run.configuration.safe_mapping(),
        "modes": {mode: _mode_summary(observations) for mode, observations in sorted(run.mode_results.items())},
    }


def _valid_aggregate_report(report: object) -> bool:
    if not isinstance(report, dict) or set(report) != {"schema_version", "evaluation_path", "configuration", "modes"}:
        return False
    if report["schema_version"] != 1 or report["evaluation_path"] not in {"REALISTIC_NO_ORACLE_RETRIEVAL", "ORACLE_TOPIC_DIAGNOSTIC"}:
        return False
    try:
        config = report["configuration"]
        if not isinstance(config, dict):
            return False
        RetrievalExperimentConfig(
            candidate_limit=config["candidate_limit"],
            final_top_k=config["final_top_k"],
            rrf_k=config["rrf_k"],
            vector_min_score=config["vector_min_score"],
            hybrid_min_rrf_score=config["hybrid_min_rrf_score"],
            max_context_chars=config["max_context_chars"],
            keyword_boosts=config["keyword_boosts"],
        ).validate()
        modes = report["modes"]
        if not isinstance(modes, dict):
            return False
        for summary in modes.values():
            if not isinstance(summary, dict) or set(summary) != {"answerable", "no_answer", "latency_ms"}:
                return False
            answerable, no_answer, latency = summary.values()
            if not all(isinstance(value, dict) for value in (answerable, no_answer, latency)):
                return False
            if set(answerable) != {"case_count", "hit_rate_at_1", "hit_rate_at_5", "hit_rate_at_10", "delivery_hit_rate", "mrr_at_10", "candidate_to_delivery_loss_count"}:
                return False
            if set(no_answer) != {"case_count", "false_evidence_acceptance_count", "retrieval_abstention_correct_count"}:
                return False
            if set(latency) != {"median", "p95"}:
                return False
    except (KeyError, TypeError, PrivateRetrievalError):
        return False
    return True


def write_aggregate_report(report: Mapping[str, object], destination: Path) -> None:
    _require_private_path(destination)
    payload = dict(report)
    if not _valid_aggregate_report(payload):
        _fail("AGGREGATE_REPORT_PRIVACY_VIOLATION")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def build_experiment_retrievers(
    *,
    store,
    embedding_provider,
    configuration: RetrievalExperimentConfig,
    ranking_limit: int = 10,
) -> dict[str, Retriever]:
    """Build question-only retrievers from existing production rows without writes."""
    configuration.validate()
    if ranking_limit < 1:
        _fail("EVALUATOR_INPUT_INVALID")
    stored_hits = store.list_chunks()
    keyword_index = KeywordSearchIndex.from_hits(stored_hits)
    vector_retriever = VectorRetriever(store=store, embedding_provider=embedding_provider)

    def cached_embedding_adjustment() -> float:
        if bool(getattr(embedding_provider, "last_query_cache_hit", False)):
            return float(getattr(embedding_provider, "last_query_latency_ms", 0.0))
        return 0.0

    def keyword(question: str) -> RetrievalResponse:
        ranked = tuple(
            keyword_index.search(
                question,
                limit=max(ranking_limit, configuration.final_top_k),
                boosts=configuration.keyword_boosts,
            )
        )
        return RetrievalResponse(ranked, ranked[: configuration.final_top_k], ranked)

    def vector(question: str) -> RetrievalResponse:
        ranked = tuple(
            vector_retriever.search(question, limit=max(ranking_limit, configuration.final_top_k), topic=None)
        )
        delivery = tuple(
            _fit_context_budget(
                [hit for hit in ranked[: configuration.final_top_k] if hit.score >= configuration.vector_min_score],
                configuration.max_context_chars,
            )
        )
        return RetrievalResponse(
            ranked,
            delivery,
            ranked,
            latency_adjustment_ms=cached_embedding_adjustment(),
        )

    def hybrid(question: str) -> RetrievalResponse:
        keyword_candidates = keyword_index.search(
            question,
            limit=configuration.candidate_limit,
            boosts=configuration.keyword_boosts,
        )
        vector_candidates = vector_retriever.search(question, limit=configuration.candidate_limit, topic=None)
        all_candidates = tuple(
            fuse_ranked_hits(
                keyword_candidates,
                vector_candidates,
                rrf_k=configuration.rrf_k,
                limit=2 * configuration.candidate_limit,
            )
        )
        ranked = all_candidates[:ranking_limit]
        delivery = tuple(
            _fit_context_budget(
                [
                    hit
                    for hit in all_candidates[: configuration.final_top_k]
                    if hit.score >= configuration.hybrid_min_rrf_score
                ],
                configuration.max_context_chars,
            )
        )
        return RetrievalResponse(
            ranked,
            delivery,
            all_candidates,
            latency_adjustment_ms=cached_embedding_adjustment(),
        )

    return {"keyword": keyword, "vector": vector, "hybrid": hybrid}


_CANDIDATE_LIMITS = (10, 20, 30, 50)
_FINAL_TOP_KS = (3, 5, 8, 10)
_RRF_KS = (1, 50, 60, 100, 200)
_BOOST_PROFILES: Mapping[str, Mapping[str, float]] = {
    "baseline": {"content": 2.0, "topic": 1.5, "source_file": 0.5},
    "content_only": {"content": 1.0, "topic": 0.0, "source_file": 0.0},
    "balanced": {"content": 1.0, "topic": 1.0, "source_file": 1.0},
    "source_aware": {"content": 1.5, "topic": 1.0, "source_file": 1.0},
}


@dataclass(frozen=True)
class DevelopmentExperimentResults:
    """Aggregate-only development output. Detailed observations remain private."""

    runs: Mapping[str, PrivateRetrievalRun]
    selected_configuration_ids: Mapping[str, str]
    recommended_configuration_id: str
    recommended_configuration: RetrievalExperimentConfig

    def aggregate_report(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "evaluation_path": "REALISTIC_NO_ORACLE_RETRIEVAL",
            "experiment_scope": "development_only",
            "selected_configuration_ids": dict(self.selected_configuration_ids),
            "recommended_configuration_id": self.recommended_configuration_id,
            "recommended_configuration": self.recommended_configuration.safe_mapping(),
            "runs": {name: summarize_run(run) for name, run in sorted(self.runs.items())},
        }


def _hybrid_summary(run: PrivateRetrievalRun) -> dict[str, object]:
    try:
        return _mode_summary(run.mode_results["hybrid"])
    except KeyError:
        _fail("HYBRID_RESULT_REQUIRED")


def _answerable_hit_count(summary: Mapping[str, object]) -> int:
    answerable = summary["answerable"]
    assert isinstance(answerable, Mapping)
    return round(float(answerable["hit_rate_at_5"]) * int(answerable["case_count"]))


def _answerable_mrr(summary: Mapping[str, object]) -> float:
    answerable = summary["answerable"]
    assert isinstance(answerable, Mapping)
    return float(answerable["mrr_at_10"])


def _development_quality(summary: Mapping[str, object]) -> tuple[float, float, float, float]:
    answerable = summary["answerable"]
    assert isinstance(answerable, Mapping)
    return (
        float(_answerable_hit_count(summary)),
        float(answerable["delivery_hit_rate"]),
        _answerable_mrr(summary),
        float(answerable["hit_rate_at_1"]),
    )


def _false_acceptance_count(summary: Mapping[str, object]) -> int:
    no_answer = summary["no_answer"]
    assert isinstance(no_answer, Mapping)
    return int(no_answer["false_evidence_acceptance_count"])


def _class_hit_counts(observations: Sequence[CaseObservation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in observations:
        if item.expected_evidence_exists:
            counts.setdefault(item.question_class, 0)
            if item.expected_rank is not None and item.expected_rank <= 5:
                counts[item.question_class] += 1
    return counts


def choose_development_candidate(
    *, baseline: PrivateRetrievalRun, candidates: Mapping[str, PrivateRetrievalRun]
) -> str:
    """Select a development direction without claiming a sealed-holdout result.

    A candidate cannot worsen no-answer evidence delivery or lose more than one
    answerable retrieval in any question class. Among eligible candidates, rank
    Hit Rate@5, MRR@10, Hit Rate@1, then lower p95 retrieval latency.
    """
    baseline_summary = _hybrid_summary(baseline)
    baseline_observations = baseline.mode_results["hybrid"]
    baseline_false_acceptance = _false_acceptance_count(baseline_summary)
    baseline_by_class = _class_hit_counts(baseline_observations)
    baseline_quality = _development_quality(baseline_summary)
    eligible: list[tuple[tuple[float, float, float, float], str]] = []

    for name, candidate in candidates.items():
        candidate_summary = _hybrid_summary(candidate)
        if _false_acceptance_count(candidate_summary) > baseline_false_acceptance:
            continue
        candidate_by_class = _class_hit_counts(candidate.mode_results["hybrid"])
        if any(candidate_by_class.get(question_class, 0) < count - 1 for question_class, count in baseline_by_class.items()):
            continue
        candidate_quality = _development_quality(candidate_summary)
        if candidate_quality <= baseline_quality and _false_acceptance_count(candidate_summary) == baseline_false_acceptance:
            continue
        answerable = candidate_summary["answerable"]
        latency = candidate_summary["latency_ms"]
        assert isinstance(answerable, Mapping)
        assert isinstance(latency, Mapping)
        eligible.append(
            (
                (
                    *candidate_quality,
                    -float(latency["p95"]),
                ),
                name,
            )
        )
    if not eligible:
        return "baseline"
    return max(eligible, key=lambda item: (item[0], item[1]))[1]


def _evaluate_configuration(
    *,
    cases: Sequence[PrivateRetrievalCase],
    store: object,
    embedding_provider: object,
    configuration: RetrievalExperimentConfig,
    modes: Sequence[str],
    ranking_limit: int,
) -> PrivateRetrievalRun:
    retrievers = build_experiment_retrievers(
        store=store,
        embedding_provider=embedding_provider,
        configuration=configuration,
        ranking_limit=ranking_limit,
    )
    selected_retrievers = {mode: retrievers[mode] for mode in modes}
    return evaluate_no_oracle(
        cases=cases,
        retrievers=selected_retrievers,
        ranking_limit=ranking_limit,
        delivery_limit=configuration.final_top_k,
        configuration=configuration,
    )


def run_development_experiments(
    *,
    cases: Sequence[PrivateRetrievalCase],
    store: object,
    embedding_provider: object,
    ranking_limit: int = 10,
) -> DevelopmentExperimentResults:
    """Run stages A-E on development cases only, never loading the holdout."""
    if len(cases) != 28 or any(case.split != "development" for case in cases):
        _fail("DEVELOPMENT_CASES_REQUIRED")
    if ranking_limit < 10:
        _fail("RANKING_LIMIT_INVALID")

    runs: dict[str, PrivateRetrievalRun] = {}
    selected: dict[str, str] = {}

    baseline = RetrievalExperimentConfig.baseline()
    runs["A-baseline"] = _evaluate_configuration(
        cases=cases,
        store=store,
        embedding_provider=embedding_provider,
        configuration=baseline,
        modes=("keyword", "vector", "hybrid"),
        ranking_limit=ranking_limit,
    )

    stage_b: dict[str, PrivateRetrievalRun] = {}
    for candidate_limit in _CANDIDATE_LIMITS:
        name = f"B-candidate-{candidate_limit}"
        config = replace(baseline, candidate_limit=candidate_limit)
        stage_b[name] = _evaluate_configuration(
            cases=cases,
            store=store,
            embedding_provider=embedding_provider,
            configuration=config,
            modes=("hybrid",),
            ranking_limit=ranking_limit,
        )
    runs.update(stage_b)
    selected["B-candidate-pool"] = choose_development_candidate(
        baseline=runs["A-baseline"], candidates=stage_b
    )
    config_b = (
        baseline
        if selected["B-candidate-pool"] == "baseline"
        else stage_b[selected["B-candidate-pool"]].configuration
    )
    baseline_b_run = (
        runs["A-baseline"]
        if selected["B-candidate-pool"] == "baseline"
        else stage_b[selected["B-candidate-pool"]]
    )

    stage_c: dict[str, PrivateRetrievalRun] = {}
    for final_top_k in _FINAL_TOP_KS:
        name = f"C-top-k-{final_top_k}"
        config = replace(config_b, final_top_k=final_top_k)
        stage_c[name] = _evaluate_configuration(
            cases=cases,
            store=store,
            embedding_provider=embedding_provider,
            configuration=config,
            modes=("hybrid",),
            ranking_limit=ranking_limit,
        )
    runs.update(stage_c)
    selected["C-final-top-k"] = choose_development_candidate(
        baseline=baseline_b_run, candidates=stage_c
    )
    config_c = (
        config_b
        if selected["C-final-top-k"] == "baseline"
        else stage_c[selected["C-final-top-k"]].configuration
    )
    baseline_c_run = (
        baseline_b_run
        if selected["C-final-top-k"] == "baseline"
        else stage_c[selected["C-final-top-k"]]
    )

    stage_d: dict[str, PrivateRetrievalRun] = {}
    for rrf_k in _RRF_KS:
        name = f"D-rrf-k-{rrf_k}"
        config = replace(config_c, rrf_k=rrf_k)
        stage_d[name] = _evaluate_configuration(
            cases=cases,
            store=store,
            embedding_provider=embedding_provider,
            configuration=config,
            modes=("hybrid",),
            ranking_limit=ranking_limit,
        )
    runs.update(stage_d)
    selected["D-rrf-k"] = choose_development_candidate(
        baseline=baseline_c_run, candidates=stage_d
    )
    config_d = (
        config_c
        if selected["D-rrf-k"] == "baseline"
        else stage_d[selected["D-rrf-k"]].configuration
    )
    baseline_d_run = (
        baseline_c_run
        if selected["D-rrf-k"] == "baseline"
        else stage_d[selected["D-rrf-k"]]
    )

    stage_e: dict[str, PrivateRetrievalRun] = {}
    for profile_name, boosts in _BOOST_PROFILES.items():
        name = f"E-boosts-{profile_name}"
        config = replace(config_d, keyword_boosts=boosts)
        stage_e[name] = _evaluate_configuration(
            cases=cases,
            store=store,
            embedding_provider=embedding_provider,
            configuration=config,
            modes=("keyword", "hybrid"),
            ranking_limit=ranking_limit,
        )
    runs.update(stage_e)
    selected["E-keyword-boosts"] = choose_development_candidate(
        baseline=baseline_d_run, candidates=stage_e
    )
    final_stage_choice = selected["E-keyword-boosts"]
    recommended_configuration = (
        config_d if final_stage_choice == "baseline" else stage_e[final_stage_choice].configuration
    )
    recommended = " | ".join(
        (
            f"B={selected['B-candidate-pool']}",
            f"C={selected['C-final-top-k']}",
            f"D={selected['D-rrf-k']}",
            f"E={final_stage_choice}",
        )
    )
    return DevelopmentExperimentResults(
        runs=runs,
        selected_configuration_ids=selected,
        recommended_configuration_id=recommended,
        recommended_configuration=recommended_configuration,
    )


def write_private_development_results(
    results: DevelopmentExperimentResults, destination: Path
) -> None:
    """Write private diagnostics only under the controller-owned evaluation root."""
    _require_private_path(destination)
    payload = {
        "aggregate": results.aggregate_report(),
        "observations": {
            name: {
                mode: [asdict(observation) for observation in observations]
                for mode, observations in sorted(run.mode_results.items())
            }
            for name, run in sorted(results.runs.items())
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
