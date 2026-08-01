import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DATASET_DIR = PROJECT_ROOT / "eval"
PRIVATE_DATASET_DIR = PUBLIC_DATASET_DIR / "private"
PUBLIC_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
EvaluationPrivacyMode = Literal["public", "private"]
_DATASET_FIELDS = {"question_id", "question", "expected_chunk_ids", "topic"}


@dataclass(frozen=True)
class EvaluationCase:
    question_id: str
    question: str
    expected_chunk_ids: tuple[str, ...]
    topic: str


@dataclass(frozen=True)
class EvaluationResult:
    question_id: str
    topic: str
    ranked_chunk_ids: tuple[str, ...]
    expected_rank: int | None
    latency_ms: float


def load_evaluation_cases(
    path: Path, *, privacy_mode: EvaluationPrivacyMode = "public"
) -> list[EvaluationCase]:
    resolved_path = _resolve_dataset_path(path, privacy_mode)

    lines = resolved_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("dataset must not be empty")

    cases: list[EvaluationCase] = []
    question_ids: set[str] = set()
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("dataset records must be valid JSON objects") from error
        if not isinstance(record, dict) or set(record) != _DATASET_FIELDS:
            raise ValueError("dataset records must be objects with exactly public fields")

        values = (record["question_id"], record["question"], record["topic"])
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("dataset values must not be blank")
        expected = record["expected_chunk_ids"]
        if (
            not isinstance(expected, list)
            or not expected
            or any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in expected)
        ):
            raise ValueError("expected_chunk_ids must not be empty")
        if len(set(expected)) != len(expected):
            raise ValueError("expected_chunk_ids must not contain duplicates")

        question_id = record["question_id"]
        if question_id in question_ids:
            raise ValueError(f"duplicate question_id: {question_id}")
        question_ids.add(question_id)
        cases.append(
            EvaluationCase(question_id, record["question"], tuple(expected), record["topic"])
        )
    return cases


def _resolve_dataset_path(path: Path, privacy_mode: EvaluationPrivacyMode) -> Path:
    if privacy_mode not in {"public", "private"}:
        raise ValueError("privacy mode must be public or private")

    resolved_path = path.resolve()
    boundary = PRIVATE_DATASET_DIR if privacy_mode == "private" else PUBLIC_DATASET_DIR
    try:
        resolved_path.relative_to(boundary)
    except ValueError as error:
        raise ValueError(f"dataset path must reference a {privacy_mode} dataset") from error

    if privacy_mode == "public":
        try:
            resolved_path.relative_to(PRIVATE_DATASET_DIR)
        except ValueError:
            pass
        else:
            raise ValueError("dataset path must reference a public dataset")
    return resolved_path


def public_sample_chunk_ids() -> frozenset[str]:
    from ingestion.ingest import generate_chunks

    with TemporaryDirectory() as output_dir:
        chunks = generate_chunks(PUBLIC_SAMPLE_DIR, output_dir)
    return frozenset(chunk.metadata.chunk_id for chunk in chunks)


def validate_public_stored_chunk_ids(
    stored_chunk_ids: set[str], public_chunk_ids: frozenset[str]
) -> None:
    missing_count = len(public_chunk_ids - stored_chunk_ids)
    unexpected_count = len(stored_chunk_ids - public_chunk_ids)
    if missing_count or unexpected_count:
        raise ValueError(
            "stored chunk IDs do not match public manifest "
            f"(missing={missing_count}, unexpected={unexpected_count})"
        )

def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def first_expected_rank(
    ranked_chunk_ids: tuple[str, ...], expected_chunk_ids: frozenset[str], k: int
) -> int | None:
    _validate_k(k)
    for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        if chunk_id in expected_chunk_ids:
            return rank
    return None


def hit_rate_at_k(results: list[EvaluationResult], k: int) -> float:
    _validate_k(k)
    if not results:
        return 0.0
    return sum(
        result.expected_rank is not None and result.expected_rank <= k for result in results
    ) / len(results)


def mean_reciprocal_rank(results: list[EvaluationResult], k: int) -> float:
    _validate_k(k)
    if not results:
        return 0.0
    return sum(
        1 / result.expected_rank
        for result in results
        if result.expected_rank and result.expected_rank <= k
    ) / len(results)
