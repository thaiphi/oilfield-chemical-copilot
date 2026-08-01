import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from oilfield_chemical_copilot.evaluation import retrieval
from oilfield_chemical_copilot.evaluation.retrieval import (
    EvaluationCase,
    EvaluationResult,
    first_expected_rank,
    hit_rate_at_k,
    load_evaluation_cases,
    mean_reciprocal_rank,
)


APPROVED_CASES = [
    EvaluationCase(
        "scale-01",
        "How should I assess scale risk from produced water analysis?",
        (
            "docs:scale_water_analysis_overview.md:document:0:d38047e6931f",
            "docs:water_analysis_interpretation.md:document:0:a36132aa7fbd",
        ),
        "scale",
    ),
    EvaluationCase(
        "dosage-01",
        "How do ppm, water barrels per day, and 42 gallons per barrel affect continuous treatment dosage?",
        ("docs:chemical_dosage_examples.md:document:0:43031497f26b",),
        "dosage",
    ),
    EvaluationCase(
        "iron-01",
        "What field checks help investigate black iron sulfide deposits and THPS treatment?",
        ("docs:iron_sulfide_overview.md:document:0:8a0c9656afa6",),
        "iron_sulfide",
    ),
    EvaluationCase(
        "corrosion-01",
        "Which observations separate corrosion under-treatment from mechanical or operating causes?",
        ("docs:corrosion_root_cause.md:document:0:154c59366030",),
        "corrosion",
    ),
    EvaluationCase(
        "paraffin-01",
        "What operating changes can make paraffin or asphaltene deposits more likely?",
        ("docs:paraffin_asphaltene_overview.md:document:0:3ded8ee54f6b",),
        "paraffin",
    ),
    EvaluationCase(
        "water-01",
        "Which ions and operating conditions frame a scale and corrosion water review?",
        (
            "docs:water_analysis_interpretation.md:document:0:a36132aa7fbd",
            "docs:scale_water_analysis_overview.md:document:0:d38047e6931f",
        ),
        "water_analysis",
    ),
    EvaluationCase(
        "scale-02",
        "What chemistry and operating changes should be screened before predicting inorganic scale?",
        ("docs:scale_water_analysis_overview.md:document:0:d38047e6931f",),
        "scale",
    ),
    EvaluationCase(
        "scale-03",
        "Why can incompatible produced waters increase deposition risk?",
        ("docs:scale_water_analysis_overview.md:document:0:d38047e6931f",),
        "scale",
    ),
    EvaluationCase(
        "dosage-02",
        "What conversion inputs are needed to estimate water-basis chemical gallons per day?",
        ("docs:chemical_dosage_examples.md:document:0:43031497f26b",),
        "dosage",
    ),
    EvaluationCase(
        "dosage-03",
        "How is a continuous ppm treatment related to daily water production?",
        ("docs:chemical_dosage_examples.md:document:0:43031497f26b",),
        "dosage",
    ),
    EvaluationCase(
        "iron-02",
        "Which history and deposit observations support an iron sulfide diagnosis?",
        ("docs:iron_sulfide_overview.md:document:0:8a0c9656afa6",),
        "iron_sulfide",
    ),
    EvaluationCase(
        "iron-03",
        "Why might THPS be discussed when black produced-water solids appear?",
        ("docs:iron_sulfide_overview.md:document:0:8a0c9656afa6",),
        "iron_sulfide",
    ),
    EvaluationCase(
        "corrosion-02",
        "What data should a corrosion root-cause investigation compare?",
        ("docs:corrosion_root_cause.md:document:0:154c59366030",),
        "corrosion",
    ),
    EvaluationCase(
        "corrosion-03",
        "How do coupons, probes, wall loss, and failure location help diagnose corrosion?",
        ("docs:corrosion_root_cause.md:document:0:154c59366030",),
        "corrosion",
    ),
    EvaluationCase(
        "paraffin-02",
        "What conditions are commonly associated with wax deposition?",
        ("docs:paraffin_asphaltene_overview.md:document:0:3ded8ee54f6b",),
        "paraffin",
    ),
    EvaluationCase(
        "paraffin-03",
        "Which changes can destabilize asphaltenes in crude oil?",
        ("docs:paraffin_asphaltene_overview.md:document:0:3ded8ee54f6b",),
        "paraffin",
    ),
    EvaluationCase(
        "water-02",
        "What does high chloride and TDS indicate during a brine review?",
        ("docs:water_analysis_interpretation.md:document:0:a36132aa7fbd",),
        "water_analysis",
    ),
    EvaluationCase(
        "water-03",
        "Which water-analysis fields frame corrosion and deposit questions?",
        ("docs:water_analysis_interpretation.md:document:0:a36132aa7fbd",),
        "water_analysis",
    ),
]


@pytest.fixture
def public_dataset_path(tmp_path: Path) -> Path:
    path = Path("eval") / f"{tmp_path.name}.jsonl"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


@pytest.fixture
def private_dataset_path(tmp_path: Path) -> Path:
    private_dir = Path("eval") / "private"
    private_dir.mkdir(exist_ok=True)
    path = private_dir / f"{tmp_path.name}.jsonl"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _dataset_record() -> str:
    return '{"question_id":"q","question":"x","expected_chunk_ids":["id"],"topic":"t"}\n'


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    text: str
    source_filename: str
    source_path: str
    metadata: dict[str, str]


def _result(ranked_hits: tuple[RetrievalHit, ...], expected: frozenset[str]) -> EvaluationResult:
    ranked_chunk_ids = tuple(hit.chunk_id for hit in ranked_hits)
    rank = first_expected_rank(ranked_chunk_ids, expected, 3)
    return EvaluationResult("q1", "scale", ranked_chunk_ids, rank, 12.5)


def test_expected_evidence_at_rank_two_has_rank_hit_rate_and_reciprocal_rank() -> None:
    hits = (
        RetrievalHit("wrong", "private text", "private.md", "/private/private.md", {}),
        RetrievalHit("expected", "private text", "private.md", "/private/private.md", {}),
    )
    result = _result(hits, frozenset({"expected"}))

    assert first_expected_rank(("wrong", "expected"), frozenset({"expected"}), 3) == 2
    assert hit_rate_at_k([result], 3) == 1.0
    assert mean_reciprocal_rank([result], 3) == 0.5


def test_absence_has_no_rank_zero_hit_rate_and_zero_reciprocal_rank() -> None:
    result = _result((RetrievalHit("wrong", "text", "file", "/file", {}),), frozenset({"expected"}))

    assert first_expected_rank(("wrong",), frozenset({"expected"}), 3) is None
    assert hit_rate_at_k([result], 3) == 0.0
    assert mean_reciprocal_rank([result], 3) == 0.0


def test_metrics_reject_k_less_than_one() -> None:
    with pytest.raises(ValueError, match="^k must be at least 1$"):
        first_expected_rank((), frozenset(), 0)

    with pytest.raises(ValueError, match="^k must be at least 1$"):
        hit_rate_at_k([], 0)

    with pytest.raises(ValueError, match="^k must be at least 1$"):
        mean_reciprocal_rank([], 0)


def test_public_dataset_matches_the_approved_manifest_and_public_sample_chunks() -> None:
    path = Path("eval/public_retrieval_dataset.jsonl")
    cases = load_evaluation_cases(path)
    assert cases == APPROVED_CASES

    expected_ids = {chunk_id for case in cases for chunk_id in case.expected_chunk_ids}
    assert expected_ids <= retrieval.public_sample_chunk_ids()

@pytest.mark.parametrize(
    "records",
    [
        [
            {"question_id": "q", "question": "x", "expected_chunk_ids": ["id"], "topic": "t"},
            {"question_id": "q", "question": "y", "expected_chunk_ids": ["id"], "topic": "t"},
        ],
        [{"question_id": "q", "question": "x", "expected_chunk_ids": [], "topic": "t"}],
    ],
)
def test_dataset_rejects_duplicate_ids_and_empty_expected_ids(
    public_dataset_path: Path, records: list[dict[str, object]]
) -> None:
    path = public_dataset_path
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    with pytest.raises(ValueError):
        load_evaluation_cases(path)


def test_dataset_rejects_existing_non_public_absolute_path(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    path.write_text(
        '{"question_id":"q","question":"x","expected_chunk_ids":["id"],"topic":"t"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="public"):
        load_evaluation_cases(path.resolve())


def test_dataset_accepts_absolute_path_inside_public_boundary(public_dataset_path: Path) -> None:
    public_dataset_path.write_text(
        '{"question_id":"q","question":"x","expected_chunk_ids":["id"],"topic":"t"}\n',
        encoding="utf-8",
    )

    assert load_evaluation_cases(public_dataset_path.resolve()) == [
        EvaluationCase("q", "x", ("id",), "t")
    ]


@pytest.mark.parametrize(
    "record",
    [
        {
            "question_id": "q",
            "question": "x",
            "expected_chunk_ids": ["id"],
            "topic": "t",
            "text": "private",
        },
        {"question_id": "q", "question": "x", "expected_chunk_ids": ["id"]},
        ["not", "an", "object"],
    ],
)
def test_dataset_requires_json_objects_with_exact_public_fields(
    public_dataset_path: Path, record: object
) -> None:
    path = public_dataset_path
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset records"):
        load_evaluation_cases(path)


@pytest.mark.parametrize(
    "contents",
    [
        "",
        '{"question_id":"q","question":"x","expected_chunk_ids":["id","id"],"topic":"t"}',
    ],
)
def test_dataset_rejects_empty_input_and_duplicate_expected_chunk_ids(
    public_dataset_path: Path, contents: str
) -> None:
    path = public_dataset_path
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="dataset|expected_chunk_ids"):
        load_evaluation_cases(path)


def test_dataset_rejects_relative_path_outside_public_boundary() -> None:
    with pytest.raises(ValueError, match="public"):
        load_evaluation_cases(Path("data/sample/README.md"))

def test_public_mode_rejects_a_dataset_under_the_private_boundary(
    private_dataset_path: Path,
) -> None:
    private_dataset_path.write_text(_dataset_record(), encoding="utf-8")

    with pytest.raises(ValueError, match="^dataset path must reference a public dataset$"):
        load_evaluation_cases(private_dataset_path)


def test_private_mode_rejects_a_dataset_outside_the_private_boundary(
    public_dataset_path: Path,
) -> None:
    public_dataset_path.write_text(_dataset_record(), encoding="utf-8")

    with pytest.raises(ValueError, match="^dataset path must reference a private dataset$"):
        load_evaluation_cases(public_dataset_path, privacy_mode="private")


def test_public_manifest_validation_rejects_mixed_ids_without_echoing_unknown_id() -> None:
    public_chunk_ids = retrieval.public_sample_chunk_ids()
    missing_id = next(iter(public_chunk_ids))
    private_like_sentinel = "private-chunk-id-must-not-appear"
    stored_chunk_ids = set(public_chunk_ids)
    stored_chunk_ids.remove(missing_id)
    stored_chunk_ids.add(private_like_sentinel)

    with pytest.raises(
        ValueError,
        match=r"^stored chunk IDs do not match public manifest \(missing=1, unexpected=1\)$",
    ) as error:
        retrieval.validate_public_stored_chunk_ids(stored_chunk_ids, public_chunk_ids)

    assert private_like_sentinel not in str(error.value)