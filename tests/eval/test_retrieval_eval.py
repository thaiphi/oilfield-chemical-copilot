import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import eval.retrieval_eval as retrieval_eval
from eval.retrieval_eval import RunProvenance, evaluate_cases, write_report
from oilfield_chemical_copilot.evaluation.retrieval import EvaluationCase, EvaluationResult
from oilfield_chemical_copilot.retrieval.models import RetrievalHit


def test_direct_cli_help_resolves_project_imports() -> None:
    result = subprocess.run(
        [sys.executable, "eval/retrieval_eval.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

def _hit(chunk_id: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text="PRIVATE EXCERPT",
        score=0.9,
        retrieval_method="fake",
        source_file="secret.md",
        source_path="C:/private/secret.md",
        topic="scale",
        parser_type="markdown",
        page_or_sheet="",
        chunk_index=0,
    )


def test_evaluate_cases_passes_questions_and_topics_and_records_ranked_ids() -> None:
    cases = [
        EvaluationCase("q1", "first question", ("expected-1",), "scale"),
        EvaluationCase("q2", "second question", ("expected-2",), "corrosion"),
    ]
    calls: list[tuple[str, str | None]] = []

    def retrieve(question: str, topic: str | None) -> list[RetrievalHit]:
        calls.append((question, topic))
        return [_hit("wrong"), _hit("expected-1" if topic == "scale" else "expected-2")]

    results = evaluate_cases(cases, retrieve, k=3)

    assert calls == [("first question", "scale"), ("second question", "corrosion")]
    assert [result.ranked_chunk_ids for result in results] == [
        ("wrong", "expected-1"),
        ("wrong", "expected-2"),
    ]
    assert [result.expected_rank for result in results] == [2, 2]
    assert all(result.latency_ms >= 0 for result in results)


def _provenance() -> RunProvenance:
    return RunProvenance(
        dataset_sha256="a" * 64,
        corpus_sha256="b" * 64,
        git_revision="unknown",
        retrieval_mode_settings={"hybrid_rrf_k": 60, "hybrid_min_rrf_score": 0.015},
        embedding_provider="deterministic",
        embedding_model="deterministic-token-hash-12",
        embedding_dimension=12,
        k=5,
        topic_filter="oracle_gold_topic",
    )


def test_public_report_includes_public_failures_and_sanitized_provenance(tmp_path: Path) -> None:
    result = EvaluationResult(
        question_id="q1",
        topic="scale",
        ranked_chunk_ids=("wrong", "expected"),
        expected_rank=None,
        latency_ms=12.5,
    )

    json_path, markdown_path = write_report(
        {"keyword": [result]}, tmp_path, privacy_mode="public", provenance=_provenance()
    )

    serialized = json_path.read_text(encoding="utf-8") + markdown_path.read_text(encoding="utf-8")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["privacy_mode"] == "public"
    assert report["provenance"]["topic_filter"] == "oracle_gold_topic"
    assert report["modes"]["keyword"]["failures"] == [
        {"question_id": "q1", "topic": "scale", "expected_rank": None}
    ]
    for forbidden in ("C:/", "PRIVATE EXCERPT", "source_path", "text"):
        assert forbidden not in serialized
    for required in ("keyword", "hit_rate_at_3", "hit_rate_at_5", "mrr_at_5"):
        assert required in serialized
    assert "| mode | questions | Hit Rate@3 | Hit Rate@5 | MRR@5 | median latency ms |" in serialized
    assert "q1" in serialized
    assert "scale" in serialized
    assert "expected_rank" in serialized
    assert "returned_chunk_ids" not in serialized



def test_evaluate_cases_records_rank_five_for_fixed_metric_window() -> None:
    case = EvaluationCase("q1", "question", ("expected",), "scale")
    results = evaluate_cases(
        [case],
        lambda _question, _topic: [
            _hit("wrong-1"), _hit("wrong-2"), _hit("wrong-3"), _hit("wrong-4"), _hit("expected")
        ],
        k=5,
    )

    assert results[0].expected_rank == 5
    summary = retrieval_eval._report_summary(results)
    assert summary["hit_rate_at_3"] == 0.0
    assert summary["hit_rate_at_5"] == 1.0
    assert summary["mrr_at_5"] == 0.2


def test_evaluate_cases_times_only_the_retriever_call(monkeypatch: pytest.MonkeyPatch) -> None:
    clock_values = iter((10.0, 15.0, 115.0))
    monkeypatch.setattr(retrieval_eval.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(
        retrieval_eval, "first_expected_rank", lambda *_args: retrieval_eval.time.perf_counter() and 1
    )

    result = evaluate_cases(
        [EvaluationCase("q1", "question", ("expected",), "scale")],
        lambda _question, _topic: [_hit("expected")],
        k=5,
    )[0]

    assert result.latency_ms == 5000.0


def test_private_report_is_aggregate_only(tmp_path: Path) -> None:
    private_result = EvaluationResult(
        question_id="PRIVATE-QUESTION-ID",
        topic="PRIVATE-TOPIC",
        ranked_chunk_ids=("PRIVATE-CHUNK-ID",),
        expected_rank=None,
        latency_ms=12.5,
    )

    json_path, markdown_path = write_report(
        {"keyword": [private_result]}, tmp_path, privacy_mode="private", provenance=_provenance()
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(report["modes"]["keyword"]) == {
        "questions", "hit_rate_at_3", "hit_rate_at_5", "mrr_at_5", "median_latency_ms"
    }
    serialized = json_path.read_text(encoding="utf-8") + markdown_path.read_text(encoding="utf-8")
    for private_value in ("PRIVATE-QUESTION-ID", "PRIVATE-TOPIC", "PRIVATE-CHUNK-ID"):
        assert private_value not in serialized

def test_cli_defaults_and_options_are_exposed() -> None:
    defaults = retrieval_eval._build_argument_parser().parse_args([])
    options = retrieval_eval._build_argument_parser().parse_args(
        ["--dataset", "eval/custom.jsonl", "--output-dir", "out", "--modes", "vector,hybrid", "--k", "5", "--privacy-mode", "private", "--database-url", "postgresql://example/eval"]
    )

    assert defaults.dataset == Path("eval/public_retrieval_dataset.jsonl")
    assert defaults.output_dir == Path("data/processed/evaluation")
    assert defaults.modes == "keyword,vector,hybrid"
    assert defaults.k == 5
    assert options.dataset == Path("eval/custom.jsonl")
    assert options.output_dir == Path("out")
    assert options.modes == "vector,hybrid"
    assert options.k == 5
    assert defaults.privacy_mode == 'public'
    assert options.privacy_mode == 'private'
    assert options.database_url == "postgresql://example/eval"


@pytest.mark.parametrize("k", (4, 6))
def test_cli_rejects_depth_other_than_fixed_metric_window(
    monkeypatch: pytest.MonkeyPatch, k: int
) -> None:
    monkeypatch.setattr(sys, "argv", ["retrieval_eval.py", "--k", str(k)])

    with pytest.raises(ValueError, match="^k must be exactly 5 for fixed evaluation metrics$"):
        retrieval_eval.main()


def test_cli_loads_chunks_once_and_wires_keyword_vector_and_hybrid_factories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = [EvaluationCase("q1", "question", ("expected",), "scale")]
    stored_hits = [_hit("expected")]
    events: list[object] = []

    class FakeStore:
        def __init__(self, database_url: str | None, *, embedding_dimension: int) -> None:
            events.append(("store", database_url, embedding_dimension))

        def list_chunks(self) -> list[RetrievalHit]:
            events.append("list_chunks")
            return stored_hits

    class FakeKeywordIndex:
        @classmethod
        def from_hits(cls, hits: list[RetrievalHit]) -> object:
            events.append(("keyword", hits))
            return cls()

        def search(self, question: str, limit: int, topic: str | None) -> list[RetrievalHit]:
            events.append(("keyword-search", question, limit, topic))
            return stored_hits

    class FakePipeline:
        def __init__(self, mode: str) -> None:
            self.mode = mode

        def retrieve(self, question: str, topic: str | None) -> list[RetrievalHit]:
            events.append(("retrieve", self.mode, question, topic))
            return stored_hits

    provider = type("Provider", (), {"dimension": 12, "model_name": "test-model"})()
    monkeypatch.setattr(sys, "argv", ["retrieval_eval.py", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(retrieval_eval, "load_evaluation_cases", lambda _path, **_kwargs: cases)
    monkeypatch.setattr(retrieval_eval, "public_sample_chunk_ids", lambda: frozenset({"expected"}))
    monkeypatch.setattr(retrieval_eval, "build_embedding_provider", lambda: provider)
    monkeypatch.setattr(retrieval_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(retrieval_eval, "KeywordSearchIndex", FakeKeywordIndex)
    monkeypatch.setattr(retrieval_eval.RetrievalSettings, "from_env", lambda: retrieval_eval.RetrievalSettings())
    monkeypatch.setattr(
        retrieval_eval,
        "build_retrieval_pipeline",
        lambda *, store, embedding_provider, settings, keyword_index=None: (
            events.append(("pipeline", store, embedding_provider, settings, keyword_index)) or FakePipeline(settings.retrieval_mode)
        ),
    )
    monkeypatch.setattr(retrieval_eval, "write_report", lambda results, output_dir, **_kwargs: (output_dir / "report.json", output_dir / "report.md"))

    retrieval_eval.main()

    assert events.count("list_chunks") == 1
    assert ("keyword", stored_hits) in events
    pipelines = [event for event in events if isinstance(event, tuple) and event[0] == "pipeline"]
    assert [(event[3].retrieval_mode, event[3].top_k) for event in pipelines] == [
        ("vector", 5), ("hybrid", 5)
    ]
    assert pipelines[0][4] is None
    assert isinstance(pipelines[1][4], FakeKeywordIndex)
    assert ("keyword-search", "question", 5, "scale") in events


def test_cli_checks_expected_ids_before_report_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = [EvaluationCase("q1", "question", ("missing",), "scale")]
    output_dir = tmp_path / "reports"

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def list_chunks(self) -> list[RetrievalHit]:
            return [_hit("other")]

    monkeypatch.setattr(sys, "argv", ["retrieval_eval.py", "--modes", "keyword", "--output-dir", str(output_dir)])
    monkeypatch.setattr(retrieval_eval, "load_evaluation_cases", lambda _path, **_kwargs: cases)
    monkeypatch.setattr(retrieval_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(retrieval_eval, "public_sample_chunk_ids", lambda: frozenset({"missing"}))
    monkeypatch.setattr(retrieval_eval, "write_report", lambda *_args, **_kwargs: pytest.fail("report must not be written"))

    with pytest.raises(ValueError, match="missing"):
        retrieval_eval.main()




def test_public_preflight_rejects_private_stored_chunk_before_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    private_sentinel = "PRIVATE-STORED-CHUNK-ID"
    events: list[str] = []
    output_dir = tmp_path / "reports"

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            events.append("store")

        def list_chunks(self) -> list[RetrievalHit]:
            events.append("list_chunks")
            return [_hit("public-chunk"), _hit(private_sentinel)]

    monkeypatch.setattr(sys, "argv", ["retrieval_eval.py", "--modes", "keyword", "--output-dir", str(output_dir)])
    monkeypatch.setattr(retrieval_eval, "load_evaluation_cases", lambda _path, **_kwargs: [])
    monkeypatch.setattr(retrieval_eval, "public_sample_chunk_ids", lambda: frozenset({"public-chunk"}))
    monkeypatch.setattr(retrieval_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(retrieval_eval, "build_embedding_provider", lambda: events.append("provider"))
    monkeypatch.setattr(retrieval_eval.KeywordSearchIndex, "from_hits", lambda _hits: events.append("keyword"))
    monkeypatch.setattr(retrieval_eval, "build_retrieval_pipeline", lambda **_kwargs: events.append("pipeline"))

    with pytest.raises(ValueError) as error:
        retrieval_eval.main()

    assert private_sentinel not in str(error.value)
    assert events == ["store", "list_chunks"]
    assert not output_dir.exists()


def test_run_provenance_has_hashes_sanitized_revision_and_retrieval_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"question_id":"q","question":"question","expected_chunk_ids":["public-chunk"],"topic":"scale"}\\n',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def list_chunks(self) -> list[RetrievalHit]:
            return [_hit("public-chunk")]

    provider = type("Provider", (), {"dimension": 12, "model_name": "test-model"})()
    monkeypatch.setattr(sys, "argv", ["retrieval_eval.py", "--dataset", str(dataset), "--modes", "keyword", "--privacy-mode", "private", "--output-dir", str(tmp_path / "reports")])
    monkeypatch.setattr(retrieval_eval, "load_evaluation_cases", lambda _path, **_kwargs: [EvaluationCase("q", "question", ("public-chunk",), "scale")])
    monkeypatch.setattr(retrieval_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(retrieval_eval, "build_embedding_provider", lambda: provider)
    monkeypatch.setattr(retrieval_eval.KeywordSearchIndex, "from_hits", lambda _hits: type("Index", (), {"search": lambda *_args, **_kwargs: []})())
    monkeypatch.setattr(retrieval_eval, "_git_revision", lambda: "unknown")
    monkeypatch.setattr(retrieval_eval, "write_report", lambda _results, _output_dir, **kwargs: captured.update(kwargs) or (_output_dir / "report.json", _output_dir / "report.md"))

    retrieval_eval.main()

    provenance = captured["provenance"]
    assert isinstance(provenance, RunProvenance)
    assert re.fullmatch(r"[0-9a-f]{64}", provenance.dataset_sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", provenance.corpus_sha256)
    assert provenance.git_revision == "unknown"
    assert provenance.embedding_provider == retrieval_eval.EmbeddingSettings.from_env().provider
    assert provenance.embedding_model == "sha256:" + hashlib.sha256(b"test-model").hexdigest()
    assert provenance.embedding_dimension == 12
    assert provenance.k == 5
    assert provenance.topic_filter == "oracle_gold_topic"
    assert provenance.retrieval_mode_settings["max_context_chars"] == 4000
    assert provenance.retrieval_mode_settings["hybrid_rrf_k"] == 60
    assert provenance.retrieval_mode_settings["hybrid_min_rrf_score"] == 0.015









def test_public_preflight_uses_configured_dimension_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []

    class FakeStore:
        def __init__(self, _database_url: str | None, *, embedding_dimension: int) -> None:
            events.append(("store", embedding_dimension))

        def list_chunks(self) -> list[RetrievalHit]:
            events.append("list_chunks")
            return [_hit("public-chunk"), _hit("unexpected-chunk")]

    monkeypatch.setattr(sys, "argv", ["retrieval_eval.py", "--modes", "keyword"])
    monkeypatch.setattr(retrieval_eval, "load_evaluation_cases", lambda _path, **_kwargs: [])
    monkeypatch.setattr(retrieval_eval, "public_sample_chunk_ids", lambda: frozenset({"public-chunk"}))
    monkeypatch.setattr(
        retrieval_eval.EmbeddingSettings,
        "from_env",
        lambda: retrieval_eval.EmbeddingSettings(dimension=777),
    )
    monkeypatch.setattr(retrieval_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(retrieval_eval, "build_embedding_provider", lambda: events.append("provider"))

    with pytest.raises(ValueError):
        retrieval_eval.main()

    assert events == [("store", 777), "list_chunks"]


def test_provenance_hashes_local_model_reference(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}", encoding="utf-8")
    model_reference = "C:/private/models/embedding.gguf"
    provenance = retrieval_eval._provenance(
        dataset,
        [],
        retrieval_eval.RetrievalSettings(),
        type("Provider", (), {"model_name": model_reference, "dimension": 12})(),
    )

    assert provenance.embedding_model == "sha256:" + hashlib.sha256(model_reference.encode()).hexdigest()
    assert model_reference not in provenance.embedding_model


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (OSError("git unavailable"), "unknown"),
        (subprocess.CompletedProcess(["git"], 0, "z" * 40, ""), "unknown"),
        (subprocess.CompletedProcess(["git"], 1, "a" * 40, "failure"), "unknown"),
        (subprocess.CompletedProcess(["git"], 0, "a" * 40, ""), "a" * 40),
    ],
)
def test_git_revision_accepts_only_successful_hexadecimal_sha(
    monkeypatch: pytest.MonkeyPatch, result: object, expected: str
) -> None:
    def run_git(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if isinstance(result, OSError):
            raise result
        assert isinstance(result, subprocess.CompletedProcess)
        return result

    monkeypatch.setattr(retrieval_eval.subprocess, "run", run_git)

    assert retrieval_eval._git_revision() == expected

def test_direct_public_cli_resolves_ingestion_import_before_database_access(tmp_path: Path) -> None:
    marker = tmp_path / "post_ingestion_marker"
    (tmp_path / "sitecustomize.py").write_text(
        """
import os
from pathlib import Path

import psycopg


def fail_connect(*_args, **_kwargs):
    Path(os.environ["RETRIEVAL_EVAL_POST_INGESTION_MARKER"]).write_text(
        "pgvector_list_chunks", encoding="utf-8"
    )
    raise RuntimeError("test database sentinel")


psycopg.connect = fail_connect
""".lstrip(),
        encoding="utf-8",
    )
    child_env = {
        **os.environ,
        "TMP": str(tmp_path),
        "TEMP": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "RETRIEVAL_EVAL_POST_INGESTION_MARKER": str(marker),
    }
    result = subprocess.run(
        [
            sys.executable,
            "eval/retrieval_eval.py",
            "--privacy-mode",
            "public",
            "--modes",
            "keyword",
            "--database-url",
            "postgresql://127.0.0.1:1/retrieval_eval?connect_timeout=1",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=30,
    )

    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "pgvector_list_chunks"