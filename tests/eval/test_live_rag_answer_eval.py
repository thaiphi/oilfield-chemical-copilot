import hashlib
import json
import sys
from pathlib import Path

import pytest

import eval.live_rag_answer_eval as live_rag_answer_eval
from oilfield_chemical_copilot.evaluation.answers import (
    AnswerEvaluationCase,
    AnswerEvaluationResult,
    DeterministicAnswerResult,
    GeneratedAnswer,
)
from oilfield_chemical_copilot.evaluation.judge import JudgeUnavailable
from oilfield_chemical_copilot.evaluation.live_rag import LiveAnswerCapture, RecordingRetriever
from oilfield_chemical_copilot.evaluation.live_rag_diagnosis import LiveFailureDiagnosis
from oilfield_chemical_copilot.retrieval.models import RetrievalHit


def _case() -> AnswerEvaluationCase:
    return AnswerEvaluationCase(
        "QUESTION-SENTINEL",
        "PUBLIC-QUESTION-SENTINEL",
        ("public-chunk",),
        True,
        True,
        False,
    )


def test_diagnostic_capture_uses_deterministic_results_without_a_judge() -> None:
    case = _case()
    capture = LiveAnswerCapture(
        GeneratedAnswer(case.question_id, "answer", "evidence", ("public-chunk",), False),
        ("public-chunk",),
        "succeeded",
    )

    results = live_rag_answer_eval._deterministic_results_from_captures([case], [capture])

    assert results == [DeterministicAnswerResult(case.question_id, "pass", "pass")]


def _hit(chunk_id: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text="EVIDENCE-SENTINEL",
        score=0.9,
        retrieval_method="stored",
        source_file="SOURCE-SENTINEL.md",
        source_path="C:/PATH-SENTINEL/secret.md",
        topic="scale",
        parser_type="markdown",
        page_or_sheet="",
        chunk_index=0,
    )






def _set_baseline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ANSWER_EVAL_JUDGE_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "granite4.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "granite-embedding:latest")


def _configure_minimal_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cases: list[AnswerEvaluationCase],
    captures: dict[str, LiveAnswerCapture],
) -> Path:
    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def list_chunks(self) -> list[RetrievalHit]:
            return [_hit("public-chunk")]

    class FakeIndex:
        @classmethod
        def from_hits(cls, _hits: list[RetrievalHit]) -> "FakeIndex":
            return cls()

    class FakeService:
        @classmethod
        def from_settings(cls, **_kwargs: object) -> "FakeService":
            return cls()

    class FakeJudge:
        def judge(self, case: AnswerEvaluationCase, **_kwargs: object) -> JudgeUnavailable:
            return JudgeUnavailable(case.question_id, "local", "local")

    output_dir = tmp_path / "reports"
    settings = live_rag_answer_eval.RetrievalSettings()
    _set_baseline_environment(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_rag_answer_eval.py",
            "--dataset",
            str(live_rag_answer_eval.PROJECT_ROOT / "eval" / "public_answer_evaluation.jsonl"),
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(live_rag_answer_eval, "load_answer_evaluation_cases", lambda _path: cases)
    monkeypatch.setattr(live_rag_answer_eval, "public_sample_chunk_ids", lambda: frozenset({"public-chunk"}))
    monkeypatch.setattr(live_rag_answer_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(live_rag_answer_eval.EmbeddingSettings, "from_env", lambda: type("Settings", (), {"dimension": 384})())
    monkeypatch.setattr(live_rag_answer_eval, "build_embedding_provider", lambda: object())
    monkeypatch.setattr(live_rag_answer_eval, "KeywordSearchIndex", FakeIndex)
    monkeypatch.setattr(live_rag_answer_eval.RetrievalSettings, "from_env", lambda: settings)
    monkeypatch.setattr(live_rag_answer_eval, "build_retrieval_pipeline", lambda **_kwargs: object())
    monkeypatch.setattr(live_rag_answer_eval, "BasicRagService", FakeService)
    monkeypatch.setattr(live_rag_answer_eval, "build_live_ollama_generator", lambda: object())
    monkeypatch.setattr(
        live_rag_answer_eval,
        "capture_live_answer",
        lambda case, _service, _generator: captures[case.question_id],
    )
    monkeypatch.setattr(live_rag_answer_eval, "AnswerJudge", FakeJudge)
    return output_dir

def test_argument_parser_defaults_to_live_rag_output_directory() -> None:
    args = live_rag_answer_eval._build_argument_parser().parse_args([])

    assert args.output_dir == Path("data/processed/evaluation/live_rag")


def test_citation_diagnostic_destination_must_be_under_private_root(tmp_path: Path) -> None:
    private_root = tmp_path / ".private"

    assert live_rag_answer_eval._require_private_diagnostic_destination(
        private_root / "citation-selection" / "capture.json", private_root
    ) == (private_root / "citation-selection" / "capture.json").resolve()
    with pytest.raises(ValueError, match="^citation diagnostics must be stored under .private$"):
        live_rag_answer_eval._require_private_diagnostic_destination(
            tmp_path / "capture.json", private_root
        )
def test_outside_dataset_path_is_rejected_before_cases_are_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    private_dataset = tmp_path / "private-answer-evaluation.jsonl"

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(
        sys, "argv", ["live_rag_answer_eval.py", "--dataset", str(private_dataset)]
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "load_answer_evaluation_cases",
        lambda _path: events.append("load") or pytest.fail("cases must not be loaded"),
    )

    monkeypatch.setattr(live_rag_answer_eval, "PgVectorStore", lambda *_args, **_kwargs: pytest.fail("store must not be constructed"))
    monkeypatch.setattr(live_rag_answer_eval, "build_embedding_provider", lambda: pytest.fail("provider must not be constructed"))

    with pytest.raises(ValueError, match="approved public answer evaluation dataset"):
        live_rag_answer_eval.main()

    assert events == []


def test_canonical_dataset_hash_mismatch_is_rejected_before_cases_are_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_baseline_environment(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["live_rag_answer_eval.py"])
    monkeypatch.setattr(live_rag_answer_eval, "_sha256", lambda _value: "0" * 64)
    monkeypatch.setattr(
        live_rag_answer_eval,
        "load_answer_evaluation_cases",
        lambda _path: pytest.fail("cases must not be loaded"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "PgVectorStore",
        lambda *_args, **_kwargs: pytest.fail("store must not be constructed"),
    )

    with pytest.raises(ValueError, match="approved canonical dataset"):
        live_rag_answer_eval.main()


@pytest.mark.parametrize(
    "settings",
    (
        live_rag_answer_eval.RetrievalSettings(top_k=6),
        live_rag_answer_eval.RetrievalSettings(min_score=0.25),
        live_rag_answer_eval.RetrievalSettings(max_context_chars=4001),
        live_rag_answer_eval.RetrievalSettings(hybrid_candidate_limit=11),
        live_rag_answer_eval.RetrievalSettings(hybrid_rrf_k=61),
        live_rag_answer_eval.RetrievalSettings(hybrid_min_rrf_score=0.02),
        live_rag_answer_eval.RetrievalSettings(retrieval_mode="vector"),
    ),
)
def test_nonbaseline_retrieval_or_embedding_settings_reject_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch, settings: object
) -> None:
    _set_baseline_environment(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["live_rag_answer_eval.py"])
    monkeypatch.setattr(
        live_rag_answer_eval.RetrievalSettings, "from_env", lambda: settings
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "load_answer_evaluation_cases",
        lambda _path: pytest.fail("cases must not be loaded"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "PgVectorStore",
        lambda *_args, **_kwargs: pytest.fail("store must not be constructed"),
    )

    with pytest.raises(ValueError, match="approved retrieval settings"):
        live_rag_answer_eval.main()


def test_nonbaseline_embedding_dimension_rejects_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_baseline_environment(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["live_rag_answer_eval.py"])
    monkeypatch.setattr(
        live_rag_answer_eval.EmbeddingSettings,
        "from_env",
        lambda: type("Settings", (), {"dimension": 385})(),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "load_answer_evaluation_cases",
        lambda _path: pytest.fail("cases must not be loaded"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "PgVectorStore",
        lambda *_args, **_kwargs: pytest.fail("store must not be constructed"),
    )

    with pytest.raises(ValueError, match="approved embedding dimension"):
        live_rag_answer_eval.main()


def test_non_ollama_provider_is_rejected_before_cases_are_loaded(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(sys, "argv", ["live_rag_answer_eval.py"])
    monkeypatch.setattr(
        live_rag_answer_eval,
        "load_answer_evaluation_cases",
        lambda _path: events.append("load") or pytest.fail("cases must not be loaded"),
    )

    monkeypatch.setattr(live_rag_answer_eval, "PgVectorStore", lambda *_args, **_kwargs: pytest.fail("store must not be constructed"))
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_embedding_provider",
        lambda: pytest.fail("embedding provider must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_live_ollama_generator",
        lambda: pytest.fail("generator must not be constructed"),
    )

    with pytest.raises(ValueError, match="LLM_PROVIDER must be 'ollama'"):
        live_rag_answer_eval.main()

    assert events == []


def test_non_ollama_judge_is_rejected_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_baseline_environment(monkeypatch)
    monkeypatch.setenv("ANSWER_EVAL_JUDGE_PROVIDER", "openai")
    monkeypatch.setattr(sys, "argv", ["live_rag_answer_eval.py"])
    monkeypatch.setattr(
        live_rag_answer_eval,
        "load_answer_evaluation_cases",
        lambda _path: pytest.fail("cases must not be loaded"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "PgVectorStore",
        lambda *_args, **_kwargs: pytest.fail("store must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_embedding_provider",
        lambda: pytest.fail("embedding provider must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_live_ollama_generator",
        lambda: pytest.fail("generator must not be constructed"),
    )

    with pytest.raises(ValueError, match="ANSWER_EVAL_JUDGE_PROVIDER must be 'ollama'"):
        live_rag_answer_eval.main()

@pytest.mark.parametrize(
    ("name", "value", "error"),
    (
        ("OLLAMA_MODEL", "other-answer-model", "OLLAMA_MODEL must be 'granite4.1:8b'"),
        ("OLLAMA_BASE_URL", "http://LOCAL-OLLAMA-URL-SENTINEL", "OLLAMA_BASE_URL must be 'http://localhost:11434'"),
        ("EMBEDDING_PROVIDER", "deterministic", "EMBEDDING_PROVIDER must be 'ollama'"),
        (
            "OLLAMA_EMBEDDING_MODEL",
            "other-embedding-model",
            "OLLAMA_EMBEDDING_MODEL must be 'granite-embedding:latest'",
        ),
    ),
)
def test_nonbaseline_model_or_embedding_configuration_rejects_before_construction(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, error: str
) -> None:
    _set_baseline_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["live_rag_answer_eval.py"])
    monkeypatch.setattr(
        live_rag_answer_eval,
        "load_answer_evaluation_cases",
        lambda _path: pytest.fail("cases must not be loaded"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "PgVectorStore",
        lambda *_args, **_kwargs: pytest.fail("store must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_embedding_provider",
        lambda: pytest.fail("embedding provider must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_live_ollama_generator",
        lambda: pytest.fail("generator must not be constructed"),
    )

    with pytest.raises(ValueError, match=error):
        live_rag_answer_eval.main()

@pytest.mark.parametrize("stored_ids", ({"public-chunk", "private-source"}, {"public-chunk"}))
def test_mixed_or_incomplete_stored_ids_reject_before_runtime_builders(
    monkeypatch: pytest.MonkeyPatch, stored_ids: set[str], tmp_path: Path
) -> None:
    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def list_chunks(self) -> list[RetrievalHit]:
            return [_hit(chunk_id) for chunk_id in stored_ids]

    _set_baseline_environment(monkeypatch)
    output_dir = tmp_path / "blocked-output"
    monkeypatch.setattr(sys, "argv", ["live_rag_answer_eval.py", "--output-dir", str(output_dir)])
    monkeypatch.setattr(live_rag_answer_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(
        live_rag_answer_eval, "public_sample_chunk_ids", lambda: frozenset({"public-chunk", "missing"})
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_embedding_provider",
        lambda: pytest.fail("embedding provider must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_live_ollama_generator",
        lambda: pytest.fail("generator must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_retrieval_pipeline",
        lambda **_kwargs: pytest.fail("pipeline must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "write_live_failure_diagnosis",
        lambda *_args, **_kwargs: pytest.fail("writer must not be called"),
    )

    with pytest.raises(ValueError, match="stored chunk IDs do not match public manifest"):
        live_rag_answer_eval.main()

    assert not output_dir.exists()

def test_complete_public_manifest_with_extra_chunk_rejects_before_runtime_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def list_chunks(self) -> list[RetrievalHit]:
            return [_hit("public-chunk"), _hit("private-chunk")]

    _set_baseline_environment(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["live_rag_answer_eval.py"])
    monkeypatch.setattr(live_rag_answer_eval, "load_answer_evaluation_cases", lambda _path: [])
    monkeypatch.setattr(live_rag_answer_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(
        live_rag_answer_eval, "public_sample_chunk_ids", lambda: frozenset({"public-chunk"})
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_embedding_provider",
        lambda: pytest.fail("embedding provider must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_live_ollama_generator",
        lambda: pytest.fail("generator must not be constructed"),
    )
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_retrieval_pipeline",
        lambda **_kwargs: pytest.fail("pipeline must not be constructed"),
    )

    with pytest.raises(ValueError, match="stored chunk IDs do not match public manifest"):
        live_rag_answer_eval.main()


def test_runner_rejects_swapped_deterministic_pairings_before_writing_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = [
        AnswerEvaluationCase("CASE-ID-SENTINEL-first", "QUESTION-SENTINEL-first", ("public-chunk",), True, True, False),
        AnswerEvaluationCase("CASE-ID-SENTINEL-second", "QUESTION-SENTINEL-second", ("public-chunk",), True, True, False),
    ]
    captures = {
        cases[0].question_id: LiveAnswerCapture(GeneratedAnswer(cases[0].question_id, "ANSWER-SENTINEL", "EVIDENCE-SENTINEL", (), False), ("public-chunk",), "succeeded"),
        cases[1].question_id: LiveAnswerCapture(GeneratedAnswer(cases[1].question_id, "ANSWER-SENTINEL", "EVIDENCE-SENTINEL", ("public-chunk",), False), ("public-chunk",), "succeeded"),
    }
    output_dir = _configure_minimal_runner(monkeypatch, tmp_path, cases, captures)

    def swapped_results(*_args: object, **_kwargs: object) -> list[AnswerEvaluationResult]:
        unavailable = JudgeUnavailable("CASE-ID-SENTINEL", "local", "local")
        return [
            AnswerEvaluationResult(cases[0].question_id, DeterministicAnswerResult(cases[0].question_id, "pass", "pass"), unavailable),
            AnswerEvaluationResult(cases[1].question_id, DeterministicAnswerResult(cases[1].question_id, "fail", "pass"), unavailable),
        ]

    monkeypatch.setattr(live_rag_answer_eval, "evaluate_cases", swapped_results)

    with pytest.raises(ValueError, match="^invalid diagnostic state$") as error:
        live_rag_answer_eval.main()

    assert not output_dir.exists()
    assert "SENTINEL" not in str(error.value)


def test_runner_rejects_contradictory_capture_before_writing_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = AnswerEvaluationCase("CASE-ID-SENTINEL", "QUESTION-SENTINEL", ("public-chunk",), True, True, False)
    captures = {
        case.question_id: LiveAnswerCapture(
            GeneratedAnswer(case.question_id, "ANSWER-SENTINEL", "EVIDENCE-SENTINEL", ("public-chunk",), False),
            (),
            "succeeded",
        )
    }
    output_dir = _configure_minimal_runner(monkeypatch, tmp_path, [case], captures)

    with pytest.raises(ValueError, match="^invalid diagnostic state$") as error:
        live_rag_answer_eval.main()

    assert not output_dir.exists()
    assert "SENTINEL" not in str(error.value)


def test_runner_rejects_unknown_category_without_diagnosis_output_or_data_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = AnswerEvaluationCase("CASE-ID-SENTINEL", "QUESTION-SENTINEL", ("public-chunk",), True, True, False)
    captures = {
        case.question_id: LiveAnswerCapture(
            GeneratedAnswer(case.question_id, "ANSWER-SENTINEL", "EVIDENCE-SENTINEL", (), False),
            ("public-chunk",),
            "succeeded",
        )
    }
    output_dir = _configure_minimal_runner(monkeypatch, tmp_path, [case], captures)
    monkeypatch.setattr(
        live_rag_answer_eval,
        "classify_live_failure",
        lambda *_args, **_kwargs: LiveFailureDiagnosis("UNKNOWN-CATEGORY-SENTINEL", None),
    )

    with pytest.raises(ValueError, match="^invalid diagnosis report$") as error:
        live_rag_answer_eval.main()

    assert not (output_dir / "live_rag_failure_diagnosis.json").exists()
    assert not (output_dir / "live_rag_failure_diagnosis.md").exists()
    assert "SENTINEL" not in str(error.value)

def test_public_runner_compares_modes_without_retaining_runtime_answer_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = [
        AnswerEvaluationCase("CASE-ID-SENTINEL-retrieval", "PUBLIC-QUESTION-SENTINEL-retrieval", ("public-chunk",), True, True, False),
        AnswerEvaluationCase("CASE-ID-SENTINEL-generation", "PUBLIC-QUESTION-SENTINEL-generation", ("public-chunk",), True, True, False),
        AnswerEvaluationCase("CASE-ID-SENTINEL-abstained", "PUBLIC-QUESTION-SENTINEL-abstained", ("public-chunk",), True, True, False),
        AnswerEvaluationCase("CASE-ID-SENTINEL-selection", "PUBLIC-QUESTION-SENTINEL-selection", ("public-chunk",), True, True, False),
        AnswerEvaluationCase("CASE-ID-SENTINEL-under", "PUBLIC-QUESTION-SENTINEL-under", ("public-chunk",), False, False, True),
    ]
    stored_hits = [_hit("public-chunk")]
    events: list[object] = []
    service_calls: list[tuple[str, str]] = []
    output_dir = tmp_path / "reports"

    class FakeStore:
        def __init__(self, database_url: str | None, *, embedding_dimension: int) -> None:
            events.append(("store", database_url, embedding_dimension))

        def list_chunks(self) -> list[RetrievalHit]:
            events.append("list_chunks")
            return stored_hits

    class FakeKeywordIndex:
        @classmethod
        def from_hits(cls, hits: list[RetrievalHit]) -> "FakeKeywordIndex":
            events.append(("keyword", hits))
            return cls()

    class FakePipeline:
        def __init__(self, mode: str) -> None:
            self.mode = mode

    class FakeService:
        def __init__(
            self,
            *,
            retriever: RecordingRetriever,
            generator: object,
            min_score: float,
            max_context_chars: int,
            apply_claim_scope_policy: bool,
        ) -> None:
            self.mode = retriever.delegate.mode
            events.append(
                (
                    "service",
                    self.mode,
                    generator,
                    min_score,
                    max_context_chars,
                    apply_claim_scope_policy,
                )
            )


        @classmethod
        def from_settings(
            cls,
            *,
            retriever: FakePipeline,
            generator: object,
            settings: object,
            apply_claim_scope_policy: bool,
        ) -> "FakeService":
            return cls(
                retriever=retriever,
                generator=generator,
                min_score=settings.evidence_threshold,
                max_context_chars=settings.max_context_chars,
                apply_claim_scope_policy=apply_claim_scope_policy,
            )
        def answer(self, question: str) -> object:
            service_calls.append((self.mode, question))
            return object()

    class FakeJudge:
        def judge(self, case: AnswerEvaluationCase, *, answer: str, evidence: str) -> JudgeUnavailable:
            unavailable = JudgeUnavailable(
                case.question_id,
                "https://JUDGE-PROVIDER-URL-SENTINEL.example",
                "RAW-JUDGE-MODEL-SENTINEL",
            )
            object.__setattr__(unavailable, "reason", "RAW-JUDGE-ERROR-SENTINEL")
            return unavailable

    def capture(case: AnswerEvaluationCase, service: FakeService, _generator: object) -> LiveAnswerCapture:
        answer = service.answer(case.question)
        assert answer is not None
        captures = {
            "CASE-ID-SENTINEL-retrieval": ((), (), True, "not_called"),
            "CASE-ID-SENTINEL-generation": (("public-chunk",), (), True, "failed"),
            "CASE-ID-SENTINEL-abstained": (("public-chunk",), (), True, "succeeded"),
            "CASE-ID-SENTINEL-selection": (("public-chunk", "other"), ("other",), False, "succeeded"),
            "CASE-ID-SENTINEL-under": (("public-chunk",), ("public-chunk",), False, "succeeded"),
        }
        retrieved_ids, cited_ids, abstained, generation_outcome = captures[case.question_id]
        return LiveAnswerCapture(
            GeneratedAnswer(
                case.question_id,
                "ANSWER-SENTINEL https://URL-SENTINEL.example KEY-SENTINEL ERROR-SENTINEL",
                "EVIDENCE-SENTINEL C:/PATH-SENTINEL SOURCE-SENTINEL",
                cited_ids,
                abstained,
            ),
            retrieved_ids,
            generation_outcome,
        )

    configured_settings = live_rag_answer_eval.RetrievalSettings()
    embedding_provider = type("Provider", (), {"dimension": 384})()
    recording_generator = object()
    _set_baseline_environment(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_rag_answer_eval.py",
            "--dataset",
            str(live_rag_answer_eval.PROJECT_ROOT / "eval" / "public_answer_evaluation.jsonl"),
            "--output-dir",
            str(output_dir),
            "--database-url",
            "postgresql://eval-user:CREDENTIAL-SENTINEL@localhost/live-rag",
        ],
    )
    monkeypatch.setattr(live_rag_answer_eval, "load_answer_evaluation_cases", lambda _path: cases)
    monkeypatch.setattr(live_rag_answer_eval, "public_sample_chunk_ids", lambda: frozenset({"public-chunk"}))
    monkeypatch.setattr(live_rag_answer_eval, "PgVectorStore", FakeStore)
    monkeypatch.setattr(live_rag_answer_eval, "build_embedding_provider", lambda: embedding_provider)
    monkeypatch.setattr(live_rag_answer_eval, "KeywordSearchIndex", FakeKeywordIndex)
    monkeypatch.setattr(live_rag_answer_eval.RetrievalSettings, "from_env", lambda: configured_settings)
    monkeypatch.setattr(
        live_rag_answer_eval,
        "build_retrieval_pipeline",
        lambda *, store, embedding_provider, settings, keyword_index=None: (
            events.append(("pipeline", store, embedding_provider, settings, keyword_index))
            or FakePipeline(settings.retrieval_mode)
        ),
    )
    monkeypatch.setattr(live_rag_answer_eval, "BasicRagService", FakeService)
    monkeypatch.setattr(live_rag_answer_eval, "build_live_ollama_generator", lambda: recording_generator)
    monkeypatch.setattr(live_rag_answer_eval, "capture_live_answer", capture)
    monkeypatch.setattr(live_rag_answer_eval, "AnswerJudge", FakeJudge)
    original_validate_manifest = live_rag_answer_eval.validate_public_stored_chunk_ids

    def validate_manifest(stored_ids: set[str], public_ids: frozenset[str]) -> None:
        events.append("manifest-validated")
        original_validate_manifest(stored_ids, public_ids)

    writer_calls: list[bool] = []
    original_writer = live_rag_answer_eval.write_live_failure_diagnosis

    def write_diagnosis(*args: object, **kwargs: object) -> tuple[Path, Path]:
        events.append("diagnosis-written")
        writer_calls.append(kwargs["verified_preflight"])
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(live_rag_answer_eval, "validate_public_stored_chunk_ids", validate_manifest)
    monkeypatch.setattr(live_rag_answer_eval, "write_live_failure_diagnosis", write_diagnosis)

    live_rag_answer_eval.main()

    assert events.count("list_chunks") == 1
    assert events.count(("keyword", stored_hits)) == 1
    pipelines = [event for event in events if isinstance(event, tuple) and event[0] == "pipeline"]
    assert [event[3].retrieval_mode for event in pipelines] == ["vector", "hybrid"]
    assert all(event[3].top_k == configured_settings.top_k for event in pipelines)
    assert all(event[3].min_score == configured_settings.min_score for event in pipelines)
    assert all(event[3].max_context_chars == configured_settings.max_context_chars for event in pipelines)
    assert all(event[3].hybrid_candidate_limit == configured_settings.hybrid_candidate_limit for event in pipelines)
    assert pipelines[0][4] is None
    assert isinstance(pipelines[1][4], FakeKeywordIndex)
    services = [event for event in events if isinstance(event, tuple) and event[0] == "service"]
    assert [event[-1] for event in services] == [False, False]
    assert service_calls == [
        (mode, case.question) for mode in ("vector", "hybrid") for case in cases
    ]
    assert writer_calls == [True]
    assert events.index("manifest-validated") < events.index("diagnosis-written")

    report_paths = sorted(output_dir.glob("*"))
    assert [path.name for path in report_paths] == [
        "answer_eval_comparison.json",
        "answer_eval_comparison.md",
        "live_rag_failure_diagnosis.json",
        "live_rag_failure_diagnosis.md",
    ]
    report = json.loads((output_dir / "answer_eval_comparison.json").read_text(encoding="utf-8"))
    assert report["provenance"] == {
        "corpus_sha256": hashlib.sha256(b"public-chunk").hexdigest(),
        "dataset_sha256": hashlib.sha256(
            (live_rag_answer_eval.PROJECT_ROOT / "eval" / "public_answer_evaluation.jsonl").read_bytes()
        ).hexdigest(),
        "embedding_provider": "ollama",
        "generation_model_sha256": hashlib.sha256(b"granite4.1:8b").hexdigest(),
        "generation_provider": "ollama",
        "judge_model_sha256": hashlib.sha256(b"granite4.1:8b").hexdigest(),
        "judge_provider": "ollama",
        "retrieval_mode_settings": {"hybrid": "hybrid", "vector": "vector"},
        "retrieval_settings": {
            "hybrid_candidate_limit": 10,
            "hybrid_min_rrf_score": 0.015,
            "hybrid_rrf_k": 60,
            "max_context_chars": 4000,
            "min_score": 0.2,
            "top_k": 5,
        },
        "temperature": 0,
        "topic_filter": "none",
    }
    for mode in ("vector", "hybrid"):
        assert report["modes"][mode]["question_count"] == 5
        assert report["modes"][mode]["judge"]["status"] == {"unavailable": 5}
        assert "providers" not in report["modes"][mode]["judge"]
    diagnosis = json.loads((output_dir / "live_rag_failure_diagnosis.json").read_text(encoding="utf-8"))
    assert set(diagnosis) == {"baseline_reproduced", "modes", "provenance", "public"}
    assert diagnosis["baseline_reproduced"] is False
    assert diagnosis["modes"] == {
        "vector": {
            "question_count": 5,
            "citation_failures": {
                "expected_citation_allowed_retrieved_not_cited": 1,
                "expected_citation_missing_abstained_after_qualifying_retrieval": 1,
                "expected_citation_missing_generation_failure": 1,
                "expected_citation_missing_no_qualifying_retrieval": 1,
                "unexpected_citation_when_abstention_expected": 1,
            },
            "abstention_failures": {
                "over_abstention_after_qualifying_retrieval": 1,
                "over_abstention_generation_failure": 1,
                "over_abstention_no_qualifying_retrieval": 1,
                "under_abstention_answered_on_insufficient_case": 1,
            },
        },
        "hybrid": {
            "question_count": 5,
            "citation_failures": {
                "expected_citation_allowed_retrieved_not_cited": 1,
                "expected_citation_missing_abstained_after_qualifying_retrieval": 1,
                "expected_citation_missing_generation_failure": 1,
                "expected_citation_missing_no_qualifying_retrieval": 1,
                "unexpected_citation_when_abstention_expected": 1,
            },
            "abstention_failures": {
                "over_abstention_after_qualifying_retrieval": 1,
                "over_abstention_generation_failure": 1,
                "over_abstention_no_qualifying_retrieval": 1,
                "under_abstention_answered_on_insufficient_case": 1,
            },
        },
    }
    serialized = "".join(path.read_text(encoding="utf-8") for path in report_paths)
    for sentinel in (
        "QUESTION-SENTINEL",
        "CASE-ID-SENTINEL",
        "PUBLIC-QUESTION-SENTINEL",
        "ANSWER-SENTINEL",
        "EVIDENCE-SENTINEL",
        "SOURCE-SENTINEL",
        "public-chunk",
        "PATH-SENTINEL",
        "http://localhost:11434",
        "postgresql://eval-user:CREDENTIAL-SENTINEL@localhost/live-rag",
        "CREDENTIAL-SENTINEL",
        "URL-SENTINEL",
        "KEY-SENTINEL",
        "ERROR-SENTINEL",
        "RAW-JUDGE-ERROR-SENTINEL",
        "RAW-JUDGE-MODEL-SENTINEL",
        "JUDGE-PROVIDER-URL-SENTINEL",
    ):
        assert sentinel not in serialized
    assert "answer_eval_comparison.json" in capsys.readouterr().out


def test_opt_in_policy_shadow_reuses_one_capture_per_case_and_writes_safe_aggregates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = [
        *[
            AnswerEvaluationCase(
                f"allow-{index}",
                f"What indicators should frame a general review {index}?",
                ("public-chunk",),
                True,
                True,
                False,
            )
            for index in range(6)
        ],
        *[
            AnswerEvaluationCase(
                f"abstain-{index}",
                f"Can this replace the complete analysis for asset {index}?",
                ("public-chunk",),
                False,
                False,
                True,
            )
            for index in range(6)
        ],
    ]
    captures = {
        case.question_id: LiveAnswerCapture(
            GeneratedAnswer(
                case.question_id,
                "ANSWER-SENTINEL",
                "EVIDENCE-SENTINEL",
                ("public-chunk",) if index < 4 or index >= 6 else ("other",),
                False,
            ),
            ("public-chunk", "other") if 4 <= index < 6 else ("public-chunk",),
            "succeeded",
        )
        for index, case in enumerate(cases)
    }
    output_dir = _configure_minimal_runner(monkeypatch, tmp_path, cases, captures)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_rag_answer_eval.py",
            "--dataset",
            str(live_rag_answer_eval.PROJECT_ROOT / "eval" / "public_answer_evaluation.jsonl"),
            "--output-dir",
            str(output_dir),
            "--abstention-policy-shadow",
            "claim_scope_v1",
        ],
    )
    classifier_inputs: list[object] = []
    original_classifier = live_rag_answer_eval.classify_claim_scope

    def classify_question_only(question: str) -> object:
        classifier_inputs.append(question)
        return original_classifier(question)

    capture_calls: list[str] = []
    original_capture = live_rag_answer_eval.capture_live_answer

    def capture_once(case: AnswerEvaluationCase, service: object, generator: object) -> LiveAnswerCapture:
        capture_calls.append(case.question_id)
        return original_capture(case, service, generator)

    monkeypatch.setattr(live_rag_answer_eval, "classify_claim_scope", classify_question_only)
    monkeypatch.setattr(live_rag_answer_eval, "capture_live_answer", capture_once)

    live_rag_answer_eval.main()

    assert classifier_inputs == [case.question for case in cases]
    assert all(isinstance(question, str) for question in classifier_inputs)
    assert capture_calls == [case.question_id for _mode in ("vector", "hybrid") for case in cases]
    assert sorted(path.name for path in output_dir.glob("*")) == [
        "answer_eval_comparison.json",
        "answer_eval_comparison.md",
        "live_rag_failure_diagnosis.json",
        "live_rag_failure_diagnosis.md",
        "live_rag_policy_investigation.json",
        "live_rag_policy_investigation.md",
    ]
    policy_report = json.loads(
        (output_dir / "live_rag_policy_investigation.json").read_text(encoding="utf-8")
    )
    assert policy_report["verified_preflight"] is True
    assert policy_report["baseline_reproduced"] is True
    for mode in ("vector", "hybrid"):
        assert policy_report["modes"][mode]["decision_categories"] == {
            "complete_input_substitution": 6,
            "general_review": 6,
        }
        assert policy_report["modes"][mode]["shadow"] == {
            "abstention": {"pass": 12},
            "citation": {"fail": 2, "pass": 10},
        }
    serialized = "".join(path.read_text(encoding="utf-8") for path in output_dir.glob("*"))
    for sentinel in ("ANSWER-SENTINEL", "EVIDENCE-SENTINEL", "public-chunk", "allow-0"):
        assert sentinel not in serialized
