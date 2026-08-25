from __future__ import annotations
import pytest

import app.streamlit_app as streamlit_app

from app.streamlit_app import (
    _answer_question,
    _agentic_routing_enabled,
    _build_rag_service,
    _citation_display,
    _database_url,
    _monitoring_database_url,
    _route_prompt,
    _route_prompt_with_outcome,
    _record_feedback,
    _record_request,
    _excerpt,
)
from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision
from oilfield_chemical_copilot.observability.aggregate_monitoring import (
    FeedbackValue,
    MonitoringOutcome,
    RetrievalMode,
)
from oilfield_chemical_copilot.rag.models import RagAnswer
from oilfield_chemical_copilot.ollama import OllamaClientError
from oilfield_chemical_copilot.rag.models import RagConfigurationError
from oilfield_chemical_copilot.rag.models import SourceEvidence
from oilfield_chemical_copilot.retrieval.pipeline import RetrievalSettings


def test_citation_display_hides_absolute_path_and_keeps_chunk_metadata() -> None:
    source = SourceEvidence(
        source_id="Source 1",
        chunk_id="scale-1",
        source_file="docs/scale.md",
        page_or_sheet="document",
        topic="scale",
        excerpt="Scale evidence",
        score=0.91,
    )

    display = _citation_display(source)

    assert display == "Source 1: docs/scale.md | document | chunk scale-1 | score 0.910 | vector: vector"
    assert "C:/" not in display


def test_excerpt_is_bounded() -> None:
    assert _excerpt("a" * 90, limit=20) == "a" * 17 + "..."


def test_lazy_openai_generator_does_not_require_key_until_generation(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from oilfield_chemical_copilot.rag.openai_client import LazyOpenAIAnswerClient

    generator = LazyOpenAIAnswerClient()

    assert generator is not None


def test_rag_service_builds_the_selected_retrieval_mode(monkeypatch) -> None:
    sentinel_generator = object()
    sentinel_chunks = [object()]
    sentinel_store = type("Store", (), {"list_chunks": lambda self: sentinel_chunks})()
    sentinel_retriever = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr("app.streamlit_app.build_answer_generator", lambda: sentinel_generator)
    monkeypatch.setattr(
        "app.streamlit_app.build_embedding_provider",
        lambda: type("Provider", (), {"dimension": 384})(),
    )
    monkeypatch.setattr(
        "app.streamlit_app.PgVectorStore", lambda *args, **kwargs: sentinel_store
    )

    class FakeKeywordIndex:
        @classmethod
        def from_hits(cls, hits):
            captured["keyword_hits"] = hits
            return "keyword-index"

    def fake_build_pipeline(*, store, embedding_provider, settings, keyword_index):
        captured["store"] = store
        captured["settings"] = settings
        captured["keyword_index"] = keyword_index
        return sentinel_retriever

    monkeypatch.setattr("app.streamlit_app.KeywordSearchIndex", FakeKeywordIndex, raising=False)
    monkeypatch.setattr(
        "app.streamlit_app.build_retrieval_pipeline", fake_build_pipeline, raising=False
    )
    monkeypatch.setattr(
        "app.streamlit_app.RetrievalSettings.from_env", lambda: RetrievalSettings()
    )
    _build_rag_service.clear()

    hybrid_service = _build_rag_service("hybrid")
    assert hybrid_service.generator is sentinel_generator
    assert captured["store"] is sentinel_store
    assert captured["settings"].retrieval_mode == "hybrid"
    assert captured["keyword_index"] == "keyword-index"
    assert captured["keyword_hits"] is sentinel_chunks

    vector_service = _build_rag_service("vector")
    assert vector_service.generator is sentinel_generator
    assert captured["settings"].retrieval_mode == "vector"
    assert captured["keyword_index"] is None
    _build_rag_service.clear()


def test_database_url_defaults_to_localhost_for_local_streamlit(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert _database_url() == "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"


def test_database_url_preserves_explicit_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://custom/value")

    assert _database_url() == "postgresql://custom/value"


def test_monitoring_database_url_is_independent_from_rag_database(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag/private")
    monkeypatch.delenv("MONITORING_DATABASE_URL", raising=False)

    assert _monitoring_database_url() == "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"

    monkeypatch.setenv("MONITORING_DATABASE_URL", "postgresql://monitoring/public")
    assert _monitoring_database_url() == "postgresql://monitoring/public"


def test_citation_display_hides_absolute_windows_path() -> None:
    source = SourceEvidence(
        source_id="Source 1",
        chunk_id="scale-1",
        source_file="C:/Users/Alice/private/scale.md",
        page_or_sheet="document",
        topic="scale",
        excerpt="Scale evidence",
        score=0.91,
    )

    display = _citation_display(source)
    assert "C:/" not in display
    assert "scale.md" in display

def test_citation_display_shows_safe_hybrid_retrieval_provenance() -> None:
    source = SourceEvidence(
        source_id="Source 1",
        chunk_id="scale-1",
        source_file="docs/scale.md",
        page_or_sheet="document",
        topic="scale",
        excerpt="Scale evidence",
        score=0.033,
        retrieval_method="hybrid",
        retrieval_sources=("keyword", "vector"),
    )

    display = _citation_display(source)

    assert display == (
        "Source 1: docs/scale.md | document | chunk scale-1 | score 0.033 | "
        "hybrid: keyword + vector"
    )


def test_answer_question_hides_ollama_retrieval_error_details(monkeypatch) -> None:
    class FailingService:
        def answer(self, _prompt: str):
            raise OllamaClientError("provider response body: private corpus excerpt")

    monkeypatch.setattr("app.streamlit_app._build_rag_service", lambda _mode: FailingService())

    with pytest.raises(RagConfigurationError, match="Ollama retrieval is unavailable") as error:
        _answer_question("How should I assess scale risk?", "hybrid")

    assert "private corpus excerpt" not in str(error.value)


def test_agentic_routing_flag_only_enables_case_insensitive_true(monkeypatch) -> None:
    monkeypatch.delenv("AGENTIC_ROUTING_ENABLED", raising=False)
    assert _agentic_routing_enabled() is False

    monkeypatch.setenv("AGENTIC_ROUTING_ENABLED", "TRUE")
    assert _agentic_routing_enabled() is True

    monkeypatch.setenv("AGENTIC_ROUTING_ENABLED", "yes")
    assert _agentic_routing_enabled() is False


def test_answer_question_builds_agentic_service_only_when_enabled(monkeypatch) -> None:
    expected = RagAnswer(text="Agentic response", sources=[], weak_evidence=False)
    rag_service = object()
    captured: dict[str, object] = {}

    class FakePlanner:
        def __init__(self, *, model: str, client: object) -> None:
            captured["model"] = model
            captured["client"] = client

    class FakeAgenticService:
        def __init__(self, *, rag_service: object, planner: object) -> None:
            captured["rag_service"] = rag_service
            captured["planner"] = planner

        def answer(self, prompt: str) -> RagAnswer:
            captured["prompt"] = prompt
            return expected

    monkeypatch.setenv("AGENTIC_ROUTING_ENABLED", "true")
    monkeypatch.setenv("OLLAMA_MODEL", "test-granite")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    monkeypatch.setattr("app.streamlit_app._build_rag_service", lambda _mode: rag_service)
    monkeypatch.setattr("app.streamlit_app.OllamaToolPlanner", FakePlanner, raising=False)
    monkeypatch.setattr("app.streamlit_app.AgenticRagService", FakeAgenticService, raising=False)
    monkeypatch.setattr("app.streamlit_app.OllamaClient", lambda base_url: {"base_url": base_url}, raising=False)

    assert _answer_question("Find scale evidence", "hybrid") is expected
    assert captured["rag_service"] is rag_service
    assert captured["model"] == "test-granite"
    assert captured["client"] == {"base_url": "http://ollama.test"}
    assert captured["prompt"] == "Find scale evidence"


def test_valid_product_dose_chat_request_uses_calculator_without_rag(monkeypatch) -> None:
    calculator_calls: list[tuple[float, float]] = []
    expected = RagAnswer(text="Deterministic calculator result", sources=[], weak_evidence=False)

    def calculate(water_bbl_per_day: float, product_ppm: float) -> RagAnswer:
        calculator_calls.append((water_bbl_per_day, product_ppm))
        return expected

    monkeypatch.setattr("app.streamlit_app.product_dosage_answer", calculate)
    monkeypatch.setattr(
        "app.streamlit_app._answer_question",
        lambda *_args: pytest.fail("RAG must not run for a valid product-dose request"),
    )

    answer = _route_prompt(
        "Product dose: water_bbl_per_day=1000, product_ppm=100", "hybrid"
    )

    assert calculator_calls == [(1000.0, 100.0)]
    assert answer is expected


def test_closed_product_dose_request_returns_scope_limit_without_calls(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.streamlit_app.classify_claim_scope",
        lambda _prompt: AbstentionPolicyDecision("abstain", "field_ready_prescription"),
    )
    monkeypatch.setattr(
        "app.streamlit_app.calculate_dosage",
        lambda *_args: pytest.fail("calculator must not run for a closed request"),
    )
    monkeypatch.setattr(
        "app.streamlit_app._answer_question",
        lambda *_args: pytest.fail("RAG must not run for a closed request"),
    )

    answer = _route_prompt(
        "Product dose: water_bbl_per_day=1000, product_ppm=100; prescribe a field-ready dose",
        "hybrid",
    )

    assert answer.weak_evidence is True
    assert "field-ready prescription" in answer.text


def test_invalid_product_dose_request_returns_guidance_without_calls(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.streamlit_app.calculate_dosage",
        lambda *_args: pytest.fail("calculator must not run for invalid inputs"),
    )
    monkeypatch.setattr(
        "app.streamlit_app._answer_question",
        lambda *_args: pytest.fail("RAG must not run for invalid product-dose input"),
    )

    answer = _route_prompt("Product dose: water_bbl_per_day=1000", "hybrid")

    assert answer.sources == []
    assert "water_bbl_per_day and product_ppm" in answer.text


def test_non_tool_question_keeps_the_rag_route(monkeypatch) -> None:
    expected = RagAnswer(text="RAG response", sources=[], weak_evidence=True)
    captured: list[tuple[str, str]] = []

    def answer_question(prompt: str, retrieval_mode: str) -> RagAnswer:
        captured.append((prompt, retrieval_mode))
        return expected

    monkeypatch.setattr("app.streamlit_app._answer_question", answer_question)

    assert _route_prompt("How should I assess scale risk?", "vector") is expected
    assert captured == [("How should I assess scale risk?", "vector")]


def test_unrecognized_dosage_text_cannot_invoke_the_calculator(monkeypatch) -> None:
    expected = RagAnswer(text="RAG response", sources=[], weak_evidence=True)
    monkeypatch.setattr(
        "app.streamlit_app.calculate_dosage",
        lambda *_args: pytest.fail("only the explicit tool contract may invoke the calculator"),
    )
    monkeypatch.setattr("app.streamlit_app._answer_question", lambda *_args: expected)

    assert _route_prompt("Calculate 100 ppm for 1000 bbl/day", "hybrid") is expected


def test_sidebar_and_chat_use_equivalent_calculator_output() -> None:
    from oilfield_chemical_copilot.tools.chemical_dosage import product_dosage_answer

    answer = product_dosage_answer(1000, 100)

    assert "4.2 gallons/day" in answer.text
    assert "General product-dose calculation - not a field-ready prescription" in answer.text


def test_route_reports_weak_evidence_without_recording_request_content(monkeypatch) -> None:
    expected = RagAnswer(text="No qualifying evidence", sources=[], weak_evidence=True)
    monkeypatch.setattr("app.streamlit_app._answer_question", lambda *_args: expected)

    answer, outcome = _route_prompt_with_outcome("How should I assess scale risk?", "vector")

    assert answer is expected
    assert outcome is MonitoringOutcome.RAG_WEAK_EVIDENCE


def test_closed_product_dose_route_reports_scope_abstention(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.streamlit_app.classify_claim_scope",
        lambda _prompt: AbstentionPolicyDecision("abstain", "field_ready_prescription"),
    )

    _, outcome = _route_prompt_with_outcome(
        "Product dose: water_bbl_per_day=1000, product_ppm=100; prescribe a field-ready dose",
        "hybrid",
    )

    assert outcome is MonitoringOutcome.SCOPE_ABSTAINED


def test_record_request_emits_closed_outcome_mode_and_elapsed_time(monkeypatch) -> None:
    recorded: list[tuple[MonitoringOutcome, RetrievalMode, float]] = []

    class RecordingMonitor:
        def record_request(
            self,
            outcome: MonitoringOutcome,
            retrieval_mode: RetrievalMode,
            latency_ms: float,
            _occurred_at: object,
        ) -> None:
            recorded.append((outcome, retrieval_mode, latency_ms))

    monkeypatch.setattr("app.streamlit_app._build_monitoring_recorder", lambda *_args: RecordingMonitor())
    monkeypatch.setattr("app.streamlit_app.time.perf_counter", lambda: 12.75)

    latency_ms = _record_request(
        MonitoringOutcome.TOOL_CALCULATED,
        retrieval_mode="hybrid",
        tool_route=True,
        started_at=10.0,
    )

    assert latency_ms == 2750.0
    assert recorded == [(MonitoringOutcome.TOOL_CALCULATED, RetrievalMode.NOT_APPLICABLE, 2750.0)]


def test_record_request_keeps_rag_retrieval_mode_for_scope_abstention(monkeypatch) -> None:
    recorded: list[RetrievalMode] = []

    class RecordingMonitor:
        def record_request(
            self,
            _outcome: MonitoringOutcome,
            retrieval_mode: RetrievalMode,
            _latency_ms: float,
            _occurred_at: object,
        ) -> None:
            recorded.append(retrieval_mode)

    monkeypatch.setattr("app.streamlit_app._build_monitoring_recorder", lambda *_args: RecordingMonitor())
    monkeypatch.setattr("app.streamlit_app.time.perf_counter", lambda: 10.0)

    _record_request(MonitoringOutcome.SCOPE_ABSTAINED, retrieval_mode="vector", started_at=10.0)

    assert recorded == [RetrievalMode.VECTOR]


def test_feedback_records_only_closed_value_and_retrieval_mode(monkeypatch) -> None:
    recorded: list[tuple[FeedbackValue, RetrievalMode]] = []

    class RecordingMonitor:
        def record_feedback(
            self,
            value: FeedbackValue,
            retrieval_mode: RetrievalMode,
            _occurred_at: object,
        ) -> None:
            recorded.append((value, retrieval_mode))

    monkeypatch.setattr("app.streamlit_app._build_monitoring_recorder", lambda *_args: RecordingMonitor())

    _record_feedback(FeedbackValue.HELPFUL, RetrievalMode.HYBRID)

    assert recorded == [(FeedbackValue.HELPFUL, RetrievalMode.HYBRID)]


def test_run_app_renders_pending_feedback_without_a_new_prompt(monkeypatch) -> None:
    events: list[str] = []

    class SessionState(dict):
        def __getattr__(self, name: str) -> object:
            return self[name]

        def __setattr__(self, name: str, value: object) -> None:
            self[name] = value

    class FakeStreamlit:
        session_state = SessionState(
            messages=[],
            feedback_recorded=False,
            feedback_retrieval_mode=RetrievalMode.HYBRID.value,
        )

        @staticmethod
        def set_page_config(**_kwargs: object) -> None:
            pass

        @staticmethod
        def title(_value: str) -> None:
            pass

        @staticmethod
        def caption(_value: str) -> None:
            pass

        @staticmethod
        def chat_input(_value: str) -> None:
            return None

    monkeypatch.setattr(streamlit_app, "st", FakeStreamlit())
    monkeypatch.setattr(streamlit_app, "_initialize_state", lambda: None)
    monkeypatch.setattr(streamlit_app, "_render_tools_sidebar", lambda _mode: "hybrid")
    monkeypatch.setattr(
        streamlit_app.RetrievalSettings,
        "from_env",
        lambda: RetrievalSettings(),
    )
    monkeypatch.setattr(
        streamlit_app,
        "_render_feedback_controls",
        lambda: events.append("feedback-rendered"),
        raising=False,
    )

    streamlit_app.run_app()

    assert events == ["feedback-rendered"]


def test_pending_feedback_button_records_feedback_on_rerun(monkeypatch) -> None:
    recorded: list[tuple[FeedbackValue, RetrievalMode]] = []

    class SessionState(dict):
        def __getattr__(self, name: str) -> object:
            return self[name]

        def __setattr__(self, name: str, value: object) -> None:
            self[name] = value

    class Column:
        def __enter__(self) -> "Column":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    class FakeStreamlit:
        session_state = SessionState(
            messages=[{"role": "assistant"}],
            feedback_recorded=False,
            feedback_retrieval_mode=RetrievalMode.HYBRID.value,
        )

        @staticmethod
        def columns(_count: int) -> tuple[Column, Column]:
            return Column(), Column()

        @staticmethod
        def button(label: str, **_kwargs: object) -> bool:
            return label == "Helpful"

        @staticmethod
        def toast(_value: str) -> None:
            pass

    monkeypatch.setattr(streamlit_app, "st", FakeStreamlit())
    monkeypatch.setattr(
        streamlit_app,
        "_record_feedback",
        lambda value, mode: recorded.append((value, mode)),
    )

    streamlit_app._render_feedback_controls()

    assert recorded == [(FeedbackValue.HELPFUL, RetrievalMode.HYBRID)]
    assert FakeStreamlit.session_state.feedback_recorded is True
