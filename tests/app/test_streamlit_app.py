from __future__ import annotations
import pytest

from app.streamlit_app import (
    _answer_question,
    _build_rag_service,
    _citation_display,
    _database_url,
    _route_prompt,
    _route_prompt_with_outcome,
    _record_request,
    _excerpt,
)
from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision
from oilfield_chemical_copilot.observability.aggregate_monitoring import MonitoringOutcome
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


def test_valid_product_dose_chat_request_uses_calculator_without_rag(monkeypatch) -> None:
    calculator_calls: list[tuple[float, float]] = []

    def calculate(water_bbl_per_day: float, product_ppm: float):
        calculator_calls.append((water_bbl_per_day, product_ppm))
        from oilfield_chemical_copilot.tools.chemical_dosage import calculate_dosage

        return calculate_dosage(water_bbl_per_day, product_ppm)

    monkeypatch.setattr("app.streamlit_app.calculate_dosage", calculate)
    monkeypatch.setattr(
        "app.streamlit_app._answer_question",
        lambda *_args: pytest.fail("RAG must not run for a valid product-dose request"),
    )

    answer = _route_prompt(
        "Product dose: water_bbl_per_day=1000, product_ppm=100", "hybrid"
    )

    assert calculator_calls == [(1000.0, 100.0)]
    assert answer.sources == []
    assert "General product-dose calculation - not a field-ready prescription" in answer.text
    assert "4.2 gallons/day" in answer.text


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
    from app.streamlit_app import _dosage_answer

    answer = _dosage_answer(1000, 100)

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


def test_record_request_emits_only_closed_outcome_and_elapsed_time(monkeypatch) -> None:
    recorded: list[tuple[MonitoringOutcome, float]] = []

    class RecordingMonitor:
        def record(self, outcome: MonitoringOutcome, latency_ms: float) -> None:
            recorded.append((outcome, latency_ms))

    monkeypatch.setattr("app.streamlit_app.REQUEST_MONITOR", RecordingMonitor())
    monkeypatch.setattr("app.streamlit_app.time.perf_counter", lambda: 12.75)

    latency_ms = _record_request(MonitoringOutcome.TOOL_CALCULATED, started_at=10.0)

    assert latency_ms == 2750.0
    assert recorded == [(MonitoringOutcome.TOOL_CALCULATED, 2750.0)]
