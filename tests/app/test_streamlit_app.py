from __future__ import annotations
import pytest

from app.streamlit_app import (
    _answer_question,
    _build_rag_service,
    _citation_display,
    _database_url,
    _excerpt,
)
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
