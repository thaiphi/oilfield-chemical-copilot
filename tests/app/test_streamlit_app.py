from __future__ import annotations
import pytest

from app.streamlit_app import _answer_question, _build_rag_service, _citation_display, _database_url, _excerpt
from oilfield_chemical_copilot.ollama import OllamaClientError
from oilfield_chemical_copilot.rag.models import RagConfigurationError
from oilfield_chemical_copilot.rag.models import SourceEvidence


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

    assert display == "Source 1: docs/scale.md | document | chunk scale-1 | score 0.910"
    assert "C:/" not in display


def test_excerpt_is_bounded() -> None:
    assert _excerpt("a" * 90, limit=20) == "a" * 17 + "..."

def test_lazy_openai_generator_does_not_require_key_until_generation(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from oilfield_chemical_copilot.rag.openai_client import LazyOpenAIAnswerClient

    generator = LazyOpenAIAnswerClient()

    assert generator is not None


def test_rag_service_uses_configured_answer_generator(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr("app.streamlit_app.build_answer_generator", lambda: sentinel)
    monkeypatch.setattr("app.streamlit_app.build_embedding_provider", lambda: type("Provider", (), {"dimension": 384})())
    monkeypatch.setattr("app.streamlit_app.PgVectorStore", lambda *args, **kwargs: object())
    monkeypatch.setattr("app.streamlit_app.BasicRetrievalPipeline", lambda **kwargs: object())
    monkeypatch.setattr(
        "app.streamlit_app.RetrievalSettings.from_env",
        lambda: type("Settings", (), {"min_score": 0.2, "max_context_chars": 4000})(),
    )
    _build_rag_service.clear()

    service = _build_rag_service()

    assert service.generator is sentinel
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



def test_answer_question_hides_ollama_retrieval_error_details(monkeypatch) -> None:
    class FailingService:
        def answer(self, _prompt: str):
            raise OllamaClientError("provider response body: private corpus excerpt")

    monkeypatch.setattr("app.streamlit_app._build_rag_service", lambda: FailingService())

    with pytest.raises(RagConfigurationError, match="Ollama retrieval is unavailable") as error:
        _answer_question("How should I assess scale risk?")

    assert "private corpus excerpt" not in str(error.value)
