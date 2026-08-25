from __future__ import annotations

import pytest

from oilfield_chemical_copilot.rag.generator_factory import build_answer_generator
from oilfield_chemical_copilot.rag.models import RagConfigurationError
from oilfield_chemical_copilot.rag.ollama_client import LazyOllamaAnswerClient
from oilfield_chemical_copilot.rag.openai_client import LazyOpenAIAnswerClient


def test_factory_builds_lazy_ollama_generator_without_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    generator = build_answer_generator()

    assert isinstance(generator, LazyOllamaAnswerClient)


def test_factory_builds_existing_lazy_openai_generator(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    generator = build_answer_generator()

    assert isinstance(generator, LazyOpenAIAnswerClient)


def test_factory_rejects_unsupported_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")

    with pytest.raises(RagConfigurationError, match="Unsupported LLM provider"):
        build_answer_generator()

def test_factory_defaults_to_lazy_ollama_generator_without_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    generator = build_answer_generator()

    assert isinstance(generator, LazyOllamaAnswerClient)