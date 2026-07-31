from __future__ import annotations

import pytest

from oilfield_chemical_copilot.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingSettings,
    OllamaEmbeddingProvider,
    build_embedding_provider,
)


class FakeOllamaClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[tuple[str, list[str]]] = []

    def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        self.calls.append((model, texts))
        return self.embeddings


def test_deterministic_embedding_provider_is_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimension=8)

    first = provider.embed_documents(["iron sulfide", "scale"])
    second = provider.embed_documents(["iron sulfide", "scale"])

    assert first == second
    assert provider.model_name == "deterministic-token-hash-8"
    assert provider.dimension == 8
    assert len(first) == 2
    assert all(len(vector) == 8 for vector in first)
    assert all(abs(sum(value * value for value in vector) - 1.0) < 1e-9 for vector in first)


def test_deterministic_embedding_provider_handles_empty_inputs() -> None:
    provider = DeterministicEmbeddingProvider(dimension=4)

    assert provider.embed_documents([]) == []
    assert len(provider.embed_query("scale")) == 4


def test_deterministic_embedding_provider_validates_dimension() -> None:
    try:
        DeterministicEmbeddingProvider(dimension=0)
    except ValueError as error:
        assert "dimension must be positive" in str(error)
    else:
        raise AssertionError("invalid dimension should fail")


def test_embedding_settings_from_env_builds_matching_provider(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "deterministic")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "8")

    provider = build_embedding_provider(EmbeddingSettings.from_env())

    assert provider.model_name == "deterministic-token-hash-8"
    assert provider.dimension == 8


def test_deterministic_embedding_provider_ranks_lexically_related_text_higher() -> None:
    provider = DeterministicEmbeddingProvider(dimension=384)
    query = provider.embed_query("scale risk from produced water analysis")
    related = provider.embed_documents(["Produced water analysis can indicate scale risk."])[0]
    unrelated = provider.embed_documents(["Paraffin deposition can restrict flow in cold wells."])[0]

    related_score = sum(a * b for a, b in zip(query, related, strict=True))
    unrelated_score = sum(a * b for a, b in zip(query, unrelated, strict=True))

    assert related_score >= 0.2
    assert related_score > unrelated_score

def test_ollama_embedding_provider_uses_model_and_validates_dimension() -> None:
    client = FakeOllamaClient([[0] * 384, [1] * 384])
    provider = OllamaEmbeddingProvider(
        model_name="granite-embedding:latest",
        dimension=384,
        client=client,
    )

    assert provider.embed_documents(["scale", "corrosion"]) == [[0.0] * 384, [1.0] * 384]
    assert client.calls == [("granite-embedding:latest", ["scale", "corrosion"])]


def test_ollama_embedding_provider_rejects_wrong_dimension() -> None:
    provider = OllamaEmbeddingProvider(
        model_name="granite-embedding:latest",
        dimension=384,
        client=FakeOllamaClient([[0.0] * 383]),
    )

    with pytest.raises(ValueError, match="expected 384, got 383"):
        provider.embed_query("scale")


def test_ollama_embedding_provider_handles_empty_document_batch() -> None:
    client = FakeOllamaClient([])
    provider = OllamaEmbeddingProvider(
        model_name="granite-embedding:latest",
        dimension=384,
        client=client,
    )

    assert provider.embed_documents([]) == []
    assert client.calls == []


def test_embedding_settings_from_env_builds_ollama_provider(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "granite-embedding:latest")

    settings = EmbeddingSettings.from_env()
    provider = build_embedding_provider(settings)

    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.model_name == "granite-embedding:latest"
    assert provider._client._base_url == "http://ollama:11434"
    assert settings.provider == "ollama"
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.ollama_embedding_model == "granite-embedding:latest"
def test_embedding_settings_defaults_to_ollama_when_environment_is_absent(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)

    settings = EmbeddingSettings.from_env()

    assert settings.provider == "ollama"
    assert settings.ollama_embedding_model == "granite-embedding:latest"


@pytest.mark.parametrize("vectors", [[[0.0] * 384], [[0.0] * 384 for _ in range(3)]])
def test_ollama_embedding_provider_rejects_response_with_wrong_vector_count(vectors) -> None:
    provider = OllamaEmbeddingProvider(
        model_name="granite-embedding:latest",
        dimension=384,
        client=FakeOllamaClient(vectors),
    )

    with pytest.raises(ValueError, match="expected 2, got"):
        provider.embed_documents(["scale", "corrosion"])
