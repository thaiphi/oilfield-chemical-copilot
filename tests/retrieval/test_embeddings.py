from __future__ import annotations

from oilfield_chemical_copilot.retrieval.embeddings import DeterministicEmbeddingProvider


def test_deterministic_embedding_provider_is_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimension=8)

    first = provider.embed_documents(["iron sulfide", "scale"])
    second = provider.embed_documents(["iron sulfide", "scale"])

    assert first == second
    assert provider.model_name == "deterministic-hash-8"
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
