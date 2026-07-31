from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from oilfield_chemical_copilot.ollama import OllamaClient

DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SENTENCE_TRANSFORMERS_DIMENSION = 384
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    dimension: int = DEFAULT_SENTENCE_TRANSFORMERS_DIMENSION
    sentence_transformers_model: str = DEFAULT_SENTENCE_TRANSFORMERS_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_embedding_model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
            dimension=_env_int("EMBEDDING_DIMENSION", DEFAULT_SENTENCE_TRANSFORMERS_DIMENSION),
            sentence_transformers_model=os.getenv(
                "SENTENCE_TRANSFORMERS_MODEL",
                DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
            ),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            ollama_embedding_model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL",
                DEFAULT_OLLAMA_EMBEDDING_MODEL,
            ),
        )


def build_embedding_provider(settings: EmbeddingSettings | None = None) -> EmbeddingProvider:
    settings = settings or EmbeddingSettings.from_env()
    if settings.provider == "deterministic":
        return DeterministicEmbeddingProvider(dimension=settings.dimension)
    if settings.provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=settings.sentence_transformers_model,
            dimension=settings.dimension,
        )
    if settings.provider == "ollama":
        return OllamaEmbeddingProvider(
            model_name=settings.ollama_embedding_model,
            dimension=settings.dimension,
            client=OllamaClient(settings.ollama_base_url),
        )
    raise ValueError(f"Unsupported embedding provider: {settings.provider}")


class DeterministicEmbeddingProvider:
    def __init__(self, dimension: int = 384) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model_name = f"deterministic-token-hash-{dimension}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            values[index] += 1.0
        return _normalize(values)


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
        dimension: int = DEFAULT_SENTENCE_TRANSFORMERS_DIMENSION,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        encoded = self._model.encode(texts, normalize_embeddings=True)
        vectors = [list(map(float, vector)) for vector in encoded]
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch for {self.model_name}: "
                    f"expected {self.dimension}, got {len(vector)}"
                )
        return vectors


class OllamaEmbeddingProvider:
    def __init__(self, model_name: str, dimension: int, client: OllamaClient) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.model_name = model_name
        self.dimension = dimension
        self._client = client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = [
            list(map(float, vector))
            for vector in self._client.embed(model=self.model_name, texts=texts)
        ]
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding vector count mismatch for {self.model_name}: "
                f"expected {len(texts)}, got {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch for {self.model_name}: "
                    f"expected {self.dimension}, got {len(vector)}"
                )
        return vectors

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
}
_TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "analyses": ("analysis",),
    "analyze": ("analysis",),
    "analysing": ("analysis",),
    "corroded": ("corrosion",),
    "corrosive": ("corrosion",),
    "deposit": ("deposit", "issue"),
    "deposition": ("deposit", "issue"),
    "deposits": ("deposit", "issue"),
    "issues": ("issue",),
    "paraffin": ("paraffin", "wax"),
    "paraffins": ("paraffin", "wax"),
    "scaled": ("scale",),
    "scales": ("scale",),
    "scaling": ("scale",),
    "troubleshoot": ("troubleshoot", "issue"),
    "troubleshooting": ("troubleshoot", "issue"),
    "sulphide": ("sulfide",),
    "sulphides": ("sulfide",),
}


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall(text.lower()):
        if raw_token in _STOPWORDS:
            continue
        for token in _TOKEN_ALIASES.get(raw_token, (raw_token,)):
            if token not in _STOPWORDS:
                tokens.append(token)
    return tokens

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]