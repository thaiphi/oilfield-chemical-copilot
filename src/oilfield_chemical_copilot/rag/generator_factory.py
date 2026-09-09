from __future__ import annotations

import os
from typing import Protocol

from oilfield_chemical_copilot.rag.models import RagConfigurationError, RagDraft
from oilfield_chemical_copilot.rag.ollama_client import LazyOllamaAnswerClient
from oilfield_chemical_copilot.rag.openai_client import LazyOpenAIAnswerClient

DEFAULT_LLM_PROVIDER = "ollama"


class AnswerGenerator(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_source_ids: set[str],
    ) -> RagDraft:
        ...


def build_answer_generator(settings=None) -> AnswerGenerator:
    if settings is not None:
        if settings.provider == "ollama":
            return LazyOllamaAnswerClient(base_url=settings.ollama_base_url,
                                          model=settings.ollama_model)
        if settings.provider == "openai" and settings.openai_api_key.strip():
            return LazyOpenAIAnswerClient(api_key=settings.openai_api_key,
                                          model=settings.openai_model)
        raise RagConfigurationError("Invalid answer generator configuration")
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    if provider == "ollama":
        return LazyOllamaAnswerClient()
    if provider == "openai":
        return LazyOpenAIAnswerClient()
    raise RagConfigurationError(f"Unsupported LLM provider: {provider}")
