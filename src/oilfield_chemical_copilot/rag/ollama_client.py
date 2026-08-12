from __future__ import annotations

import json
import os

from oilfield_chemical_copilot.ollama import OllamaClient, OllamaClientError
from oilfield_chemical_copilot.rag.models import RagDraft, RagGenerationError

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "granite4.1:8b"

ANSWER_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object", "additionalProperties": False,
    "required": ["answer", "why_this_matters", "cited_source_ids", "recommended_next_checks", "limitations"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "why_this_matters": {"type": "string", "minLength": 1},
        "cited_source_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "recommended_next_checks": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "limitations": {"type": "string", "minLength": 1},
    },
}


class OllamaAnswerClient:
    def __init__(self, *, model: str, client: OllamaClient, generation_options: dict[str, object] | None = None) -> None:
        self.model = model
        self.client = client
        self.generation_options = generation_options

    def generate(self, *, system_prompt: str, user_prompt: str, allowed_source_ids: set[str]) -> RagDraft:
        kwargs = {
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_schema": ANSWER_RESPONSE_SCHEMA,
        }
        if self.generation_options is not None:
            kwargs["generation_options"] = self.generation_options
        failure: Exception | None = None
        for _ in range(2):
            try:
                output_text = self.client.chat(**kwargs)
                return _parse_and_validate_draft(output_text, allowed_source_ids)
            except (
                OllamaClientError,
                RagGenerationError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                failure = error
        raise RagGenerationError("Ollama answer generation failed") from failure


class LazyOllamaAnswerClient:
    def __init__(self, *, base_url: str | None = None, model: str | None = None, client: OllamaClient | None = None, generation_options: dict[str, object] | None = None) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.client = client
        self.generation_options = generation_options
        self._adapter: OllamaAnswerClient | None = None

    def generate(self, *, system_prompt: str, user_prompt: str, allowed_source_ids: set[str]) -> RagDraft:
        if self._adapter is None:
            client = self.client or OllamaClient(self.base_url)
            self._adapter = OllamaAnswerClient(model=self.model, client=client, generation_options=self.generation_options)
        return self._adapter.generate(system_prompt=system_prompt, user_prompt=user_prompt, allowed_source_ids=allowed_source_ids)


def _parse_and_validate_draft(output_text: str, allowed_source_ids: set[str]) -> RagDraft:
    payload = json.loads(output_text)
    if not isinstance(payload, dict):
        raise ValueError("Ollama answer must be a JSON object")
    draft = RagDraft(answer=_required_string(payload, "answer"), why_this_matters=_required_string(payload, "why_this_matters"), cited_source_ids=_required_string_list(payload, "cited_source_ids"), recommended_next_checks=_required_string_list(payload, "recommended_next_checks"), limitations=_required_string(payload, "limitations"))
    draft.validate(allowed_source_ids)
    return draft


def _required_string(payload: dict[object, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise ValueError(f"Ollama answer field {field} must be a string")
    return value


def _required_string_list(payload: dict[object, object], field: str) -> list[str]:
    value = payload[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Ollama answer field {field} must be a list of strings")
    return value
