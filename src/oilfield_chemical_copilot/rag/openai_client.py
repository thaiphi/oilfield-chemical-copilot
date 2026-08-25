from __future__ import annotations

import json
import os
from typing import Any

from oilfield_chemical_copilot.rag.models import RagConfigurationError, RagDraft, RagGenerationError

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "why_this_matters": {"type": "string"},
        "cited_source_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "recommended_next_checks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
        "limitations": {"type": "string"},
    },
    "required": [
        "answer",
        "why_this_matters",
        "cited_source_ids",
        "recommended_next_checks",
        "limitations",
    ],
}


class OpenAIAnswerClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, client=None) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        if not self.api_key:
            raise RagConfigurationError("OPENAI_API_KEY is required for answer generation")
        if not self.model:
            raise RagConfigurationError("OPENAI_MODEL is required for answer generation")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, timeout=30.0)
        self.client = client

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_source_ids: set[str],
    ) -> RagDraft:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "oilfield_rag_answer",
                        "schema": ANSWER_SCHEMA,
                        "strict": True,
                    }
                },
                temperature=0.0,
                store=False,
            )
        except Exception as error:  # OpenAI SDK errors vary by transport and version.
            raise RagGenerationError("OpenAI answer generation failed") from error
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RagGenerationError("OpenAI response did not include text output")
        try:
            payload = json.loads(output_text)
            draft = RagDraft(
                answer=str(payload["answer"]),
                why_this_matters=str(payload["why_this_matters"]),
                cited_source_ids=[str(source_id) for source_id in payload["cited_source_ids"]],
                recommended_next_checks=[
                    str(check) for check in payload["recommended_next_checks"]
                ],
                limitations=str(payload["limitations"]),
            )
            draft.validate(allowed_source_ids)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RagGenerationError("OpenAI response did not match the answer contract") from error
        return draft

class LazyOpenAIAnswerClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, client=None) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client
        self._adapter: OpenAIAnswerClient | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_source_ids: set[str],
    ) -> RagDraft:
        if self._adapter is None:
            self._adapter = OpenAIAnswerClient(
                api_key=self.api_key,
                model=self.model,
                client=self.client,
            )
        return self._adapter.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_source_ids=allowed_source_ids,
        )