from __future__ import annotations

import json

import pytest

from oilfield_chemical_copilot.rag.models import RagConfigurationError, RagGenerationError
from oilfield_chemical_copilot.rag.openai_client import OpenAIAnswerClient


class FakeResponses:
    def __init__(self, output_text: str | None = None, error: Exception | None = None) -> None:
        self.output_text = output_text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return type("FakeResponse", (), {"output_text": self.output_text})()


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def test_openai_adapter_requests_structured_json_and_parses_draft() -> None:
    responses = FakeResponses(
        json.dumps(
            {
                "answer": "Use water analysis to screen scale risk.",
                "why_this_matters": "Scale can restrict flow.",
                "cited_source_ids": ["Source 1"],
                "recommended_next_checks": ["Check calcium", "Check sulfate", "Check temperature"],
                "limitations": "Sample evidence only.",
            }
        )
    )
    adapter = OpenAIAnswerClient(api_key="sk-test", model="gpt-test", client=FakeClient(responses))

    draft = adapter.generate(system_prompt="system", user_prompt="user", allowed_source_ids={"Source 1"})

    assert draft.answer.startswith("Use water analysis")
    assert responses.calls[0]["model"] == "gpt-test"
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"
    assert responses.calls[0]["store"] is False


def test_openai_adapter_rejects_missing_api_key_without_logging_secret() -> None:
    with pytest.raises(RagConfigurationError, match="OPENAI_API_KEY"):
        OpenAIAnswerClient(api_key="", model="gpt-test")


def test_openai_adapter_rejects_malformed_or_uncited_response() -> None:
    adapter = OpenAIAnswerClient(
        api_key="sk-test",
        model="gpt-test",
        client=FakeClient(FakeResponses('{"answer": "missing fields"}')),
    )

    with pytest.raises(RagGenerationError):
        adapter.generate(system_prompt="system", user_prompt="user", allowed_source_ids={"Source 1"})

def test_openai_adapter_rejects_valid_json_with_no_citations() -> None:
    adapter = OpenAIAnswerClient(
        api_key="sk-test",
        model="gpt-test",
        client=FakeClient(
            FakeResponses(
                json.dumps(
                    {
                        "answer": "Uncited answer.",
                        "why_this_matters": "Citations are required.",
                        "cited_source_ids": [],
                        "recommended_next_checks": ["One", "Two", "Three"],
                        "limitations": "None.",
                    }
                )
            )
        ),
    )

    with pytest.raises(RagGenerationError, match="at least one cited source"):
        adapter.generate(system_prompt="system", user_prompt="user", allowed_source_ids={"Source 1"})