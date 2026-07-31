from __future__ import annotations

import json

import pytest

from oilfield_chemical_copilot.ollama import OllamaClientError
from oilfield_chemical_copilot.rag.models import RagGenerationError
from oilfield_chemical_copilot.rag.ollama_client import OllamaAnswerClient


class FakeOllamaClient:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.response_schema: dict[str, object] | None = None
    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        self.response_schema = response_schema

        return self.output_text


class FailingOllamaClient:
    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        raise OllamaClientError("transport detail")


def _valid_draft_json() -> str:
    return json.dumps(
        {
            "answer": "Use water analysis to screen scale risk.",
            "why_this_matters": "Scale can restrict flow.",
            "cited_source_ids": ["Source 1"],
            "recommended_next_checks": ["Check calcium", "Check sulfate", "Check temperature"],
            "limitations": "Sample evidence only.",
        }
    )


def test_ollama_adapter_parses_and_validates_cited_draft() -> None:
    adapter = OllamaAnswerClient(
        model="granite4.1:8b",
        client=FakeOllamaClient(_valid_draft_json()),
    )

    draft = adapter.generate(
        system_prompt="system",
        user_prompt="user",
        allowed_source_ids={"Source 1"},
    )

    assert draft.cited_source_ids == ["Source 1"]


def test_ollama_adapter_requests_schema_and_rejects_two_next_checks() -> None:
    payload = json.loads(_valid_draft_json())
    payload["recommended_next_checks"] = ["Check calcium", "Check sulfate"]
    client = FakeOllamaClient(json.dumps(payload))
    adapter = OllamaAnswerClient(model="granite4.1:8b", client=client)

    with pytest.raises(RagGenerationError, match="Ollama answer generation failed"):
        adapter.generate(
            system_prompt="system",
            user_prompt="user",
            allowed_source_ids={"Source 1"},
        )

    assert client.response_schema is not None
    checks_schema = client.response_schema["properties"]["recommended_next_checks"]
    assert set(client.response_schema["required"]) == set(client.response_schema["properties"])
    assert checks_schema["minItems"] == 3
    assert checks_schema["maxItems"] == 3
def test_ollama_adapter_rejects_chunk_id_citation() -> None:
    payload = json.loads(_valid_draft_json())
    payload["cited_source_ids"] = ["scale-1"]
    adapter = OllamaAnswerClient(
        model="granite4.1:8b",
        client=FakeOllamaClient(json.dumps(payload)),
    )

    with pytest.raises(RagGenerationError, match="Ollama answer generation failed"):
        adapter.generate(
            system_prompt="system",
            user_prompt="user",
            allowed_source_ids={"Source 1"},
        )



def test_ollama_adapter_hides_transport_details() -> None:
    adapter = OllamaAnswerClient(model="granite4.1:8b", client=FailingOllamaClient())

    with pytest.raises(RagGenerationError, match="Ollama answer generation failed"):
        adapter.generate(
            system_prompt="system",
            user_prompt="user",
            allowed_source_ids={"Source 1"},
        )

def test_ollama_adapter_hides_malformed_response_details() -> None:
    adapter = OllamaAnswerClient(
        model="granite4.1:8b",
        client=FakeOllamaClient('{"answer": "missing required fields"}'),
    )

    with pytest.raises(RagGenerationError, match="Ollama answer generation failed"):
        adapter.generate(
            system_prompt="system",
            user_prompt="user",
            allowed_source_ids={"Source 1"},
        )

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("answer", 42),
        ("why_this_matters", False),
        ("limitations", None),
        ("cited_source_ids", ["Source 1", 2]),
        ("recommended_next_checks", ["Check calcium", 2, "Check temperature"]),
    ],
)
def test_ollama_adapter_rejects_non_string_draft_fields(field: str, value: object) -> None:
    payload = json.loads(_valid_draft_json())
    payload[field] = value
    adapter = OllamaAnswerClient(
        model="granite4.1:8b",
        client=FakeOllamaClient(json.dumps(payload)),
    )

    with pytest.raises(RagGenerationError, match="Ollama answer generation failed"):
        adapter.generate(
            system_prompt="system",
            user_prompt="user",
            allowed_source_ids={"Source 1"},
        )


def test_ollama_adapter_rejects_string_valued_next_checks() -> None:
    payload = json.loads(_valid_draft_json())
    payload["recommended_next_checks"] = "one"
    adapter = OllamaAnswerClient(
        model="granite4.1:8b",
        client=FakeOllamaClient(json.dumps(payload)),
    )

    with pytest.raises(RagGenerationError, match="Ollama answer generation failed"):
        adapter.generate(
            system_prompt="system",
            user_prompt="user",
            allowed_source_ids={"Source 1"},
        )