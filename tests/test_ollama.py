import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from oilfield_chemical_copilot.ollama import OllamaClient, OllamaClientError


class _Response:
    def __init__(self, payload: object, raw: bytes | None = None) -> None:
        self._body = raw if raw is not None else json.dumps(payload).encode()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _fake_opener(payload: object, path: str):
    def opener(request: Request, timeout: float) -> _Response:
        assert request.full_url == f"http://ollama.test{path}"
        assert request.method == "POST"
        assert request.headers["Content-type"] == "application/json"
        assert timeout > 0
        opener.request_payload = json.loads(request.data)
        return _Response(payload)

    opener.request_payload = None
    return opener


def test_embed_posts_batch_and_returns_float_vectors() -> None:
    opener = _fake_opener({"embeddings": [[1, 2.5]]}, "/api/embed")
    client = OllamaClient("http://ollama.test", opener=opener)

    assert client.embed(model="granite-embedding:latest", texts=["scale"]) == [[1.0, 2.5]]
    assert opener.request_payload == {"model": "granite-embedding:latest", "input": ["scale"]}


def test_chat_posts_non_streaming_request_and_returns_content() -> None:
    opener = _fake_opener({"message": {"content": '{"answer": "ok"}'}}, "/api/chat")
    client = OllamaClient("http://ollama.test", opener=opener)

    assert client.chat(model="granite4.1:8b", system_prompt="system", user_prompt="user") == '{"answer": "ok"}'
    assert opener.request_payload == {
        "model": "granite4.1:8b",
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    }



def test_chat_posts_supplied_response_schema_as_format() -> None:
    schema = {
        "type": "object",
        "properties": {"recommended_next_checks": {"type": "array", "minItems": 3, "maxItems": 3}},
        "required": ["recommended_next_checks"],
    }
    opener = _fake_opener({"message": {"content": "{}"}}, "/api/chat")
    client = OllamaClient("http://ollama.test", opener=opener)

    client.chat(
        model="granite4.1:8b",
        system_prompt="system",
        user_prompt="user",
        response_schema=schema,
    )

    assert opener.request_payload["format"] == schema


def test_chat_posts_optional_generation_options() -> None:
    opener = _fake_opener({"message": {"content": "{}"}}, "/api/chat")
    client = OllamaClient("http://ollama.test", opener=opener)

    client.chat(
        model="granite4.1:8b",
        system_prompt="system",
        user_prompt="user",
        generation_options={"temperature": 0},
    )

    assert opener.request_payload["options"] == {"temperature": 0}
@pytest.mark.parametrize("payload", [{}, {"embeddings": []}, {"embeddings": [[]]}, {"message": {}}])
def test_client_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(OllamaClientError):
        OllamaClient("http://ollama.test", opener=_fake_opener(payload, "/api/embed")).embed(
            model="granite-embedding:latest", texts=["scale"]
        )


def test_chat_rejects_malformed_payload() -> None:
    with pytest.raises(OllamaClientError):
        OllamaClient("http://ollama.test", opener=_fake_opener({"message": {}}, "/api/chat")).chat(
            model="granite4.1:8b", system_prompt="system", user_prompt="user"
        )


@pytest.mark.parametrize("vector", [["1"], [True], [1, False]])
def test_embed_rejects_non_numeric_vector_elements(vector: list[object]) -> None:
    with pytest.raises(OllamaClientError):
        OllamaClient("http://ollama.test", opener=_fake_opener({"embeddings": [vector]}, "/api/embed")).embed(
            model="model", texts=["text"]
        )


@pytest.mark.parametrize("raw_body", [b"not json", b"secret response body"])
def test_transport_errors_hide_response_body(raw_body: bytes) -> None:
    def opener(request: Request, timeout: float) -> _Response:
        if raw_body.startswith(b"secret"):
            raise HTTPError(request.full_url, 500, "failure", {}, _Response({}, raw_body))
        return _Response({}, raw_body)

    with pytest.raises(OllamaClientError) as raised:
        OllamaClient("http://ollama.test", opener=opener).embed(model="model", texts=["text"])

    error = raised.value
    assert error.__cause__ is None
    assert "secret response body" not in str(error)


def test_client_wraps_oserror_without_exposing_details() -> None:
    def opener(request: Request, timeout: float) -> _Response:
        raise OSError("connection refused: secret response body")

    with pytest.raises(OllamaClientError) as raised:
        OllamaClient("http://ollama.test", opener=opener).embed(model="model", texts=["text"])

    assert raised.value.__cause__ is None
    assert "secret response body" not in str(raised.value)



def test_malformed_json_error_has_no_context_or_body() -> None:
    def opener(request: Request, timeout: float) -> _Response:
        return _Response({}, b"malformed provider body")

    with pytest.raises(OllamaClientError) as raised:
        OllamaClient("http://ollama.test", opener=opener).embed(model="model", texts=["text"])

    assert "malformed provider body" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_http_error_has_no_context_or_body() -> None:
    def opener(request: Request, timeout: float) -> _Response:
        raise HTTPError(request.full_url, 500, "failure", {}, _Response({}, b"http provider body"))

    with pytest.raises(OllamaClientError) as raised:
        OllamaClient("http://ollama.test", opener=opener).embed(model="model", texts=["text"])

    assert "http provider body" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
