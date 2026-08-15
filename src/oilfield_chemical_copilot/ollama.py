import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class OllamaClientError(Exception):
    pass


@dataclass(frozen=True)
class OllamaToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class OllamaToolResponse:
    content: str
    tool_calls: tuple[OllamaToolCall, ...]


class OllamaClient:
    def __init__(self, base_url: str, opener: Callable[..., object] = urlopen) -> None:
        self._base_url = base_url.rstrip("/")
        self._opener = opener

    def embed(self, *, model: str, texts: Sequence[str]) -> list[list[float]]:
        return _parse_embeddings(self._post("/api/embed", {"model": model, "input": list(texts)}))

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
        generation_options: dict[str, object] | None = None,
    ) -> str:
        response_format = response_schema if response_schema is not None else "json"
        payload = {
            "model": model,
            "stream": False,
            "format": response_format,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if generation_options is not None:
            payload["options"] = generation_options
        payload = self._post("/api/chat", payload)
        return _parse_chat_content(payload)

    def chat_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        generation_options: dict[str, object] | None = None,
    ) -> OllamaToolResponse:
        payload: dict[str, object] = {
            "model": model,
            "stream": False,
            "messages": messages,
            "tools": tools,
        }
        if generation_options is not None:
            payload["options"] = generation_options
        return _parse_tool_response(self._post("/api/chat", payload))

    def _post(self, path: str, payload: dict[str, object]) -> object:
        request = Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        failure = None
        try:
            with self._opener(request, timeout=30) as response:
                parsed_payload = json.loads(response.read())
        except (OSError, HTTPError, json.JSONDecodeError, TypeError, ValueError):
            failure = OllamaClientError("Ollama request failed")
        if failure is not None:
            raise failure
        return parsed_payload
def _parse_embeddings(payload: object) -> list[list[float]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), list):
        raise OllamaClientError("Invalid Ollama embeddings response")
    embeddings = payload["embeddings"]
    if not embeddings or not all(
        isinstance(vector, list) and vector and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in vector
        )
        for vector in embeddings
    ):
        raise OllamaClientError("Invalid Ollama embeddings response")
    return [[float(value) for value in vector] for vector in embeddings]


def _parse_chat_content(payload: object) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict) or not isinstance(payload["message"].get("content"), str):
        raise OllamaClientError("Invalid Ollama chat response")
    return payload["message"]["content"]


def _parse_tool_response(payload: object) -> OllamaToolResponse:
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
        raise OllamaClientError("Invalid Ollama tool response")
    message = payload["message"]
    content = message.get("content")
    tool_calls = message.get("tool_calls", [])
    if not isinstance(content, str) or not isinstance(tool_calls, list):
        raise OllamaClientError("Invalid Ollama tool response")
    parsed_calls: list[OllamaToolCall] = []
    for call in tool_calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise OllamaClientError("Invalid Ollama tool response")
        function = call["function"]
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise OllamaClientError("Invalid Ollama tool response")
        parsed_calls.append(OllamaToolCall(name=name, arguments=arguments))
    return OllamaToolResponse(content=content, tool_calls=tuple(parsed_calls))
