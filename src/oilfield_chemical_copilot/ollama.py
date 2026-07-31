import json
from collections.abc import Callable, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class OllamaClientError(Exception):
    pass


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
    ) -> str:
        response_format = response_schema if response_schema is not None else "json"
        payload = self._post(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "format": response_format,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        return _parse_chat_content(payload)

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
