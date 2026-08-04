"""Privacy-safe structured answer-quality judging."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from oilfield_chemical_copilot.evaluation.answers import AnswerEvaluationCase
from oilfield_chemical_copilot.ollama import OllamaClient

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "granite4.1:8b"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
_RUBRIC_FIELDS = (
    "groundedness",
    "relevance",
    "limitation_awareness",
    "operational_certainty",
)
_SCORE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_RUBRIC_FIELDS),
    "properties": {field: {"type": "integer", "minimum": 1, "maximum": 5} for field in _RUBRIC_FIELDS},
}
_SAFE_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class JudgeProvider(Protocol):
    provider_name: str
    model_name: str

    def judge(self, *, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True)
class JudgeScores:
    groundedness: int
    relevance: int
    limitation_awareness: int
    operational_certainty: int

    def __post_init__(self) -> None:
        values = (
            self.groundedness,
            self.relevance,
            self.limitation_awareness,
            self.operational_certainty,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5 for value in values):
            raise ValueError("judge scores must be integers from 1 through 5")

    @classmethod
    def from_json(cls, response: str) -> JudgeScores:
        payload = json.loads(response, object_pairs_hook=_object_without_duplicate_keys)
        if not isinstance(payload, dict) or set(payload) != set(_RUBRIC_FIELDS):
            raise ValueError("judge response must contain exactly the rubric scores")
        return cls(**payload)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("judge response must not contain duplicate keys")
        payload[key] = value
    return payload


@dataclass(frozen=True)
class JudgeAvailable:
    question_id: str
    provider: str
    model: str
    scores: JudgeScores
    status: Literal["available"] = "available"


@dataclass(frozen=True)
class JudgeUnavailable:
    question_id: str
    provider: str
    model: str
    status: Literal["unavailable"] = "unavailable"


JudgeResult = JudgeAvailable | JudgeUnavailable


class _UnavailableJudgeProvider:
    def __init__(self, provider_name: str, model_name: str) -> None:
        self.provider_name = provider_name
        self.model_name = model_name

    def judge(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("judge is unavailable")


class AnswerJudge:
    """Judge runtime material while retaining only report-safe result metadata."""

    def __init__(self, provider: JudgeProvider | None = None) -> None:
        if provider is not None:
            self._provider = provider
            return
        provider_name, model_name = _configured_identity()
        try:
            self._provider = _provider_from_environment()
        except Exception:
            self._provider = _UnavailableJudgeProvider(provider_name, model_name)

    def judge(self, case: AnswerEvaluationCase, *, answer: str, evidence: str) -> JudgeResult:
        provider, model = _report_identity(self._provider)
        try:
            response = self._provider.judge(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_user_prompt(case, answer=answer, evidence=evidence),
            )
            scores = JudgeScores.from_json(response)
        except Exception:
            return JudgeUnavailable(case.question_id, provider, model)
        return JudgeAvailable(case.question_id, provider, model, scores)


class OllamaJudgeProvider:
    provider_name = "ollama"

    def __init__(self, *, model: str, client: OllamaClient) -> None:
        self.model_name = model
        self._client = client

    def judge(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._client.chat(
            model=self.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=_SCORE_SCHEMA,
            generation_options={"temperature": 0},
        )


class OpenAIJudgeProvider:
    provider_name = "openai"

    def __init__(self, *, api_key: str, model: str, client: object | None = None) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.model_name = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=30.0)
        self._client = client

    def judge(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._client.responses.create(
            model=self.model_name,
            instructions=system_prompt,
            input=user_prompt,
            text={"format": {"type": "json_schema", "name": "answer_judge", "schema": _SCORE_SCHEMA, "strict": True}},
            temperature=0.0,
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise ValueError("OpenAI judge response did not include text")
        return output_text


def _configured_identity() -> tuple[str, str]:
    provider = os.getenv("ANSWER_EVAL_JUDGE_PROVIDER", "ollama").lower()
    model = os.getenv(
        "ANSWER_EVAL_OPENAI_MODEL" if provider == "openai" else "OLLAMA_MODEL",
        DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_OLLAMA_MODEL,
    )
    return provider, model


def _provider_from_environment() -> JudgeProvider:
    provider, _ = _configured_identity()
    if provider == "ollama":
        return OllamaJudgeProvider(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            client=OllamaClient(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)),
        )
    if provider == "openai":
        return OpenAIJudgeProvider(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("ANSWER_EVAL_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        )
    raise ValueError("unsupported answer judge provider")


def _report_identity(provider: JudgeProvider) -> tuple[str, str]:
    name = str(getattr(provider, "provider_name", ""))
    safe_provider = name if _SAFE_PROVIDER.fullmatch(name) else "unknown"
    model_name = str(getattr(provider, "model_name", "unknown"))
    return safe_provider, "sha256:" + hashlib.sha256(model_name.encode("utf-8")).hexdigest()


_SYSTEM_PROMPT = """Score the response only against the supplied public question and evidence.
Return only the required JSON object. Use integer scores from 1 to 5:
- groundedness: 1=unsupported or contradicts evidence; 5=claims are directly supported.
- relevance: 1=does not address the question; 5=directly addresses the question.
- limitation_awareness: 1=omits material uncertainty; 5=states material evidence limits.
- operational_certainty: 1=overconfident or unsafe certainty; 5=certainty is calibrated to evidence.
"""


def _user_prompt(case: AnswerEvaluationCase, *, answer: str, evidence: str) -> str:
    return (
        f"Case ID: {case.question_id}\n"
        f"Question:\n{case.question}\n"
        f"Evidence is sufficient: {case.evidence_sufficient}\n"
        f"Generated answer:\n{answer}\n"
        f"Public evidence:\n{evidence}"
    )
