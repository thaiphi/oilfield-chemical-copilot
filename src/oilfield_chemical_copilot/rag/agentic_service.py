from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol

from oilfield_chemical_copilot.evaluation.abstention_policy import classify_claim_scope
from oilfield_chemical_copilot.ollama import OllamaClient, OllamaClientError
from oilfield_chemical_copilot.rag.formatter import scope_limited_answer
from oilfield_chemical_copilot.rag.models import RagAnswer
from oilfield_chemical_copilot.tools.chemical_dosage import (
    DosageResult,
    calculate_dosage,
    product_dosage_answer_from_result,
)


SEARCH_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "Search the indexed knowledge base for troubleshooting evidence.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 500}},
        },
    },
}
PRODUCT_DOSE_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "calculate_product_dose",
        "description": "Calculate a general product-ppm water-basis dose from explicit inputs.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["water_bbl_per_day", "product_ppm"],
            "properties": {
                "water_bbl_per_day": {"type": "number", "exclusiveMinimum": 0},
                "product_ppm": {"type": "number", "minimum": 0},
            },
        },
    },
}
TOOL_SCHEMAS = [SEARCH_TOOL_SCHEMA, PRODUCT_DOSE_TOOL_SCHEMA]
_PLANNER_SYSTEM_INSTRUCTION = (
    "Choose at most one listed tool. Never call an unlisted tool. "
    "Do not answer the user directly."
)


@dataclass(frozen=True)
class ToolSelection:
    name: str
    arguments: dict[str, object]


class ToolPlanner(Protocol):
    def select_tool(self, question: str) -> tuple[ToolSelection, ...]: ...


class OllamaToolPlanner:
    def __init__(self, *, model: str, client: OllamaClient) -> None:
        self.model = model
        self.client = client

    def select_tool(self, question: str) -> tuple[ToolSelection, ...]:
        response = self.client.chat_with_tools(
            model=self.model,
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM_INSTRUCTION},
                {"role": "user", "content": question},
            ],
            tools=TOOL_SCHEMAS,
            generation_options={"temperature": 0},
        )
        return tuple(
            ToolSelection(name=tool_call.name, arguments=tool_call.arguments)
            for tool_call in response.tool_calls
        )


class AgenticRagService:
    def __init__(self, *, rag_service, planner: ToolPlanner, calculator=calculate_dosage) -> None:
        self.rag_service = rag_service
        self.planner = planner
        self.calculator = calculator

    def answer(self, question: str, topic: str | None = None) -> RagAnswer:
        claim_scope = classify_claim_scope(question)
        if claim_scope.action == "abstain":
            return scope_limited_answer(category=claim_scope.category)
        try:
            selections = self.planner.select_tool(question)
        except OllamaClientError:
            return self._rag_fallback(question, topic)
        if len(selections) != 1:
            return self._rag_fallback(question, topic)
        selection = selections[0]
        if selection.name == "search_knowledge":
            query = _search_query(selection.arguments)
            if query is None:
                return self._rag_fallback(question, topic)
            return self.rag_service.answer(question, topic=topic, retrieval_query=query)
        if selection.name == "calculate_product_dose":
            inputs = _dose_inputs(selection.arguments)
            if inputs is None:
                return self._rag_fallback(question, topic)
            try:
                result = self.calculator(**inputs)
            except ValueError:
                return self._rag_fallback(question, topic)
            if not isinstance(result, DosageResult):
                return self._rag_fallback(question, topic)
            return product_dosage_answer_from_result(result)
        return self._rag_fallback(question, topic)

    def _rag_fallback(self, question: str, topic: str | None) -> RagAnswer:
        return self.rag_service.answer(question, topic=topic)


def _search_query(arguments: dict[str, object]) -> str | None:
    if set(arguments) != {"query"}:
        return None
    query = arguments["query"]
    if not isinstance(query, str):
        return None
    query = query.strip()
    return query if 1 <= len(query) <= 500 else None


def _dose_inputs(arguments: dict[str, object]) -> dict[str, float] | None:
    if set(arguments) != {"water_bbl_per_day", "product_ppm"}:
        return None
    water_bbl_per_day = _finite_number(arguments["water_bbl_per_day"])
    product_ppm = _finite_number(arguments["product_ppm"])
    if water_bbl_per_day is None or product_ppm is None:
        return None
    if water_bbl_per_day <= 0 or product_ppm < 0:
        return None
    return {"water_bbl_per_day": water_bbl_per_day, "product_ppm": product_ppm}


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        return None
    return float(value)
