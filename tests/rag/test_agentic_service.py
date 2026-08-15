from __future__ import annotations

from oilfield_chemical_copilot.ollama import OllamaClientError, OllamaToolCall, OllamaToolResponse
from oilfield_chemical_copilot.rag.agentic_service import (
    AgenticRagService,
    OllamaToolPlanner,
    TOOL_SCHEMAS,
    ToolSelection,
)
from oilfield_chemical_copilot.rag.models import RagAnswer
from oilfield_chemical_copilot.tools.chemical_dosage import calculate_dosage


class FakePlanner:
    def __init__(self, selections: tuple[ToolSelection, ...]) -> None:
        self.selections = selections
        self.calls: list[str] = []

    def select_tool(self, question: str) -> tuple[ToolSelection, ...]:
        self.calls.append(question)
        return self.selections


class FakeRagService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []
        self.answer_value = RagAnswer(text="Grounded RAG answer", sources=[], weak_evidence=False)

    def answer(
        self,
        question: str,
        topic: str | None = None,
        retrieval_query: str | None = None,
    ) -> RagAnswer:
        self.calls.append((question, topic, retrieval_query))
        return self.answer_value


class FakeOllamaClient:
    def __init__(self, response: OllamaToolResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat_with_tools(self, **kwargs) -> OllamaToolResponse:
        self.calls.append(kwargs)
        return self.response


def test_closed_scope_skips_planner_tool_and_rag() -> None:
    planner = FakePlanner(())
    rag_service = FakeRagService()

    service = AgenticRagService(
        rag_service=rag_service,
        planner=planner,
        calculator=lambda *_args: (_ for _ in ()).throw(AssertionError("tool must not run")),
    )

    answer = service.answer("Can you prescribe a field-ready dosage?")

    assert answer.weak_evidence is True
    assert planner.calls == []
    assert rag_service.calls == []


def test_search_uses_selected_query_and_preserves_the_original_question() -> None:
    planner = FakePlanner((ToolSelection("search_knowledge", {"query": "produced water scale"}),))
    rag_service = FakeRagService()
    service = AgenticRagService(rag_service=rag_service, planner=planner)

    answer = service.answer("How should I assess scale risk from produced water analysis?", topic="scale")

    assert answer is rag_service.answer_value
    assert rag_service.calls == [
        ("How should I assess scale risk from produced water analysis?", "scale", "produced water scale")
    ]


def test_valid_dose_selection_returns_a_deterministic_result() -> None:
    calculator_calls: list[tuple[float, float]] = []

    def calculator(water_bbl_per_day: float, product_ppm: float):
        calculator_calls.append((water_bbl_per_day, product_ppm))
        return calculate_dosage(water_bbl_per_day, product_ppm)

    service = AgenticRagService(
        rag_service=FakeRagService(),
        planner=FakePlanner(
            (ToolSelection("calculate_product_dose", {"water_bbl_per_day": 1000, "product_ppm": 100}),)
        ),
        calculator=calculator,
    )

    answer = service.answer("Calculate a general product-dose estimate.")

    assert calculator_calls == [(1000.0, 100.0)]
    assert answer.sources == []
    assert "4.2 gallons/day" in answer.text
    assert "not a field-ready prescription" in answer.text


def test_invalid_selection_executes_no_tool_and_falls_back_to_rag() -> None:
    selection_sets = (
        (ToolSelection("water_analysis", {}),),
        (ToolSelection("search_knowledge", {"query": 4}),),
        (
            ToolSelection("search_knowledge", {"query": "scale"}),
            ToolSelection("calculate_product_dose", {"water_bbl_per_day": 1, "product_ppm": 1}),
        ),
    )
    for selections in selection_sets:
        planner = FakePlanner(selections)
        rag_service = FakeRagService()
        service = AgenticRagService(
            rag_service=rag_service,
            planner=planner,
            calculator=lambda *_args: (_ for _ in ()).throw(AssertionError("tool must not run")),
        )

        answer = service.answer("How should I assess scale risk from produced water analysis?")

        assert answer is rag_service.answer_value
        assert rag_service.calls == [("How should I assess scale risk from produced water analysis?", None, None)]


def test_ollama_planner_uses_fixed_tool_registry_and_maps_one_selection() -> None:
    client = FakeOllamaClient(
        OllamaToolResponse(
            content="",
            tool_calls=(OllamaToolCall("search_knowledge", {"query": "scale evidence"}),),
        )
    )
    planner = OllamaToolPlanner(model="granite4.1:8b", client=client)

    selections = planner.select_tool("How should I assess scale risk?")

    assert selections == (ToolSelection("search_knowledge", {"query": "scale evidence"}),)
    assert client.calls == [
        {
            "model": "granite4.1:8b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Choose at most one listed tool. Never call an unlisted tool. "
                        "Do not answer the user directly."
                    ),
                },
                {"role": "user", "content": "How should I assess scale risk?"},
            ],
            "tools": TOOL_SCHEMAS,
            "generation_options": {"temperature": 0},
        }
    ]


def test_planner_transport_error_falls_back_without_exposing_provider_text() -> None:
    class FailingPlanner:
        def select_tool(self, _question: str) -> tuple[ToolSelection, ...]:
            raise OllamaClientError("private provider response")

    rag_service = FakeRagService()
    answer = AgenticRagService(rag_service=rag_service, planner=FailingPlanner()).answer(
        "How should I assess scale risk from produced water analysis?"
    )

    assert answer is rag_service.answer_value
    assert "private provider response" not in answer.text
    assert rag_service.calls == [("How should I assess scale risk from produced water analysis?", None, None)]
