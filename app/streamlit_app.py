from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
LOCAL_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
_PRODUCT_DOSE_REQUEST = re.compile(r"^\s*product\s+dose\s*:", re.IGNORECASE)
_PRODUCT_DOSE_INPUT = re.compile(
    r"\b(water_bbl_per_day|product_ppm)\s*=\s*([^,;\s]+)", re.IGNORECASE
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from oilfield_chemical_copilot.ollama import OllamaClient, OllamaClientError
from oilfield_chemical_copilot.evaluation.abstention_policy import classify_claim_scope
from oilfield_chemical_copilot.observability.aggregate_monitoring import (
    AggregateMonitor,
    MonitoringOutcome,
)
from oilfield_chemical_copilot.rag.formatter import scope_limited_answer
from oilfield_chemical_copilot.rag.agentic_service import AgenticRagService, OllamaToolPlanner
from oilfield_chemical_copilot.rag.generator_factory import build_answer_generator
from oilfield_chemical_copilot.rag.models import RagAnswer, RagConfigurationError, SourceEvidence
from oilfield_chemical_copilot.rag.ollama_client import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
)
from oilfield_chemical_copilot.rag.service import BasicRagService
from oilfield_chemical_copilot.retrieval.embeddings import build_embedding_provider
from oilfield_chemical_copilot.retrieval.keyword import KeywordSearchIndex
from oilfield_chemical_copilot.retrieval.pipeline import RetrievalSettings, build_retrieval_pipeline
from oilfield_chemical_copilot.storage.pgvector import PgVectorStore
from oilfield_chemical_copilot.tools.chemical_dosage import calculate_dosage, product_dosage_answer
from oilfield_chemical_copilot.tools.water_analysis import summarize_water_analysis

REQUEST_MONITOR = AggregateMonitor()


def _database_url() -> str:
    return os.getenv("DATABASE_URL") or LOCAL_DATABASE_URL


def _agentic_routing_enabled() -> bool:
    return os.getenv("AGENTIC_ROUTING_ENABLED", "").strip().lower() == "true"


def _initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask about oilfield production-chemistry troubleshooting. "
                    "This app retrieves indexed source chunks and answers with citations."
                ),
                "sources": [],
            }
        ]


@st.cache_resource(show_spinner=False)
def _build_rag_service(retrieval_mode: str) -> BasicRagService:
    settings = replace(RetrievalSettings.from_env(), retrieval_mode=retrieval_mode)
    embedding_provider = build_embedding_provider()
    store = PgVectorStore(_database_url(), embedding_dimension=embedding_provider.dimension)
    keyword_index = (
        KeywordSearchIndex.from_hits(store.list_chunks())
        if settings.retrieval_mode == "hybrid"
        else None
    )
    retriever = build_retrieval_pipeline(
        store=store,
        embedding_provider=embedding_provider,
        settings=settings,
        keyword_index=keyword_index,
    )
    return BasicRagService.from_settings(
        retriever=retriever,
        generator=build_answer_generator(),
        settings=settings,
    )


def _answer_question(prompt: str, retrieval_mode: str) -> RagAnswer:
    service = _build_rag_service(retrieval_mode)
    try:
        if _agentic_routing_enabled():
            planner = OllamaToolPlanner(
                model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
                client=OllamaClient(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)),
            )
            return AgenticRagService(rag_service=service, planner=planner).answer(prompt)
        return service.answer(prompt)
    except OllamaClientError as error:
        raise RagConfigurationError("Ollama retrieval is unavailable. Check local Ollama configuration.") from error


def _route_prompt(prompt: str, retrieval_mode: str) -> RagAnswer:
    return _route_prompt_with_outcome(prompt, retrieval_mode)[0]


def _route_prompt_with_outcome(
    prompt: str, retrieval_mode: str
) -> tuple[RagAnswer, MonitoringOutcome]:
    if not _PRODUCT_DOSE_REQUEST.match(prompt):
        claim_scope = classify_claim_scope(prompt)
        answer = _answer_question(prompt, retrieval_mode)
        if claim_scope.action == "abstain":
            return answer, MonitoringOutcome.SCOPE_ABSTAINED
        outcome = (
            MonitoringOutcome.RAG_WEAK_EVIDENCE
            if answer.weak_evidence
            else MonitoringOutcome.RAG_ANSWERED
        )
        return answer, outcome
    decision = classify_claim_scope(prompt)
    if decision.action == "abstain":
        return scope_limited_answer(category=decision.category), MonitoringOutcome.SCOPE_ABSTAINED
    inputs = _parse_product_dose_inputs(prompt)
    if inputs is None:
        return _tool_input_guidance(), MonitoringOutcome.TOOL_INPUT_INVALID
    try:
        return product_dosage_answer(**inputs), MonitoringOutcome.TOOL_CALCULATED
    except ValueError:
        return _tool_input_guidance(), MonitoringOutcome.TOOL_INPUT_INVALID


def _record_request(outcome: MonitoringOutcome, *, started_at: float) -> float:
    latency_ms = max(0.0, (time.perf_counter() - started_at) * 1000)
    REQUEST_MONITOR.record(outcome, latency_ms)
    return latency_ms


def _parse_product_dose_inputs(prompt: str) -> dict[str, float] | None:
    values: dict[str, str] = {}
    for name, value in _PRODUCT_DOSE_INPUT.findall(prompt):
        normalized_name = name.lower()
        if normalized_name in values:
            return None
        values[normalized_name] = value
    if set(values) != {"water_bbl_per_day", "product_ppm"}:
        return None
    try:
        return {name: float(value) for name, value in values.items()}
    except ValueError:
        return None


def _tool_input_guidance() -> RagAnswer:
    return RagAnswer(
        text=(
            "Answer:\nProvide a product-ppm water-basis calculation request with both "
            "water_bbl_per_day and product_ppm.\n\n"
            "Why this matters:\nThe calculator accepts only explicit, reviewable units.\n\n"
            "Evidence from retrieved sources:\n- No retrieval or calculation was run.\n\n"
            "Recommended next checks:\n"
            "1. Provide water_bbl_per_day as a positive number.\n"
            "2. Provide product_ppm as a nonnegative number.\n"
            "3. Use the Product dose: request form for this general calculation.\n\n"
            "Limitations:\nThis calculator does not accept active-ingredient dosing or field-ready treatment requests."
        ),
        sources=[],
        weak_evidence=True,
    )


def _citation_display(source: SourceEvidence) -> str:
    source_file = _safe_source_file(source.source_file)
    retrieval_sources = source.retrieval_sources or (source.retrieval_method,)
    return (
        f"{source.source_id}: {source_file} | {source.page_or_sheet} | "
        f"chunk {source.chunk_id} | score {source.score:.3f} | "
        f"{source.retrieval_method}: {' + '.join(retrieval_sources)}"
    )


def _safe_source_file(source_file: str) -> str:
    if PureWindowsPath(source_file).is_absolute() or PurePosixPath(source_file).is_absolute():
        return source_file.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return source_file


def _excerpt(text: str, *, limit: int = 280) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    if limit <= 3:
        return normalized[:limit]
    return normalized[: limit - 3] + "..."


def _render_tools_sidebar(default_retrieval_mode: str) -> str:
    with st.sidebar:
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            ("hybrid", "vector"),
            index=("hybrid", "vector").index(default_retrieval_mode),
        )
        st.header("Tools")
        st.caption("Deterministic calculator. Chat calculations require explicit product-ppm inputs.")

        st.subheader("Chemical Dosage")
        water_bbl_per_day = st.number_input(
            "Water rate, bbl/day", min_value=0.1, value=1000.0, step=100.0
        )
        product_ppm = st.number_input("Product dose, ppm", min_value=0.0, value=50.0, step=5.0)
        if st.button("Estimate dosage"):
            start = time.perf_counter()
            result = calculate_dosage(water_bbl_per_day, product_ppm)
            _record_request(MonitoringOutcome.TOOL_CALCULATED, started_at=start)
            st.info(f"{result.label}: {result.product_gallons_per_day:g} gallons/day")

        st.subheader("Water Analysis")
        chloride_mg_l = st.number_input(
            "Chloride, mg/L", min_value=0.0, value=35000.0, step=1000.0
        )
        hardness_mg_l = st.number_input(
            "Hardness, mg/L as CaCO3", min_value=0.0, value=2500.0, step=100.0
        )
        if st.button("Summarize water"):
            result = summarize_water_analysis(chloride_mg_l, hardness_mg_l)
            st.info(result.summary)
    return retrieval_mode


def _render_message(message: dict[str, object]) -> None:
    with st.chat_message(str(message["role"])):
        st.write(str(message["content"]))
        sources = message.get("sources") or []
        if sources:
            with st.expander("Source citations"):
                for source in sources:
                    st.markdown(f"**{_citation_display(source)}**")
                    st.write(_excerpt(source.excerpt))


def run_app() -> None:
    st.set_page_config(page_title="Oilfield Chemical Copilot", layout="wide")
    _initialize_state()
    st.title("Oilfield Chemical Troubleshooting Copilot")
    st.caption("Basic RAG with source-grounded answers for production chemistry.")
    default_retrieval_mode = RetrievalSettings.from_env().retrieval_mode
    retrieval_mode = _render_tools_sidebar(default_retrieval_mode)

    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input("Describe the production-chemistry problem...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    start = time.perf_counter()
    with st.chat_message("assistant"):
        try:
            answer, outcome = _route_prompt_with_outcome(prompt, retrieval_mode)
            latency_ms = _record_request(outcome, started_at=start)
            st.write(answer.text)
            st.caption(f"Response latency: {latency_ms:.0f} ms")
            if answer.sources:
                with st.expander("Source citations"):
                    for source in answer.sources:
                        st.markdown(f"**{_citation_display(source)}**")
                        st.write(_excerpt(source.excerpt))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer.text, "sources": answer.sources}
            )
        except (RagConfigurationError, ValueError) as error:
            _record_request(MonitoringOutcome.RAG_CONFIGURATION_ERROR, started_at=start)
            safe_message = f"Configuration needed before RAG can run: {error}"
            st.warning(safe_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": safe_message, "sources": []}
            )

    feedback_cols = st.columns(2)
    with feedback_cols[0]:
        if st.button("Helpful", key=f"helpful-{len(st.session_state.messages)}"):
            st.toast("Feedback logging starts in Milestone 8.")
    with feedback_cols[1]:
        if st.button("Needs work", key=f"needs-work-{len(st.session_state.messages)}"):
            st.toast("Feedback logging starts in Milestone 8.")


if __name__ == "__main__":
    run_app()
