from __future__ import annotations

import os
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
LOCAL_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from oilfield_chemical_copilot.ollama import OllamaClientError
from oilfield_chemical_copilot.rag.models import RagAnswer, RagConfigurationError, SourceEvidence
from oilfield_chemical_copilot.rag.generator_factory import build_answer_generator
from oilfield_chemical_copilot.rag.service import BasicRagService
from oilfield_chemical_copilot.retrieval.embeddings import build_embedding_provider
from oilfield_chemical_copilot.retrieval.pipeline import BasicRetrievalPipeline, RetrievalSettings
from oilfield_chemical_copilot.storage.pgvector import PgVectorStore
from oilfield_chemical_copilot.tools.chemical_dosage import calculate_dosage
from oilfield_chemical_copilot.tools.water_analysis import summarize_water_analysis



def _database_url() -> str:
    return os.getenv("DATABASE_URL") or LOCAL_DATABASE_URL


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
def _build_rag_service() -> BasicRagService:
    settings = RetrievalSettings.from_env()
    embedding_provider = build_embedding_provider()
    store = PgVectorStore(_database_url(), embedding_dimension=embedding_provider.dimension)
    retriever = BasicRetrievalPipeline(
        store=store,
        embedding_provider=embedding_provider,
        settings=settings,
    )
    generator = build_answer_generator()
    return BasicRagService.from_settings(retriever=retriever, generator=generator, settings=settings)


def _answer_question(prompt: str) -> RagAnswer:
    service = _build_rag_service()
    try:
        return service.answer(prompt)
    except OllamaClientError as error:
        raise RagConfigurationError("Ollama retrieval is unavailable. Check local Ollama configuration.") from error


def _citation_display(source: SourceEvidence) -> str:
    source_file = _safe_source_file(source.source_file)
    return (
        f"{source.source_id}: {source_file} | {source.page_or_sheet} | "
        f"chunk {source.chunk_id} | score {source.score:.3f}"
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


def _render_tools_sidebar() -> None:
    with st.sidebar:
        st.header("Tools")
        st.caption("Standalone calculators. LLM tool routing starts in Milestone 5.")

        st.subheader("Chemical Dosage")
        volume_bbl = st.number_input("Fluid volume, bbl", min_value=0.0, value=1000.0, step=100.0)
        target_ppm = st.number_input("Target dosage, ppm", min_value=0.0, value=50.0, step=5.0)
        active_fraction = st.number_input(
            "Active fraction",
            min_value=0.01,
            max_value=1.0,
            value=0.25,
            step=0.01,
        )
        if st.button("Estimate dosage"):
            result = calculate_dosage(volume_bbl, target_ppm, active_fraction)
            st.info(result.summary)

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
    _render_tools_sidebar()

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
            answer = _answer_question(prompt)
            latency_ms = int((time.perf_counter() - start) * 1000)
            st.write(answer.text)
            st.caption(f"RAG latency: {latency_ms} ms")
            if answer.sources:
                with st.expander("Source citations"):
                    for source in answer.sources:
                        st.markdown(f"**{_citation_display(source)}**")
                        st.write(_excerpt(source.excerpt))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer.text, "sources": answer.sources}
            )
        except (RagConfigurationError, ValueError) as error:
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
