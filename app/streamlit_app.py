from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from oilfield_chemical_copilot.tools.chemical_dosage import calculate_dosage
from oilfield_chemical_copilot.tools.water_analysis import summarize_water_analysis


st.set_page_config(
    page_title="Oilfield Chemical Troubleshooting Copilot",
    page_icon="OC",
    layout="wide",
)


def _initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask about oilfield production-chemistry troubleshooting. "
                    "This scaffold does not run RAG yet."
                ),
            }
        ]


def _placeholder_answer(prompt: str) -> str:
    # TODO: Replace with hybrid retrieval, tool selection, and OpenAI answer generation.
    return (
        "Scaffold response only. The future assistant will retrieve relevant chunks, "
        "call chemistry tools when useful, and cite source files for this question: "
        f"{prompt}"
    )


_initialize_state()

st.title("Oilfield Chemical Troubleshooting Copilot")
st.caption("LLM Zoomcamp 2026 capstone scaffold: RAG + tool calling for production chemistry.")

with st.sidebar:
    st.header("Tools")
    st.caption("Placeholders for future tool-calling routes.")

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
    chloride_mg_l = st.number_input("Chloride, mg/L", min_value=0.0, value=35000.0, step=1000.0)
    hardness_mg_l = st.number_input("Hardness, mg/L as CaCO3", min_value=0.0, value=2500.0, step=100.0)
    if st.button("Summarize water"):
        result = summarize_water_analysis(chloride_mg_l, hardness_mg_l)
        st.info(result.summary)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Describe the production-chemistry problem...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    start = time.perf_counter()
    answer = _placeholder_answer(prompt)
    latency_ms = int((time.perf_counter() - start) * 1000)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Placeholder latency: {latency_ms} ms")

        with st.expander("Source citations"):
            st.write("TODO: Show retrieved source chunks with file, page/sheet, and chunk ID.")

        feedback_cols = st.columns(2)
        with feedback_cols[0]:
            if st.button("Helpful", key=f"helpful-{len(st.session_state.messages)}"):
                st.toast("TODO: Log positive feedback.")
        with feedback_cols[1]:
            if st.button("Needs work", key=f"needs-work-{len(st.session_state.messages)}"):
                st.toast("TODO: Log negative feedback.")
