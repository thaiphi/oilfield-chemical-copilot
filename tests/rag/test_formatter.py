from __future__ import annotations

from oilfield_chemical_copilot.rag.formatter import format_answer, weak_evidence_answer
from oilfield_chemical_copilot.rag.models import FALLBACK_MESSAGE, RagDraft, SourceEvidence


def _source(label: str = "Source 1") -> SourceEvidence:
    return SourceEvidence(
        source_id=label,
        chunk_id="scale-1",
        source_file="docs/scale.md",
        page_or_sheet="document",
        topic="scale",
        excerpt="Scale inhibitors are selected from water analysis evidence.",
        score=0.91,
    )


def test_format_answer_renders_required_sections_and_metadata_citations() -> None:
    answer = format_answer(
        RagDraft(
            answer="Check the water analysis for scale tendency.",
            why_this_matters="Scale can restrict production and damage equipment.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review calcium", "Review sulfate", "Confirm temperature"],
            limitations="Only one sample source was retrieved.",
        ),
        [_source()],
    )

    assert answer.text.startswith("Answer:\nCheck the water analysis")
    assert "Why this matters:\nScale can restrict" in answer.text
    assert "Evidence from retrieved sources:\n- Source 1: docs/scale.md, document, chunk scale-1" in answer.text
    assert "Recommended next checks:\n1. Review calcium\n2. Review sulfate\n3. Confirm temperature" in answer.text
    assert "C:/" not in answer.text


def test_weak_evidence_answer_uses_exact_fallback_sentence() -> None:
    answer = weak_evidence_answer(limitations="No retrieved chunks met the evidence threshold.")

    assert answer.text.splitlines()[1] == FALLBACK_MESSAGE
    assert answer.sources == []

def test_format_answer_rejects_successful_answer_without_citations() -> None:
    draft = RagDraft(
        answer="This answer has evidence available but cites none.",
        why_this_matters="Uncited answers are not grounded.",
        cited_source_ids=[],
        recommended_next_checks=["Check one", "Check two", "Check three"],
        limitations="None stated.",
    )

    import pytest

    with pytest.raises(Exception, match="at least one cited source"):
        format_answer(draft, [_source()])