from __future__ import annotations

from oilfield_chemical_copilot.rag.formatter import (
    format_answer,
    select_supported_sources,
    weak_evidence_answer,
)
from oilfield_chemical_copilot.rag.models import FALLBACK_MESSAGE, RagDraft, SourceEvidence


def _source(label: str = "Source 1") -> SourceEvidence:
    return SourceEvidence(
        source_id=label,
        chunk_id="scale-1",
        source_file="C:/private/docs/scale.md",
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
        question="Which water-analysis fields should be reviewed for scale risk?",
    )

    assert answer.text.startswith("Answer:\nCheck the water analysis")
    assert "Why this matters:\nScale can restrict" in answer.text
    assert "Evidence from retrieved sources:\n- Source 1: scale.md, document, chunk scale-1" in answer.text
    assert "Recommended next checks:\n1. Review calcium\n2. Review sulfate\n3. Confirm temperature" in answer.text
    assert "C:/" not in answer.text


def test_weak_evidence_answer_uses_exact_fallback_sentence() -> None:
    answer = weak_evidence_answer(limitations="No retrieved chunks met the evidence threshold.")

    assert answer.text.splitlines()[1] == FALLBACK_MESSAGE
    assert answer.sources == []

def test_format_answer_rejects_answer_without_any_supported_source() -> None:
    draft = RagDraft(
        answer="This answer has evidence available but cites none.",
        why_this_matters="Uncited answers are not grounded.",
        cited_source_ids=[],
        recommended_next_checks=["Check one", "Check two", "Check three"],
        limitations="None stated.",
    )

    import pytest

    with pytest.raises(Exception, match="at least one cited source"):
        format_answer(
            draft,
            [_source()],
            question="Which paraffin properties should be reviewed?",
        )


def test_selector_replaces_unrelated_model_citations_with_the_best_supported_source() -> None:
    sources = [
        SourceEvidence(
            source_id="Source 1",
            chunk_id="dosage",
            source_file="docs/chemical_dosage_examples.md",
            page_or_sheet="document",
            topic="dosage",
            excerpt="A continuous treatment estimate uses ppm, water barrels per day, and 42 gallons per barrel.",
            score=0.72,
        ),
        SourceEvidence(
            source_id="Source 2",
            chunk_id="corrosion",
            source_file="docs/corrosion_root_cause.md",
            page_or_sheet="document",
            topic="corrosion",
            excerpt="Corrosion reviews compare acid gas exposure, bacteria indicators, and inhibitor residuals.",
            score=0.95,
        ),
    ]
    draft = RagDraft(
        answer="Frame inhibitor dosage with ppm and water barrels per day.",
        why_this_matters="A continuous treatment estimate needs those inputs.",
        cited_source_ids=["Source 2"],
        recommended_next_checks=["Check ppm", "Check water rate", "Check active fraction"],
        limitations="General review only.",
    )

    selected = select_supported_sources(
        question="What public inputs frame an inhibitor dosage review?",
        draft=draft,
        sources=sources,
    )

    assert [source.source_id for source in selected] == ["Source 1"]


def test_selector_requires_a_topical_source_anchor_before_using_answer_overlap() -> None:
    sources = [
        SourceEvidence(
            source_id="Dosage",
            chunk_id="dosage",
            source_file="docs/chemical_dosage_examples.md",
            page_or_sheet="document",
            topic="dosage",
            excerpt="A continuous treatment estimate uses ppm, water barrels per day, and 42 gallons per barrel.",
            score=0.64,
        ),
        SourceEvidence(
            source_id="README",
            chunk_id="readme",
            source_file="README.md",
            page_or_sheet="document",
            topic="unknown",
            excerpt="Dosage, water analysis, corrosion, and scale are all covered by this public project.",
            score=0.62,
        ),
    ]
    draft = RagDraft(
        answer="Review dosage alongside water analysis, corrosion, and scale context.",
        why_this_matters="General operating context affects treatment review.",
        cited_source_ids=["README"],
        recommended_next_checks=["Review water", "Review corrosion", "Review scale"],
        limitations="General review only.",
    )

    selected = select_supported_sources(
        question="What public inputs frame an inhibitor dosage review?",
        draft=draft,
        sources=sources,
    )

    assert [source.source_id for source in selected] == ["Dosage"]


def test_selector_fails_closed_when_no_source_directly_supports_the_answer() -> None:
    draft = RagDraft(
        answer="Evaluate paraffin cloud point and crude composition.",
        why_this_matters="Those factors determine wax deposition behavior.",
        cited_source_ids=["Source 1"],
        recommended_next_checks=["Check one", "Check two", "Check three"],
        limitations="General review only.",
    )

    selected = select_supported_sources(
        question="Which paraffin properties should be reviewed?",
        draft=draft,
        sources=[_source()],
    )

    assert selected == []
