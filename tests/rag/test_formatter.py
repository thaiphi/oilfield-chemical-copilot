from __future__ import annotations

from dataclasses import replace

import pytest

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


def test_format_answer_replaces_a_field_treatment_directive_in_next_checks() -> None:
    answer = format_answer(
        RagDraft(
            answer="Oil-wet solids can restrict injectivity.",
            why_this_matters="Restriction can reduce disposal capacity.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=[
                "Confirm the deposit composition.",
                "Implement downhole scale inhibition measures.",
                "Review the injection pressure trend.",
            ],
            limitations="General evidence only.",
        ),
        [_source()],
        question="How can oil-wet iron sulfide solids affect water-disposal injectivity?",
    )

    assert "Implement downhole scale inhibition measures" not in answer.text
    assert "Obtain qualified engineering review before any field treatment change." in answer.text


def test_format_answer_replaces_a_plural_chemical_treatment_directive() -> None:
    answer = format_answer(
        RagDraft(
            answer="Wax can deposit when equipment cools below the cloud point.",
            why_this_matters="Deposits can restrict flow.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=[
                "Confirm the cloud point.",
                "Monitor flowline temperatures.",
                "Consider implementing hot oiling or chemical treatments if deposition is observed.",
            ],
            limitations="General evidence only.",
        ),
        [_source()],
        question="What is the wax appearance temperature?",
    )

    assert "chemical treatments" not in answer.text
    assert "Obtain qualified engineering review before any field treatment change." in answer.text


def test_format_answer_fails_closed_for_an_unsupported_numeric_claim() -> None:
    answer = format_answer(
        RagDraft(
            answer="A colloidal instability index above 0.9 proves instability.",
            why_this_matters="Unsupported thresholds can drive incorrect conclusions.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=[
                "Review the cited analytical method.",
                "Confirm the representative fluid sample.",
                "Obtain qualified engineering review before any field treatment change.",
            ],
            limitations="The cited source does not define a numeric cutoff.",
        ),
        [_source()],
        question="What does a colloidal instability index greater than 0.9 indicate?",
    )

    assert answer.weak_evidence is True
    assert "numeric claim" in answer.text
    assert answer.sources == []


def test_format_answer_allows_a_numeric_claim_found_in_cited_evidence() -> None:
    answer = format_answer(
        RagDraft(
            answer="The cited threshold is 0.9.",
            why_this_matters="The value is grounded in the cited evidence.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review the cited method.", "Confirm the sample.", "Review limits."],
            limitations="General evidence only.",
        ),
        [replace(_source(), excerpt="The cited threshold is 0.9.")],
        question="What does the cited threshold mean?",
    )

    assert answer.weak_evidence is False
    assert "The cited threshold is 0.9." in answer.text


def test_format_answer_rejects_a_numeric_claim_embedded_in_a_larger_source_number() -> None:
    answer = format_answer(
        RagDraft(
            answer="The study evaluated 50 samples.",
            why_this_matters="The sample count must be grounded exactly.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review the method.", "Confirm the sample set.", "Review limits."],
            limitations="General evidence only.",
        ),
        [replace(_source(), excerpt="The study evaluated 150 samples.")],
        question="How many samples were evaluated?",
    )

    assert answer.weak_evidence is True
    assert "numeric claim" in answer.text
    assert answer.sources == []


def test_format_answer_allows_equivalent_decimal_numeric_tokens() -> None:
    answer = format_answer(
        RagDraft(
            answer="The study evaluated 50.0 samples.",
            why_this_matters="Equivalent numeric representations should remain grounded.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review the method.", "Confirm the sample set.", "Review limits."],
            limitations="General evidence only.",
        ),
        [replace(_source(), excerpt="The study evaluated 50 samples.")],
        question="How many samples were evaluated?",
    )

    assert answer.weak_evidence is False
    assert "50.0 samples" in answer.text


def test_format_answer_allows_equivalent_decimal_measurements() -> None:
    answer = format_answer(
        RagDraft(
            answer="The cited concentration was 50.0 ppm.",
            why_this_matters="Equivalent measurement representations should remain grounded.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review the method.", "Confirm the sample.", "Review limits."],
            limitations="General evidence only.",
        ),
        [replace(_source(), excerpt="The cited concentration was 50 ppm.")],
        question="What concentration was cited?",
    )

    assert answer.weak_evidence is False
    assert "50.0 ppm" in answer.text


def test_format_answer_allows_equivalent_decimal_comparators() -> None:
    answer = format_answer(
        RagDraft(
            answer="The cited concentration was above 50.0 ppm.",
            why_this_matters="Equivalent comparator values should remain grounded.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review the method.", "Confirm the sample.", "Review limits."],
            limitations="General evidence only.",
        ),
        [replace(_source(), excerpt="The cited concentration was above 50 ppm.")],
        question="What concentration threshold was cited?",
    )

    assert answer.weak_evidence is False
    assert "above 50.0 ppm" in answer.text


@pytest.mark.parametrize(
    ("answer_text", "excerpt", "question"),
    [
        (
            "The synthetic laboratory result was 75 ppm.",
            "The laboratory result was 75 mg/L using the stated method.",
            "What result did the synthetic laboratory report?",
        ),
        (
            "The synthetic temperature was 75 C.",
            "The synthetic temperature was 75 F during the observation.",
            "What temperature did the synthetic observation report?",
        ),
        (
            "Above 4 hours proves the synthetic process is stable.",
            "The observation window was between 4 and 9 hours without a stability conclusion.",
            "What does the synthetic observation window establish?",
        ),
        (
            "The synthetic trend proves deposition risk.",
            "The synthetic trend may be associated with deposition risk when mixing occurs.",
            "Does the synthetic trend prove deposition risk?",
        ),
        (
            "The synthetic result of 11 mg/L confirms treatment effectiveness.",
            "The sample result was 11 mg/L during the review period.",
            "Does the synthetic result confirm treatment effectiveness?",
        ),
    ],
)
def test_format_answer_fails_closed_when_a_technical_claim_changes_evidence_meaning(
    answer_text: str, excerpt: str, question: str
) -> None:
    answer = format_answer(
        RagDraft(
            answer=answer_text,
            why_this_matters="Technical meaning must remain grounded.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review the method.", "Confirm the context.", "Review limits."],
            limitations="Synthetic public regression coverage.",
        ),
        [replace(_source(), excerpt=excerpt)],
        question=question,
    )

    assert answer.weak_evidence is True
    assert answer.sources == []


def test_format_answer_fails_closed_when_conflicting_evidence_is_presented_as_settled() -> None:
    answer = format_answer(
        RagDraft(
            answer="The synthetic deposit is mineral scale.",
            why_this_matters="Conflicting evidence must remain visible.",
            cited_source_ids=["Source 1", "Source 2"],
            recommended_next_checks=["Review both methods.", "Confirm the sample.", "Review limits."],
            limitations="Synthetic public regression coverage.",
        ),
        [
            replace(_source("Source 1"), excerpt="One examination noted mineral particles."),
            replace(
                _source("Source 2"),
                excerpt="A second examination was insufficient and did not identify the deposit class.",
            ),
        ],
        question="What deposit type do the synthetic examinations establish?",
    )

    assert answer.weak_evidence is True
    assert answer.sources == []


@pytest.mark.parametrize(
    ("answer_text", "excerpt", "question"),
    [
        (
            "The synthetic pressure equals 7 psi.",
            "The synthetic pressure must remain below 7 psi.",
            "What synthetic pressure limit is supported?",
        ),
        (
            "The synthetic pressure is 2 bar.",
            "The synthetic pressure was 2 psi during the inspection.",
            "What pressure did the synthetic inspection report?",
        ),
        (
            "The synthetic deposit forms at 14 ppm.",
            "The synthetic deposit forms at 14 ppm only when the brine contains the stated tracer.",
            "When can the synthetic deposit form?",
        ),
    ],
)
def test_format_answer_fails_closed_for_unpreserved_semantic_qualifiers(
    answer_text: str, excerpt: str, question: str
) -> None:
    answer = format_answer(
        RagDraft(
            answer=answer_text,
            why_this_matters="Synthetic public regression coverage.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review the method.", "Confirm the context.", "Review limits."],
            limitations="Synthetic public regression coverage.",
        ),
        [replace(_source(), excerpt=excerpt)],
        question=question,
    )

    assert answer.weak_evidence is True
    assert answer.sources == []


def test_format_answer_fails_closed_for_single_source_conflict_presented_as_settled() -> None:
    answer = format_answer(
        RagDraft(
            answer="The synthetic deposit is mineral scale.",
            why_this_matters="Conflicting evidence must remain visible.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review both methods.", "Confirm the sample.", "Review limits."],
            limitations="Synthetic public regression coverage.",
        ),
        [
            replace(
                _source(),
                excerpt="One synthetic examination noted mineral particles, but a second examination could not confirm the deposit class.",
            ),
        ],
        question="What deposit type do the synthetic examinations establish?",
    )

    assert answer.weak_evidence is True
    assert answer.sources == []


def test_format_answer_allows_an_explicitly_grounded_absent_threshold_statement() -> None:
    answer = format_answer(
        RagDraft(
            answer="There is no established synthetic cutoff.",
            why_this_matters="The source leaves the cutoff undefined.",
            cited_source_ids=["Source 1"],
            recommended_next_checks=["Review the method.", "Confirm the context.", "Review limits."],
            limitations="Synthetic public regression coverage.",
        ),
        [replace(_source(), excerpt="No established synthetic cutoff is defined by the evidence.")],
        question="What synthetic cutoff does the evidence establish?",
    )

    assert answer.weak_evidence is False
    assert [source.source_id for source in answer.sources] == ["Source 1"]


def test_selector_preserves_validated_model_citations() -> None:
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

    assert [source.source_id for source in selected] == ["Source 2"]


def test_selector_preserves_a_validated_citation_even_when_terms_are_broad() -> None:
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

    assert [source.source_id for source in selected] == ["README"]


def test_selector_preserves_a_validated_retrieved_source() -> None:
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

    assert [source.source_id for source in selected] == ["Source 1"]
