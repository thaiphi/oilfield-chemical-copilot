import inspect
import json
from pathlib import Path

import pytest

from oilfield_chemical_copilot.evaluation.abstention_policy import (
    AbstentionPolicyDecision,
    classify_claim_scope,
)


@pytest.mark.parametrize(
    ("category", "questions"),
    [
        ("complete_input_substitution", [
            "Can this replace a complete field analysis?",
            "Could the public sample substitute for a full input set?",
        ]),
        ("field_ready_prescription", [
            "Can you prescribe a field-ready dosage?",
            "What final treatment plan should we use?",
            "What exact chemical dose should I inject for Well A tomorrow?",
        ]),
        ("site_specific_determination", [
            "Can this determine the root cause at a named asset?",
            "Could the material confirm the mechanism for a specific deposit?",
        ]),
    ],
)
def test_classifies_closed_abstention_categories(category, questions):
    for question in questions:
        decision = classify_claim_scope(question)
        assert decision == AbstentionPolicyDecision("abstain", category)


@pytest.mark.parametrize(
    "question",
    [
        "Which indicators should frame an initial review?",
        "What inputs are useful for a general screen?",
        "What public evidence should be checked for scale risk?",
        "What topics should a general review program include?",
        "What should a general review program recommend?",
    ],
)
def test_allows_general_review_questions(question):
    assert classify_claim_scope(question) == AbstentionPolicyDecision("allow", "general_review")


def test_precedence_chooses_complete_input_substitution():
    assert classify_claim_scope(
        "Can this replace a complete input set and prescribe a final treatment plan?"
    ).category == "complete_input_substitution"


def test_explicit_field_ready_recommendation_keeps_field_ready_category():
    assert classify_claim_scope(
        "Should you recommend a field-ready treatment plan?"
    ) == AbstentionPolicyDecision("abstain", "field_ready_prescription")


@pytest.mark.parametrize(
    "question",
    [
        "Can broad findings infer a treatment approach?",
        "Provide a general review and select the treatment plan.",
        "For a general review, identify the root cause.",
        "Recommend a treatment without site data or laboratory analysis.",
    ],
)
def test_abstains_for_unresolved_scope_boundaries(question):
    assert classify_claim_scope(question).action == "abstain"


@pytest.mark.parametrize("question", ["confirm the trend", "review site-specific indicators"])
def test_near_misses_remain_general_review(question):
    assert classify_claim_scope(question).category == "general_review"


def test_normalization_handles_case_punctuation_and_hyphens():
    assert classify_claim_scope("CAN THIS REPLACE a COMPLETE, FULL-analysis?").category == (
        "complete_input_substitution"
    )


@pytest.mark.parametrize("value", ["", "   ", "!?--", None, 42])
def test_rejects_malformed_input_without_echoing_value(value):
    with pytest.raises(ValueError, match="question must be a non-blank string") as error:
        classify_claim_scope(value)
    assert repr(value) not in str(error.value)


def test_signature_accepts_only_question_text():
    parameters = inspect.signature(classify_claim_scope).parameters
    assert list(parameters) == ["question"]
    assert all(parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters.values())


def test_repeated_calls_are_deterministic():
    question = "Can the public sample determine a site-specific root cause?"
    assert [classify_claim_scope(question) for _ in range(20)] == [
        classify_claim_scope(question)
    ] * 20


def test_frozen_canonical_questions_are_compatibility_characterization_only():
    dataset = Path("eval/public_answer_evaluation.jsonl")
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    decisions = [classify_claim_scope(case["question"]) for case in cases]
    assert len(decisions) == 12
    assert sum(decision.action == "allow" for decision in decisions) == 6
    assert sum(decision.action == "abstain" for decision in decisions) == 6
