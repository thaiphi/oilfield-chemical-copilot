"""Evaluation-only claim-scope abstention policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ClaimScopeCategory = Literal[
    "general_review",
    "site_specific_determination",
    "field_ready_prescription",
    "complete_input_substitution",
]
PolicyAction = Literal["allow", "abstain"]


@dataclass(frozen=True)
class AbstentionPolicyDecision:
    action: PolicyAction
    category: ClaimScopeCategory

    def __post_init__(self) -> None:
        expected_action: PolicyAction = "allow" if self.category == "general_review" else "abstain"
        if self.action != expected_action:
            raise ValueError("policy action does not match claim-scope category")


def _normalize(question: str) -> str:
    lowered = question.lower().replace("-", " ")
    punctuation_free = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", punctuation_free).strip()


def _contains(text: str, *phrases: str) -> bool:
    return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)


def _has_unresolved_scope_boundary(text: str) -> bool:
    return (
        _contains(text, "should")
        and _contains(text, "determine", "confirm", "predict", "diagnose", "establish")
        and _contains(text, "site specific", "named asset", "specific deposit", "mechanism", "root cause")
    ) or (
        _contains(text, "should")
        and _contains(text, "recommend")
        and _contains(text, "field ready", "final", "prescribe")
        and _contains(text, "dosage", "dose", "treatment", "treatment plan")
    ) or (
        _contains(text, "infer") and _contains(text, "treatment", "approach")
    ) or (
        _contains(text, "general review")
        and _contains(text, "select", "treatment plan", "root cause")
    ) or (
        _contains(text, "without", "no")
        and _contains(text, "site data", "laboratory analysis", "field data")
        and _contains(text, "select", "recommend", "treatment")
    )


def classify_claim_scope(question: str) -> AbstentionPolicyDecision:
    """Classify normalized question text without evaluation or runtime inputs."""
    if not isinstance(question, str):
        raise ValueError("question must be a non-blank string")
    text = _normalize(question)
    if not text:
        raise ValueError("question must be a non-blank string")
    if _contains(text, "replace", "substitute") and _contains(
        text, "complete", "full"
    ) and _contains(text, "analysis", "data set", "input set"):
        return AbstentionPolicyDecision("abstain", "complete_input_substitution")
    if _contains(text, "field ready", "final", "prescribe") and _contains(
        text, "dosage", "dose", "treatment", "treatment plan"
    ):
        return AbstentionPolicyDecision("abstain", "field_ready_prescription")
    if _contains(text, "determine", "confirm", "predict", "diagnose", "establish") and _contains(
        text, "site specific", "named asset", "specific deposit", "mechanism", "root cause"
    ):
        return AbstentionPolicyDecision("abstain", "site_specific_determination")
    if _has_unresolved_scope_boundary(text):
        return AbstentionPolicyDecision("abstain", "site_specific_determination")
    return AbstentionPolicyDecision("allow", "general_review")
