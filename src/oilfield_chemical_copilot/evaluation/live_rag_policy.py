"""Privacy-safe aggregate scoring for the claim-scope policy shadow."""

from __future__ import annotations

import json
from hashlib import sha256
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from oilfield_chemical_copilot.evaluation.abstention_policy import (
    AbstentionPolicyDecision,
    ClaimScopeCategory,
    classify_claim_scope,
)
from oilfield_chemical_copilot.evaluation.answers import (
    AnswerEvaluationCase,
    DeterministicAnswerResult,
    Status,
    _validated_comparison_provenance,
    evaluate_answer,
)
from oilfield_chemical_copilot.evaluation.live_rag import LiveAnswerCapture

POLICY_NAME = "claim_scope"
POLICY_VERSION = "v1"

_MODES = ("vector", "hybrid")
_CATEGORIES = frozenset(ClaimScopeCategory.__args__)


@dataclass(frozen=True)
class PolicyDeterministicMetrics:
    """Safe deterministic statuses with no case or runtime material."""

    citation_status: Status
    abstention_status: Status


@dataclass(frozen=True)
class PolicyCounterfactualScore:
    """Safe result of reconciling one runtime capture with a policy decision."""

    decision: AbstentionPolicyDecision
    control: PolicyDeterministicMetrics
    shadow: PolicyDeterministicMetrics
    pairing_identity: str = field(repr=False)


def _invalid_counterfactual() -> ValueError:
    return ValueError("invalid policy counterfactual")


def _invalid_report() -> ValueError:
    return ValueError("invalid policy investigation report")


def _safe_metrics(result: DeterministicAnswerResult) -> PolicyDeterministicMetrics:
    return PolicyDeterministicMetrics(result.citation_status, result.abstention_status)


def _pairing_identity(question_id: str) -> str:
    return sha256(question_id.encode("utf-8")).hexdigest()


def _valid_metrics(metrics: object) -> bool:
    return (
        isinstance(metrics, PolicyDeterministicMetrics)
        and metrics.citation_status in {"pass", "fail"}
        and metrics.abstention_status in {"pass", "fail"}
    )


def score_policy_counterfactual(
    decision: AbstentionPolicyDecision,
    case: AnswerEvaluationCase,
    capture: LiveAnswerCapture,
    control: DeterministicAnswerResult,
) -> PolicyCounterfactualScore:
    """Score an in-memory shadow result without invoking any runtime component."""
    if (
        not isinstance(decision, AbstentionPolicyDecision)
        or decision.category not in _CATEGORIES
        or not isinstance(case, AnswerEvaluationCase)
        or not isinstance(capture, LiveAnswerCapture)
        or not isinstance(control, DeterministicAnswerResult)
        or case.question_id != capture.answer.question_id
        or case.question_id != control.question_id
        or decision != classify_claim_scope(case.question)
    ):
        raise _invalid_counterfactual()
    expected_control = evaluate_answer(
        case,
        cited_evidence_ids=capture.answer.cited_evidence_ids,
        abstained=capture.answer.abstained,
    )
    if control != expected_control:
        raise _invalid_counterfactual()
    shadow = evaluate_answer(
        case,
        cited_evidence_ids=() if decision.action == "abstain" else capture.answer.cited_evidence_ids,
        abstained=True if decision.action == "abstain" else capture.answer.abstained,
    )
    return PolicyCounterfactualScore(
        decision,
        _safe_metrics(control),
        _safe_metrics(shadow),
        _pairing_identity(case.question_id),
    )


def _validated_modes(value: object) -> Mapping[str, list[object]]:
    if not isinstance(value, Mapping) or set(value) != set(_MODES):
        raise _invalid_report()
    if any(type(value[mode]) is not list for mode in _MODES):
        raise _invalid_report()
    return value


def _validated_score(score: object) -> PolicyCounterfactualScore:
    if (
        not isinstance(score, PolicyCounterfactualScore)
        or score.decision.category not in _CATEGORIES
        or not _valid_metrics(score.control)
        or not _valid_metrics(score.shadow)
        or len(score.pairing_identity) != 64
        or any(character not in "0123456789abcdef" for character in score.pairing_identity)
    ):
        raise _invalid_report()
    return score


def _status_counts(scores: list[PolicyCounterfactualScore], field: Literal["control", "shadow"]):
    metrics = [getattr(score, field) for score in scores]
    return {
        "citation": dict(sorted(Counter(metric.citation_status for metric in metrics).items())),
        "abstention": dict(sorted(Counter(metric.abstention_status for metric in metrics).items())),
    }


def _is_baseline(metrics: dict[str, dict[str, int]]) -> bool:
    return metrics == {
        "citation": {"fail": 8, "pass": 4},
        "abstention": {"fail": 6, "pass": 6},
    }


def _is_expected_shadow(metrics: dict[str, dict[str, int]]) -> bool:
    return metrics == {
        "citation": {"fail": 2, "pass": 10},
        "abstention": {"pass": 12},
    }


def _summary(
    scores: list[object],
) -> tuple[dict[str, object], tuple[tuple[str, str], ...], tuple[str, ...]]:
    if len(scores) != 12:
        raise _invalid_report()
    validated = [_validated_score(score) for score in scores]
    decisions = tuple((score.decision.action, score.decision.category) for score in validated)
    identities = tuple(score.pairing_identity for score in validated)
    if len(set(identities)) != 12:
        raise _invalid_report()
    action_counts = Counter(action for action, _ in decisions)
    if action_counts != {"allow": 6, "abstain": 6}:
        raise _invalid_report()
    control = _status_counts(validated, "control")
    shadow = _status_counts(validated, "shadow")
    if not _is_expected_shadow(shadow):
        raise _invalid_report()
    return (
        {
            "question_count": 12,
            "decision_categories": dict(sorted(Counter(category for _, category in decisions).items())),
            "control": control,
            "shadow": shadow,
        },
        decisions,
        identities,
    )


def write_live_rag_policy_investigation(
    scores_by_mode: Mapping[str, list[PolicyCounterfactualScore]],
    output_dir: Path,
    provenance: Mapping[str, object],
    *,
    verified_preflight: bool,
) -> tuple[Path, Path]:
    """Write the fixed-schema aggregate-only policy investigation reports."""
    if type(verified_preflight) is not bool or not isinstance(output_dir, Path):
        raise _invalid_report()
    modes = _validated_modes(scores_by_mode)
    try:
        safe_provenance = _validated_comparison_provenance(provenance)
    except (TypeError, ValueError):
        raise _invalid_report() from None

    summaries: dict[str, dict[str, object]] = {}
    decisions_by_mode: dict[str, tuple[tuple[str, str], ...]] = {}
    identities_by_mode: dict[str, tuple[str, ...]] = {}
    for mode in _MODES:
        summary, decisions, identities = _summary(modes[mode])
        summaries[mode] = summary
        decisions_by_mode[mode] = decisions
        identities_by_mode[mode] = identities
    if (
        decisions_by_mode["vector"] != decisions_by_mode["hybrid"]
        or identities_by_mode["vector"] != identities_by_mode["hybrid"]
    ):
        raise _invalid_report()

    baseline_reproduced = verified_preflight and all(
        _is_baseline(summary["control"]) for summary in summaries.values()
    )
    report = {
        "public": True,
        "provenance": safe_provenance,
        "verified_preflight": verified_preflight,
        "baseline_reproduced": baseline_reproduced,
        "policy": {"name": POLICY_NAME, "version": POLICY_VERSION},
        "modes": {mode: summaries[mode] for mode in _MODES},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "live_rag_policy_investigation.json"
    markdown_path = output_dir / "live_rag_policy_investigation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Live RAG abstention-policy investigation",
        "",
        "Public: true",
        f"Verified preflight: {str(verified_preflight).lower()}",
        f"Baseline reproduced: {str(baseline_reproduced).lower()}",
        f"Policy: {POLICY_NAME} {POLICY_VERSION}",
        "",
    ]
    for mode in _MODES:
        summary = summaries[mode]
        lines.extend(
            [
                f"## {mode}",
                "",
                f"Question count: {summary['question_count']}",
                f"Decision categories: {json.dumps(summary['decision_categories'], sort_keys=True)}",
                f"Control: {json.dumps(summary['control'], sort_keys=True)}",
                f"Shadow: {json.dumps(summary['shadow'], sort_keys=True)}",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
