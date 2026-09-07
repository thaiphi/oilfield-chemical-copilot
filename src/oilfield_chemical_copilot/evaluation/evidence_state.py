"""Private E1a evidence-state classification primitives.

The classifier sees a user question and frozen C1-delivered passages only.
Gold claims and passage-level support labels are used strictly after inference
to score the private experiment.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol, Sequence

from oilfield_chemical_copilot.evaluation.claim_support_depth import (
    ClaimSupportLabel,
    RequiredClaimFixture,
)
from oilfield_chemical_copilot.evaluation.private_retrieval import PRIVATE_RETRIEVAL_ROOT
from oilfield_chemical_copilot.ollama import OllamaClientError

EvidenceState = Literal["SUFFICIENT", "PARTIALLY_SUFFICIENT", "INSUFFICIENT"]
_STATES: tuple[EvidenceState, ...] = (
    "SUFFICIENT",
    "PARTIALLY_SUFFICIENT",
    "INSUFFICIENT",
)
_PUBLIC_MINIMUM_COHORT_SIZE = 10
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

E1A_SYSTEM_PROMPT = """Classify whether the supplied evidence context can support the user's question.
Do not answer the question. Do not add facts. Do not infer missing technical details.

Use these states:
- SUFFICIENT: the evidence directly supports all material parts needed to answer the question.
- PARTIALLY_SUFFICIENT: the evidence supports a material part but misses another material part.
- INSUFFICIENT: the evidence does not directly support a material answer to the question.

Return only the required JSON object."""


class EvidenceStateError(ValueError):
    """A sanitized E1a contract or classifier failure."""


class EvidenceStateClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, object] | None = None,
        generation_options: dict[str, object] | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class DeliveredEvidence:
    """One C1 passage with only the data the classifier is allowed to see."""

    rank: int
    passage_text: str

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1 or not self.passage_text.strip():
            raise EvidenceStateError("E1A_DELIVERED_EVIDENCE_INVALID")


@dataclass(frozen=True)
class EvidenceStateObservation:
    """Private case result. Never write this outside the private evaluation root."""

    question_id: str
    actual_state: EvidenceState
    predicted_state: EvidenceState


@dataclass(frozen=True)
class EvidenceStateRun:
    observations: tuple[EvidenceStateObservation, ...]
    model: str
    input_contract_sha256: str

    def private_result(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "experiment": "E1A_EVIDENCE_STATE_CLASSIFICATION",
            "model": self.model,
            "input_contract_sha256": self.input_contract_sha256,
            "aggregate": self.aggregate_report(),
            "observations": [asdict(observation) for observation in self.observations],
        }

    def aggregate_report(self) -> dict[str, object]:
        if not self.observations:
            raise EvidenceStateError("E1A_OBSERVATIONS_REQUIRED")
        confusion = {
            expected: {
                observed: sum(
                    item.actual_state == expected and item.predicted_state == observed
                    for item in self.observations
                )
                for observed in _STATES
            }
            for expected in _STATES
        }
        metrics = _classification_metrics(confusion)
        return {
            "case_count": len(self.observations),
            "overall": {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "sufficient_precision": metrics["sufficient_precision"],
                "unsafe_false_sufficient": _public_unsafe_false_sufficient(metrics),
            },
            "class_metrics": _public_class_metrics(metrics["class_metrics"]),
            "confusion_matrix": _public_confusion_matrix(confusion),
            "privacy": {
                "case_level_results": "PRIVATE_ONLY",
                "small_cells": "SUPPRESSED_WHEN_SUPPORT_BELOW_10",
            },
        }


class LocalEvidenceStateClassifier:
    """Fail-closed local classifier with a narrow structured-output contract."""

    def __init__(self, *, model: str, client: EvidenceStateClient) -> None:
        if not isinstance(model, str) or not model.strip():
            raise EvidenceStateError("E1A_MODEL_REQUIRED")
        self._model = model
        self._client = client

    @property
    def model(self) -> str:
        """Return the exact model identifier used for classifier calls."""
        return self._model

    def classify(self, *, question: str, evidence: Sequence[DeliveredEvidence]) -> EvidenceState:
        if not question.strip():
            raise EvidenceStateError("E1A_CLASSIFIER_INPUT_INVALID")
        try:
            raw = self._client.chat(
                model=self._model,
                system_prompt=E1A_SYSTEM_PROMPT,
                user_prompt=_classifier_prompt(question=question, evidence=evidence),
                response_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["evidence_state"],
                    "properties": {"evidence_state": {"type": "string", "enum": list(_STATES)}},
                },
                generation_options={"temperature": 0},
            )
        except (OllamaClientError, OSError, TypeError, ValueError) as error:
            raise EvidenceStateError("E1A_CLASSIFIER_UNAVAILABLE") from error
        return _parse_state(raw)


def derive_gold_state(
    *,
    fixture: RequiredClaimFixture,
    label: ClaimSupportLabel,
    retained_ranks: Sequence[int],
) -> EvidenceState:
    """Derive an evaluator-only state from sealed D2 claims and blind labels."""
    fixture.require_ready()
    if label.question_id != fixture.question_id:
        raise EvidenceStateError("E1A_GOLD_STATE_INPUT_INVALID")
    if any(type(rank) is not int or rank < 1 for rank in retained_ranks):
        raise EvidenceStateError("E1A_GOLD_STATE_INPUT_INVALID")
    required = {claim.claim_id for claim in fixture.required_claims}
    covered = set().union(
        *(
            set(item.claim_ids)
            for item in label.claim_support_by_rank
            if item.rank in set(retained_ranks)
        )
    )
    if not covered.issubset(required):
        raise EvidenceStateError("E1A_GOLD_STATE_INPUT_INVALID")
    if covered == required:
        return "SUFFICIENT"
    if covered:
        return "PARTIALLY_SUFFICIENT"
    return "INSUFFICIENT"


def classify_frozen_c1_contexts(
    *,
    contexts: Sequence[tuple[str, str, Sequence[DeliveredEvidence]]],
    gold_states: dict[str, EvidenceState],
    classifier: LocalEvidenceStateClassifier,
    model: str,
    input_contract_sha256: str,
) -> EvidenceStateRun:
    """Run E1a without exposing evaluator-only gold labels to the classifier."""
    if (
        not contexts
        or not isinstance(model, str)
        or model != classifier.model
        or not isinstance(input_contract_sha256, str)
        or not _SHA256_HEX.fullmatch(input_contract_sha256)
    ):
        raise EvidenceStateError("E1A_RUN_INPUT_INVALID")
    ids = [question_id for question_id, _, _ in contexts]
    if len(ids) != len(set(ids)) or set(ids) != set(gold_states):
        raise EvidenceStateError("E1A_RUN_INPUT_INVALID")
    observations: list[EvidenceStateObservation] = []
    for question_id, question, evidence in contexts:
        predicted = classifier.classify(question=question, evidence=evidence)
        observations.append(
            EvidenceStateObservation(
                question_id=question_id,
                actual_state=gold_states[question_id],
                predicted_state=predicted,
            )
        )
    return EvidenceStateRun(
        observations=tuple(observations),
        model=model,
        input_contract_sha256=input_contract_sha256,
    )


def write_private_evidence_state_result(run: EvidenceStateRun, destination: Path) -> None:
    """Persist E1a case-level results only under the local private root."""
    _require_private_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(run.private_result(), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_input_contract_sha256(contract: dict[str, object]) -> str:
    """Return a stable digest for the private frozen E1a input contract."""
    return sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _classifier_prompt(*, question: str, evidence: Sequence[DeliveredEvidence]) -> str:
    passages = (
        "\n\n".join(f"Passage {item.rank}:\n{item.passage_text.strip()}" for item in evidence)
        if evidence
        else "No evidence passage was delivered."
    )
    return f"User question:\n{question.strip()}\n\nEvidence context:\n{passages}"


def _parse_state(raw: str) -> EvidenceState:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise EvidenceStateError("E1A_CLASSIFIER_RESPONSE_INVALID") from error
    if not isinstance(payload, dict) or set(payload) != {"evidence_state"}:
        raise EvidenceStateError("E1A_CLASSIFIER_RESPONSE_INVALID")
    state = payload["evidence_state"]
    if state not in _STATES:
        raise EvidenceStateError("E1A_CLASSIFIER_RESPONSE_INVALID")
    return state


def _classification_metrics(
    confusion: dict[EvidenceState, dict[EvidenceState, int]],
) -> dict[str, object]:
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[state][state] for state in _STATES)
    class_metrics: dict[EvidenceState, dict[str, float | int]] = {}
    for state in _STATES:
        true_positive = confusion[state][state]
        support = sum(confusion[state].values())
        predicted = sum(confusion[actual][state] for actual in _STATES)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[state] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    false_sufficient = sum(
        confusion[actual]["SUFFICIENT"]
        for actual in ("PARTIALLY_SUFFICIENT", "INSUFFICIENT")
    )
    nonsufficient = sum(
        sum(confusion[actual].values())
        for actual in ("PARTIALLY_SUFFICIENT", "INSUFFICIENT")
    )
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(item["f1"] for item in class_metrics.values()) / len(_STATES),
        "sufficient_precision": class_metrics["SUFFICIENT"]["precision"],
        "unsafe_false_sufficient_rate": false_sufficient / nonsufficient if nonsufficient else 0.0,
        "nonsufficient_support": nonsufficient,
        "class_metrics": class_metrics,
    }


def _public_class_metrics(metrics: dict[EvidenceState, dict[str, float | int]]) -> dict[str, object]:
    return {
        state: (
            {
                "status": "SUPPRESSED_SMALL_COHORT",
                "minimum_public_cohort_size": _PUBLIC_MINIMUM_COHORT_SIZE,
            }
            if int(values["support"]) < _PUBLIC_MINIMUM_COHORT_SIZE
            else {
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
            }
        )
        for state, values in metrics.items()
    }


def _public_confusion_matrix(
    confusion: dict[EvidenceState, dict[EvidenceState, int]],
) -> dict[str, object]:
    if any(sum(row.values()) < _PUBLIC_MINIMUM_COHORT_SIZE for row in confusion.values()):
        return {"status": "SUPPRESSED_SMALL_COHORT"}
    return confusion


def _public_unsafe_false_sufficient(metrics: dict[str, object]) -> dict[str, object]:
    rate = float(metrics["unsafe_false_sufficient_rate"])
    support = int(metrics["nonsufficient_support"])
    if support < _PUBLIC_MINIMUM_COHORT_SIZE:
        return {
            "status": (
                "UNSAFE_FALSE_SUFFICIENT_OBSERVED"
                if rate > 0
                else "NO_UNSAFE_FALSE_SUFFICIENT_OBSERVED"
            ),
            "quantitative_detail": "SUPPRESSED_SMALL_COHORT",
        }
    return {"rate": rate}


def _require_private_path(path: Path) -> None:
    try:
        path.resolve().relative_to(PRIVATE_RETRIEVAL_ROOT.resolve())
    except ValueError as error:
        raise EvidenceStateError("E1A_PRIVATE_PATH_REQUIRED") from error
