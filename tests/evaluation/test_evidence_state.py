from __future__ import annotations

import pytest


class _StaticClient:
    def chat(self, **_: object) -> str:
        return '{"evidence_state":"SUFFICIENT"}'


class _CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **_: object) -> str:
        self.calls += 1
        return '{"evidence_state":"SUFFICIENT"}'


def _run_inputs():
    from oilfield_chemical_copilot.evaluation.evidence_state import DeliveredEvidence

    return {
        "contexts": (("case-1", "What is the condition?", (DeliveredEvidence(1, "Evidence."),)),),
        "gold_states": {"case-1": "SUFFICIENT"},
    }


def _frozen_input_contract(*, model: str = "frozen-model") -> dict[str, object]:
    return {
        "model": model,
        "contexts": [
            {
                "question_id": "case-1",
                "question": "What is the condition?",
                "evidence": [{"rank": 1, "passage_text": "Evidence."}],
            }
        ],
    }


def test_frozen_run_rejects_a_contract_with_a_different_model() -> None:
    from oilfield_chemical_copilot.evaluation.evidence_state import (
        EvidenceStateError,
        LocalEvidenceStateClassifier,
        classify_frozen_c1_contexts,
    )

    with pytest.raises(EvidenceStateError, match="E1A_RUN_INPUT_INVALID"):
        classify_frozen_c1_contexts(
            **_run_inputs(),
            classifier=LocalEvidenceStateClassifier(model="frozen-model", client=_StaticClient()),
            frozen_input_contract=_frozen_input_contract(model="different-model"),
        )


def test_frozen_run_records_the_digest_of_the_exact_model_and_context_contract() -> None:
    from oilfield_chemical_copilot.evaluation.evidence_state import (
        LocalEvidenceStateClassifier,
        canonical_input_contract_sha256,
        classify_frozen_c1_contexts,
    )

    frozen_input_contract = _frozen_input_contract()
    run = classify_frozen_c1_contexts(
        **_run_inputs(),
        classifier=LocalEvidenceStateClassifier(model="frozen-model", client=_StaticClient()),
        frozen_input_contract=frozen_input_contract,
    )

    assert run.model == "frozen-model"
    assert run.input_contract_sha256 == canonical_input_contract_sha256(frozen_input_contract)


def test_frozen_run_rejects_a_contract_with_different_delivered_context() -> None:
    from oilfield_chemical_copilot.evaluation.evidence_state import (
        EvidenceStateError,
        LocalEvidenceStateClassifier,
        classify_frozen_c1_contexts,
    )

    frozen_input_contract = _frozen_input_contract()
    frozen_input_contract["contexts"] = [
        {
            "question_id": "case-1",
            "question": "A different question?",
            "evidence": [{"rank": 1, "passage_text": "Evidence."}],
        }
    ]

    with pytest.raises(EvidenceStateError, match="E1A_RUN_INPUT_INVALID"):
        classify_frozen_c1_contexts(
            **_run_inputs(),
            classifier=LocalEvidenceStateClassifier(model="frozen-model", client=_StaticClient()),
            frozen_input_contract=frozen_input_contract,
        )


def test_frozen_run_rejects_invalid_gold_state_before_classifier_inference() -> None:
    from oilfield_chemical_copilot.evaluation.evidence_state import (
        EvidenceStateError,
        LocalEvidenceStateClassifier,
        classify_frozen_c1_contexts,
    )

    client = _CountingClient()
    with pytest.raises(EvidenceStateError, match="E1A_RUN_INPUT_INVALID"):
        classify_frozen_c1_contexts(
            contexts=_run_inputs()["contexts"],
            gold_states={"case-1": "PARTIAL"},  # type: ignore[dict-item]
            classifier=LocalEvidenceStateClassifier(model="frozen-model", client=client),
            frozen_input_contract=_frozen_input_contract(),
        )

    assert client.calls == 0
