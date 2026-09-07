from __future__ import annotations

import pytest


class _StaticClient:
    def chat(self, **_: object) -> str:
        return '{"evidence_state":"SUFFICIENT"}'


def _run_inputs():
    from oilfield_chemical_copilot.evaluation.evidence_state import DeliveredEvidence

    return {
        "contexts": (("case-1", "What is the condition?", (DeliveredEvidence(1, "Evidence."),)),),
        "gold_states": {"case-1": "SUFFICIENT"},
    }


def test_frozen_run_rejects_a_model_other_than_the_classifier_model() -> None:
    from oilfield_chemical_copilot.evaluation.evidence_state import (
        EvidenceStateError,
        LocalEvidenceStateClassifier,
        classify_frozen_c1_contexts,
    )

    with pytest.raises(EvidenceStateError, match="E1A_RUN_INPUT_INVALID"):
        classify_frozen_c1_contexts(
            **_run_inputs(),
            classifier=LocalEvidenceStateClassifier(model="frozen-model", client=_StaticClient()),
            model="different-model",
            input_contract_sha256="a" * 64,
        )


def test_frozen_run_records_the_classifier_model_with_a_valid_contract_digest() -> None:
    from oilfield_chemical_copilot.evaluation.evidence_state import (
        LocalEvidenceStateClassifier,
        classify_frozen_c1_contexts,
    )

    run = classify_frozen_c1_contexts(
        **_run_inputs(),
        classifier=LocalEvidenceStateClassifier(model="frozen-model", client=_StaticClient()),
        model="frozen-model",
        input_contract_sha256="a" * 64,
    )

    assert run.model == "frozen-model"
    assert run.input_contract_sha256 == "a" * 64


@pytest.mark.parametrize("digest", ["A" * 64, "g" * 64, "a" * 63])
def test_frozen_run_requires_a_lowercase_hex_contract_digest(digest: str) -> None:
    from oilfield_chemical_copilot.evaluation.evidence_state import (
        EvidenceStateError,
        LocalEvidenceStateClassifier,
        classify_frozen_c1_contexts,
    )

    with pytest.raises(EvidenceStateError, match="E1A_RUN_INPUT_INVALID"):
        classify_frozen_c1_contexts(
            **_run_inputs(),
            classifier=LocalEvidenceStateClassifier(model="frozen-model", client=_StaticClient()),
            model="frozen-model",
            input_contract_sha256=digest,
        )
