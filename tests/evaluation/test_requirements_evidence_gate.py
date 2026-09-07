from __future__ import annotations

import pytest

def _requirements():
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        Requirement,
        RequirementFixture,
    )

    return RequirementFixture(
        question_id="private-case",
        requirements=(
            Requirement(requirement_id="r1", requirement="Define the condition."),
            Requirement(requirement_id="r2", requirement="State its operational consequence."),
        ),
    )


def _evidence():
    from oilfield_chemical_copilot.evaluation.evidence_state import DeliveredEvidence

    return (
        DeliveredEvidence(rank=1, passage_text="Private passage one."),
        DeliveredEvidence(rank=3, passage_text="Private passage three."),
    )


def test_requirement_fixture_requires_one_to_three_ordered_unique_requirements() -> None:
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        Requirement,
        RequirementFixture,
        RequirementsEvidenceGateError,
    )

    with pytest.raises(RequirementsEvidenceGateError, match="E1A4_REQUIREMENT_FIXTURE_INVALID"):
        RequirementFixture(
            question_id="private-case",
            requirements=(
                Requirement(requirement_id="r1", requirement="One."),
                Requirement(requirement_id="r1", requirement="Two."),
            ),
        )


def test_support_result_requires_exact_requirement_set_and_direct_rank_provenance() -> None:
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        RequirementSupport,
        RequirementSupportResult,
        RequirementsEvidenceGateError,
        validate_support_result,
    )

    with pytest.raises(RequirementsEvidenceGateError, match="E1A4_SUPPORT_RESULT_INVALID"):
        validate_support_result(
            fixture=_requirements(),
            evidence=_evidence(),
            result=RequirementSupportResult(
                question_id="private-case",
                requirement_support=(
                    RequirementSupport(
                        requirement_id="r1",
                        status="SUPPORTED",
                        supporting_ranks=(),
                    ),
                    RequirementSupport(
                        requirement_id="r2",
                        status="UNSUPPORTED",
                        supporting_ranks=(),
                    ),
                ),
            ),
        )

    with pytest.raises(RequirementsEvidenceGateError, match="E1A4_SUPPORT_RESULT_INVALID"):
        validate_support_result(
            fixture=_requirements(),
            evidence=_evidence(),
            result=RequirementSupportResult(
                question_id="private-case",
                requirement_support=(
                    RequirementSupport(
                        requirement_id="r2",
                        status="UNSUPPORTED",
                        supporting_ranks=(),
                    ),
                    RequirementSupport(
                        requirement_id="r1",
                        status="SUPPORTED",
                        supporting_ranks=(2,),
                    ),
                ),
            ),
        )


def test_non_support_must_not_claim_passage_provenance() -> None:
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        RequirementSupport,
        RequirementSupportResult,
        RequirementsEvidenceGateError,
        validate_support_result,
    )

    with pytest.raises(RequirementsEvidenceGateError, match="E1A4_SUPPORT_RESULT_INVALID"):
        validate_support_result(
            fixture=_requirements(),
            evidence=_evidence(),
            result=RequirementSupportResult(
                question_id="private-case",
                requirement_support=(
                    RequirementSupport(
                        requirement_id="r1",
                        status="SUPPORTED",
                        supporting_ranks=(1,),
                    ),
                    RequirementSupport(
                        requirement_id="r2",
                        status="UNCLEAR",
                        supporting_ranks=(3,),
                    ),
                ),
            ),
        )


def test_supported_rank_must_belong_to_the_delivered_context() -> None:
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        RequirementSupport,
        RequirementSupportResult,
        RequirementsEvidenceGateError,
        validate_support_result,
    )

    with pytest.raises(RequirementsEvidenceGateError, match="E1A4_SUPPORT_RESULT_INVALID"):
        validate_support_result(
            fixture=_requirements(),
            evidence=_evidence(),
            result=RequirementSupportResult(
                question_id="private-case",
                requirement_support=(
                    RequirementSupport(
                        requirement_id="r1",
                        status="SUPPORTED",
                        supporting_ranks=(2,),
                    ),
                    RequirementSupport(
                        requirement_id="r2",
                        status="UNSUPPORTED",
                        supporting_ranks=(),
                    ),
                ),
            ),
        )


def test_empty_context_can_only_produce_an_abstention_first_result() -> None:
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        RequirementSupport,
        RequirementSupportResult,
        derive_answer_boundary,
        derive_evidence_state,
    )

    result = RequirementSupportResult(
        question_id="private-case",
        requirement_support=(
            RequirementSupport(requirement_id="r1", status="UNSUPPORTED", supporting_ranks=()),
            RequirementSupport(requirement_id="r2", status="UNCLEAR", supporting_ranks=()),
        ),
    )

    assert derive_evidence_state(fixture=_requirements(), evidence=(), result=result) == "INSUFFICIENT"
    assert derive_answer_boundary("INSUFFICIENT") == "ABSTENTION_REQUIRED"


def test_state_derivation_rejects_supported_result_without_delivered_evidence() -> None:
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        RequirementSupport,
        RequirementSupportResult,
        RequirementsEvidenceGateError,
        derive_evidence_state,
    )

    result = RequirementSupportResult(
        question_id="private-case",
        requirement_support=(
            RequirementSupport(requirement_id="r1", status="SUPPORTED", supporting_ranks=(1,)),
            RequirementSupport(requirement_id="r2", status="UNSUPPORTED", supporting_ranks=()),
        ),
    )

    with pytest.raises(RequirementsEvidenceGateError, match="E1A4_SUPPORT_RESULT_INVALID"):
        derive_evidence_state(fixture=_requirements(), evidence=(), result=result)


@pytest.mark.parametrize(
    ("statuses", "expected_state", "expected_boundary"),
    [
        (("SUPPORTED", "SUPPORTED"), "SUFFICIENT", "FULL_ANSWER_PERMITTED"),
        (("SUPPORTED", "UNCLEAR"), "PARTIALLY_SUFFICIENT", "SUPPORTED_ONLY_RESPONSE_REQUIRED"),
        (("UNSUPPORTED", "UNCLEAR"), "INSUFFICIENT", "ABSTENTION_REQUIRED"),
    ],
)
def test_controller_derives_evidence_state_and_answer_boundary(
    statuses: tuple[str, str], expected_state: str, expected_boundary: str
) -> None:
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        RequirementSupport,
        RequirementSupportResult,
        derive_answer_boundary,
        derive_evidence_state,
    )

    result = RequirementSupportResult(
        question_id="private-case",
        requirement_support=(
            RequirementSupport(
                requirement_id="r1",
                status=statuses[0],
                supporting_ranks=(1,) if statuses[0] == "SUPPORTED" else (),
            ),
            RequirementSupport(
                requirement_id="r2",
                status=statuses[1],
                supporting_ranks=(3,) if statuses[1] == "SUPPORTED" else (),
            ),
        ),
    )

    state = derive_evidence_state(fixture=_requirements(), evidence=_evidence(), result=result)

    assert state == expected_state
    assert derive_answer_boundary(state) == expected_boundary


def test_gate_observation_preserves_the_validated_result_and_controller_decisions() -> None:
    from oilfield_chemical_copilot.evaluation.requirements_evidence_gate import (
        RequirementSupport,
        RequirementSupportResult,
        RequirementsGateObservation,
        derive_gate_observation,
    )

    result = RequirementSupportResult(
        question_id="private-case",
        requirement_support=(
            RequirementSupport(requirement_id="r1", status="SUPPORTED", supporting_ranks=(1,)),
            RequirementSupport(requirement_id="r2", status="UNCLEAR", supporting_ranks=()),
        ),
    )

    observation = derive_gate_observation(
        fixture=_requirements(), evidence=_evidence(), result=result
    )

    assert isinstance(observation, RequirementsGateObservation)
    assert observation.question_id == "private-case"
    assert observation.requirement_support_result == result
    assert observation.evidence_state == "PARTIALLY_SUFFICIENT"
    assert observation.answer_boundary == "SUPPORTED_ONLY_RESPONSE_REQUIRED"
