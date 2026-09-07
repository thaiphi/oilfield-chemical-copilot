from __future__ import annotations

import pytest

def _case(
    index: int,
    *,
    topic: str,
    role: str,
    question_form: str = "definition_mechanism",
    evidence_depth: str = "single_claim",
):
    from oilfield_chemical_copilot.evaluation.e1a4_population import E1A4PopulationCase

    return E1A4PopulationCase(
        question_id=f"q{index:02d}",
        question=f"Private question {index}?",
        topic=topic,
        expected_source="private.pdf",
        expected_locator=f"page:{index}",
        expected_source_role=role,
        question_form=question_form,
        evidence_depth=evidence_depth,
    )


def _state(question_id: str, state: str):
    from oilfield_chemical_copilot.evaluation.e1a4_selection import E1A4GoldState

    return E1A4GoldState(question_id=question_id, state=state)


def _full_cases():
    from itertools import product

    topics = ("iron_sulfide", "scale", "corrosion", "paraffin")
    roles = ("foundational", "supporting")
    forms = ("definition_mechanism", "diagnostic_interpretive", "operational_procedural")
    depths = ("single_claim", "multi_claim")
    return tuple(
        _case(index, topic=topic, role=role, question_form=form, evidence_depth=depth)
        for index, (topic, role, form, depth, _) in enumerate(
            product(topics, roles, forms, depths, (1, 2))
        )
    )


def test_gold_state_set_requires_independent_blind_label_provenance() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_selection import (
        E1A4GoldState,
        E1A4GoldStateSet,
        E1A4SelectionError,
    )

    with pytest.raises(E1A4SelectionError, match="E1A4_GOLD_STATE_PROVENANCE_INVALID"):
        E1A4GoldStateSet(
            blind_label_sha256="not-a-digest",
            blind_review_contract_sha256="b" * 64,
            states=(E1A4GoldState(question_id="q1", state="SUFFICIENT"),),
        )


def test_selector_returns_balanced_coverage_only_from_independent_gold_states() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_selection import (
        E1A4GoldStateSet,
        select_balanced_case_ids,
    )

    states = ("SUFFICIENT", "PARTIALLY_SUFFICIENT", "INSUFFICIENT")
    cases = _full_cases()
    gold = E1A4GoldStateSet(
        blind_label_sha256="a" * 64,
        blind_review_contract_sha256="b" * 64,
        states=tuple(_state(case.question_id, states[index // 32]) for index, case in enumerate(cases)),
    )

    selected = select_balanced_case_ids(cases=cases, gold_states=gold)
    assert len(selected) == 30
    assert set(selected).issubset({case.question_id for case in cases})


def test_selector_rejects_truncated_population_or_gold_state_set() -> None:
    from oilfield_chemical_copilot.evaluation.e1a4_selection import (
        E1A4GoldStateSet,
        E1A4SelectionError,
        select_balanced_case_ids,
    )

    states = ("SUFFICIENT", "PARTIALLY_SUFFICIENT", "INSUFFICIENT")
    cases = _full_cases()[:30]
    gold = E1A4GoldStateSet(
        blind_label_sha256="a" * 64,
        blind_review_contract_sha256="b" * 64,
        states=tuple(_state(case.question_id, states[index // 10]) for index, case in enumerate(cases)),
    )

    with pytest.raises(E1A4SelectionError, match="E1A4_SELECTION_INPUT_INVALID"):
        select_balanced_case_ids(cases=cases, gold_states=gold)
