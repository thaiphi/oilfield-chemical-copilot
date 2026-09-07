"""Deterministic selection from independently derived E1a-4 gold states."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence

from oilfield_chemical_copilot.evaluation.e1a4_population import (
    E1A4PopulationCase,
    SOURCE_ROLES,
    TOPICS,
    has_exact_population_grid,
)

EvidenceState = Literal["SUFFICIENT", "PARTIALLY_SUFFICIENT", "INSUFFICIENT"]
STATE_ORDER: tuple[EvidenceState, ...] = (
    "SUFFICIENT",
    "PARTIALLY_SUFFICIENT",
    "INSUFFICIENT",
)
PER_STATE_QUOTA = 10


class E1A4SelectionError(ValueError):
    """Raised with a safe E1a-4 selection-contract error code."""


def _fail(code: str) -> None:
    raise E1A4SelectionError(code)


def _sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class E1A4GoldState:
    """One evaluator-only state derived from blind canonical-claim support."""

    question_id: str
    state: EvidenceState

    def __post_init__(self) -> None:
        if not isinstance(self.question_id, str) or not self.question_id.strip() or self.state not in STATE_ORDER:
            _fail("E1A4_GOLD_STATE_INVALID")


@dataclass(frozen=True)
class E1A4GoldStateSet:
    """Gold states bound to the separate blind review, never candidate output."""

    blind_label_sha256: str
    blind_review_contract_sha256: str
    states: tuple[E1A4GoldState, ...]

    def __post_init__(self) -> None:
        if (
            not _sha256(self.blind_label_sha256)
            or not _sha256(self.blind_review_contract_sha256)
            or not self.states
            or len({state.question_id for state in self.states}) != len(self.states)
        ):
            _fail("E1A4_GOLD_STATE_PROVENANCE_INVALID")


def select_balanced_case_ids(
    *,
    cases: Sequence[E1A4PopulationCase],
    gold_states: E1A4GoldStateSet,
) -> tuple[str, ...]:
    """Return the first lexicographic, 10/10/10, role/topic-complete tuple."""
    case_by_id = {case.question_id: case for case in cases}
    state_by_id = {item.question_id: item.state for item in gold_states.states}
    if (
        len(case_by_id) != len(cases)
        or not has_exact_population_grid(cases)
        or len(gold_states.states) != len(cases)
        or set(case_by_id) != set(state_by_id)
        or any(state not in STATE_ORDER for state in state_by_id.values())
    ):
        _fail("E1A4_SELECTION_INPUT_INVALID")

    buckets = {
        state: tuple(sorted(question_id for question_id, observed in state_by_id.items() if observed == state))
        for state in STATE_ORDER
    }
    if any(len(bucket) < PER_STATE_QUOTA for bucket in buckets.values()):
        _fail("E1A4_BALANCED_SELECTION_UNAVAILABLE")

    def can_cover(selected: tuple[str, ...], remaining: tuple[str, ...]) -> bool:
        available = selected + remaining
        return (
            {case_by_id[question_id].topic for question_id in available} == set(TOPICS)
            and {case_by_id[question_id].expected_source_role for question_id in available}
            == set(SOURCE_ROLES)
        )

    def choose_state(
        state_index: int,
        start: int,
        chosen: tuple[str, ...],
        selected: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        state = STATE_ORDER[state_index]
        bucket = buckets[state]
        required = PER_STATE_QUOTA - len(chosen)
        if required == 0:
            merged = selected + chosen
            if state_index == len(STATE_ORDER) - 1:
                return merged if can_cover(merged, ()) else None
            remaining = bucket[start:] + tuple(
                question_id
                for later_state in STATE_ORDER[state_index + 1 :]
                for question_id in buckets[later_state]
            )
            return choose_state(state_index + 1, 0, (), merged) if can_cover(merged, remaining) else None
        if len(bucket) - start < required:
            return None
        remaining = bucket[start:] + tuple(
            question_id
            for later_state in STATE_ORDER[state_index + 1 :]
            for question_id in buckets[later_state]
        )
        if not can_cover(selected + chosen, remaining):
            return None
        for index in range(start, len(bucket)):
            result = choose_state(state_index, index + 1, chosen + (bucket[index],), selected)
            if result is not None:
                return result
        return None

    selected = choose_state(0, 0, (), ())
    if selected is None:
        _fail("E1A4_BALANCED_SELECTION_UNAVAILABLE")
    return selected
