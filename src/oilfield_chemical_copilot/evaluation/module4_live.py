"""In-memory actual-RAG evaluation for the Module 4 case contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from oilfield_chemical_copilot.evaluation.answers import AnswerEvaluationCase, evaluate_answer
from oilfield_chemical_copilot.evaluation.live_rag import (
    RecordingAnswerGenerator,
    capture_live_answer,
)
from oilfield_chemical_copilot.evaluation.module4_contract import Module4Case
from oilfield_chemical_copilot.evaluation.module4_reports import ModeSummary
from oilfield_chemical_copilot.evaluation.retrieval import (
    EvaluationResult,
    first_expected_rank,
    hit_rate_at_k,
    mean_reciprocal_rank,
)


class Module4RuntimeError(ValueError):
    """A sanitized Module 4 runtime error."""


@dataclass(frozen=True)
class ModeRuntime:
    service: object
    generator: RecordingAnswerGenerator


ServiceBuilder = Callable[[str], ModeRuntime]


def _answer_case(case: Module4Case) -> AnswerEvaluationCase:
    return AnswerEvaluationCase(
        case.case_id,
        case.question,
        case.expected_chunk_ids,
        not case.expect_abstention,
        case.expect_citations,
        case.expect_abstention,
    )


def _summarize_mode(cases: tuple[Module4Case, ...], runtime: ModeRuntime) -> ModeSummary:
    retrieval_results: list[EvaluationResult] = []
    citation_counts: Counter[str] = Counter()
    abstention_counts: Counter[str] = Counter()
    for case in cases:
        started_at = perf_counter()
        try:
            capture = capture_live_answer(_answer_case(case), runtime.service, runtime.generator)
        except Exception as error:
            raise Module4RuntimeError("RUNTIME_UNAVAILABLE") from error
        if capture.generation_outcome == "failed":
            raise Module4RuntimeError("RUNTIME_UNAVAILABLE")
        deterministic = evaluate_answer(
            _answer_case(case),
            cited_evidence_ids=capture.answer.cited_evidence_ids,
            abstained=capture.answer.abstained,
        )
        citation_counts[deterministic.citation_status] += 1
        abstention_counts[deterministic.abstention_status] += 1
        if not case.expect_abstention:
            expected_rank = first_expected_rank(
                capture.retrieved_evidence_ids,
                frozenset(case.expected_chunk_ids),
                5,
            )
            retrieval_results.append(
                EvaluationResult(
                    case.case_id,
                    case.topic,
                    capture.retrieved_evidence_ids,
                    expected_rank,
                    (perf_counter() - started_at) * 1000,
                )
            )
    return ModeSummary(
        len(retrieval_results),
        hit_rate_at_k(retrieval_results, 5),
        mean_reciprocal_rank(retrieval_results, 5),
        citation_counts["pass"],
        citation_counts["fail"],
        abstention_counts["pass"],
        abstention_counts["fail"],
    )


def evaluate_module4_modes(
    cases: tuple[Module4Case, ...], *, build_service: ServiceBuilder
) -> dict[str, ModeSummary]:
    try:
        return {
            mode: _summarize_mode(cases, build_service(mode))
            for mode in ("vector", "hybrid")
        }
    except Module4RuntimeError:
        raise
    except Exception as error:
        raise Module4RuntimeError("RUNTIME_UNAVAILABLE") from error
