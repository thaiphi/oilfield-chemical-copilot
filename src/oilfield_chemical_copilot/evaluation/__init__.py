"""Public retrieval evaluation primitives."""

from .retrieval import (
    EvaluationCase,
    EvaluationResult,
    first_expected_rank,
    hit_rate_at_k,
    load_evaluation_cases,
    mean_reciprocal_rank,
)

__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "first_expected_rank",
    "hit_rate_at_k",
    "load_evaluation_cases",
    "mean_reciprocal_rank",
]
