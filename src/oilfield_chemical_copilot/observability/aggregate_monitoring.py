from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from threading import Lock


class MonitoringOutcome(str, Enum):
    RAG_ANSWERED = "rag_answered"
    RAG_WEAK_EVIDENCE = "rag_weak_evidence"
    SCOPE_ABSTAINED = "scope_abstained"
    TOOL_CALCULATED = "tool_calculated"
    TOOL_INPUT_INVALID = "tool_input_invalid"
    RAG_CONFIGURATION_ERROR = "rag_configuration_error"


@dataclass(frozen=True)
class LatencySummary:
    count: int
    minimum_ms: float | None
    average_ms: float | None
    maximum_ms: float | None


@dataclass(frozen=True)
class AggregateMonitoringSnapshot:
    total_requests: int
    outcome_counts: dict[str, int]
    latency: LatencySummary


class AggregateMonitor:
    """Process-local aggregate monitoring that never retains request content."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._outcome_counts = {outcome.value: 0 for outcome in MonitoringOutcome}
        self._total_requests = 0
        self._latency_total_ms = 0.0
        self._latency_minimum_ms: float | None = None
        self._latency_maximum_ms: float | None = None

    def record(self, outcome: MonitoringOutcome, latency_ms: Real) -> None:
        if not isinstance(outcome, MonitoringOutcome):
            raise ValueError("outcome must be a MonitoringOutcome")
        if isinstance(latency_ms, bool) or not isinstance(latency_ms, Real):
            raise ValueError("latency_ms must be a finite nonnegative number")
        normalized_latency = float(latency_ms)
        if normalized_latency < 0 or not isfinite(normalized_latency):
            raise ValueError("latency_ms must be a finite nonnegative number")

        with self._lock:
            self._outcome_counts[outcome.value] += 1
            self._total_requests += 1
            self._latency_total_ms += normalized_latency
            self._latency_minimum_ms = (
                normalized_latency
                if self._latency_minimum_ms is None
                else min(self._latency_minimum_ms, normalized_latency)
            )
            self._latency_maximum_ms = (
                normalized_latency
                if self._latency_maximum_ms is None
                else max(self._latency_maximum_ms, normalized_latency)
            )

    def snapshot(self) -> AggregateMonitoringSnapshot:
        with self._lock:
            return AggregateMonitoringSnapshot(
                total_requests=self._total_requests,
                outcome_counts=dict(self._outcome_counts),
                latency=LatencySummary(
                    count=self._total_requests,
                    minimum_ms=self._latency_minimum_ms,
                    average_ms=(
                        self._latency_total_ms / self._total_requests
                        if self._total_requests
                        else None
                    ),
                    maximum_ms=self._latency_maximum_ms,
                ),
            )
