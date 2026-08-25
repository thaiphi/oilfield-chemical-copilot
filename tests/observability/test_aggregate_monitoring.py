from __future__ import annotations

from datetime import datetime, timezone
from math import inf, nan

import pytest

from oilfield_chemical_copilot.observability.aggregate_monitoring import (
    AggregateMonitor,
    FeedbackValue,
    MonitoringOutcome,
    RetrievalMode,
)


def test_monitor_aggregates_only_closed_outcomes_and_latency() -> None:
    monitor = AggregateMonitor()

    monitor.record(MonitoringOutcome.RAG_ANSWERED, 12.5)
    monitor.record(MonitoringOutcome.TOOL_CALCULATED, 7.5)
    monitor.record(MonitoringOutcome.RAG_ANSWERED, 20.0)

    snapshot = monitor.snapshot()

    assert snapshot.total_requests == 3
    assert snapshot.outcome_counts == {
        "rag_answered": 2,
        "rag_weak_evidence": 0,
        "scope_abstained": 0,
        "tool_calculated": 1,
        "tool_input_invalid": 0,
        "rag_configuration_error": 0,
    }
    assert snapshot.latency.count == 3
    assert snapshot.latency.minimum_ms == 7.5
    assert snapshot.latency.average_ms == pytest.approx(40 / 3)
    assert snapshot.latency.maximum_ms == 20.0


@pytest.mark.parametrize("latency_ms", [-1, nan, inf, True, "12"])
def test_monitor_rejects_invalid_latency(latency_ms: object) -> None:
    monitor = AggregateMonitor()

    with pytest.raises(ValueError, match="latency_ms"):
        monitor.record(MonitoringOutcome.RAG_ANSWERED, latency_ms)


def test_monitor_rejects_arbitrary_payloads_and_keeps_no_request_content() -> None:
    monitor = AggregateMonitor()

    with pytest.raises(ValueError, match="MonitoringOutcome"):
        monitor.record("prompt=private water analysis", 10.0)  # type: ignore[arg-type]

    snapshot = monitor.snapshot()
    assert not hasattr(snapshot, "prompt")
    assert not hasattr(snapshot, "answer")
    assert not hasattr(snapshot, "payload")
    assert not hasattr(monitor, "events")


def test_monitoring_value_types_are_closed_and_do_not_accept_payloads() -> None:
    assert RetrievalMode.HYBRID.value == "hybrid"
    assert RetrievalMode.NOT_APPLICABLE.value == "not_applicable"
    assert FeedbackValue.HELPFUL.value == "helpful"
    assert FeedbackValue.NEEDS_WORK.value == "needs_work"

    with pytest.raises(ValueError):
        RetrievalMode("private-source")
    with pytest.raises(ValueError):
        FeedbackValue("free-text-comment")


def test_utc_hour_boundary_is_represented_without_request_content() -> None:
    from oilfield_chemical_copilot.observability.persistence import HourlyRequestMetric

    metric = HourlyRequestMetric.from_values(
        outcome=MonitoringOutcome.RAG_ANSWERED,
        retrieval_mode=RetrievalMode.HYBRID,
        latency_ms=12.5,
        occurred_at=datetime(2026, 8, 15, 10, 47, 30, tzinfo=timezone.utc),
    )

    assert metric.bucket_start == datetime(2026, 8, 15, 10, tzinfo=timezone.utc)
    assert metric.outcome is MonitoringOutcome.RAG_ANSWERED
    assert metric.retrieval_mode is RetrievalMode.HYBRID
    assert metric.latency_ms == 12.5
    assert not hasattr(metric, "prompt")
    assert not hasattr(metric, "answer")
    assert not hasattr(metric, "payload")
