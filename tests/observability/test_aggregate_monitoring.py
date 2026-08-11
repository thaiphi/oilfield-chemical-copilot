from __future__ import annotations

from math import inf, nan

import pytest

from oilfield_chemical_copilot.observability.aggregate_monitoring import (
    AggregateMonitor,
    MonitoringOutcome,
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
