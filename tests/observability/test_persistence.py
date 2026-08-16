from __future__ import annotations

from datetime import datetime, timezone

from oilfield_chemical_copilot.observability.aggregate_monitoring import (
    AggregateMonitor,
    FeedbackValue,
    MonitoringOutcome,
    RetrievalMode,
)
from oilfield_chemical_copilot.observability.persistence import (
    HourlyFeedbackMetric,
    HourlyRequestMetric,
    PostgresMonitoringRepository,
    SafeMonitoringRecorder,
)


NOW = datetime(2026, 8, 15, 10, 47, tzinfo=timezone.utc)


class FailingRepository:
    def record_request(self, metric: HourlyRequestMetric) -> None:
        raise OSError("database unavailable")

    def record_feedback(self, metric: HourlyFeedbackMetric) -> None:
        raise OSError("database unavailable")


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...]) -> None:
        self.executed.append((query, params))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
        self.committed = False

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True


def test_recorder_keeps_process_aggregate_when_repository_is_unavailable() -> None:
    recorder = SafeMonitoringRecorder(AggregateMonitor(), FailingRepository())

    recorder.record_request(
        MonitoringOutcome.RAG_ANSWERED,
        RetrievalMode.VECTOR,
        9.0,
        NOW,
    )

    snapshot = recorder.snapshot()
    assert snapshot.total_requests == 1
    assert snapshot.outcome_counts[MonitoringOutcome.RAG_ANSWERED.value] == 1


def test_feedback_metric_accepts_only_closed_values_and_utc_hours() -> None:
    metric = HourlyFeedbackMetric.from_values(
        value=FeedbackValue.HELPFUL,
        retrieval_mode=RetrievalMode.HYBRID,
        occurred_at=NOW,
    )

    assert metric.bucket_start == datetime(2026, 8, 15, 10, tzinfo=timezone.utc)
    assert not hasattr(metric, "comment")
    assert not hasattr(metric, "session_id")


def test_postgres_repository_upserts_request_aggregate_with_closed_parameters() -> None:
    connection = FakeConnection()
    repository = PostgresMonitoringRepository("postgresql://test", connect=lambda _: connection)
    metric = HourlyRequestMetric.from_values(
        outcome=MonitoringOutcome.RAG_ANSWERED,
        retrieval_mode=RetrievalMode.HYBRID,
        latency_ms=9.0,
        occurred_at=NOW,
    )

    repository.record_request(metric)

    query, parameters = connection.cursor_instance.executed[0]
    assert "insert into monitoring_request_hourly" in query.lower()
    assert "on conflict" in query.lower()
    assert parameters == (
        metric.bucket_start,
        "rag_answered",
        "hybrid",
        9.0,
        9.0,
        9.0,
    )
    assert connection.committed is True
    assert "conversations" not in query.lower()
    assert "tool_calls" not in query.lower()


def test_postgres_repository_upserts_feedback_aggregate_with_closed_parameters() -> None:
    connection = FakeConnection()
    repository = PostgresMonitoringRepository("postgresql://test", connect=lambda _: connection)
    metric = HourlyFeedbackMetric.from_values(
        value=FeedbackValue.HELPFUL,
        retrieval_mode=RetrievalMode.VECTOR,
        occurred_at=NOW,
    )

    repository.record_feedback(metric)

    query, parameters = connection.cursor_instance.executed[0]
    assert "insert into monitoring_feedback_hourly" in query.lower()
    assert "on conflict" in query.lower()
    assert parameters == (metric.bucket_start, "helpful", "vector")
    assert connection.committed is True
