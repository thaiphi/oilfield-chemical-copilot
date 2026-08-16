from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from typing import Callable, Protocol

import psycopg

from oilfield_chemical_copilot.observability.aggregate_monitoring import (
    AggregateMonitor,
    AggregateMonitoringSnapshot,
    FeedbackValue,
    MonitoringOutcome,
    RetrievalMode,
)


def _hour_bucket(occurred_at: datetime) -> datetime:
    if not isinstance(occurred_at, datetime) or occurred_at.tzinfo is None:
        raise ValueError("occurred_at must be timezone-aware")
    if occurred_at.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    utc_time = occurred_at.astimezone(timezone.utc)
    return utc_time.replace(minute=0, second=0, microsecond=0)


def _finite_nonnegative_latency(latency_ms: Real) -> float:
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, Real):
        raise ValueError("latency_ms must be a finite nonnegative number")
    normalized = float(latency_ms)
    if normalized < 0 or not isfinite(normalized):
        raise ValueError("latency_ms must be a finite nonnegative number")
    return normalized


@dataclass(frozen=True)
class HourlyRequestMetric:
    bucket_start: datetime
    outcome: MonitoringOutcome
    retrieval_mode: RetrievalMode
    latency_ms: float

    @classmethod
    def from_values(
        cls,
        *,
        outcome: MonitoringOutcome,
        retrieval_mode: RetrievalMode,
        latency_ms: Real,
        occurred_at: datetime,
    ) -> HourlyRequestMetric:
        if not isinstance(outcome, MonitoringOutcome):
            raise ValueError("outcome must be a MonitoringOutcome")
        if not isinstance(retrieval_mode, RetrievalMode):
            raise ValueError("retrieval_mode must be a RetrievalMode")
        return cls(
            bucket_start=_hour_bucket(occurred_at),
            outcome=outcome,
            retrieval_mode=retrieval_mode,
            latency_ms=_finite_nonnegative_latency(latency_ms),
        )


@dataclass(frozen=True)
class HourlyFeedbackMetric:
    bucket_start: datetime
    value: FeedbackValue
    retrieval_mode: RetrievalMode

    @classmethod
    def from_values(
        cls,
        *,
        value: FeedbackValue,
        retrieval_mode: RetrievalMode,
        occurred_at: datetime,
    ) -> HourlyFeedbackMetric:
        if not isinstance(value, FeedbackValue):
            raise ValueError("value must be a FeedbackValue")
        if not isinstance(retrieval_mode, RetrievalMode):
            raise ValueError("retrieval_mode must be a RetrievalMode")
        return cls(bucket_start=_hour_bucket(occurred_at), value=value, retrieval_mode=retrieval_mode)


class MonitoringRepository(Protocol):
    def record_request(self, metric: HourlyRequestMetric) -> None: ...

    def record_feedback(self, metric: HourlyFeedbackMetric) -> None: ...


REQUEST_UPSERT_SQL = """
insert into monitoring_request_hourly (
    bucket_start,
    outcome,
    retrieval_mode,
    request_count,
    latency_count,
    latency_total_ms,
    latency_minimum_ms,
    latency_maximum_ms
)
values (%s, %s, %s, 1, 1, %s, %s, %s)
on conflict (bucket_start, outcome, retrieval_mode) do update set
    request_count = monitoring_request_hourly.request_count + excluded.request_count,
    latency_count = monitoring_request_hourly.latency_count + excluded.latency_count,
    latency_total_ms = monitoring_request_hourly.latency_total_ms + excluded.latency_total_ms,
    latency_minimum_ms = least(
        monitoring_request_hourly.latency_minimum_ms,
        excluded.latency_minimum_ms
    ),
    latency_maximum_ms = greatest(
        monitoring_request_hourly.latency_maximum_ms,
        excluded.latency_maximum_ms
)
"""

FEEDBACK_UPSERT_SQL = """
insert into monitoring_feedback_hourly (
    bucket_start,
    feedback_value,
    retrieval_mode,
    feedback_count
)
values (%s, %s, %s, 1)
on conflict (bucket_start, feedback_value, retrieval_mode) do update set
    feedback_count = monitoring_feedback_hourly.feedback_count + excluded.feedback_count
"""


class PostgresMonitoringRepository:
    """Persist validated aggregate monitoring records with parameterized upserts."""

    def __init__(
        self,
        database_url: str,
        *,
        connect: Callable[..., object] = psycopg.connect,
    ) -> None:
        self._database_url = database_url
        self._connect = connect

    def record_request(self, metric: HourlyRequestMetric) -> None:
        self._execute(
            REQUEST_UPSERT_SQL,
            (
                metric.bucket_start,
                metric.outcome.value,
                metric.retrieval_mode.value,
                metric.latency_ms,
                metric.latency_ms,
                metric.latency_ms,
            ),
        )

    def record_feedback(self, metric: HourlyFeedbackMetric) -> None:
        self._execute(
            FEEDBACK_UPSERT_SQL,
            (metric.bucket_start, metric.value.value, metric.retrieval_mode.value),
        )

    def _execute(self, query: str, parameters: tuple[object, ...]) -> None:
        with self._connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, parameters)
            connection.commit()


class SafeMonitoringRecorder:
    """Record only validated aggregate metrics without affecting user responses."""

    def __init__(
        self,
        aggregate_monitor: AggregateMonitor,
        repository: MonitoringRepository | None = None,
    ) -> None:
        self._aggregate_monitor = aggregate_monitor
        self._repository = repository

    def record_request(
        self,
        outcome: MonitoringOutcome,
        retrieval_mode: RetrievalMode,
        latency_ms: Real,
        occurred_at: datetime,
    ) -> None:
        metric = HourlyRequestMetric.from_values(
            outcome=outcome,
            retrieval_mode=retrieval_mode,
            latency_ms=latency_ms,
            occurred_at=occurred_at,
        )
        self._aggregate_monitor.record(metric.outcome, metric.latency_ms)
        if self._repository is None:
            return
        try:
            self._repository.record_request(metric)
        except (OSError, psycopg.Error):
            return

    def record_feedback(
        self,
        value: FeedbackValue,
        retrieval_mode: RetrievalMode,
        occurred_at: datetime,
    ) -> None:
        metric = HourlyFeedbackMetric.from_values(
            value=value,
            retrieval_mode=retrieval_mode,
            occurred_at=occurred_at,
        )
        if self._repository is None:
            return
        try:
            self._repository.record_feedback(metric)
        except (OSError, psycopg.Error):
            return

    def snapshot(self) -> AggregateMonitoringSnapshot:
        return self._aggregate_monitor.snapshot()
