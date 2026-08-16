from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oilfield_chemical_copilot.observability.aggregate_monitoring import (  # noqa: E402
    FeedbackValue,
    MonitoringOutcome,
    RetrievalMode,
)
from oilfield_chemical_copilot.observability.persistence import (  # noqa: E402
    HourlyFeedbackMetric,
    HourlyRequestMetric,
    MonitoringRepository,
    PostgresMonitoringRepository,
)


DEMO_START = datetime(2026, 1, 15, 8, tzinfo=timezone.utc)


def build_demo_metrics() -> tuple[tuple[HourlyRequestMetric, ...], tuple[HourlyFeedbackMetric, ...]]:
    requests = (
        HourlyRequestMetric.from_values(
            outcome=MonitoringOutcome.RAG_ANSWERED,
            retrieval_mode=RetrievalMode.VECTOR,
            latency_ms=420.0,
            occurred_at=DEMO_START,
        ),
        HourlyRequestMetric.from_values(
            outcome=MonitoringOutcome.RAG_ANSWERED,
            retrieval_mode=RetrievalMode.HYBRID,
            latency_ms=560.0,
            occurred_at=DEMO_START + timedelta(hours=1),
        ),
        HourlyRequestMetric.from_values(
            outcome=MonitoringOutcome.RAG_WEAK_EVIDENCE,
            retrieval_mode=RetrievalMode.VECTOR,
            latency_ms=300.0,
            occurred_at=DEMO_START + timedelta(hours=2),
        ),
        HourlyRequestMetric.from_values(
            outcome=MonitoringOutcome.SCOPE_ABSTAINED,
            retrieval_mode=RetrievalMode.NOT_APPLICABLE,
            latency_ms=25.0,
            occurred_at=DEMO_START + timedelta(hours=3),
        ),
        HourlyRequestMetric.from_values(
            outcome=MonitoringOutcome.TOOL_CALCULATED,
            retrieval_mode=RetrievalMode.NOT_APPLICABLE,
            latency_ms=18.0,
            occurred_at=DEMO_START + timedelta(hours=4),
        ),
        HourlyRequestMetric.from_values(
            outcome=MonitoringOutcome.RAG_CONFIGURATION_ERROR,
            retrieval_mode=RetrievalMode.HYBRID,
            latency_ms=120.0,
            occurred_at=DEMO_START + timedelta(hours=5),
        ),
    )
    feedback = (
        HourlyFeedbackMetric.from_values(
            value=FeedbackValue.HELPFUL,
            retrieval_mode=RetrievalMode.VECTOR,
            occurred_at=DEMO_START + timedelta(hours=1),
        ),
        HourlyFeedbackMetric.from_values(
            value=FeedbackValue.NEEDS_WORK,
            retrieval_mode=RetrievalMode.HYBRID,
            occurred_at=DEMO_START + timedelta(hours=2),
        ),
    )
    return requests, feedback


def seed_demo_metrics(repository: MonitoringRepository) -> None:
    requests, feedback = build_demo_metrics()
    for metric in requests:
        repository.record_request(metric)
    for metric in feedback:
        repository.record_feedback(metric)


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed fixed synthetic aggregate monitoring metrics.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        parser.error("DATABASE_URL is required")
    seed_demo_metrics(PostgresMonitoringRepository(args.database_url))
    print("Synthetic monitoring demo seed completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
