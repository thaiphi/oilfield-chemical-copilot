from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from oilfield_chemical_copilot.observability.persistence import (
    HourlyFeedbackMetric,
    HourlyRequestMetric,
)
from monitoring.seed_demo_metrics import build_demo_metrics, seed_demo_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_SCRIPT = PROJECT_ROOT / "monitoring" / "seed_demo_metrics.py"


class RecordingRepository:
    def __init__(self) -> None:
        self.requests: list[HourlyRequestMetric] = []
        self.feedback: list[HourlyFeedbackMetric] = []

    def record_request(self, metric: HourlyRequestMetric) -> None:
        self.requests.append(metric)

    def record_feedback(self, metric: HourlyFeedbackMetric) -> None:
        self.feedback.append(metric)


def test_demo_metrics_are_fixed_aggregate_only_values() -> None:
    requests, feedback = build_demo_metrics()

    assert requests
    assert feedback
    assert all(metric.bucket_start.tzinfo is not None for metric in requests)
    assert {metric.retrieval_mode.value for metric in requests} <= {
        "vector",
        "hybrid",
        "not_applicable",
    }
    assert not any(hasattr(metric, "prompt") or hasattr(metric, "answer") for metric in requests)
    assert not any(hasattr(metric, "comment") or hasattr(metric, "session_id") for metric in feedback)


def test_seed_demo_metrics_uses_the_safe_repository_contract() -> None:
    repository = RecordingRepository()

    seed_demo_metrics(repository)

    assert len(repository.requests) == len(build_demo_metrics()[0])
    assert len(repository.feedback) == len(build_demo_metrics()[1])


def test_seed_script_runs_without_pythonpath_configuration() -> None:
    result = subprocess.run(
        [sys.executable, str(SEED_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        env=os.environ | {"PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Seed fixed synthetic aggregate monitoring metrics." in result.stdout
