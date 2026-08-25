from pathlib import Path
import json

import pytest

from production_hardening.aggregate_report import (
    AggregateReport,
    ReportSchemaError,
    write_aggregate_report,
)


def test_writes_a_valid_aggregate_only_36_count_report(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    report = AggregateReport(
        task=1,
        status="pass",
        counts={"tests_run": 36, "checks_failed": 0},
        gates={"privacy_guard_passed": True},
    )

    write_aggregate_report(destination, report)

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "counts": {"checks_failed": 0, "tests_run": 36},
        "gates": {"privacy_guard_passed": True},
        "status": "pass",
        "task": 1,
    }


@pytest.mark.parametrize(
    "unsafe_key",
    [
        "question",
        "expected_label",
        "source_content",
        "file_path",
        "url",
        "credential",
        "raw_error",
    ],
)
def test_rejects_private_report_keys_with_a_sanitized_code(unsafe_key: str, tmp_path: Path) -> None:
    report = AggregateReport(
        task=1,
        status="pass",
        counts={unsafe_key: 1},
        gates={"privacy_guard_passed": True},
    )

    with pytest.raises(ReportSchemaError, match="^REPORT_SCHEMA_VIOLATION$"):
        write_aggregate_report(tmp_path / "report.json", report)


def test_rejects_non_enum_status_and_negative_counts(tmp_path: Path) -> None:
    invalid_status = AggregateReport(1, "unknown", {"tests_run": 1}, {})
    negative_count = AggregateReport(1, "pass", {"tests_run": -1}, {})

    for report in (invalid_status, negative_count):
        with pytest.raises(ReportSchemaError, match="^REPORT_SCHEMA_VIOLATION$"):
            write_aggregate_report(tmp_path / "report.json", report)


def test_sanitizes_destination_write_failures(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("x", encoding="utf-8")
    destination = blocked_parent / "report.json"
    report = AggregateReport(1, "pass", {"tests_run": 1}, {"privacy_guard_passed": True})

    with pytest.raises(ValueError, match="^REPORT_WRITE_FAILURE$") as error:
        write_aggregate_report(destination, report)

    assert str(destination) not in str(error.value)
