from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping


_STATUS = frozenset(("pass", "fail", "blocked"))
_FORBIDDEN_KEY_PARTS = (
    "question", "label", "source", "path", "url", "credential", "secret",
    "error", "exception", "trace",
)


class ReportSchemaError(ValueError):
    def __init__(self) -> None:
        super().__init__("REPORT_SCHEMA_VIOLATION")


class ReportWriteError(ValueError):
    def __init__(self) -> None:
        super().__init__("REPORT_WRITE_FAILURE")


@dataclass(frozen=True)
class AggregateReport:
    task: int
    status: Literal["pass", "fail", "blocked"]
    counts: Mapping[str, int]
    gates: Mapping[str, bool]


def _validate(report: AggregateReport) -> None:
    if report.status not in _STATUS or not isinstance(report.task, int):
        raise ReportSchemaError()
    for mapping, expected_type in ((report.counts, int), (report.gates, bool)):
        for key, value in mapping.items():
            if not isinstance(key, str) or any(part in key.lower() for part in _FORBIDDEN_KEY_PARTS):
                raise ReportSchemaError()
            if type(value) is not expected_type:
                raise ReportSchemaError()
            if expected_type is int and value < 0:
                raise ReportSchemaError()


def write_aggregate_report(destination: Path, report: AggregateReport) -> None:
    _validate(report)
    try:
        payload = {
            "task": report.task,
            "status": report.status,
            "counts": dict(report.counts),
            "gates": dict(report.gates),
        }
        destination.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError, OverflowError):
        raise ReportWriteError() from None
