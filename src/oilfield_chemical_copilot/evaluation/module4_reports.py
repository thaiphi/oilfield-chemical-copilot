"""Aggregate-only report schema for Module 4 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Literal, Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODES = ("vector", "hybrid")
_SCOPES = ("public", "local")
_STATUSES = ("success", "unavailable", "failed")


class Module4ReportError(ValueError):
    """A sanitized Module 4 report error."""


@dataclass(frozen=True)
class ModeSummary:
    retrieval_case_count: int
    hit_rate_at_5: float
    mrr_at_5: float
    citation_pass: int
    citation_fail: int
    abstention_pass: int
    abstention_fail: int


def _fail(code: str) -> None:
    raise Module4ReportError(code)


def _validate_summary(summary: ModeSummary) -> None:
    counts = (
        summary.retrieval_case_count,
        summary.citation_pass,
        summary.citation_fail,
        summary.abstention_pass,
        summary.abstention_fail,
    )
    if any(type(count) is not int or count < 0 for count in counts):
        _fail("COUNT_INVALID")
    metrics = (summary.hit_rate_at_5, summary.mrr_at_5)
    if any(
        type(metric) not in {int, float} or not math.isfinite(float(metric)) or not 0 <= metric <= 1
        for metric in metrics
    ):
        _fail("METRIC_INVALID")
    if summary.citation_pass + summary.citation_fail != summary.abstention_pass + summary.abstention_fail:
        _fail("OUTCOME_COUNTS_INVALID")


def _summary_mapping(summary: ModeSummary) -> dict[str, object]:
    return {
        "retrieval_case_count": summary.retrieval_case_count,
        "retrieval": {
            "hit_rate_at_5": float(summary.hit_rate_at_5),
            "mrr_at_5": float(summary.mrr_at_5),
        },
        "deterministic": {
            "citations": {"pass": summary.citation_pass, "fail": summary.citation_fail},
            "abstention": {"pass": summary.abstention_pass, "fail": summary.abstention_fail},
        },
    }


def build_module4_report(
    *,
    scope: Literal["public", "local"],
    dataset_sha256: str,
    modes: Mapping[str, ModeSummary],
    status: Literal["success", "unavailable", "failed"] = "success",
) -> dict[str, object]:
    if scope not in _SCOPES:
        _fail("SCOPE_INVALID")
    if status not in _STATUSES:
        _fail("STATUS_INVALID")
    if not isinstance(dataset_sha256, str) or _SHA256.fullmatch(dataset_sha256) is None:
        _fail("DATASET_DIGEST_INVALID")
    if set(modes) != set(_MODES):
        _fail("MODE_SET_INVALID")
    for mode in _MODES:
        _validate_summary(modes[mode])
    return {
        "scope": scope,
        "dataset_sha256": dataset_sha256,
        "status": status,
        "modes": {mode: _summary_mapping(modes[mode]) for mode in _MODES},
    }


def _is_mode_mapping(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "retrieval_case_count",
        "retrieval",
        "deterministic",
    }:
        return False
    count = value["retrieval_case_count"]
    retrieval = value["retrieval"]
    deterministic = value["deterministic"]
    if type(count) is not int or count < 0:
        return False
    if not isinstance(retrieval, dict) or set(retrieval) != {"hit_rate_at_5", "mrr_at_5"}:
        return False
    if any(
        type(metric) not in {int, float} or not math.isfinite(float(metric)) or not 0 <= metric <= 1
        for metric in retrieval.values()
    ):
        return False
    if not isinstance(deterministic, dict) or set(deterministic) != {"citations", "abstention"}:
        return False
    outcomes: list[int] = []
    for key in ("citations", "abstention"):
        counts = deterministic[key]
        if not isinstance(counts, dict) or set(counts) != {"pass", "fail"}:
            return False
        if any(type(count) is not int or count < 0 for count in counts.values()):
            return False
        outcomes.append(counts["pass"] + counts["fail"])
    return outcomes[0] == outcomes[1]


def _is_safe_report(report: object) -> bool:
    if not isinstance(report, dict) or set(report) != {"scope", "dataset_sha256", "status", "modes"}:
        return False
    scope = report["scope"]
    dataset_sha256 = report["dataset_sha256"]
    status = report["status"]
    modes = report["modes"]
    return (
        scope in _SCOPES
        and isinstance(dataset_sha256, str)
        and _SHA256.fullmatch(dataset_sha256) is not None
        and status in _STATUSES
        and isinstance(modes, dict)
        and set(modes) == set(_MODES)
        and all(_is_mode_mapping(modes[mode]) for mode in _MODES)
    )


def write_module4_report(report: Mapping[str, object], destination: Path) -> None:
    payload = dict(report)
    if not _is_safe_report(payload):
        _fail("UNSAFE_REPORT")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
