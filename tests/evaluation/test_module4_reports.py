from __future__ import annotations

import json
from pathlib import Path

import pytest

from oilfield_chemical_copilot.evaluation.module4_reports import (
    ModeSummary,
    Module4ReportError,
    build_module4_report,
    write_module4_report,
)


def _modes() -> dict[str, ModeSummary]:
    return {
        "vector": ModeSummary(4, 0.75, 0.625, 3, 1, 4, 0),
        "hybrid": ModeSummary(4, 1.0, 1.0, 4, 0, 4, 0),
    }


def test_build_module4_report_contains_only_approved_aggregate_fields(tmp_path: Path) -> None:
    report = build_module4_report(scope="local", dataset_sha256="a" * 64, modes=_modes())
    destination = tmp_path / "aggregate.json"

    write_module4_report(report, destination)

    assert report["scope"] == "local"
    assert report["modes"]["hybrid"]["retrieval"] == {
        "hit_rate_at_5": 1.0,
        "mrr_at_5": 1.0,
    }
    assert json.loads(destination.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    "unsafe_report",
    [
        {"scope": "local", "nested": {"question": "private prompt"}},
        {"scope": "local", "dataset_sha256": "a" * 64, "answer": "private answer"},
        {"scope": "local", "dataset_sha256": "a" * 64, "path": "C:/private"},
    ],
)
def test_report_rejects_private_text_at_any_depth(
    tmp_path: Path, unsafe_report: dict[str, object]
) -> None:
    with pytest.raises(Module4ReportError, match="^UNSAFE_REPORT$"):
        write_module4_report(unsafe_report, tmp_path / "aggregate.json")


def test_build_module4_report_rejects_invalid_modes_and_count_mismatch() -> None:
    with pytest.raises(Module4ReportError, match="^MODE_SET_INVALID$"):
        build_module4_report(scope="public", dataset_sha256="a" * 64, modes={"vector": _modes()["vector"]})

    with pytest.raises(Module4ReportError, match="^OUTCOME_COUNTS_INVALID$"):
        build_module4_report(
            scope="public",
            dataset_sha256="a" * 64,
            modes={"vector": ModeSummary(1, 1.0, 1.0, 1, 0, 0, 0), "hybrid": _modes()["hybrid"]},
        )
