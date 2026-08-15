from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

import ingestion.publish_orchestration_metrics as metrics_module
from ingestion.publish_orchestration_metrics import (
    load_orchestration_metrics,
    publish_orchestration_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_report(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "orchestration_run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_report() -> dict[str, object]:
    return {
        "actual_indexed_chunks": 3,
        "chunks": 3,
        "embedding_model": "granite-embedding:latest",
        "expected_indexed_chunks": 3,
        "source_files": 2,
        "status": "success",
    }


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, str]] = []

    def run(self, data: object, *, table_name: str, write_disposition: str) -> None:
        self.calls.append((data, table_name, write_disposition))


def test_publish_orchestration_metrics_loads_one_allowlisted_record(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path, _valid_report())
    pipeline = _FakePipeline()
    captured: dict[str, object] = {}

    def pipeline_factory(**kwargs: object) -> _FakePipeline:
        captured["pipeline"] = kwargs
        return pipeline

    def destination_factory(*, credentials: str) -> str:
        captured["credentials"] = credentials
        return "postgres-destination"

    publish_orchestration_metrics(
        report_path=report_path,
        database_url="postgresql://test",
        pipeline_factory=pipeline_factory,
        destination_factory=destination_factory,
    )

    assert captured["credentials"] == "postgresql://test"
    assert captured["pipeline"] == {
        "pipeline_name": "oilfield_chemical_copilot_orchestration",
        "destination": "postgres-destination",
        "dataset_name": "orchestration",
    }
    assert pipeline.calls == [([_valid_report()], "ingestion_runs", "append")]


def test_load_orchestration_metrics_rejects_unallowlisted_field_before_publication(
    tmp_path: Path,
) -> None:
    payload = _valid_report() | {"source_file": "not-allowed"}

    with pytest.raises(ValueError, match="Invalid orchestration metrics report") as error:
        load_orchestration_metrics(_write_report(tmp_path, payload))

    assert "not-allowed" not in str(error.value)


def test_main_hides_dlt_exception_details(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def raise_dlt_error(**_kwargs: object) -> None:
        raise RuntimeError("private dlt destination detail")

    monkeypatch.setattr(metrics_module, "publish_orchestration_metrics", raise_dlt_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_orchestration_metrics.py",
            "--input",
            "orchestration_run.json",
            "--database-url",
            "postgresql://test",
        ],
    )

    with pytest.raises(SystemExit) as error:
        metrics_module.main()

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "dlt publication failed" in captured.err
    assert "private dlt destination detail" not in captured.err


def test_project_declares_dlt_postgres_destination_extra() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert any(dependency.startswith("dlt[postgres]") for dependency in pyproject["project"]["dependencies"])
