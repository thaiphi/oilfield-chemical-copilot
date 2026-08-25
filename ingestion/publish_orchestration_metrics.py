from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import dlt
from dlt.destinations import postgres


_ALLOWED_KEYS = {
    "status",
    "source_files",
    "chunks",
    "expected_indexed_chunks",
    "actual_indexed_chunks",
    "embedding_model",
}
_COUNT_KEYS = {
    "source_files",
    "chunks",
    "expected_indexed_chunks",
    "actual_indexed_chunks",
}


def load_orchestration_metrics(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Invalid orchestration metrics report") from error
    if not isinstance(payload, dict) or set(payload) != _ALLOWED_KEYS:
        raise ValueError("Invalid orchestration metrics report")
    if payload["status"] != "success":
        raise ValueError("Invalid orchestration metrics report")
    if not isinstance(payload["embedding_model"], str) or not payload["embedding_model"].strip():
        raise ValueError("Invalid orchestration metrics report")
    if any(
        isinstance(payload[key], bool)
        or not isinstance(payload[key], int)
        or payload[key] < 0
        for key in _COUNT_KEYS
    ):
        raise ValueError("Invalid orchestration metrics report")
    return payload


def publish_orchestration_metrics(
    *,
    report_path: Path,
    database_url: str,
    pipeline_factory: Callable[..., object] = dlt.pipeline,
    destination_factory: Callable[..., object] = postgres,
) -> None:
    record = load_orchestration_metrics(report_path)
    pipeline = pipeline_factory(
        pipeline_name="oilfield_chemical_copilot_orchestration",
        destination=destination_factory(credentials=database_url),
        dataset_name="orchestration",
    )
    pipeline.run([record], table_name="ingestion_runs", write_disposition="append")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish aggregate orchestration metrics with dlt.")
    parser.add_argument("--input", required=True, type=Path, help="Aggregate validation report path.")
    parser.add_argument("--database-url", required=True, help="PostgreSQL URL.")
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    try:
        publish_orchestration_metrics(report_path=args.input, database_url=args.database_url)
    except Exception:
        parser.error("dlt publication failed")
    print("Published aggregate orchestration metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
