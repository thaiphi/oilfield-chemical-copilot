from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

from ingestion.index_chunks import load_chunks_jsonl


def _input_chunk_ids(chunks_path: Path) -> set[str]:
    return {chunk.metadata.chunk_id for chunk in load_chunks_jsonl(chunks_path)}


def _source_file_count(chunks_path: Path) -> int:
    return len({chunk.metadata.source_file for chunk in load_chunks_jsonl(chunks_path)})


def validate_indexed_chunk_count(
    *, chunks_path: Path, database_url: str, embedding_model: str
) -> tuple[int, int]:
    chunk_ids = _input_chunk_ids(chunks_path)
    if not chunk_ids:
        return (0, 0)
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "select count(*) from chunks where embedding_model = %s and chunk_id = any(%s)",
                (embedding_model, sorted(chunk_ids)),
            )
            actual = int(cursor.fetchone()[0])
    expected = len(chunk_ids)
    if actual != expected:
        raise ValueError(f"Indexed chunk count mismatch: expected {expected}, got {actual}")
    return (expected, actual)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate aggregate PGVector index counts.")
    parser.add_argument("--input", required=True, type=Path, help="Path to chunks JSONL.")
    parser.add_argument("--database-url", required=True, help="PostgreSQL URL.")
    parser.add_argument("--embedding-model", required=True, help="Embedding model label.")
    parser.add_argument("--report", type=Path, help="Optional aggregate JSON report path.")
    return parser


def _write_report(
    *, report_path: Path, source_files: int, expected: int, actual: int, embedding_model: str
) -> None:
    report = {
        "actual_indexed_chunks": actual,
        "chunks": expected,
        "embedding_model": embedding_model,
        "expected_indexed_chunks": expected,
        "source_files": source_files,
        "status": "success",
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    try:
        expected, actual = validate_indexed_chunk_count(
            chunks_path=args.input,
            database_url=args.database_url,
            embedding_model=args.embedding_model,
        )
        source_files = _source_file_count(args.input)
        if args.report is not None:
            _write_report(
                report_path=args.report,
                source_files=source_files,
                expected=expected,
                actual=actual,
                embedding_model=args.embedding_model,
            )
    except psycopg.Error:
        parser.error("PGVector count validation failed")
    except (OSError, ValueError):
        parser.error("Index validation failed")
    print(f"Validated expected={expected} actual={actual}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
