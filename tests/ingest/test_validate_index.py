from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg
import pytest

import ingestion.validate_index as validate_index_module
from ingestion.validate_index import validate_indexed_chunk_count


def _write_chunks(tmp_path: Path, ids: tuple[str, ...]) -> Path:
    rows = [
        {
            "text": "Synthetic public chunk",
            "metadata": {
                "chunk_id": chunk_id,
                "source_file": f"docs/source-{index}.md",
                "source_path": f"C:/synthetic/source-{index}.md",
                "topic": "scale",
                "parser_type": "text",
                "page_or_sheet": "document",
                "chunk_index": index,
                "extra": {},
            },
        }
        for index, chunk_id in enumerate(ids)
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


class _FakeCursor:
    def __init__(self, returned_count: int) -> None:
        self.returned_count = returned_count
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, parameters: tuple[object, ...]) -> None:
        self.calls.append((query, parameters))

    def fetchone(self) -> tuple[int]:
        return (self.returned_count,)


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return self._cursor


def _install_counting_connection(
    monkeypatch: pytest.MonkeyPatch, *, returned_count: int
) -> _FakeCursor:
    cursor = _FakeCursor(returned_count)
    monkeypatch.setattr(
        validate_index_module.psycopg,
        "connect",
        lambda _database_url: _FakeConnection(cursor),
    )
    return cursor


def test_validate_indexed_chunk_count_uses_unique_input_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks_path = _write_chunks(tmp_path, ids=("scale-1", "scale-1", "scale-2"))
    cursor = _install_counting_connection(monkeypatch, returned_count=2)

    assert validate_indexed_chunk_count(
        chunks_path=chunks_path,
        database_url="postgresql://test",
        embedding_model="granite-embedding:latest",
    ) == (2, 2)
    assert cursor.calls[0][1] == ("granite-embedding:latest", ["scale-1", "scale-2"])


def test_validate_indexed_chunk_count_rejects_a_mismatch_without_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks_path = _write_chunks(tmp_path, ids=("nonpublic-like-id", "scale-2"))
    _install_counting_connection(monkeypatch, returned_count=1)

    with pytest.raises(ValueError, match="Indexed chunk count mismatch") as error:
        validate_indexed_chunk_count(
            chunks_path=chunks_path,
            database_url="postgresql://test",
            embedding_model="granite-embedding:latest",
        )

    assert "nonpublic-like-id" not in str(error.value)


def test_main_writes_an_allowlisted_aggregate_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "orchestration_run.json"
    monkeypatch.setattr(validate_index_module, "validate_indexed_chunk_count", lambda **_kwargs: (2, 2))
    monkeypatch.setattr(validate_index_module, "_source_file_count", lambda _path: 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_index.py",
            "--input",
            "chunks.jsonl",
            "--database-url",
            "postgresql://test",
            "--embedding-model",
            "granite-embedding:latest",
            "--report",
            str(report_path),
        ],
    )

    assert validate_index_module.main() == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "actual_indexed_chunks": 2,
        "chunks": 2,
        "embedding_model": "granite-embedding:latest",
        "expected_indexed_chunks": 2,
        "source_files": 1,
        "status": "success",
    }


def test_main_hides_database_error_details(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def raise_database_error(**_kwargs: object) -> tuple[int, int]:
        raise psycopg.OperationalError("private connection detail")

    monkeypatch.setattr(validate_index_module, "validate_indexed_chunk_count", raise_database_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_index.py",
            "--input",
            "chunks.jsonl",
            "--database-url",
            "postgresql://test",
            "--embedding-model",
            "granite-embedding:latest",
        ],
    )

    with pytest.raises(SystemExit) as error:
        validate_index_module.main()

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "PGVector count validation failed" in captured.err
    assert "private connection detail" not in captured.err
