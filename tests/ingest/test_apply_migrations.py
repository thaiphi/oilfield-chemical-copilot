from __future__ import annotations

from pathlib import Path

from ingestion import apply_migrations as migrations


class FakeCursor:
    def __init__(self) -> None:
        self.executed = []
        self._selecting_completed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None) -> None:
        self.executed.append((str(query), params))
        self._selecting_completed = "select filename from schema_migrations" in str(query)

    def fetchall(self):
        if self._selecting_completed:
            return [("0001_existing.sql",)]
        return []


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True


def test_apply_migrations_skips_completed_files(monkeypatch, tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "0001_existing.sql").write_text("select 1", encoding="utf-8")
    (migrations_dir / "0002_pending.sql").write_text("select 2", encoding="utf-8")
    connection = FakeConnection()

    monkeypatch.setattr(migrations.psycopg, "connect", lambda database_url: connection)
    monkeypatch.setattr(migrations, "register_vector", lambda connection: None)

    applied = migrations.apply_migrations("postgresql://example", migrations_dir)

    assert applied == ["0002_pending.sql"]
    assert connection.committed is True
    executed_sql = [query for query, _ in connection.cursor_instance.executed]
    assert "select 1" not in executed_sql
    assert "select 2" in executed_sql