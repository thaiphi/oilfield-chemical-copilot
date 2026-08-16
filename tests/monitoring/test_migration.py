from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = PROJECT_ROOT / "db" / "migrations" / "0003_module_5_monitoring.sql"


def test_module_5_migration_contains_only_aggregate_monitoring_tables() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8")
    lowered = sql.lower()

    assert "monitoring_request_hourly" in lowered
    assert "monitoring_feedback_hourly" in lowered
    assert "jsonb" not in lowered
    assert "user_message" not in lowered
    assert "assistant_message" not in lowered
    assert "session_id" not in lowered
    assert "tool_calls" not in lowered
