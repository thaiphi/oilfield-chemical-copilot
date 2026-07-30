from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import psycopg
from pgvector.psycopg import register_vector

MIGRATIONS_DIR = PROJECT_ROOT / "db" / "migrations"

CREATE_LEDGER_SQL = """
create table if not exists schema_migrations (
    filename text primary key,
    applied_at timestamptz not null default now()
)
"""


def apply_migrations(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    migration_paths = sorted(migrations_dir.glob("*.sql"))
    if not migration_paths:
        raise ValueError(f"No SQL migrations found in {migrations_dir}")
    applied: list[str] = []
    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(CREATE_LEDGER_SQL)
            cursor.execute("select filename from schema_migrations")
            completed = {row[0] for row in cursor.fetchall()}
            for migration_path in migration_paths:
                if migration_path.name in completed:
                    continue
                cursor.execute(migration_path.read_text(encoding="utf-8-sig"))
                cursor.execute(
                    "insert into schema_migrations (filename) values (%s)",
                    (migration_path.name,),
                )
                applied.append(migration_path.name)
        connection.commit()
    return applied


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply repository SQL migrations to PostgreSQL.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="PostgreSQL URL.")
    parser.add_argument("--migrations-dir", type=Path, default=MIGRATIONS_DIR)
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    if not args.database_url:
        parser.error("DATABASE_URL is required")
    try:
        applied = apply_migrations(args.database_url, args.migrations_dir)
    except (OSError, ValueError, psycopg.Error) as error:
        parser.error(str(error))
    if applied:
        print(f"Applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        print("No pending migrations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())