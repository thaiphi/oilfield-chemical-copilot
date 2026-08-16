from __future__ import annotations

import argparse
import os
from typing import Callable

import psycopg
from psycopg import sql


ROLE_NAME = "grafana_reader"


def configure_grafana_role(
    database_url: str,
    password: str,
    *,
    connect: Callable[..., object] = psycopg.connect,
) -> None:
    if not database_url or not password:
        raise ValueError("database URL and Grafana password are required")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                do $role$
                begin
                    if not exists (select 1 from pg_roles where rolname = 'grafana_reader') then
                        create role grafana_reader login;
                    end if;
                end
                $role$;
                """
            )
            cursor.execute(
                sql.SQL("alter role {} with login password {}").format(
                    sql.Identifier(ROLE_NAME),
                    sql.Literal(password),
                )
            )
            cursor.execute("revoke all privileges on all tables in schema public from grafana_reader")
            cursor.execute("grant usage on schema public to grafana_reader")
            cursor.execute("grant select on table monitoring_request_hourly to grafana_reader")
            cursor.execute("grant select on table monitoring_feedback_hourly to grafana_reader")
        connection.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure the Grafana read-only database role.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--password", default=os.getenv("GRAFANA_DB_PASSWORD"))
    args = parser.parse_args()
    try:
        configure_grafana_role(args.database_url, args.password)
    except (OSError, ValueError, psycopg.Error):
        print("Grafana role bootstrap failed.")
        return 1
    print("Grafana role bootstrap completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
