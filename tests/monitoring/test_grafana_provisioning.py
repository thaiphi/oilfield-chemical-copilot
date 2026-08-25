from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_PATH = PROJECT_ROOT / "monitoring" / "grafana" / "dashboards" / "module5-monitoring.json"
DATASOURCE_PATH = PROJECT_ROOT / "monitoring" / "grafana" / "datasources" / "postgres.yml"
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"
ROLE_BOOTSTRAP_PATH = PROJECT_ROOT / "monitoring" / "bootstrap_grafana_role.py"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"


def test_dashboard_has_six_panels_and_queries_only_aggregate_tables() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    rendered = json.dumps(dashboard).lower()
    query_text = "\n".join(
        target["rawSql"].lower()
        for panel in dashboard["panels"]
        for target in panel["targets"]
    )

    assert dashboard["uid"] == "oilfield-module5-monitoring"
    assert len(dashboard["panels"]) == 6
    assert dashboard["time"] == {
        "from": "2026-01-15T07:00:00.000Z",
        "to": "2026-01-15T15:00:00.000Z",
    }
    assert [panel["gridPos"] for panel in dashboard["panels"]] == [
        {"h": 8, "w": 12, "x": 0, "y": 0},
        {"h": 8, "w": 12, "x": 12, "y": 0},
        {"h": 8, "w": 12, "x": 0, "y": 8},
        {"h": 8, "w": 12, "x": 12, "y": 8},
        {"h": 8, "w": 12, "x": 0, "y": 16},
        {"h": 8, "w": 12, "x": 12, "y": 16},
    ]
    assert "monitoring_request_hourly" in rendered
    assert "monitoring_feedback_hourly" in rendered
    outcome_mix = next(panel for panel in dashboard["panels"] if panel["id"] == 4)
    assert outcome_mix["options"]["reduceOptions"]["values"] is True
    assert outcome_mix["options"]["displayLabels"] == ["percent"]
    for forbidden in (
        "conversations",
        "retrieved_chunks",
        "tool_calls",
        "user_message",
        "assistant_message",
        "prompt",
        "answer",
        "jsonb",
    ):
        assert forbidden not in query_text


def test_every_panel_has_a_parameter_aware_description() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    required_terms = {
        1: ("request_count", "outcome", "UTC"),
        2: ("latency_total_ms", "latency_count", "milliseconds"),
        3: ("latency_minimum_ms", "latency_maximum_ms", "milliseconds"),
        4: ("outcome", "request_count", "percentage"),
        5: ("retrieval_mode", "request_count", "not_applicable"),
        6: ("helpful_rate", "feedback_count", "needs_work"),
    }

    for panel in dashboard["panels"]:
        description = panel.get("description", "")

        assert description
        assert all(term in description for term in required_terms[panel["id"]])


def test_datasource_uses_the_grafana_read_only_role() -> None:
    datasource = DATASOURCE_PATH.read_text(encoding="utf-8")

    assert "name: Oilfield Monitoring" in datasource
    assert "uid: oilfield-monitoring" in datasource
    assert "user: grafana_reader" in datasource
    assert "password: $GRAFANA_DB_PASSWORD" in datasource
    assert "url: postgres:5432" in datasource
    assert "database: $MONITORING_POSTGRES_DB" in datasource


def test_compose_keeps_demo_seed_explicit_and_grafana_provisioned() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "grafana-role-init:" in compose
    assert "monitoring-migrate:" in compose
    assert "monitoring-demo-seed:" in compose
    assert 'profiles: ["demo"]' in compose
    assert "./monitoring/grafana:/etc/grafana/provisioning:ro" in compose
    assert "MONITORING_PERSISTENCE_ENABLED: \"true\"" in compose
    assert "MONITORING_DATABASE_URL" in compose
    assert compose.count("image: oilfield-chemical-copilot:local") == 4


def test_grafana_is_localhost_bound_with_anonymous_viewer_access() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert 'GF_AUTH_ANONYMOUS_ENABLED: "true"' in compose
    assert "GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer" in compose
    assert '"127.0.0.1:${GRAFANA_PORT:-3000}:3000"' in compose


def test_role_bootstrap_grants_only_monitoring_table_access() -> None:
    bootstrap = ROLE_BOOTSTRAP_PATH.read_text(encoding="utf-8").lower()

    assert "grafana_reader" in bootstrap
    assert "monitoring_request_hourly" in bootstrap
    assert "monitoring_feedback_hourly" in bootstrap
    assert "conversations" not in bootstrap
    assert "retrieved_chunks" not in bootstrap
    assert "tool_calls" not in bootstrap


def test_dockerfile_copies_monitoring_scripts_for_compose_services() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "COPY monitoring ./monitoring" in dockerfile


def test_docker_build_context_includes_monitoring_scripts() -> None:
    dockerignore = DOCKERIGNORE_PATH.read_text(encoding="utf-8")

    assert "monitoring/" not in dockerignore.splitlines()
