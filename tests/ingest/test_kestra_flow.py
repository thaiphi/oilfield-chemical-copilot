from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLOW_PATH = PROJECT_ROOT / "flows" / "kestra" / "ingest.yml"
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"


def test_kestra_flow_uses_five_ordered_public_sample_tasks() -> None:
    flow = FLOW_PATH.read_text(encoding="utf-8")

    assert "id: inventory" in flow
    assert "id: parse_chunk" in flow
    assert "id: embed_load" in flow
    assert "id: validate_counts" in flow
    assert "id: publish_metrics" in flow
    assert flow.index("id: inventory") < flow.index("id: parse_chunk")
    assert flow.index("id: parse_chunk") < flow.index("id: embed_load")
    assert flow.index("id: embed_load") < flow.index("id: validate_counts")
    assert flow.index("id: validate_counts") < flow.index("id: publish_metrics")
    assert "/app/data/sample" in flow
    nonpublic_root = "data/" + "private"
    assert nonpublic_root not in flow
    assert "echo " not in flow


def test_kestra_flow_hands_off_only_declared_public_artifacts() -> None:
    flow = FLOW_PATH.read_text(encoding="utf-8")

    assert flow.count("containerImage: oilfield-chemical-copilot:local") == 5
    assert flow.count("uv run --project /app python") == 5
    assert "chunks.jsonl: \"{{ outputs.parse_chunk.outputFiles['chunks.jsonl'] }}\"" in flow
    assert "orchestration_run.json: \"{{ outputs.validate_counts.outputFiles['orchestration_run.json'] }}\"" in flow
    assert "--embedding-model \"$OLLAMA_EMBEDDING_MODEL\"" in flow
    assert "--output-dir {{ workingDir }}" in flow
    assert "--report {{ workingDir }}/orchestration_run.json" in flow
    assert "outputDir" not in flow
    assert "publish_orchestration_metrics.py" in flow


def test_kestra_validation_runs_from_the_project_root() -> None:
    flow = FLOW_PATH.read_text(encoding="utf-8")

    assert "cd /app && uv run --project /app python -m ingestion.validate_index" in flow


def test_kestra_runtime_uses_named_project_image_and_no_nonpublic_copy() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    dockerignore = DOCKERIGNORE.read_text(encoding="utf-8")

    assert "COPY data/sample ./data/sample" in dockerfile
    nonpublic_copy = "COPY data/" + "private"
    assert nonpublic_copy not in dockerfile
    nonpublic_root = "." + "private/"
    assert nonpublic_root in dockerignore
    assert "data/" + "private/" in dockerignore
    assert ".env" in dockerignore
    assert "image: oilfield-chemical-copilot:local" in compose
    assert "/var/run/docker.sock:/var/run/docker.sock" in compose


def test_kestra_runtime_configures_persistent_orchestration_state() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "KESTRA_CONFIGURATION: |" in compose
    assert "kestra-db-init:" in compose
    assert "CREATE DATABASE kestra" in compose
    assert 'entrypoint: ["/bin/sh", "-ec"]' in compose
    assert "url: jdbc:postgresql://postgres:5432/kestra" in compose
    assert "repository:\n            type: postgres" in compose
    assert "queue:\n            type: postgres" in compose
    assert "storage:\n            type: local" in compose
    assert 'base-path: "/app/storage"' in compose
    assert "/tmp/kestra-wd:/tmp/kestra-wd" in compose
    assert "basic-auth:" in compose
    assert "enabled: false" in compose
    assert "username: ${KESTRA_USERNAME:-admin@kestra.io}" in compose
    assert "password: ${KESTRA_PASSWORD:-Kestra123}" in compose
    assert "kestra-db-init:\n        condition: service_completed_successfully" in compose
