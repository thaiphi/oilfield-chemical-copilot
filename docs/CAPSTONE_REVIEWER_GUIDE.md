# Capstone Reviewer Guide

This guide is the public path for reviewing the Oilfield Chemical Troubleshooting
Copilot at a submitted Git commit. It runs the committed synthetic sample corpus
only. It is a software and grounding demonstration, not chemistry validation,
operational advice, or a production deployment.

## Prerequisites

- Git
- Python 3.11 and `uv`
- Docker Desktop with Docker Compose
- Ollama running locally

Pull the local answer and embedding models before starting the application:

```powershell
ollama pull granite4.1:8b
ollama pull granite-embedding:latest
```

If Ollama is not already running as a local service, start it in a separate
terminal with `ollama serve`.

## Public Setup

Clone the submitted public repository, check out the submitted commit, and run
the following commands from the repository root:

```powershell
Copy-Item .env.example .env
uv sync
docker compose up -d --build postgres migrate monitoring-migrate grafana-role-init grafana
```

Parse the committed public sample corpus and index it with local Ollama
embeddings:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
$env:EMBEDDING_PROVIDER = "ollama"
$env:OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"
uv run python ingestion/ingest.py --data-dir data/sample --output-dir data/processed --max-files 20
uv run python ingestion/index_chunks.py --input data/processed/chunks.jsonl --database-url $env:DATABASE_URL
```

Start the interface after indexing completes:

```powershell
docker compose up -d --build app
```

Open http://localhost:8501 and ask a public-sample question such as:

```text
How should I assess scale risk from produced water analysis?
```

The expected behavior is a source-grounded response with visible citations, or a
safe weak-evidence response when the indexed material is insufficient. Do not
treat either output as a field treatment recommendation.

## Public Monitoring Demo

The dashboard uses deliberately fixed synthetic aggregate events so a reviewer
can inspect the six panels without recording application content:

```powershell
docker compose --profile demo run --rm monitoring-demo-seed
```

Open http://localhost:3000. The dashboard is bound to the local machine and
shows response volume, latency, outcome mix, retrieval volume, and feedback
aggregates. It is a reproducible review artifact, not hosted monitoring or an
alerting service.

On a fresh local database volume, one seed invocation writes six request events
and two feedback events. The seed is intentionally additive when rerun, so a
reviewer who wants the canonical counts should use a fresh local review setup
rather than rerunning the command against prior demo data.

## Verification

Run the public checks after the setup and demo:

```powershell
uv run pytest
node --test .codex/workflows/tests/*.test.mjs
uv run ruff check .
git diff --check
git status --short
```

The final two commands should show no whitespace errors and only the reviewer's
expected local generated files, if any. Generated outputs are intentionally not
part of the submitted source revision.

## What This Review Proves

The public path demonstrates the committed interface, ingestion route, local
embedding and answer providers, source citation behavior, evaluation artifacts,
and aggregate-only monitoring. The related evidence map is in
[CAPSTONE_EVIDENCE.md](CAPSTONE_EVIDENCE.md).

It does not prove private-corpus quality, complete chemistry coverage, safe
field dosage selection, hosted availability, or production operational
readiness.

## Teardown

When the review is complete, stop local services without deleting volumes:

```powershell
docker compose down
```
