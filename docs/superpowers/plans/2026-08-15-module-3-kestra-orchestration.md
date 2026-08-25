# Module 3 Kestra Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Kestra ingestion scaffold with a real, local, public-sample workflow that inventories, parses and chunks, embeds and loads PGVector, validates aggregate indexed counts, and demonstrates dlt publication of an allowlisted run record.

**Architecture:** Kestra Docker-runner tasks use the explicit `oilfield-chemical-copilot:local` project image. The image contains only `data/sample`; task artifacts pass public outputs between stages, while local Ollama and PGVector are accessed through `host.docker.internal`. A count-only validator writes a fixed-schema aggregate report after the load stage; a dlt publisher validates that report and appends it to the separate `orchestration.ingestion_runs` dataset.

**Tech Stack:** Python 3.11+, psycopg, PGVector, existing Granite/Ollama embedding provider, Docker Compose, Kestra shell task runner, pytest, Ruff.

## Global Constraints

- Accept and process only `data/sample`; do not introduce a nonpublic data-mode input, mount, artifact, report, or Git path.
- Use `granite-embedding:latest` and a 384-dimensional embedding contract for the live public run.
- Do not print or persist chunk text, source paths, chunk IDs, vectors, credentials, raw provider errors, or absolute local paths in public reports.
- dlt accepts only the validator's fixed aggregate report and writes only the allowlisted fields to its separate `orchestration` dataset.
- A task failure stops its dependent stages; no task runs arbitrary user-supplied shell text.
- Existing ingestion CLIs remain reusable outside Kestra.
- Do not commit until the user explicitly requests it.

---

### Task 1: Add Aggregate Index Validation

**Files:**
- Create: `ingestion/validate_index.py`
- Create: `tests/ingest/test_validate_index.py`

**Interfaces:**
- Produces `validate_indexed_chunk_count(*, chunks_path: Path, database_url: str, embedding_model: str) -> tuple[int, int]`.
- Consumes the existing `load_chunks_jsonl(path)` function and the `chunks` table populated by `ingestion/index_chunks.py`.

- [x] **Step 1: Write failing validator tests**

Create fake cursor and connection helpers, then add these tests:

```python
def test_validate_indexed_chunk_count_uses_unique_input_ids(tmp_path, monkeypatch) -> None:
    chunks_path = _write_chunks(tmp_path, ids=("scale-1", "scale-1", "scale-2"))
    _install_counting_connection(monkeypatch, returned_count=2)

    assert validate_indexed_chunk_count(
        chunks_path=chunks_path,
        database_url="postgresql://test",
        embedding_model="granite-embedding:latest",
    ) == (2, 2)


def test_validate_indexed_chunk_count_rejects_a_mismatch_without_ids(tmp_path, monkeypatch) -> None:
    chunks_path = _write_chunks(tmp_path, ids=("private-like-id", "scale-2"))
    _install_counting_connection(monkeypatch, returned_count=1)

    with pytest.raises(ValueError, match="Indexed chunk count mismatch") as error:
        validate_indexed_chunk_count(
            chunks_path=chunks_path,
            database_url="postgresql://test",
            embedding_model="granite-embedding:latest",
        )

    assert "private-like-id" not in str(error.value)
```

Also test that `--report` writes exactly the six allowlisted aggregate fields for a match, the CLI prints only expected and actual counts, and a `psycopg.Error` becomes the fixed message `PGVector count validation failed`.

- [x] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/ingest/test_validate_index.py -q`

Expected: collection fails because `ingestion.validate_index` does not exist.

- [x] **Step 3: Implement the count-only validator**

Implement these helpers:

```python
def _input_chunk_ids(chunks_path: Path) -> set[str]:
    return {chunk.metadata.chunk_id for chunk in load_chunks_jsonl(chunks_path)}


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
```

Add an argparse entry point accepting `--input`, `--database-url`, `--embedding-model`, and an optional `--report` path. When `--report` is supplied after a successful match, write JSON with exactly `status`, `source_files`, `chunks`, `expected_indexed_chunks`, `actual_indexed_chunks`, and `embedding_model`. Derive `source_files` as an aggregate count of distinct `LoadedChunk.metadata.source_file` values. Catch `psycopg.Error` in `main()` and call `parser.error("PGVector count validation failed")`; never include database or provider details in that error.

- [x] **Step 4: Run focused validation**

Run:

```powershell
uv run pytest tests/ingest/test_validate_index.py tests/ingest/test_index_chunks.py -q
uv run ruff check ingestion/validate_index.py tests/ingest/test_validate_index.py
```

Expected: all tests pass and Ruff reports no findings.

### Task 2: Add The dlt Aggregate-Record Publisher

**Files:**
- Create: `ingestion/publish_orchestration_metrics.py`
- Create: `tests/ingest/test_publish_orchestration_metrics.py`

**Interfaces:**
- Produces `load_orchestration_metrics(path: Path) -> dict[str, object]` and `publish_orchestration_metrics(*, report_path: Path, database_url: str) -> None`.
- Consumes only the validator's fixed-schema report and dlt's installed PostgreSQL destination.

- [x] **Step 1: Write failing dlt publisher tests**

Add a valid report fixture with only `status`, `source_files`, `chunks`, `expected_indexed_chunks`, `actual_indexed_chunks`, and `embedding_model`. Inject a fake dlt pipeline and assert it receives a one-record list and:

```python
pipeline.run(
    [valid_record],
    table_name="ingestion_runs",
    write_disposition="append",
)
```

Add a report with an extra `source_file` key and assert `load_orchestration_metrics()` raises `ValueError("Invalid orchestration metrics report")` before the fake pipeline is constructed. Add CLI failure coverage proving a dlt exception becomes exactly `dlt publication failed`.

- [x] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/ingest/test_publish_orchestration_metrics.py -q`

Expected: collection fails because `ingestion.publish_orchestration_metrics` does not exist.

- [x] **Step 3: Implement fixed-schema dlt publication**

Create this allowlist and type checks:

```python
_ALLOWED_KEYS = {
    "status",
    "source_files",
    "chunks",
    "expected_indexed_chunks",
    "actual_indexed_chunks",
    "embedding_model",
}
```

Require `status == "success"`, nonnegative integer count fields, and a nonblank string embedding-model label. Build the destination with `postgres(credentials=database_url)`, then create:

```python
dlt.pipeline(
    pipeline_name="oilfield_chemical_copilot_orchestration",
    destination=postgres(credentials=database_url),
    dataset_name="orchestration",
)
```

Call `run()` once with the validated record and `table_name="ingestion_runs"`. The CLI accepts `--input` and `--database-url`; it must hide all dlt and database exception details.

- [x] **Step 4: Run focused dlt validation**

Run:

```powershell
uv run pytest tests/ingest/test_publish_orchestration_metrics.py -q
uv run ruff check ingestion/publish_orchestration_metrics.py tests/ingest/test_publish_orchestration_metrics.py
```

Expected: all tests pass and Ruff reports no findings.

### Task 3: Make The Kestra Runtime And Flow Executable

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `flows/kestra/ingest.yml`
- Create: `tests/ingest/test_kestra_flow.py`

**Interfaces:**
- Produces the explicit task-runner image `oilfield-chemical-copilot:local`.
- Produces Kestra tasks `inventory`, `parse_chunk`, `embed_load`, `validate_counts`, and `publish_metrics`.
- Consumes only public task artifacts and fixed environment variable names `DATABASE_URL`, `OLLAMA_BASE_URL`, and `OLLAMA_EMBEDDING_MODEL`.

- [x] **Step 1: Write failing flow-contract tests**

Add text-based configuration tests with exact assertions:

```python
def test_kestra_flow_uses_four_ordered_public_sample_tasks() -> None:
    flow = FLOW_PATH.read_text(encoding="utf-8")

    assert 'id: inventory' in flow
    assert 'id: parse_chunk' in flow
    assert 'id: embed_load' in flow
    assert 'id: validate_counts' in flow
    assert 'id: publish_metrics' in flow
    assert flow.index('id: inventory') < flow.index('id: parse_chunk')
    assert flow.index('id: parse_chunk') < flow.index('id: embed_load')
    assert flow.index('id: embed_load') < flow.index('id: validate_counts')
    assert flow.index('id: validate_counts') < flow.index('id: publish_metrics')
    assert '/app/data/sample' in flow
    nonpublic_root = 'data/' + 'private'
    assert nonpublic_root not in flow
    assert 'echo ' not in flow


def test_kestra_runtime_uses_a_named_project_image_and_no_private_copy() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert 'COPY data/sample ./data/sample' in dockerfile
    nonpublic_copy = 'COPY data/' + 'private'
    assert nonpublic_copy not in dockerfile
    assert 'image: oilfield-chemical-copilot:local' in compose
    assert '/var/run/docker.sock:/var/run/docker.sock' in compose
```

Assert that `embed_load` and `validate_counts` refer to the `parse_chunk` `chunks.jsonl` artifact, `publish_metrics` refers to the `validate_counts` aggregate report artifact, all five tasks use the project image, and the index/validator commands specify `granite-embedding:latest`.

- [x] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/ingest/test_kestra_flow.py -q`

Expected: failures because the scaffold has placeholder commands, no named image, no sample data copy, and no artifact handoff.

- [x] **Step 3: Implement the Docker runtime boundary**

In `Dockerfile`, add only the public sample corpus after copying project code:

```dockerfile
COPY data/sample ./data/sample
```

In the Compose `app` service, add:

```yaml
image: oilfield-chemical-copilot:local
```

In the Compose `kestra` service, add the Docker socket mount:

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
  - kestra-data:/app/storage
  - ./flows/kestra:/app/flows:ro
  - ./data:/app/data:ro
```

Do not add task-runner volume mounts and do not copy any private source directory into the image.

- [x] **Step 4: Implement the real flow**

Replace the scaffold with four sequential `io.kestra.plugin.scripts.shell.Commands` tasks. Every task uses:

```yaml
containerImage: oilfield-chemical-copilot:local
env:
  DATABASE_URL: postgresql://postgres:postgres@host.docker.internal:5432/oilfield_copilot
  OLLAMA_BASE_URL: http://host.docker.internal:11434
  OLLAMA_EMBEDDING_MODEL: granite-embedding:latest
```

Use these commands and public artifact boundaries:

```yaml
- id: inventory
  commands:
    - uv run python /app/ingestion/inventory.py --data-dir /app/data/sample --output-dir {{ outputDir }} --summary-only
  outputFiles:
    - inventory.csv
    - inventory_summary.md

- id: parse_chunk
  commands:
    - uv run python /app/ingestion/ingest.py --data-dir /app/data/sample --output-dir {{ outputDir }}
  outputFiles:
    - chunks.jsonl
    - ingestion_report.json

- id: embed_load
  inputFiles:
    chunks.jsonl: "{{ outputs.parse_chunk.outputFiles['chunks.jsonl'] }}"
  commands:
    - uv run python /app/ingestion/index_chunks.py --input {{ workingDir }}/chunks.jsonl --database-url "$DATABASE_URL" --embedding-provider ollama --embedding-model "$OLLAMA_EMBEDDING_MODEL"

- id: validate_counts
  inputFiles:
    chunks.jsonl: "{{ outputs.parse_chunk.outputFiles['chunks.jsonl'] }}"
  commands:
    - uv run python /app/ingestion/validate_index.py --input {{ workingDir }}/chunks.jsonl --database-url "$DATABASE_URL" --embedding-model "$OLLAMA_EMBEDDING_MODEL" --report {{ outputDir }}/orchestration_run.json
  outputFiles:
    - orchestration_run.json

- id: publish_metrics
  inputFiles:
    orchestration_run.json: "{{ outputs.validate_counts.outputFiles['orchestration_run.json'] }}"
  commands:
    - uv run python /app/ingestion/publish_orchestration_metrics.py --input {{ workingDir }}/orchestration_run.json --database-url "$DATABASE_URL"
```

The Kestra task list is sequential; artifact references make `embed_load` and `validate_counts` depend on successful `parse_chunk` completion, while `publish_metrics` depends on successful count validation.

- [x] **Step 5: Run flow and Compose configuration validation**

Run:

```powershell
uv run pytest tests/ingest/test_kestra_flow.py -q
docker compose config --quiet
uv run ruff check tests/ingest/test_kestra_flow.py
```

Expected: flow tests and Ruff pass; Compose renders without configuration errors.

### Task 4: Verify The Complete Public Orchestration Boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/PROJECT_STATUS.md`
- Create: `docs/superpowers/reports/2026-08-15-module-3-kestra-live-run.md`

**Interfaces:**
- Consumes public test evidence and one local Kestra execution.
- Produces aggregate-only execution evidence and accurate Module 3 status.

- [x] **Step 1: Run complete public verification before live execution**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
git diff --check
```

Expected: all public checks pass before Docker/Kestra execution.

- [x] **Step 2: Build and start the local Kestra prerequisites**

Run:

```powershell
docker compose build app
docker compose up -d postgres kestra
docker compose ps
```

Wait until PostgreSQL and Kestra report healthy or running status. Confirm only that local Ollama is reachable and `granite-embedding:latest` is present; do not record model output.

- [x] **Step 3: Execute the public Kestra flow once**

Upload `flows/kestra/ingest.yml` through the local Kestra UI or API, execute it with no data-mode override, and wait for the five tasks to finish. Record only: flow success status, per-task success status, source-file count, chunk count, expected indexed count, actual indexed count, configured embedding-model label, dlt publication success, and sanitized error category if a task fails.

- [x] **Step 4: Document reproducible public operation**

Update `README.md` with the public-only prerequisites, the local image build command, Kestra URL, and the rule that nonpublic source material is outside this flow. Update `docs/PROJECT_STATUS.md` to state either successful live evidence or the exact unresolved local prerequisite.

Create the aggregate-only report at `docs/superpowers/reports/2026-08-15-module-3-kestra-live-run.md`. It must not include question text, chunk text, filenames, paths, chunk IDs, vectors, credentials, or raw provider errors.

- [x] **Step 5: Complete the practical teaching review and request commit approval**

Explain the five task boundaries, artifact handoff, retry/failure behavior, dlt's separate destination dataset, and why private source material stays outside the initial flow. Mark Module 3 ready for lock only after the public checks and one live execution pass. Do not stage, commit, or push until the user explicitly approves.

## Plan Self-Review

- **Spec coverage:** Task 1 implements aggregate validation; Task 2 implements fixed-schema dlt publication; Task 3 implements the named-image runtime, public-only flow, artifact handoff, task order, and configuration tests; Task 4 covers public verification, one live execution, documentation, privacy-safe reporting, and teaching review.
- **Placeholder scan:** The plan has concrete paths, commands, task IDs, environment variables, interfaces, and test cases. It does not defer implementation details.
- **Type consistency:** `parse_chunk` produces `chunks.jsonl`; `embed_load` and `validate_counts` both consume that exact artifact. `validate_counts` produces `orchestration_run.json`, which is the only input accepted by `publish_metrics`. The validator and publisher CLI options are used consistently across Tasks 1 through 4.

## Live Execution Notes

The executable flow has five sequential tasks: `inventory`, `parse_chunk`, `embed_load`, `validate_counts`, and `publish_metrics`. Kestra's Docker runner writes and receives declared artifacts through `workingDir`; no `outputDir` expression is used. The validation task runs its module from `/app` so its repository-level import is available. The publisher depends on dlt's PostgreSQL extra, declared as `dlt[postgres]`.
