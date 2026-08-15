# Module 3 Kestra Orchestration Design

## Goal

Turn the existing Kestra ingestion scaffold into a real, local, sample-only workflow that inventories, parses and chunks, embeds and indexes, then validates the public corpus in PGVector.

## Scope

The flow operates only on the repository's public `data/sample` corpus. It does not accept a nonpublic data-mode input, mount a nonpublic directory, create public artifacts containing nonpublic source material, or add telemetry. The dlt workshop is included only for a fixed aggregate execution record; later monitoring work remains out of scope.

## Architecture

Kestra runs each ingestion stage in the project's explicit local image. The image contains the public sample corpus and the existing Python ingestion commands. Kestra task artifacts carry only public generated files between stages; the database and Ollama endpoints are reached through `host.docker.internal` from task-runner containers. After validation, a dlt task loads one allowlisted aggregate record into a separate PostgreSQL `orchestration` dataset.

```text
public sample corpus
    -> inventory artifact
    -> parsed chunk artifact
    -> Granite embeddings + PGVector upsert
    -> aggregate count validation
    -> dlt aggregate run record
```

The flow has four ordered tasks:

1. `inventory` writes the existing inventory outputs for `data/sample`.
2. `parse_chunk` writes `chunks.jsonl` and its ingestion report.
3. `embed_load` receives `chunks.jsonl` as a Kestra input artifact, calls the existing Ollama index command, and upserts vectors into PGVector.
4. `validate_counts` receives the same artifact and verifies that every input chunk ID is present under the configured embedding-model label. It emits an aggregate-only JSON report.
5. `publish_metrics` receives that JSON report and uses dlt to append one allowlisted execution record to `orchestration.ingestion_runs`.

## Runtime Boundaries

- Docker Compose gives the application an explicit `oilfield-chemical-copilot:local` image name so the Kestra Docker runner can use it.
- The Kestra service receives Docker-socket access solely to launch its task-runner containers. No task volume mounts are enabled or needed.
- The public sample corpus is copied into the project image. Private data is excluded from the image and from the flow.
- Task-runner containers receive the local database and Ollama endpoints through fixed environment variables. They do not receive credentials beyond the existing local database connection required for indexing.
- Each task fails immediately on a nonzero ingestion command. A later task runs only after its dependency succeeds.

## Validation Interface

Add a small validation command that consumes a chunks JSONL path, database URL, and embedding-model label. It returns aggregate counts only:

```python
def validate_indexed_chunk_count(
    *, chunks_path: Path, database_url: str, embedding_model: str
) -> tuple[int, int]:
    """Return expected unique input IDs and matching indexed-row count."""
```

It raises a safe error when the counts differ. Its optional report contains only `status`, `source_files`, `chunks`, `expected_indexed_chunks`, `actual_indexed_chunks`, and `embedding_model`. It does not print chunk IDs, source paths, chunk text, embeddings, credentials, or provider responses.

## dlt Workshop Boundary

The dlt task demonstrates a source-to-destination load and its generated destination lineage without treating it as the monitoring module. `ingestion/publish_orchestration_metrics.py` accepts only a validator report with the fixed allowlist above, validates its schema, and runs a local dlt PostgreSQL pipeline with dataset name `orchestration` and table name `ingestion_runs`.

It does not consume chat conversations, RAG answers, questions, source documents, tool calls, feedback, provider responses, source identifiers, or arbitrary JSON. dlt internal tables and load metadata remain in its separate dataset; no dashboard, alerting, or raw-content telemetry is introduced.

## Test Strategy

- Unit-test the aggregate validator with a fake database boundary and empty, duplicate, and mismatch cases.
- Unit-test the dlt publisher with a fake pipeline and reject every unallowlisted report field before dlt is called.
- Test the Kestra flow as configuration: no placeholder commands, sample-only input, explicit task dependencies, artifact handoff, project image, expected environment names, and no private-path reference.
- Keep the full application test suite passing.
- Run one local Kestra execution only after Docker Postgres, Kestra, and Ollama are healthy. Store a public, aggregate-only report with task status and counts.

## Acceptance Criteria

- The flow executes the required inventory -> parse/chunk -> embed/load -> validate -> dlt-publish sequence instead of `echo` placeholders.
- It indexes the public sample corpus with `granite-embedding:latest` and validates the resulting aggregate count.
- A failed stage prevents dependent stages from running.
- No private corpus path, content, chunk ID, vector, credential, raw provider error, or local absolute path enters the flow configuration, dlt record, public report, or Git history.
- Module 3 remains unlocked until public tests, one live public flow execution, and the practical teaching review are complete.
