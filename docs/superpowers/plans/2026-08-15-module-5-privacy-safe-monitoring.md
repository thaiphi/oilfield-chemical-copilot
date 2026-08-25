# Module 5 Privacy-Safe Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver persistent, privacy-safe monitoring, aggregate feedback, and a reproducible six-panel Grafana dashboard for the Module 5 capstone requirement.

**Architecture:** Keep the existing process-local `AggregateMonitor` as the closed-contract guard. Add a PostgreSQL repository that accepts only closed outcome, retrieval mode, UTC hour, aggregate latency, and feedback value; Streamlit sends those fields after each completed response. Docker Compose provisions a read-only Grafana role, datasource, dashboard, and optional fixed synthetic demo aggregates.

**Tech Stack:** Python 3.11, Streamlit, psycopg 3, PostgreSQL 16/PGVector, Grafana, Docker Compose, pytest, Ruff.

## Global Constraints

- Persist no prompt, answer, excerpt, source identifier, user/session identifier, tool argument/result, raw error, token/cost, trace ID, free-text field, or JSON payload.
- Do not write to the legacy raw-content tables: `conversations`, `feedback`, `latency_events`, `retrieved_chunks`, or `tool_calls`.
- Aggregate request rows use only UTC hourly buckets, a closed `MonitoringOutcome`, a closed `RetrievalMode`, and latency summary fields.
- Aggregate feedback rows use only UTC hourly buckets, `helpful`/`needs_work`, a closed retrieval mode, and a count.
- Monitoring failure must never change an answer, a claim-scope abstention, or a calculator result.
- Grafana configuration and demo fixtures are tracked; actual activity and all private corpus/evaluation material remain untracked.
- The demo seed inserts deterministic synthetic aggregates only when explicitly invoked, never during app startup.
- No RAG, retrieval, evaluator, chemistry guidance, or tool behavior changes are in scope.
- Do not commit unless the user explicitly asks for a commit.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/oilfield_chemical_copilot/observability/aggregate_monitoring.py` | Existing process-local closed outcomes; extend with closed retrieval and feedback value types only. |
| `src/oilfield_chemical_copilot/observability/persistence.py` | Immutable aggregate metric records, PostgreSQL upserts, and a no-op-safe recorder facade. |
| `tests/observability/test_aggregate_monitoring.py` | Closed-value and payload-denial unit coverage. |
| `tests/observability/test_persistence.py` | SQL parameter and repository behavior with fake connections. |
| `db/migrations/0003_module_5_monitoring.sql` | Aggregate tables, constraints, indexes, and no raw-content writes. |
| `monitoring/bootstrap_grafana_role.py` | Creates or updates the Grafana read-only role and grants it only the two aggregate tables. |
| `monitoring/seed_demo_metrics.py` | Explicit deterministic synthetic dashboard seed command. |
| `tests/monitoring/test_demo_seed.py` | Demo data is fixed, labelled, and aggregate-only. |
| `app/streamlit_app.py` | Records completed outcomes once and writes aggregate feedback only. |
| `tests/app/test_streamlit_app.py` | Route-to-metric and feedback UI behavior. |
| `monitoring/grafana/datasources/postgres.yml` | Provisioned PostgreSQL datasource using the read-only role. |
| `monitoring/grafana/dashboards/dashboard.yml` | Grafana dashboard provider. |
| `monitoring/grafana/dashboards/module5-monitoring.json` | Six tracked SQL panels that reference aggregate tables only. |
| `tests/monitoring/test_grafana_provisioning.py` | Dashboard/datasource safety and six-panel structural tests. |
| `docker-compose.yml` | Grafana role bootstrap and explicit demo-seed service/profile. |
| `README.md` | Reviewer instructions, synthetic screenshot, data contract, and privacy statement. |
| `docs/PROJECT_STATUS.md` / `docs/LEARNING_ROADMAP.md` | Module 5 progress and eventual lock evidence. |

## Task 1: Define Closed Persistent-Metric Contracts

**Files:**
- Modify: `src/oilfield_chemical_copilot/observability/aggregate_monitoring.py`
- Create: `src/oilfield_chemical_copilot/observability/persistence.py`
- Modify: `tests/observability/test_aggregate_monitoring.py`
- Create: `tests/observability/test_persistence.py`

**Interfaces:**
- Consumes: `MonitoringOutcome` and finite nonnegative latency validation from `AggregateMonitor`.
- Produces: `RetrievalMode`, `FeedbackValue`, `HourlyRequestMetric`, `HourlyFeedbackMetric`, `MonitoringRepository`, and `SafeMonitoringRecorder`.
- Later callers use `SafeMonitoringRecorder.record_request(outcome, retrieval_mode, latency_ms, occurred_at)` and `SafeMonitoringRecorder.record_feedback(value, retrieval_mode, occurred_at)`.

- [x] **Step 1: Write failing contract tests**

```python
def test_request_metric_rounds_to_utc_hour_and_rejects_unapproved_values() -> None:
    metric = HourlyRequestMetric.from_values(
        outcome=MonitoringOutcome.RAG_ANSWERED,
        retrieval_mode=RetrievalMode.HYBRID,
        latency_ms=12.5,
        occurred_at=datetime(2026, 8, 15, 10, 47, tzinfo=timezone.utc),
    )
    assert metric.bucket_start == datetime(2026, 8, 15, 10, tzinfo=timezone.utc)
    assert not hasattr(metric, "payload")


def test_recorder_does_not_propagate_repository_failure() -> None:
    recorder = SafeMonitoringRecorder(AggregateMonitor(), FailingRepository())
    recorder.record_request(MonitoringOutcome.RAG_ANSWERED, RetrievalMode.VECTOR, 9.0, NOW)
    assert recorder.snapshot().total_requests == 1
```

- [x] **Step 2: Run the new contract tests and confirm failure**

Run: `uv run pytest tests/observability/test_aggregate_monitoring.py tests/observability/test_persistence.py -q`

Expected: FAIL because the closed persistent-metric types and recorder do not exist.

- [x] **Step 3: Implement the minimum closed contract**

```python
class RetrievalMode(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    NOT_APPLICABLE = "not_applicable"


class FeedbackValue(str, Enum):
    HELPFUL = "helpful"
    NEEDS_WORK = "needs_work"


@dataclass(frozen=True)
class HourlyRequestMetric:
    bucket_start: datetime
    outcome: MonitoringOutcome
    retrieval_mode: RetrievalMode
    latency_ms: float
```

Normalize accepted timestamps to an aware UTC hour. Reject naive timestamps, booleans, nonfinite or negative latency, and values outside the enums. `SafeMonitoringRecorder` records in-memory first, calls the optional repository second, catches only `psycopg.Error`/`OSError` from persistence, and exposes no payload-accepting method.

- [x] **Step 4: Implement fake-connection repository tests**

Use a fake psycopg connection/cursor. Assert the repository receives parameterized values matching only the aggregate record fields. Assert no SQL string names any legacy raw-content table and no parameter contains request text, source values, JSON, or a session identifier.

- [x] **Step 5: Run focused tests**

Run: `uv run pytest tests/observability/test_aggregate_monitoring.py tests/observability/test_persistence.py -q`

Expected: PASS.

## Task 2: Add Aggregate PostgreSQL Storage And Demo Seed

**Files:**
- Create: `db/migrations/0003_module_5_monitoring.sql`
- Modify: `src/oilfield_chemical_copilot/observability/persistence.py`
- Create: `monitoring/seed_demo_metrics.py`
- Create: `tests/monitoring/test_demo_seed.py`
- Modify: `tests/ingest/test_apply_migrations.py`

**Interfaces:**
- Consumes: Task 1 metric records and `MonitoringRepository`.
- Produces: `PostgresMonitoringRepository(database_url)` and `seed_demo_metrics(database_url, occurred_at)`.
- Later callers use the repository through `SafeMonitoringRecorder`, never execute SQL in Streamlit.

- [x] **Step 1: Write failing migration and repository tests**

```python
def test_module_5_migration_contains_only_aggregate_monitoring_tables() -> None:
    sql = MODULE_5_MIGRATION.read_text(encoding="utf-8")
    assert "monitoring_request_hourly" in sql
    assert "monitoring_feedback_hourly" in sql
    assert "conversations" not in sql
    assert "jsonb" not in sql.lower()


def test_postgres_repository_upserts_hourly_request_aggregates() -> None:
    repository = PostgresMonitoringRepository("postgresql://test", connect=fake_connect)
    repository.record_request(METRIC)
    assert "on conflict" in fake_cursor.executed[0][0].lower()
```

- [x] **Step 2: Run the migration/repository tests and confirm failure**

Run: `uv run pytest tests/observability/test_persistence.py tests/monitoring/test_demo_seed.py tests/ingest/test_apply_migrations.py -q`

Expected: FAIL because migration `0003`, aggregate upserts, and demo seed do not exist.

- [x] **Step 3: Create the idempotent aggregate migration**

```sql
create table if not exists monitoring_request_hourly (
    bucket_start timestamptz not null,
    outcome text not null check (outcome in (
        'rag_answered',
        'rag_weak_evidence',
        'scope_abstained',
        'tool_calculated',
        'tool_input_invalid',
        'rag_configuration_error'
    )),
    retrieval_mode text not null check (retrieval_mode in ('vector', 'hybrid', 'not_applicable')),
    request_count bigint not null check (request_count >= 0),
    latency_count bigint not null check (latency_count >= 0),
    latency_total_ms double precision not null check (latency_total_ms >= 0),
    latency_minimum_ms double precision,
    latency_maximum_ms double precision,
    primary key (bucket_start, outcome, retrieval_mode)
);
```

Create the feedback table with `bucket_start`, `feedback_value`, `retrieval_mode`, `feedback_count`, validation checks, and a composite primary key. Add only time-oriented indexes needed by the panel queries. Do not alter legacy scaffold tables.

- [x] **Step 4: Implement atomic parameterized upserts and fixed synthetic seeding**

`PostgresMonitoringRepository.record_request()` increments request/latency counts and sums, and uses `least`/`greatest` for latency extrema. `record_feedback()` increments only the aggregate counter. The seed script contains a fixed tuple of closed metrics across a named UTC demo window and invokes the same repository API; it accepts only `--database-url` and does not read application data.

- [x] **Step 5: Run focused tests**

Run: `uv run pytest tests/observability/test_persistence.py tests/monitoring/test_demo_seed.py tests/ingest/test_apply_migrations.py -q`

Expected: PASS.

## Task 3: Integrate Streamlit Outcomes And Feedback

**Files:**
- Modify: `app/streamlit_app.py`
- Modify: `tests/app/test_streamlit_app.py`
- Modify: `src/oilfield_chemical_copilot/observability/persistence.py`

**Interfaces:**
- Consumes: `SafeMonitoringRecorder`, `RetrievalMode`, `FeedbackValue`, and existing `_route_prompt_with_outcome()` return values.
- Produces: one `record_request` call per completed route and one aggregate feedback write per browser-session response/feedback value.

- [x] **Step 1: Write failing app tests**

```python
def test_record_request_uses_hybrid_for_rag_and_not_applicable_for_tool(monkeypatch) -> None:
    _record_request(MonitoringOutcome.RAG_ANSWERED, retrieval_mode="hybrid", started_at=10.0)
    _record_request(MonitoringOutcome.TOOL_CALCULATED, retrieval_mode="hybrid", started_at=10.0)
    assert recorded_modes == [RetrievalMode.HYBRID, RetrievalMode.NOT_APPLICABLE]


def test_feedback_records_closed_aggregate_without_message_content(monkeypatch) -> None:
    _record_feedback(FeedbackValue.HELPFUL, RetrievalMode.VECTOR)
    assert writes == [(FeedbackValue.HELPFUL, RetrievalMode.VECTOR)]
```

- [x] **Step 2: Run the app tests and confirm failure**

Run: `uv run pytest tests/app/test_streamlit_app.py -q`

Expected: FAIL because the existing `_record_request` has no retrieval mode and the buttons only show a placeholder toast.

- [x] **Step 3: Add lazy recorder construction and route instrumentation**

Implement `_build_monitoring_recorder()` with `@st.cache_resource`. It always returns an in-memory recorder; it attaches `PostgresMonitoringRepository(_database_url())` only when `MONITORING_PERSISTENCE_ENABLED=true`. Convert accepted RAG modes to `RetrievalMode`; force calculator paths to `NOT_APPLICABLE`. Keep telemetry failures inside the recorder so the existing response behavior is unchanged.

- [x] **Step 4: Replace placeholder feedback with aggregate-only feedback**

After a successfully completed assistant response, save only its closed outcome and normalized retrieval mode in Streamlit session state. Make each feedback button call `_record_feedback()` once for that response/value combination, show a neutral confirmation, and disable/reject repeated clicks in that browser session. Do not store message text, citation data, request value, or an identifier in session state for feedback.

- [x] **Step 5: Run focused app tests**

Run: `uv run pytest tests/app/test_streamlit_app.py tests/observability/test_aggregate_monitoring.py tests/observability/test_persistence.py -q`

Expected: PASS.

## Task 4: Provision Grafana With Six Safe Panels

**Files:**
- Create: `monitoring/bootstrap_grafana_role.py`
- Create: `monitoring/grafana/datasources/postgres.yml`
- Create: `monitoring/grafana/dashboards/dashboard.yml`
- Create: `monitoring/grafana/dashboards/module5-monitoring.json`
- Create: `tests/monitoring/test_grafana_provisioning.py`
- Modify: `docker-compose.yml`
- Modify: `.env.example`

**Interfaces:**
- Consumes: Task 2 aggregate tables and `GRAFANA_DB_PASSWORD` environment value.
- Produces: local Grafana datasource named `Oilfield Monitoring`, dashboard UID `oilfield-module5-monitoring`, and optional Compose profile `demo`.

- [x] **Step 1: Write failing provisioning tests**

```python
def test_dashboard_has_six_panels_and_queries_only_aggregate_tables() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    assert len(dashboard["panels"]) == 6
    rendered = json.dumps(dashboard).lower()
    assert "monitoring_request_hourly" in rendered
    assert "monitoring_feedback_hourly" in rendered
    for forbidden in ("conversations", "retrieved_chunks", "tool_calls", "prompt", "answer"):
        assert forbidden not in rendered
```

- [x] **Step 2: Run the provisioning tests and confirm failure**

Run: `uv run pytest tests/monitoring/test_grafana_provisioning.py -q`

Expected: FAIL because no Grafana provisioning or dashboard exists.

- [x] **Step 3: Add least-privilege role bootstrap**

Implement `bootstrap_grafana_role.py` with psycopg SQL composition. It obtains the app database URL and `GRAFANA_DB_PASSWORD`, creates or updates the `grafana_reader` login role, revokes public table access, grants schema usage and `SELECT` only on `monitoring_request_hourly` and `monitoring_feedback_hourly`, then commits. It prints only a fixed success/failure category and never prints a URL or password.

- [x] **Step 4: Add datasource, dashboard provider, and six panels**

The datasource uses `grafana_reader`, `${GRAFANA_DB_PASSWORD}`, PostgreSQL service host, and database variables supplied by Compose. Define these six panels exactly: request volume by outcome; average latency by mode; latency min/max; outcome mix; mode volume; helpful-rate plus feedback counts. SQL selects only aggregate columns, uses Grafana time macros, aliases visible labels, and labels all times UTC. Do not use raw query text, raw errors, or legacy tables.

- [x] **Step 5: Add Compose services and environment documentation**

Add `grafana-role-init` after `migrate` and before Grafana. Add `monitoring-demo-seed` behind a `demo` profile; it runs only when the reviewer invokes the profile and exits after seeding. Add `GRAFANA_DB_PASSWORD` with a local-development default to `.env.example`; never include the actual user `.env` file.

- [x] **Step 6: Run focused provisioning tests**

Run: `uv run pytest tests/monitoring/test_grafana_provisioning.py tests/monitoring/test_demo_seed.py -q`

Expected: PASS.

## Task 5: Verify The Containerized Reviewer Path

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/reports/2026-08-15-module-5-monitoring-verification.md`
- Modify: `tests/monitoring/test_grafana_provisioning.py`

**Interfaces:**
- Consumes: completed Docker Compose stack, Grafana provisioning, and explicit demo seed.
- Produces: a sanitized verification report and README screenshot of synthetic telemetry.

- [x] **Step 1: Write a failing Compose-structure test**

```python
def test_compose_keeps_demo_seed_explicit_and_grafana_provisioned() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "monitoring-demo-seed:" in compose
    assert "profiles: [\"demo\"]" in compose
    assert "./monitoring/grafana:/etc/grafana/provisioning:ro" in compose
```

- [x] **Step 2: Run the completed Compose-structure test**

Run: `uv run pytest tests/monitoring/test_grafana_provisioning.py -q`

Expected: PASS. A failure means the Task 4 Compose configuration does not meet the reviewer-path contract and must be corrected before the live container run.

- [x] **Step 3: Run the reproducible local review sequence**

Run:

```powershell
docker compose up -d --build postgres migrate grafana-role-init app grafana
docker compose --profile demo run --rm monitoring-demo-seed
docker compose ps
```

Query the two aggregate tables only to confirm nonzero demo rows. Open Grafana at `http://localhost:3000`, verify the six named panels resolve, then capture one dashboard screenshot. Do not capture or publish database credentials, private local paths, browser prompts, or source content.

- [x] **Step 4: Add reviewer documentation and safe verification report**

Document the exact startup, explicit demo seed, Grafana URL, chart list, synthetic-data label, teardown, and privacy contract. The report records only service health, panel count, and aggregate demo-validation status. It excludes query/answer content, database URLs, passwords, private paths, and raw SQL output.

- [x] **Step 5: Run focused documentation/provisioning checks**

Run: `uv run pytest tests/monitoring/test_grafana_provisioning.py tests/app/test_streamlit_app.py -q`

Expected: PASS.

## Task 6: Complete Module 5 Verification And Teaching Review

**Files:**
- Create: `docs/superpowers/reports/2026-08-15-module-5-teaching-review.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `docs/LEARNING_ROADMAP.md`
- Modify: `docs/COURSE_ALIGNED_PLAN.md`

**Interfaces:**
- Consumes: verified Task 1-5 artifacts and the Module 5 design.
- Produces: a teaching review that explains metrics, aggregation, persistence, feedback, Grafana, and the non-hosted GitHub reviewer model.

- [x] **Step 1: Write the teaching-review acceptance checklist**

```markdown
- Explain why an hourly aggregate is a metric, not an event log.
- Identify the difference between a request outcome and user feedback.
- Explain why Grafana needs a datasource and dashboard definition in Git.
- Verify why synthetic demo telemetry is useful for peer review but not production evidence.
- State why raw conversation tables remain unused.
```

- [x] **Step 2: Run full automated verification**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
git diff --check
```

Expected: all hook tests pass, all Python tests pass with only existing documented skips, Ruff reports no violations, and `git diff --check` returns no whitespace errors.

- [x] **Step 3: Review privacy and Git boundaries**

Run `git status --short`, inspect staged paths before any requested commit, and verify no `.private` path, raw source material, runtime output, or user `.env` file is present. Confirm dashboard/report artifacts contain only synthetic demo data and aggregate-safe values.

- [x] **Step 4: Update status only after successful verification**

Mark Module 5 as implemented with the exact verification results. Do not lock it or claim production readiness until the user approves the final evidence review.

- [x] **Step 5: Request final Module 5 review and lock approval**

Present the dashboard evidence, test results, privacy results, and teaching review. Do not commit or push unless separately requested.

## Plan Self-Review

- Spec coverage: Tasks 1-3 implement the safe data contract, persistence, app instrumentation, and feedback. Tasks 4-5 provide the six reproducible Grafana panels, Docker path, demo seed, and reviewer evidence. Task 6 provides full verification, privacy review, and teaching/roadmap evidence.
- Placeholder scan: no task defers a behavior or depends on an unnamed interface; every implementation step names its files, contract, and validation command.
- Type consistency: `MonitoringOutcome`, `RetrievalMode`, and `FeedbackValue` are the only categories passed from Streamlit into `SafeMonitoringRecorder`; persistence receives immutable hourly metric records; Grafana reads only the two aggregate tables.
- Scope: OpenTelemetry, raw trace storage, hosted dashboards, alerts, RAG changes, and production deployment remain excluded.
