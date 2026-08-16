# Module 5 Privacy-Safe Monitoring Design

## Goal

Complete the LLM Zoomcamp monitoring objective with collected user feedback and a reproducible dashboard containing at least five charts, while preserving the project's rule that real request content is never persisted or published.

## Course And Review Model

The capstone is peer-reviewed from a public GitHub repository and a fixed commit. A reviewer must be able to clone that commit, start the documented Docker Compose stack, open the local Grafana instance, and see the dashboard without access to this developer's local activity. The repository will therefore track dashboard provisioning, dashboard JSON, a privacy-safe database schema, and a clearly labelled synthetic demo seed. It will not require a hosted Grafana deployment.

The design follows the Module 5 concepts of instrumenting request outcomes, storing metrics, collecting feedback, and displaying system health. It uses the existing custom aggregate monitor rather than an OpenTelemetry trace exporter because the existing application does not emit raw trace attributes and a custom exporter would add complexity without increasing the capstone evidence. The design retains the course distinction: a request is observed through bounded metrics, not by storing its conversation text.

## Scope

In scope:

- Persist closed request outcomes and latency summaries in PostgreSQL.
- Persist aggregate helpful and needs-work feedback counts in PostgreSQL.
- Add a Streamlit feedback interaction tied to the latest completed response.
- Track Grafana datasource and dashboard provisioning in Git.
- Provide six privacy-safe dashboard panels and a synthetic, explicitly demo-only seed path.
- Add unit, repository, route, migration, provisioning, and Docker Compose validation.
- Document reviewer startup, demo seed, screenshot capture, and the privacy boundary.

Out of scope:

- Persisting prompts, answers, source excerpts, source identifiers, tool arguments, tool results, user/session identifiers, model text, raw errors, tokens, costs, or OpenTelemetry trace IDs.
- Activating or writing to the existing `conversations`, `feedback`, `latency_events`, `retrieved_chunks`, or `tool_calls` tables, because their schemas can retain raw content or identifiers.
- Public Grafana hosting, authentication, alerting, or production deployment.
- Changing RAG retrieval, generation, claim-scope policy, tool behavior, or Module 4 results.

## Architecture

```text
Streamlit route outcome + elapsed time
        |
        v
Safe telemetry recorder
  validates closed outcome and retrieval mode
        |
        v
PostgreSQL hourly aggregate tables
  request count + latency count/sum/min/max
  feedback count by helpful/needs_work
        |
        v
Grafana PostgreSQL datasource (read-only)
        |
        v
Six provisioned dashboard panels
```

`AggregateMonitor` remains the process-local guard and testable accumulator. A new persistence adapter receives only its explicit closed outcome, retrieval mode, latency, and the UTC hour bucket. It updates aggregate rows with parameterized SQL. There is no event table and no payload parameter in the persistence interface.

The Streamlit application determines the response outcome at the existing routing boundary. It records the request once after an answer, weak-evidence fallback, claim-scope abstention, valid/invalid tool result, or configuration error. Feedback buttons appear only after a completed assistant response and increment an aggregate feedback bucket. Browser session state may prevent duplicate clicks during the open browser session, but no session identifier is persisted.

## Persistent Data Contract

`monitoring_request_hourly` uses a composite key of `bucket_start`, `outcome`, and `retrieval_mode`:

- `bucket_start`: UTC hour, used only for time-series aggregation.
- `outcome`: one member of `MonitoringOutcome`.
- `retrieval_mode`: `vector`, `hybrid`, or `not_applicable`.
- `request_count`: positive aggregate count.
- `latency_count`, `latency_total_ms`, `latency_minimum_ms`, `latency_maximum_ms`: aggregate latency summary.

`monitoring_feedback_hourly` uses a composite key of `bucket_start`, `feedback_value`, and `retrieval_mode`:

- `feedback_value`: `helpful` or `needs_work`.
- `feedback_count`: positive aggregate count.

Both tables reject unapproved enum-like values with database checks. Neither table includes a foreign key to a conversation or any free-text/JSON column. PostgreSQL timestamps are stored as UTC hour boundaries; Grafana labels the dashboard as UTC.

## Dashboard

Grafana is provisioned by Docker Compose from tracked files under `monitoring/grafana/`. A dedicated PostgreSQL read-only dashboard role may select only the two monitoring aggregate tables. The dashboard has six panels:

1. Request volume per hour, grouped by outcome.
2. Average response latency per hour, grouped by retrieval mode.
3. Minimum and maximum response latency per hour.
4. Successful, weak-evidence, scope-abstained, and configuration-error outcome mix.
5. Retrieval-mode request volume for vector, hybrid, and not-applicable routes.
6. Helpful-rate trend with helpful and needs-work feedback counts.

The checked-in dashboard definition contains no operational data. The optional demo command inserts fixed synthetic aggregates into a separate, clearly labelled `demo` time window so reviewers can inspect every panel immediately. It must never run automatically in the normal application startup path.

## Failure Handling

- Invalid outcome, retrieval mode, feedback value, latency, or non-hour timestamp is rejected before a database write.
- A telemetry write failure never changes the RAG answer or tool response. The application records only the existing safe configuration/error outcome in memory and shows no database details to the user.
- Database migrations are idempotent and create only the new aggregate tables, indexes, and least-privilege Grafana role.
- Grafana may fail independently of the app. Docker health and provisioning checks reveal the issue without exposing query or answer content.

## Verification And Review Evidence

- Unit tests prove the recorder accepts only the fixed data contract and cannot accept payloads or identifiers.
- Repository tests use a fake connection to assert parameterized aggregate upserts and deny raw-content columns.
- Streamlit tests prove each completed route records one aggregate and feedback produces only an aggregate count.
- Migration tests prove the raw-content scaffold tables are never a write target.
- Provisioning tests parse the datasource and dashboard definitions, assert six panels, approved SQL tables only, and no private/raw-content strings.
- Docker validation starts PostgreSQL, app, and Grafana; a synthetic demo seed produces non-empty dashboard queries.
- A README screenshot captures the local dashboard after the synthetic demo seed. It is labelled as synthetic demo telemetry.
- `git status`, staged-file checks, and report review confirm no private corpus, raw question, answer, source, or local path is committed.

## Design Self-Review

- No placeholders: data fields, panel set, explicit out-of-scope data, and verification evidence are defined.
- Consistency: every persistent field is an aggregate-safe value used by the dashboard; no UI or database path activates raw logging scaffolds.
- Scope: this is the monitoring module only. It does not modify RAG, retrieval quality, or chemistry guidance.
- Ambiguity resolved: GitHub reviewers reproduce Grafana locally from Docker Compose and inspect a synthetic demo dataset; public hosted Grafana is not required.
