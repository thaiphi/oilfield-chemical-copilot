# Capstone Evidence Map

This map gives a reviewer a public, source-controlled route through the
capstone. It deliberately distinguishes verified software behavior from claims
that require field engineering, a broader corpus, or production operations.

| Review dimension | Implemented evidence | Reviewer check | Boundary |
| --- | --- | --- | --- |
| Problem description | [README.md](../README.md) describes a source-grounded troubleshooting copilot for general oilfield-chemical review. | Read the project overview and scope limits. | It does not make field-ready treatment decisions. |
| Retrieval flow | Keyword, vector, and hybrid RRF retrieval are implemented in `src/oilfield_chemical_copilot/rag/`. | Run the public setup in [CAPSTONE_REVIEWER_GUIDE.md](CAPSTONE_REVIEWER_GUIDE.md) and inspect cited responses. | Retrieval returns evidence; it does not establish chemistry correctness. |
| Retrieval evaluation | [Module 4 public evaluation](superpowers/reports/2026-08-15-module-4-public-evaluation.md) records public vector and hybrid retrieval aggregates. | Review the report and its linked tests. | The public baseline is small and does not select an operational retrieval winner. |
| LLM evaluation | [Module 4 teaching review](superpowers/reports/2026-08-15-module-4-teaching-review.md) documents deterministic checks and a structured local judge. | Review the aggregate reports and run tests. | The judge is advisory quality evidence, not a subject-matter authority. |
| Interface | `app/streamlit_app.py` exposes the RAG and bounded calculator routes. | Open http://localhost:8501 after the public setup. | Closed-scope questions abstain before retrieval and generation. |
| Ingestion pipeline | `ingestion/` parses and indexes the committed public sample; `flows/kestra/ingest.yml` provides a public ingestion orchestration flow. | Run the parser/indexer commands or review the Kestra flow. | The capstone walkthrough uses only the committed sample. |
| Monitoring | Module 5 adds aggregate-only persistence, feedback, and six Grafana panels. | Run `monitoring-demo-seed`, then open http://localhost:3000. | No application content is stored in the monitoring tables. |
| Containerization | `Dockerfile` and `docker-compose.yml` package the app, database, migration, Kestra, and Grafana services. | Run the Compose commands in the reviewer guide. | Ollama remains a documented local prerequisite outside Compose. |
| Reproducibility | `pyproject.toml`, `uv.lock`, public sample files, Compose configuration, and the reviewer guide define the path. | Run `uv sync`, public ingestion, tests, lint, and whitespace checks. | Generated local outputs are excluded from the submitted revision. |
| Best practices | Hybrid RRF is implemented and tested; claim-scope routing, citation selection, and aggregate-safe monitoring protect the production boundary. | Compare code, tests, and linked Module 4 and Module 5 evidence. | reranking and query rewriting are intentionally deferred because no measured retrieval gap currently justifies them. |

## Review Outcome

A successful review should confirm that the public codebase can be set up,
indexed, queried, evaluated, and monitored locally with Granite/Ollama and
Docker. It should also confirm that the evidence is appropriately limited:
the capstone is a reproducible learning project, not a hosted production system
or an operational chemical-treatment authority.

## Dashboard Review Surface

The fixed synthetic dashboard presents six views of closed aggregate metrics:

- Request volume by outcome
- Average response latency by retrieval mode
- Response latency minimum and maximum
- Outcome mix
- Retrieval mode volume
- Helpful rate and feedback volume

One invocation on a fresh local database writes six request events and two
feedback events. The synthetic seed is additive if it is rerun, so the canonical
counts apply to a fresh reviewer setup; repeated local runs intentionally show
the cumulative aggregate instead.
