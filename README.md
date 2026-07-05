# Oilfield Chemical Troubleshooting Copilot

Scaffold for an LLM Zoomcamp 2026 capstone project: an end-to-end RAG and tool-calling assistant for oilfield production-chemistry troubleshooting.

This first pass creates the repository shape, local app shell, data-mode conventions, database schema scaffold, Docker Compose services, Kestra ingestion flow placeholder, monitoring plan, and evaluation script placeholders. It does not implement the full RAG pipeline yet.

## Planned Capabilities

- Ingest PDFs, DOCX, XLSX/CSV, and nested folders.
- Support `data/sample` for public sample data and `data/private` for a full private corpus.
- Chunk documents with `source_file`, `topic`, `page_or_sheet`, and `chunk_id` metadata.
- Store chunks and embeddings in PostgreSQL with PGVector.
- Provide hybrid retrieval with keyword search plus vector search.
- Expose tool-calling helpers for chemical dosage calculations and water-analysis interpretation.
- Serve a Streamlit chat UI with source citations and feedback controls.
- Log conversations, feedback, latency, retrieved chunks, and tool calls.
- Provide evaluation scripts for retrieval quality and LLM answer quality.
- Orchestrate ingestion with Kestra: parse -> chunk -> embed -> load_pgvector.
- Monitor operational metrics with Grafana-compatible dashboards.

## Repo Layout

```text
app/                         Streamlit chat UI scaffold
data/sample/                 Public sample dataset included in the repo
data/private/                Private corpus location, gitignored except .gitkeep
db/migrations/               PostgreSQL + PGVector schema scaffold
eval/                        Retrieval and answer-quality evaluation placeholders
flows/kestra/                Kestra ingestion flow scaffold
monitoring/grafana/          Grafana dashboard/provisioning plan
src/oilfield_chemical_copilot/
  ingest/                    File discovery, parser, and chunking scaffolds
  observability/             Logging contracts for conversations and traces
  retrieval/                 Hybrid retrieval placeholder
  storage/                   PGVector storage placeholder
  tools/                     Tool-calling helper scaffolds
tests/                       Smoke tests for scaffold contracts
```

## Local Run

```powershell
cp .env.example .env
uv sync
uv run streamlit run app/streamlit_app.py
```

With Docker Compose:

```powershell
cp .env.example .env
docker compose up --build
```

Then open:

- Streamlit: http://localhost:8501
- Kestra: http://localhost:8080
- Grafana: http://localhost:3000
- Postgres: `localhost:5432`

## Data Modes

- `data/sample`: public sample files that can be committed.
- `data/private`: full private corpus, excluded from Git by `.gitignore`.

Set `DATA_MODE=sample` or `DATA_MODE=private` in `.env` before running ingestion.

## LLM Zoomcamp 2026 Mapping

- Introduction and environment: Python project managed with `uv`, `.env.example`, and Docker Compose.
- Search and retrieval: planned keyword search with `minsearch` plus vector retrieval through PGVector.
- Vector databases: `db/migrations/0001_oilfield_chemical_copilot_schema.sql` prepares PGVector storage.
- LLM integration: OpenAI API settings are included; answer generation is still a TODO.
- Evaluation: `eval/retrieval_eval.py` and `eval/answer_eval.py` define the planned evaluation entrypoints.
- Monitoring: `monitoring/grafana/README.md` documents the Grafana-compatible dashboard plan.
- Orchestration: `flows/kestra/ingest.yml` sketches parse, chunk, embed, and load steps.
- Capstone deployment: Docker Compose includes app, Postgres/PGVector, Kestra, and Grafana services.

## Capstone Rubric Coverage

- Problem framing: production-chemistry troubleshooting for oilfield operations.
- Data preparation: parser and chunking scaffolds for PDF, DOCX, XLSX, CSV, and nested folders.
- Retrieval quality: hybrid retrieval is planned with source metadata and evaluation scripts.
- LLM answer quality: answer evaluation script placeholder is included.
- Tool use: chemical dosage and water-analysis helper scaffolds are included.
- Monitoring: conversation, latency, retrieval, feedback, and tool-call logging tables are scaffolded.
- Reproducibility: `pyproject.toml`, `.env.example`, Dockerfile, and Docker Compose are included.

## Next Implementation Steps

- Implement parser extraction for PDF, DOCX, spreadsheet, and CSV files.
- Add chunking strategy and embedding generation.
- Implement PGVector loading and hybrid retrieval.
- Wire OpenAI response generation with source-grounded citations.
- Persist Streamlit conversations, feedback, latency, retrieval, and tool-call events.
- Add real sample documents and a small labeled evaluation dataset.
