# Oilfield Chemical Troubleshooting Copilot

LLM Zoomcamp 2026 capstone project for an oilfield production-chemistry troubleshooting RAG assistant. The repository now includes the local data inventory, sample parser/chunker, deterministic/local embedding providers, keyword retrieval, and PGVector storage/search needed for Milestones 1 through 3.

## Implemented Capabilities

- Inventory PDFs, DOCX, XLSX/CSV, text, Markdown, and nested folders without reading private file contents.
- Support `data/sample` for public sample data and `data/private` for a full private corpus.
- Parse sample Markdown, text, CSV, XLSX, DOCX, and PDF files into deterministic chunks.
- Write `chunks.jsonl` with `source_file`, `source_path`, `topic`, `parser_type`, `page_or_sheet`, `chunk_index`, and `chunk_id` metadata.
- Generate deterministic test embeddings or local sentence-transformer embeddings.
- Store chunks and 384-dimensional embeddings in PostgreSQL with PGVector.
- Run keyword search with `minsearch` and vector search through PGVector.

## Planned Capabilities

- Fuse keyword and vector results into hybrid retrieval.
- Add source-grounded answer generation and citations.
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
data/processed/              Generated inventory and chunk reports, kept local
ingestion/inventory.py       Metadata-only recursive inventory CLI
ingestion/ingest.py          Sample parser and chunking CLI
ingestion/index_chunks.py    Chunk embedding and PGVector indexing CLI
ingestion/apply_migrations.py SQL migration runner for existing databases
db/migrations/               PostgreSQL + PGVector schema migrations
eval/                        Retrieval and answer-quality evaluation placeholders
flows/kestra/                Kestra ingestion flow scaffold
monitoring/grafana/          Grafana dashboard/provisioning plan
src/oilfield_chemical_copilot/
  ingest/                    File discovery, parsing, and chunking helpers
  observability/             Logging contracts for conversations and traces
  retrieval/                 Embedding, keyword, vector, and hit models
  storage/                   PGVector storage and search helpers
  tools/                     Tool-calling helper scaffolds
tests/                       Unit and opt-in integration tests
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

## Corpus Inventory

Milestone 1 inventories file metadata only. It recursively records paths, sizes, MIME guesses, topic and parser classifications, ingestion priority, and OCR candidates. It does not parse file contents, create embeddings, load a database, or copy source files.

Public sample data:

```powershell
uv run python ingestion/inventory.py --data-dir data/sample --output-dir data/processed --max-files 20
```

Private files kept under the local repository:

```powershell
uv run python ingestion/inventory.py --data-dir data/private --output-dir data/processed --summary-only
```

An external mounted Google Drive folder:

```powershell
uv run python ingestion/inventory.py --data-dir "G:\My Drive\Operational Challenges Chenical_Electronic Handouts" --output-dir data/processed --summary-only
```

The command creates `data/processed/inventory.csv` and `data/processed/inventory_summary.md`. These generated reports are gitignored because private inventories can expose file names and absolute local paths.

## Sample Parsing and Chunking

Milestone 2 parses supported sample files and writes metadata-rich chunks for retrieval work. It supports Markdown, text, CSV, XLSX, DOCX, and PDF files. This step does not create embeddings, load PGVector, call OpenAI, or run the RAG application.

```powershell
uv run python ingestion/ingest.py --data-dir data/sample --output-dir data/processed --max-files 20
```

The command creates `data/processed/chunks.jsonl`. The file is gitignored because private runs can include source names, local paths, and extracted text from private documents.

## Retrieval Indexing and Search

Milestone 3 adds local embedding providers, keyword search, and PGVector loading/search. The default indexing provider is deterministic so tests and dry runs do not download models or call APIs. For local semantic embeddings, use `--embedding-provider sentence-transformers`, which uses `SENTENCE_TRANSFORMERS_MODEL` and stores 384-dimensional vectors by default.

Dry run without database writes:

```powershell
uv run python ingestion/index_chunks.py --input data/processed/chunks.jsonl --dry-run
```

Start PGVector, apply migrations, and index sample chunks:

```powershell
docker compose up -d postgres
docker compose run --rm migrate
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
uv run python ingestion/index_chunks.py --input data/processed/chunks.jsonl --database-url $env:DATABASE_URL
```

`docker compose run --rm migrate` applies all SQL migrations to existing volumes. Milestone 3 changes the embedding column to `vector(384)` and refuses to run when `chunks` already contains rows. For an old local volume with disposable sample data, reset the Docker volume and re-index from `chunks.jsonl`; for any non-disposable database, back it up and plan a controlled re-embedding migration.

Run the live PGVector integration test only when Docker Postgres is available. It defaults to a disposable `oilfield_copilot_test` database and rejects database names that do not end in `_test` before truncating test rows.

```powershell
$env:RUN_PGVECTOR_INTEGRATION = "1"
$env:TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot_test"
uv run pytest -m integration tests/storage/test_pgvector_integration.py
```

## LLM Zoomcamp 2026 Mapping

- Introduction and environment: Python project managed with `uv`, `.env.example`, `uv.lock`, and Docker Compose.
- Search and retrieval: implemented keyword search with `minsearch`; vector retrieval uses PGVector and filters by embedding model.
- Vector databases: PGVector migrations create durable chunk and 384-dimensional embedding storage.
- LLM integration: OpenAI API settings are included; answer generation is still a TODO.
- Evaluation: `eval/retrieval_eval.py` and `eval/answer_eval.py` define the planned evaluation entrypoints.
- Monitoring: `monitoring/grafana/README.md` documents the Grafana-compatible dashboard plan.
- Orchestration: `flows/kestra/ingest.yml` sketches parse, chunk, embed, and load steps.
- Capstone deployment: Docker Compose includes app, migration, Postgres/PGVector, Kestra, and Grafana services.

## Capstone Rubric Coverage

- Problem framing: production-chemistry troubleshooting for oilfield operations.
- Data preparation: inventory plus parser/chunker coverage for PDF, DOCX, XLSX, CSV, text, Markdown, and nested folders.
- Retrieval quality: keyword and vector retrieval primitives are implemented with source metadata and test coverage.
- LLM answer quality: answer evaluation script placeholder is included.
- Tool use: chemical dosage and water-analysis helper scaffolds are included.
- Monitoring: conversation, latency, retrieval, feedback, and tool-call logging tables are scaffolded.
- Reproducibility: `pyproject.toml`, `uv.lock`, `.env.example`, Dockerfile, and Docker Compose are included.

## Next Implementation Steps

- Fuse keyword and vector results with a tested hybrid ranking strategy.
- Wire source-grounded OpenAI answer generation with citations.
- Persist Streamlit conversations, feedback, latency, retrieval, and tool-call events.
- Add a small labeled retrieval and answer-quality evaluation dataset.