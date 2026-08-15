# Oilfield Chemical Troubleshooting Copilot

LLM Zoomcamp 2026 capstone project for an oilfield production-chemistry troubleshooting RAG assistant. The repository now includes the local data inventory, sample parser/chunker, deterministic/local embedding providers, keyword, vector, and hybrid retrieval, PGVector storage/search, a retrieval evaluator, and a basic source-grounded RAG app needed for Milestones 1 through 4.

## Implemented Capabilities

- Inventory PDFs, DOCX, XLSX/CSV, text, Markdown, and nested folders without reading private file contents.
- Support `data/sample` for public sample data and `data/private` for a full private corpus.
- Parse sample Markdown, text, CSV, XLSX, DOCX, and PDF files into deterministic chunks.
- Write `chunks.jsonl` with `source_file`, `source_path`, `topic`, `parser_type`, `page_or_sheet`, `chunk_index`, and `chunk_id` metadata.
- Generate deterministic test embeddings or local sentence-transformer embeddings.
- Store chunks and 384-dimensional embeddings in PostgreSQL with PGVector.
- Run keyword search with `minsearch`, vector search through PGVector, and RRF-based hybrid retrieval.
- Evaluate keyword, vector, and hybrid retrieval with fixed `k=5`, provenance, and public/private privacy boundaries.
- Ask questions in a Streamlit RAG app with a claim-scope gate before retrieval. Closed-scope questions return a safe response without retrieval or generation; general-review questions retrieve source chunks, call the configured provider (defaulting to local Ollama), and return cited answers or a weak-evidence fallback.

## Planned Capabilities

- Expose tool-calling helpers for chemical dosage calculations and water-analysis interpretation.
- Log conversations, feedback, latency, retrieved chunks, and tool calls.

- Orchestrate public-sample ingestion with Kestra: inventory -> parse/chunk -> embed/load -> validate -> publish aggregate metrics.
- Monitor operational metrics with Grafana-compatible dashboards.

## Repo Layout

```text
app/                         Streamlit RAG chat UI
data/sample/                 Public sample dataset included in the repo
data/private/                Private corpus location, gitignored except .gitkeep
data/processed/              Generated inventory and chunk reports, kept local
ingestion/inventory.py       Metadata-only recursive inventory CLI
ingestion/ingest.py          Sample parser and chunking CLI
ingestion/index_chunks.py    Chunk embedding and PGVector indexing CLI
ingestion/apply_migrations.py SQL migration runner for existing databases
db/migrations/               PostgreSQL + PGVector schema migrations
eval/                        Retrieval evaluator and public evaluation dataset
flows/kestra/                Public-sample Kestra ingestion flow
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

For a fresh database, run parsing and indexing before expecting cited answers:

```powershell
uv run python ingestion/ingest.py --data-dir data/sample --output-dir data/processed --max-files 20
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
uv run python ingestion/index_chunks.py --input data/processed/chunks.jsonl --database-url $env:DATABASE_URL
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

Learning note: Milestone 4 defaults to local Ollama embeddings using `granite-embedding:latest`. Use `EMBEDDING_PROVIDER=deterministic` for reproducible offline tests or dry runs, or `EMBEDDING_PROVIDER=sentence-transformers` for local sentence-transformer embeddings.

## Retrieval Indexing and Search

Milestone 3 adds local embedding providers, keyword search, and PGVector loading/search. The default indexing provider is local Ollama using `granite-embedding:latest`. Select `--embedding-provider deterministic` for offline tests and dry runs, or `--embedding-provider sentence-transformers` for the configured sentence-transformer model.

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

## Kestra Public-Sample Orchestration

Kestra runs the same public ingestion commands as five explicit task boundaries: inventory, parse/chunk, embed/load, count validation, and aggregate-metrics publication. It is an ingestion coordinator, not part of the question-answering path.

Start the local prerequisites and open Kestra at http://localhost:8080:

```powershell
docker build --tag oilfield-chemical-copilot:local .
docker compose up -d postgres kestra
```

Register and execute `flows/kestra/ingest.yml` in the local Kestra UI. The flow embeds only `data/sample`, validates the indexed count against its generated public chunk manifest, and uses `granite-embedding:latest`. Its final dlt task appends only six aggregate fields to `orchestration.ingestion_runs`: status, source-file count, chunk counts, and embedding-model label.

Private source material is outside this flow and is excluded from the worker-image build context. Do not substitute a private data directory or place private files in Kestra artifacts.

Run the live PGVector integration test only when Docker Postgres is available. It defaults to a disposable `oilfield_copilot_test` database and rejects database names that do not end in `_test` before truncating test rows.

```powershell
$env:RUN_PGVECTOR_INTEGRATION = "1"
$env:TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot_test"
uv run pytest -m integration tests/storage/test_pgvector_integration.py
```


## Basic RAG App

Milestone 4 wires the Module 1 RAG loop into Streamlit:

```text
question -> explicit product-dose contract -> claim-scope gate -> validated calculation OR general RAG -> retrieval -> bounded evidence prompt -> Ollama structured draft -> deterministic answer with citations
```

The app uses only retrieved source chunks for citations. It hides absolute `source_path` values from user-visible answers and shows source filename, page/sheet, chunk ID, score, and a bounded excerpt. The claim-scope gate runs before retrieval and before an explicit product-dose calculation: it returns a scope-limited response with no calculator, retriever, or generator call for site-specific determinations, field-ready prescriptions, and attempts to replace a complete analysis. If a general-review question has no retrieved chunk above `HYBRID_MIN_RRF_SCORE` in hybrid mode or `RAG_MIN_SCORE` in vector mode, the app does not call the answer provider and returns:

```text
I do not have enough retrieved evidence to answer confidently.
```

Ollama is the default local answer provider. Start the local service and download the configured answer and embedding models:

```powershell
ollama serve
ollama pull granite4.1:8b
ollama pull granite-embedding:latest
```

In a second terminal, parse the public sample corpus and re-index it with Ollama embeddings:

```powershell
uv run python ingestion/ingest.py --data-dir data/sample --output-dir data/processed --max-files 20
docker compose up -d postgres
docker compose run --rm migrate
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
uv run python ingestion/index_chunks.py --input data/processed/chunks.jsonl --database-url $env:DATABASE_URL
```

Re-index after changing embedding providers or models: PGVector search is model-labelled, so it returns rows only for the query embedding's model label.

Then launch Streamlit:

```powershell
cp .env.example .env
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
$env:LLM_PROVIDER = "ollama"
uv run streamlit run app/streamlit_app.py
```

To use OpenAI instead, set `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY` (optionally set `OPENAI_MODEL`). Warning: selecting OpenAI sends retrieved source excerpts, including private corpus content, to OpenAI.

Run the complete local Ollama smoke test only after Docker Postgres and Ollama are reachable:

```powershell
$env:RUN_OLLAMA_INTEGRATION = "1"
$env:TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot_test"
uv run pytest -m integration tests/rag/test_ollama_integration.py -v
```

Module 1 learning note: hybrid retrieval combines lexical precision for exact identifiers such as a chemical or product name with semantic recall for related language. Reciprocal rank fusion (RRF) uses `1 / (60 + keyword rank) + 1 / (60 + vector rank)`, where each rank starts at one. This is a ranking score, not a percentage or cosine similarity. Set `RETRIEVAL_MODE=vector` to compare vector-only retrieval; `RAG_MIN_SCORE` applies only in that mode.

## Retrieval Evaluation

The retrieval evaluator compares keyword, vector, and hybrid modes at an exact fixed `k=5`. The public evaluation set measures retrieval only: it does not establish chemistry truth or production readiness.

Public mode requires a database containing **only the complete derived `data/sample` chunk manifest**. With Docker PGVector running and that public-only sample index available, run:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
uv run python eval/retrieval_eval.py --privacy-mode public --dataset eval/public_retrieval_dataset.jsonl --output-dir data/processed/evaluation/public --modes keyword,vector,hybrid --k 5
```

Public mode applies the `oracle_gold_topic` filter. It writes local, gitignored reports with sanitized provenance and, for public failures only, public question and topic identifiers with rank. The reports must not contain absolute paths.

Private datasets belong under the gitignored `eval/private/` directory. Run private evaluation explicitly with `--privacy-mode private` and a private dataset path. Private reports are aggregate-only: they must not include question IDs, topics, chunk IDs, text, or paths.

Hit Rate@k is the share of evaluation questions whose expected evidence appears in the first *k* retrieved chunks. MRR is the average reciprocal rank of the first expected evidence chunk, so it rewards placing evidence earlier in the results.

Current public baseline (18 questions): keyword, vector, and hybrid each achieved Hit Rate@3 `1.000`, Hit Rate@5 `1.000`, and MRR@5 `1.000`, with no observed failures. This perfect, small, topic-filtered baseline does not justify a retrieval change and does not establish chemistry truth, private-corpus performance, or production readiness.

Learning checkpoint: high Hit Rate with low MRR means the evidence was found but ranked poorly, and MRR exposes that weakness.
## Grounded Answer Evaluation

Module 2 evaluates public synthetic answers in two layers. Deterministic checks verify that expected citations are present and allowed, and that insufficient evidence causes abstention. The structured judge then rates groundedness, relevance, limitation awareness, and operational certainty from 1 to 5. The first layer is a contract check; the second is advisory quality evidence, not chemistry validation.

The judge uses local Ollama with `granite4.1:8b` by default. It sends structured JSON requests at temperature `0`. Set `ANSWER_EVAL_JUDGE_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `ANSWER_EVAL_OPENAI_MODEL` to use OpenAI instead; OpenAI receives the public fixture's answer and evidence text. Reports retain public case IDs, deterministic status counts, judge status counts, provider labels, hashed model labels, and aggregate scores only. They never serialize questions, answers, evidence excerpts, source IDs, paths, credentials, or raw provider errors.

Run the public evaluator after local Ollama is reachable:

```powershell
$env:ANSWER_EVAL_JUDGE_PROVIDER = "ollama"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "granite4.1:8b"
uv run python eval/answer_eval.py --dataset eval/public_answer_evaluation.jsonl --answers eval/public_generated_answers.jsonl --output-dir data/processed/evaluation/answers/public
```

The committed answer fixture is synthetic. Its baseline verifies evaluator wiring and safe reporting, not the live RAG application's answer quality. A judge can also be biased, particularly when it is similar to the model that generated an answer. Citation validity proves only structural grounding, not chemical correctness or operational safety.

## Live RAG Answer Comparison

The live comparison evaluates the actual `BasicRagService` in `vector` and `hybrid` modes against the same public questions. Unlike the synthetic answer fixture above, it calls the local RAG path, captures each generated draft and its cited public evidence only in memory, and writes aggregate-only results. It does not use the fixture's prewritten answers or evidence.

This is intentionally an end-to-end retrieval comparison, so `service.answer(question)` is called without an `oracle_gold_topic` filter. The evaluator must measure what the live service retrieves from the public-only index; an oracle topic would make that comparison less representative. The run is a small public baseline only. It does not establish chemistry correctness, operational readiness, or a winning retrieval mode.

Before running it, make sure Docker/Postgres and local Ollama are available, pull both required models, and rebuild a public-only sample index with the Ollama embedding model:

```powershell
# Terminal 1: leave this running if Ollama is not already running as a service.
ollama serve

# Terminal 2
ollama pull granite4.1:8b
ollama pull granite-embedding:latest
docker compose up -d postgres
docker compose run --rm migrate
uv run python ingestion/ingest.py --data-dir data/sample --output-dir data/processed --max-files 20
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
$env:EMBEDDING_PROVIDER = "ollama"
$env:OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"
uv run python ingestion/index_chunks.py --input data/processed/chunks.jsonl --database-url $env:DATABASE_URL
```

The runner rejects mixed or incomplete databases. With those prerequisites satisfied, use this exact command:

```powershell
$env:LLM_PROVIDER = "ollama"
$env:ANSWER_EVAL_JUDGE_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "granite4.1:8b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:EMBEDDING_PROVIDER = "ollama"
$env:OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"
uv run python eval/live_rag_answer_eval.py --dataset eval/public_answer_evaluation.jsonl --output-dir data/processed/evaluation/live_rag --database-url $env:DATABASE_URL
```

The aggregate-only comparison report records the `vector`/`hybrid` relationship and the numeric `top_k`, `min_score`, `max_context_chars`, `hybrid_candidate_limit`, `hybrid_rrf_k`, and `hybrid_min_rrf_score` settings used for the run. It does not record database or Ollama URLs.

Completed small public baseline: both `vector` and `hybrid` ran 12 questions. In each mode, deterministic citations had 4 passes and 8 failures, abstention had 6 passes and 6 failures, and all 12 judge results were available. Judge aggregates were:

| Mode | Groundedness | Relevance | Limitation awareness | Operational certainty |
| --- | ---: | ---: | ---: | ---: |
| Vector | 3.9166666666666665 | 4.833333333333333 | 3.9166666666666665 | 3.0833333333333335 |
| Hybrid | 3.75 | 5 | 3.75 | 3.0 |

This is a small public baseline, not a winner selection, chemistry validation, or production-readiness claim.

### Approved Live Failure Diagnosis

The approved aggregate diagnosis reproduced the baseline. `vector` and `hybrid` each ran 12 questions and had category-identical failures: citation failures were `expected_citation_allowed_retrieved_not_cited` (1), `expected_citation_mixed_with_disallowed` (1), and `unexpected_citation_when_abstention_expected` (6); abstention failures were `under_abstention_answered_on_insufficient_case` (6). These aggregates do not select a retrieval winner or establish a retrieval cause.

### Claim-Scope Policy Investigation

The approved local, public policy run reproduced the control baseline in both modes and evaluated `claim_scope v1` in shadow mode. Each mode had 12 paired questions and the same six allow/six abstain decisions. Control metrics remained citation 4 pass/8 fail and abstention 6 pass/6 fail. Shadow metrics were citation 10 pass/2 fail and abstention 12 pass/0 fail.

The policy subsequently passed a sealed local 36-case v2 holdout with 36/36 exact action and category matches, zero false allows, zero false abstains, and zero stratum failures. It now runs before RAG in the production service. This is bounded evaluation evidence only: it does not establish chemistry correctness, operational safety, generalization, private-corpus quality, or a retrieval-mode winner.

### Citation Selection

The separate citation-selection investigation found that the two remaining allowed-case failures were answer-path selection failures, not retrieval failures: the allowed evidence had already been retrieved. The answer formatter now requires the question to match a source filename or declared topic before answer-content overlap can select that source. This prevents a broad `README` or water-analysis source from winning merely because a generated answer contains general oilfield terms.

A new local ID-only diagnostic capture reran the six evidence-sufficient public cases with the unchanged public corpus, retrieval settings, Granite model, and vector/hybrid modes. Both modes retrieved allowed evidence for all six cases and produced allowed-only citations for all six. The evaluator intentionally bypasses the production claim-scope gate to retain the historical baseline, so its six overclaim cases remain unchanged in that capture. The diagnostic is Git-excluded and contains no answer or source text. This is citation-structure evidence only, not chemistry validation or production readiness.

### Chemical-Dose Tool Boundary

The first tool-calling milestone is an allowlisted deterministic contract, `chemical_dosage.product_ppm_water_basis` version `v1`. An explicit chat request must use `Product dose:` and provide `water_bbl_per_day` and `product_ppm`. It computes `product_ppm * water_bbl_per_day * 42 / 1,000,000` as product gallons per day. The calculator rejects invalid values instead of clamping or guessing, and results are labeled as general calculations rather than field-ready prescriptions.

The app classifies a recognized tool request before parsing or calculation. Closed claims invoke no calculator, retriever, or generator; valid general requests invoke the calculator without RAG; all other questions retain the normal RAG route. The tool does not support active-ingredient ppm, active fraction, water analysis, arbitrary model-selected functions, or model-generated executable arguments.

### Aggregate-Safe Monitoring

Module 1 now includes a process-local aggregate monitor for six closed outcomes: successful and weak-evidence RAG responses, claim-scope abstentions, valid and invalid product-dose routes, and RAG configuration failures. It stores only outcome counts plus count/minimum/average/maximum response latency. The monitor has no event list or payload API and cannot retain prompts, answers, excerpts, source paths, tool inputs, identifiers, or raw error text. It is intentionally in-memory only; it does not write to the existing database logging tables because those tables permit raw content.
## LLM Zoomcamp 2026 Mapping

- Introduction and environment: Python project managed with `uv`, `.env.example`, `uv.lock`, and Docker Compose.
- Search and retrieval: implemented keyword search with `minsearch`; vector retrieval uses PGVector and filters by embedding model.
- Vector databases: PGVector migrations create durable chunk and 384-dimensional embedding storage.
- LLM integration: OpenAI Responses API adapter generates structured drafts for source-grounded answers.
- Evaluation: `eval/retrieval_eval.py` evaluates keyword, vector, and hybrid retrieval at fixed `k=5` with public/private report boundaries; `eval/answer_eval.py` completes synthetic-answer contract and judge evaluation; `eval/live_rag_answer_eval.py` completed a small public vector-versus-hybrid baseline without an oracle topic filter.
- Monitoring: `monitoring/grafana/README.md` documents the Grafana-compatible dashboard plan.
- Orchestration: `flows/kestra/ingest.yml` sketches parse, chunk, embed, and load steps.
- Capstone deployment: Docker Compose includes app, migration, Postgres/PGVector, Kestra, and Grafana services.

## Capstone Rubric Coverage

- Problem framing: production-chemistry troubleshooting for oilfield operations.
- Data preparation: inventory plus parser/chunker coverage for PDF, DOCX, XLSX, CSV, text, Markdown, and nested folders.
- Retrieval quality: keyword, vector, and hybrid retrieval are implemented with source metadata, test coverage, and a privacy-hardened retrieval evaluator.
- LLM answer quality: public synthetic deterministic and structured-judge answer evaluation is complete; the live public vector-versus-hybrid baseline is complete; neither evaluation is chemistry validation or production readiness.
- Tool use: a scope-gated, deterministic product-ppm water-basis dosage calculator is available through the sidebar and an explicit chat contract; water-analysis tooling remains deferred.
- Monitoring: aggregate-safe, process-local response and routing signals are implemented; raw-content database logging remains unused by the runtime.
- Reproducibility: `pyproject.toml`, `uv.lock`, `.env.example`, Dockerfile, and Docker Compose are included.

## Next Implementation Steps

### Immediate Quality Task

- Review the completed Module 1 boundary as a whole: claim-scope abstention, citation selection, deterministic dosage routing, and aggregate-safe monitoring. Do not add persistence or dashboards until a separate data-retention design is approved.

### Later Deferred Branch

- Monitor citation selection on larger approved public and private corpora before considering retrieval changes; keep retrieval frozen unless new evidence identifies a retrieval gap.
