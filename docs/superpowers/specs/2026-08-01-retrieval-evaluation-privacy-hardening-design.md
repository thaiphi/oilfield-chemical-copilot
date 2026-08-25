# Retrieval Evaluation Privacy Hardening Design

Project: Oilfield Chemical Troubleshooting Copilot — retrieval evaluation privacy hardening

Task class: Design specification

Brief: Record approved Option A for correcting the final-review findings without changing retrieval behavior.

Scope: Public-corpus verification, public/private dataset boundaries, privacy-specific report schemas, reproducibility provenance, exact `k=5`, and README current-state/baseline corrections.

Validation: Requirements are mapped to the 2026-07-31 final-review findings and cross-checked against the current evaluator, evaluation models, tests, `.gitignore`, and README.

Report: This document is the approved implementation contract. It does not authorize implementation, staging, commits, or pushes.

Return: A later implementation role must deliver code, tests, and README changes that satisfy every decision and invariant below.

## Context and Decision

The 2026-07-31 public retrieval baseline completed with perfect scores on the small public set, but final review found five release-blocking gaps: a mixed database could influence public metrics and leak chunk IDs; the promised private path was neither ignored nor aggregate-only; reports lacked reproducibility provenance; the CLI accepted depths above five while failure classification used the requested depth; and README still described implemented work as planned.

Approved Option A hardens the existing evaluator. It does not add a new retrieval algorithm or a second evaluation framework.

## Goals

- Make a public run fail closed when the database is not exactly the known public evaluation corpus.
- Keep private evaluation datasets only under an explicitly ignored directory.
- Make private reports aggregate-only by construction.
- Record enough sanitized provenance to reproduce and interpret a run.
- Fix the evaluation window at exactly five retrieved results.
- Correct README capability, workflow, and baseline statements.

## Non-Goals

- No new dependencies.
- No tool calling.
- No reranking.
- No retrieval tuning or retrieval-algorithm changes.
- No ingestion, indexing, migration, or database mutation.
- No source excerpts, source paths, or answer-quality evaluation.
- No publication or commit of private datasets or generated reports.

## Dataset and Corpus Boundaries

### Public mode

Public mode uses a dataset beneath repository `eval/`, excluding `eval/private/`. The existing `eval/public_retrieval_dataset.jsonl` remains the default.

For this fixed baseline, the sorted unique union of every `expected_chunk_ids` value in the public dataset is the complete public corpus manifest. The database corpus is valid only when its complete chunk-ID set equals that manifest:

- a manifest ID absent from the database is a missing public chunk and rejects the run;
- a database ID absent from the manifest is an unexpected chunk and rejects the run;
- either condition rejects the run before constructing the keyword index, constructing a retrieval pipeline, embedding a query, or running any search.

The rejection message reports counts only. It must not print unexpected IDs because an unexpected ID may be derived from a private source name. The evaluator writes no report for a rejected run.

This exact-set decision intentionally makes corpus expansion explicit. Adding a legitimate public chunk requires a reviewed update to the public evaluation dataset/manifest and a new baseline; silently evaluating against a larger corpus is forbidden.

### Private mode

Private mode is selected explicitly with `--privacy-mode private`. Its dataset must resolve beneath `eval/private/`; symlink or traversal escape is rejected. `.gitignore` ignores `eval/private/**` with no tracked private-data exception. Private data must not be stored in another evaluation directory.

Private mode uses the same input record fields internally—`question_id`, `question`, `expected_chunk_ids`, and `topic`—because scoring requires them. It does not require the private database chunk IDs to equal the expected-ID union; private corpora may contain unlabeled distractors. It records only an irreversible corpus hash and aggregate chunk count.

### Path resolution

Dataset authorization is based on `Path.resolve()` and containment after resolution:

- `public`: beneath `eval/` and not beneath `eval/private/`;
- `private`: beneath `eval/private/` only.

An absolute path, relative traversal, or resolved symlink outside the selected boundary is rejected before reading the file.

## Exact Retrieval Window

The evaluation depth is exactly `k=5` in every mode. The CLI retains `--k` for command compatibility, but argparse permits only the value `5`; the default is `5`. Runtime helpers also reject any value other than five so programmatic callers cannot bypass the CLI rule.

The evaluator stores at most five returned chunk IDs and computes `expected_rank` only within those five. Hit Rate@3, Hit Rate@5, and MRR@5 remain the reported metrics. An expected hit at rank six is a miss, appears as a public failure, and cannot be hidden by requesting a deeper retrieval depth.

## Topic Filter

The current evaluator passes each gold record’s `topic` to every retriever. This is an oracle topic filter, not an inferred production filter. The behavior remains unchanged, and every report must disclose it with the stable value `gold_topic` plus human-readable Markdown explaining that the gold topic was passed to retrieval. Baselines from this evaluator must be described as topic-filtered.

## Sanitized Provenance

Every successful public or private report includes the following provenance and no filesystem paths:

- `dataset_sha256`: lowercase SHA-256 of the exact dataset bytes;
- `corpus_sha256`: lowercase SHA-256 of UTF-8 canonical corpus identity, defined as sorted unique chunk IDs joined by `\n` with a final `\n`;
- `corpus_chunk_count`: number of unique database chunk IDs included in the corpus hash;
- `git_revision`: the exact 40-character lowercase revision from `git rev-parse HEAD`; the run fails rather than writing incomplete provenance if it cannot be resolved;
- embedding `provider`, `model`, and integer `dimension` from resolved embedding settings/provider;
- retrieval `k`, always integer `5`;
- retrieval `topic_filter`, always `gold_topic`;
- `keyword_score_threshold`, always JSON `null` because keyword mode has no score threshold;
- resolved `vector_score_threshold` from `RetrievalSettings.min_score`;
- resolved `hybrid_candidate_limit`;
- resolved `hybrid_rrf_k`;
- resolved `hybrid_score_threshold` from `RetrievalSettings.hybrid_min_rrf_score`.

Hashes identify run inputs without serializing dataset questions, source-derived IDs, filenames, or paths. No timestamp, host name, username, database URL, environment-variable dump, or output path is included.

## Report Contracts

Both JSON variants use schema version `1`, UTF-8, a final newline, sorted JSON object keys, stable retrieval-mode order (`keyword`, `vector`, `hybrid` when selected), six decimal places for metric values, and three decimal places for median latency. A repeated render from identical in-memory inputs produces byte-identical output.

### Public JSON

```json
{
  "schema_version": 1,
  "privacy_mode": "public",
  "provenance": {
    "dataset_sha256": "64 lowercase hexadecimal characters",
    "corpus_sha256": "64 lowercase hexadecimal characters",
    "corpus_chunk_count": 6,
    "git_revision": "40 lowercase hexadecimal characters",
    "embedding": {
      "provider": "ollama",
      "model": "granite-embedding:latest",
      "dimension": 384
    },
    "retrieval": {
      "k": 5,
      "topic_filter": "gold_topic",
      "keyword_score_threshold": null,
      "vector_score_threshold": 0.2,
      "hybrid_candidate_limit": 10,
      "hybrid_rrf_k": 60,
      "hybrid_score_threshold": 0.015
    }
  },
  "modes": {
    "keyword": {
      "questions": 18,
      "hit_rate_at_3": 1.0,
      "hit_rate_at_5": 1.0,
      "mrr_at_5": 1.0,
      "median_latency_ms": 0.0,
      "failures": []
    }
  }
}
```

Each selected public mode has exactly `questions`, `hit_rate_at_3`, `hit_rate_at_5`, `mrr_at_5`, `median_latency_ms`, and `failures`. A failure has exactly `question_id`, `topic`, `expected_rank`, and `returned_chunk_ids`. Public reports may contain these public identifiers, but never question text.

### Private JSON

```json
{
  "schema_version": 1,
  "privacy_mode": "private",
  "provenance": {
    "dataset_sha256": "64 lowercase hexadecimal characters",
    "corpus_sha256": "64 lowercase hexadecimal characters",
    "corpus_chunk_count": 100,
    "git_revision": "40 lowercase hexadecimal characters",
    "embedding": {
      "provider": "ollama",
      "model": "granite-embedding:latest",
      "dimension": 384
    },
    "retrieval": {
      "k": 5,
      "topic_filter": "gold_topic",
      "keyword_score_threshold": null,
      "vector_score_threshold": 0.2,
      "hybrid_candidate_limit": 10,
      "hybrid_rrf_k": 60,
      "hybrid_score_threshold": 0.015
    }
  },
  "modes": {
    "keyword": {
      "questions": 18,
      "hit_rate_at_3": 0.5,
      "hit_rate_at_5": 0.6,
      "mrr_at_5": 0.4,
      "median_latency_ms": 0.0
    }
  }
}
```

Each selected private mode has exactly the five aggregate fields shown. Private JSON has no `failures` key and no per-question collection.

### Markdown

Public and private Markdown contain the sanitized provenance fields and one aggregate mode table. Public Markdown may add a failure table containing public question IDs, topics, expected rank, and returned public chunk IDs. Private Markdown contains the sentence `Per-question details are suppressed in private mode.` and no failure table.

## No-Private-Data Rules

These are hard invariants, not best-effort redaction:

- No report in any mode contains question text, hit text, excerpts, `source_file`, `source_path`, dataset path, output path, database URL, or absolute/local path.
- Public reports may include only public question IDs, public topics, public chunk IDs, and aggregate/provenance fields allowed by the schema.
- Private reports contain no question IDs, topics, expected or returned chunk IDs, text, filenames, paths, per-question ranks, or failure rows.
- Errors from public mixed-corpus checks and private runs contain counts and fixed labels only; they do not echo private/source-derived values or paths.
- Retrieval hits are never serialized directly.
- Private dataset bytes remain under ignored `eval/private/`; generated reports remain under already ignored `data/processed/`.
- README and other tracked documentation never copy private values or private report contents.

## CLI Contract

Public run:

```powershell
uv run python eval/retrieval_eval.py --privacy-mode public --dataset eval/public_retrieval_dataset.jsonl --output-dir data/processed/evaluation/public --modes keyword,vector,hybrid --k 5
```

Private run:

```powershell
uv run python eval/retrieval_eval.py --privacy-mode private --dataset eval/private/retrieval_dataset.jsonl --output-dir data/processed/evaluation/private --modes keyword,vector,hybrid --k 5
```

`--privacy-mode` has choices `public` and `private` and defaults to `public`. The private command is documentation only; no private dataset is created or committed.

## README Corrections

README must:

- move hybrid retrieval and retrieval evaluation from planned to implemented/current state;
- describe the RAG pipeline as hybrid by default, with vector as an available comparison mode;
- describe `eval/retrieval_eval.py` as implemented, not planned;
- remove hybrid fusion and creation of a small labeled retrieval dataset from next steps;
- document both explicit privacy modes and the ignored `eval/private/` boundary;
- state that public mode rejects a database containing any unexpected chunk ID before retrieval;
- state that private reports are aggregate-only and list the prohibited fields;
- disclose the gold-topic oracle filter and exact `k=5`;
- record the current 18-question public baseline: keyword, vector, and hybrid each recorded Hit Rate@3 `1.000`, Hit Rate@5 `1.000`, and MRR@5 `1.000`, with no observed failures;
- state that this perfect, small, topic-filtered public baseline does not justify a retrieval change and does not establish chemistry truth, private-corpus performance, or production readiness.

## Decision Boundaries

- A mixed public/private database is not filtered and scored; it is rejected. Filtering could conceal contamination and make the corpus hash misleading.
- Public corpus identity is the exact expected-ID union for this fixed baseline, not every row that happens to be in the configured database.
- Private mode does not publish failure categories because categories would require per-question topic or identifier disclosure. Diagnosis stays local and outside generated reports.
- Provenance reports resolved settings; it does not freeze RRF/threshold values beyond the exact `k=5`. Comparisons are valid only when reported provenance is considered.
- The gold-topic filter remains for baseline continuity but must be disclosed. Removing it is a separate approved evaluation design.
- Existing retrieval, embedding, RRF, threshold, and topic-filter behavior remains unchanged.

## Acceptance Criteria

- A public database with one extra chunk ID is rejected before any search, and the ID is absent from stdout/stderr and files.
- A public database missing one manifest ID is rejected before any search and before report creation.
- `--k 4`, `--k 6`, and programmatic `k != 5` are rejected; a rank-six hit is an @5 miss/public failure.
- Public and private datasets cannot cross their resolved path boundaries.
- `eval/private/**` is ignored.
- Public reports follow the public schema; private reports follow the aggregate-only schema.
- Both reports contain complete sanitized provenance and topic-filter disclosure.
- Sentinel private question IDs, topics, chunk IDs, text, and paths are absent from private JSON and Markdown.
- Sentinel source text and paths are absent from public JSON and Markdown.
- README accurately reflects current implementation and the baseline interpretation.
- Targeted tests, full pytest, and scoped Ruff validation pass without adding dependencies.
