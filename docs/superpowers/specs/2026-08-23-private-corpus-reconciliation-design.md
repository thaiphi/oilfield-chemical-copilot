# Private Corpus Reconciliation Design

**Date:** 2026-08-23

**Status:** Approved workflow; written design awaiting final review

**Parent experiment:** E1a-4 Requirements-Aware Evidence Gate

## Goal

Build a resumable, metadata-only reconciliation that maps the approved Google
Drive oilfield corpus to local files, parsed documents, the verified private
PostgreSQL/PGVector index, evaluation topic and source-role assignments, and
unused evidence locators. The immediate decision is why the E1a-4 allocator
returns `E1A4_ALLOCATION_UNAVAILABLE` even though the approved index contains
4,797 chunks from 198 sources.

The reconciliation must determine where capacity is lost before any download,
ingestion, reindexing, topic or role change, freshness exception, or E1a-4
population authoring is proposed.

## Non-Goals

- Do not download, parse, ingest, embed, reindex, retrieve, rerank, or call a
  model during reconciliation.
- Do not inspect or persist PDF text, extracted passages, tables, images,
  questions, canonical claims, support labels, or answers.
- Do not modify the current corpus, PostgreSQL rows, PGVector embeddings,
  E1a-3 artifacts, production retrieval, or E1a-4 contracts.
- Do not add Qdrant or another vector store. Exact reconciliation uses stable
  identifiers, cryptographic hashes, provenance, and transactional state.
- Do not infer that additional PDFs are required merely because allocation is
  unavailable.

## Existing Evidence

- The approved private index contract covers 4,797 chunks from 198 sources.
- E1a-3 sealed a 96-case population with 48 foundational and 48 supporting
  allocations after excluding title-only records and correcting foundational
  locator eligibility.
- E1a-4 requires an exact 96-slot grid: four topics, two source roles, three
  question forms, two evidence depths, and two replicates.
- Each topic and source-role cell therefore requires 12 fresh substantive
  locators.
- The E1a-4 allocator currently fails after mandatory E1a-3 locator exclusion,
  and no E1a-4 sampling-frame file exists.
- Google Drive metadata shows broad oilfield-document coverage and duplicate
  titles, but Drive inventory has not been reconciled to the 198 indexed
  sources.

## Approved Corpus Boundary

The current 198 indexed sources are trusted anchors, not proof that every Drive
file is approved or indexed.

1. Match each indexed anchor to Google Drive using exact identity evidence.
2. Include the Drive parent folders containing confirmed anchor matches as
   discovery scope.
3. Treat other files in those folders as reconciliation candidates only.
4. Search outside confirmed folders only for the four approved E1a-4 topics.
5. Never download, ingest, or mark an unmatched file eligible automatically.
6. Require an explicit review decision before adding a new folder or source
   category to the approved corpus.
7. Exclude unrelated personal, operational, commercial, and equipment material.

## Architecture

### Authoritative Working State

Use a private local SQLite database as the resumable system of record:

```text
.private/corpus-reconciliation/v1/reconciliation.sqlite
```

Configure SQLite with foreign keys enabled, write-ahead logging, and full
synchronous commits. Every stage writes small idempotent transactions. A
process or connection failure may leave incomplete records, but must not lose
previously committed work or convert incomplete work into success.

PostgreSQL/PGVector and Google Drive are read-only sources. Neither is the
reconciliation system of record.

### Portable Audit Snapshots

Successful checkpoints export canonical JSONL files:

```text
.private/corpus-reconciliation/v1/snapshots/drive-inventory.jsonl
.private/corpus-reconciliation/v1/snapshots/local-inventory.jsonl
.private/corpus-reconciliation/v1/snapshots/index-inventory.jsonl
.private/corpus-reconciliation/v1/snapshots/document-matches.jsonl
.private/corpus-reconciliation/v1/snapshots/review-decisions.jsonl
.private/corpus-reconciliation/v1/snapshots/locator-capacity.jsonl
.private/corpus-reconciliation/v1/snapshots/snapshot-binding.json
```

Each export is written to a same-directory temporary file, flushed, closed,
validated, and atomically replaced. Every snapshot receives a SHA-256 manifest
beside it under `.private/corpus-reconciliation/v1/snapshots/`. The binding
artifact seals the run ID, schema version, index-contract digest, E1a-3
allocation digest, and a digest over all six canonical snapshot payloads. A
snapshot set is valid only when all six JSONL files, the binding artifact, and
all seven manifests exist, validate, and match the active SQLite run's current
canonical state.

### Tracked Public Output

Tracked reports may contain only aggregate counts and closed statuses:

- Drive candidates, exact matches, duplicates, Drive-only, local-only,
  parsed-not-indexed, index-only, ambiguous, ineligible, and complete.
- Whether each public topic/source-role stratum has sufficient fresh capacity.
- Safe error codes and contract-validation status.

Tracked output must not contain filenames, Drive IDs, folder names, local
paths, source IDs, locators, hashes, timestamps, per-document statuses, or
small reconstructive cells.

## SQLite Data Model

### `runs`

- `run_id` primary key
- `schema_version`
- `status`: `IN_PROGRESS`, `BLOCKED`, `COMPLETE`, or `INVALID`
- `index_contract_sha256`
- `e1a3_allocation_sha256`
- `created_at` and `updated_at`

Only one unfinished run is allowed for the same contract pair. A new run may
not silently supersede an unfinished or invalid run.

### `checkpoints`

- `run_id` and `stage` composite primary key
- `status`: `NOT_STARTED`, `IN_PROGRESS`, `COMPLETE`, or `BLOCKED`
- provider page token when applicable
- committed record count
- closed `error_code`
- `updated_at`

The saved provider page token is the required continuation point for the next
page in the same run. Each submitted page identifies the token used to request
it, and the controller rejects any mismatch before committing records. If a
Drive token expires, start a new versioned inventory run; do not silently mark
the interrupted run complete from a different page.

### `drive_files`

- `drive_file_id` primary key
- private display name
- MIME type and byte size
- provider checksum algorithm and digest when available
- modified timestamp
- parent folder IDs
- discovery scope and scan status

Never store OAuth tokens, bearer URLs, permission principals, file contents,
or fetched text.

### `local_files`

- normalized private relative path primary key
- SHA-256, any provider-compatible checksum calculated locally, and byte size
- file type
- parser status
- page or sheet count when already available from ingestion metadata

The reconciliation does not open documents to extract content. Calculating a
file checksum is allowed. When Drive exposes MD5 or another provider checksum,
the local inventory calculates the same algorithm solely for exact identity
comparison; SHA-256 remains mandatory for sealing and integrity manifests.
Existing parser results may be read without parsing again.

### `index_sources`

- stable indexed source ID primary key
- private source provenance
- parser type
- assigned topic
- chunk count
- embedding-model label
- index inventory contract binding

The index read is metadata-only and begins only after exact index-contract
verification.

### `index_locators`

- indexed source ID and normalized locator composite primary key
- assigned topic
- assigned source role
- substantive eligibility status
- E1a-3 use status
- E1a-4 availability status

The table stores locator identity and closed status only, never chunk text.

### `document_matches`

- Drive file ID
- local-file key
- indexed source ID
- match method
- match status
- review requirement
- exact evidence fields used by the match

Accepted automatic match methods are `DRIVE_ID_PROVENANCE`,
`EXACT_CONTENT_HASH`, and `EXACT_INGESTION_PROVENANCE`. Filename and size may
create a review candidate but can never create an accepted automatic match.

### `review_decisions`

- stable decision ID
- candidate match identity
- decision: `ACCEPT`, `REJECT`, `DUPLICATE_ALIAS`, `INELIGIBLE`, or
  `NEEDS_SOURCE_OWNER_REVIEW`
- reviewer identifier
- closed reason code
- decision timestamp

Review decisions contain no free text. Correcting a decision creates a new
superseding record rather than mutating audit history.

## Reconciliation Stages

### Stage 0: Preflight

1. Confirm the private root is Git-ignored and contains no tracked files.
2. Verify the approved index contract before reading index metadata.
3. Verify the sealed E1a-3 allocation and manifest without changing them.
4. Confirm the E1a-4 sampling frame is absent or is a complete verified set.
5. Create or resume the matching SQLite run.

Failure at this stage initializes no Drive scan and no database inventory.

### Stage 1: Drive Metadata Inventory

Discover the approved boundary in bounded pages. Bind each page to the saved
request token, upsert each record by Drive file ID, and commit the records plus
the next token atomically. Record duplicate-title groups only as candidates;
title equality is not document identity.

### Stage 2: Local Metadata Inventory

Inventory the approved local source roots. Calculate SHA-256 and upsert by
normalized relative path. Record already-existing parser outcomes and
page/sheet counts without parsing files again.

### Stage 3: Index Metadata Inventory

Read the verified index metadata into `index_sources` and `index_locators`.
Bind every row to the approved index-contract digest. Reject a mixed embedding
model, malformed provenance, duplicate locator, or source-count mismatch.

### Stage 4: Exact Document Matching

Apply this precedence:

1. Exact recorded Drive ID in ingestion provenance.
2. Exact provider checksum compared with the same algorithm calculated from
   the local file.
3. Exact content SHA-256 recorded by ingestion.
4. Exact existing ingestion provenance binding local file to indexed source.
5. Filename plus size only as an ambiguous review candidate.

A lower-precedence rule cannot override a conflicting higher-precedence rule.
One Drive file may have duplicate aliases, but one indexed source cannot map to
two different content hashes.

### Stage 5: Eligibility Reconciliation

For every accepted source match, reconcile:

- existing topic assignment;
- foundational or supporting source role;
- title-only or substantive eligibility;
- locator normalization;
- locator-level topic assignment for multi-topic foundational documents; and
- exact E1a-3 locator use.

Do not rewrite the sealed E1a-3 role configuration. Any proposed E1a-4 mapping
correction is a new versioned artifact requiring separate review.

### Stage 6: Capacity Calculation And Dry Run

Calculate unused substantive locator capacity for all eight public
topic/source-role strata. Each stratum must provide at least 12 unique locator
keys. Then call the existing deterministic allocator in no-write mode and
require an exact 96-slot result.

The dry run records only private allocations and a public closed status. It may
not create the E1a-4 sampling-frame artifacts.

### Stage 7: Decision

Classify the blocker as exactly one or more of:

- `DRIVE_TO_LOCAL_GAP`
- `LOCAL_TO_PARSE_GAP`
- `PARSE_TO_INDEX_GAP`
- `DUPLICATE_IDENTITY_GAP`
- `TOPIC_MAPPING_GAP`
- `SOURCE_ROLE_MAPPING_GAP`
- `SUBSTANTIVE_ELIGIBILITY_GAP`
- `PRIOR_LOCATOR_CAPACITY_GAP`
- `ALLOCATION_IMPLEMENTATION_GAP`
- `TRUE_APPROVED_CORPUS_CAPACITY_GAP`

Each class has a separate follow-on approval. Reconciliation itself authorizes
no correction.

### Stage 8: Seal Reconciliation Evidence

When all stages validate and every ambiguous match has one valid current closed
review decision, export the canonical snapshot set. Validate set equality,
digests, cross-file relationships, and the immutable active-run binding before
marking the SQLite run `COMPLETE`. Only then may a new plan authorize a mapping
correction, controlled ingestion, or another E1a-4 sampling-frame attempt.

## Restart And Recovery

- Every source scan is idempotent and uses upserts with stable primary keys.
- Every provider page or local batch commits independently.
- A stage becomes `COMPLETE` only after its record-count and contract checks
  pass.
- Resuming revalidates the run bindings before using saved state.
- Expired Drive tokens require a new versioned inventory run; the interrupted
  run cannot be completed from a different pagination chain.
- Database disconnection leaves the current stage `IN_PROGRESS` or `BLOCKED`.
- JSONL exports never resume by appending to a partial file; they are rebuilt
  from committed SQLite rows and atomically published.
- Recovery never deletes prior audit records or silently starts a new run.

## Privacy And Security

- All working state, snapshots, manifests, and review decisions remain under
  `.private/corpus-reconciliation/v1/`.
- No credential, OAuth token, database URL, bearer URL, permission principal,
  content excerpt, or raw provider error is stored.
- CLI output uses closed error codes and aggregate counts only.
- Logs must not contain private paths, filenames, source IDs, Drive IDs,
  locators, or hashes.
- The tracked plan may record only the aggregate decision and next approval
  gate.

## Failure Behavior

Fail closed on:

- index-contract or E1a-3 manifest mismatch;
- unsupported schema version;
- duplicate primary identity with conflicting metadata;
- one indexed source associated with conflicting content hashes;
- malformed or missing topic, role, or locator status;
- ambiguous automatic match;
- incomplete Drive pagination without a resumable checkpoint;
- partial snapshot or manifest set;
- any attempt to write outside the private root; or
- any attempt to run allocation before reconciliation completes.

No fallback source, filename-only acceptance, model-based identity decision,
silent record drop, or partial success is allowed.

## Implementation Boundary

Planned tracked files:

```text
src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py
eval/reconcile_private_corpus.py
tests/evaluation/test_corpus_reconciliation.py
docs/superpowers/plans/2026-08-23-private-corpus-reconciliation.md
```

Planned private files exist only beneath
`.private/corpus-reconciliation/v1/`.

Implementation will use Python standard-library SQLite and the existing Google
Drive and PostgreSQL boundaries. It will not add a workflow framework, Qdrant,
another vector index, or a second ingestion pipeline.

## Verification

Tests must prove:

- schema creation and version rejection;
- resumable page and batch commits;
- exact-match precedence and conflict rejection;
- duplicate aliases without duplicate canonical documents;
- filename-only matches require review;
- no content or credential fields can be serialized;
- index and E1a-3 digest binding;
- exact locator normalization and prior-use exclusion;
- eight-stratum capacity calculation and exact 96-slot dry run;
- atomic JSONL snapshot sealing and partial-set rejection;
- safe CLI errors without private identifiers or paths; and
- restart after simulated Drive and PostgreSQL disconnections.

Focused tests, the full Python suite, Ruff, whitespace checks, private ignore
checks, and a tracked-output privacy scan are required before completion.

## Acceptance Criteria

The reconciliation stage is complete only when:

1. Every approved indexed source has exactly one closed reconciliation status.
2. Every discovered Drive and local candidate has a closed status or an
   explicit review requirement.
3. Duplicate identity groups have one canonical content identity.
4. Index metadata exactly matches the approved contract.
5. Every eligible locator has a topic, source role, substantive status, and
   E1a-3-use status.
6. The capacity report covers all eight topic/source-role strata.
7. The allocator dry run either returns the exact deterministic 96-slot grid or
   a closed blocker classification.
8. SQLite state and all JSONL snapshots validate after a simulated restart.
9. No private material appears in Git or tracked reports.
10. The outcome proposes no ingestion or evaluation-rule change without its
    own approval.
