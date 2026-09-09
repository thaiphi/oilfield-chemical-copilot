# Controlled Corpus V2 Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Build a resumable, provenance-bound Corpus V2 pipeline from the user-approved 385 Drive identities through a separate PGVector release, with safe promotion and rollback to the frozen legacy corpus.

**Architecture:** Corpus V2 is a new private release pipeline. A steward-approved source register drives immutable Drive acquisition; a SQLite ledger records every fact and exception decision; canonical manifests bind snapshot bytes, extraction, chunks, embeddings, candidate index, evaluation, and promotion. The existing 198-source legacy database remains read-only and is never opened by V2 write paths.

**Tech Stack:** Python 3.11, PostgreSQL/PGVector, psycopg, SQLite, existing parser/chunker modules, Google Drive API client, Streamlit, pytest, Ruff.

**Spec:** [2026-09-07-controlled-corpus-v2-rebuild-design.md](../specs/2026-09-07-controlled-corpus-v2-rebuild-design.md)

## Global Constraints

- The approved 385-document Google Drive source register is authoritative. No code may infer membership from file names, paths, topics, or model output.
- Preserve the legacy 198-source / 4,797-chunk database and its configuration as read-only. Never apply V2 migrations, upserts, deletes, grants, or schema changes to it.
- Corpus V2 uses an independently provisioned PostgreSQL database and dedicated operational role. Its database name must differ from the legacy database name.
- All detailed source IDs, names, paths, bytes, revision data, hashes, extracted text, chunks, embeddings, review decisions, and evaluation cases remain in an ignored private root.
- Public reports contain release ID and aggregates only. They contain no reconstructive document-level data.
- The controller is deterministic and fail-closed. It records parser facts but cannot infer an approved identity, criticality, or exclusion disposition.
- Do not use Astra or any other LLM for identity, provenance, criticality, or final disposition. Astra may only review architecture and aggregate gate evidence.
- Do not add Qdrant, a second vector store, a mixed-release chunks table, automatic promotion, or a legacy fallback inside V2 evaluation.
- Never tune on E1a-3/E1a-4 sealed artifacts or an unseen Corpus V2 acceptance cohort.
- Do not run Drive acquisition, private parsing, embedding, database writes, index creation, or production promotion until the release owner supplies the approved release configuration and explicitly authorizes the operational task.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| \`src/oilfield_chemical_copilot/corpus_v2/models.py\` | Immutable records, enums, and explicit release configuration. |
| \`src/oilfield_chemical_copilot/corpus_v2/canonical.py\` | Canonical JSONL/JSON serialization, SHA-256 digests, and manifest verification. |
| \`src/oilfield_chemical_copilot/corpus_v2/ledger.py\` | Resumable SQLite state, append-only decisions, and stage transitions. |
| \`src/oilfield_chemical_copilot/corpus_v2/drive.py\` | Narrow Google Drive metadata/download adapter and snapshot acquisition. |
| \`src/oilfield_chemical_copilot/corpus_v2/processing.py\` | Versioned extraction, disposition candidates, and V2 chunk construction. |
| \`src/oilfield_chemical_copilot/corpus_v2/store.py\` | V2-only PGVector schema, upsert, inventory, and exact release validation. |
| \`src/oilfield_chemical_copilot/corpus_v2/release.py\` | Legacy guard, release binding, promotion state, and public aggregate reporting. |
| \`ingestion/corpus_v2.py\` | Explicit stage CLI; no implicit all-in-one build command. |
| \`ingestion/corpus_v2_apply_migrations.py\` | V2-only schema installer with target-identity checks. |
| \`ingestion/corpus_v2_legacy_guard.py\` | Read-only legacy fingerprint and backup/restore rehearsal contract. |
| \`db/corpus_v2_migrations/0001_corpus_v2_release.sql\` | V2 candidate database schema only. |
| \`app/streamlit_app.py\` | Explicit release configuration and release-scoped RAG cache key. |
| \`tests/corpus_v2/\` | Unit and adversarial tests for every V2 contract. |
| \`tests/app/test_streamlit_app.py\` | Runtime release selection and cache-isolation tests. |
| \`docs/superpowers/reports/2026-09-07-corpus-v2-rebuild-status.md\` | Aggregate-only closure and future execution status. |

## Required Release Configuration

The controller accepts one canonical private \`release-config.v1.json\`, supplied by the
release owner. It must contain:

\`\`\`json
{
  "release_id": "corpus-v2-20260907-initial",
  "approved_source_register_sha256": "64 lowercase hexadecimal characters",
  "critical_source_register_sha256": "64 lowercase hexadecimal characters",
  "drive_acquisition_mode": "pinned-revision",
  "drive_export_policy_version": "v1",
  "parser_policy_version": "v1",
  "chunk_policy_version": "v1",
  "embedding_provider": "approved provider identifier",
  "embedding_model": "approved exact model identifier",
  "embedding_dimension": 384,
  "candidate_database_name": "corpus_v2_initial",
  "evaluation_spec_sha256": "64 lowercase hexadecimal characters",
  "canary_minutes": 0,
  "rollback_trigger_policy_version": "v1"
}
\`\`\`

This is a field-shape example only; the accepted private file must contain concrete
values approved by the release owner. The controller rejects missing, extra,
noncanonical, or unapproved fields and never manufactures a configuration or applies
defaults to it.

### Task 1: Establish Corpus V2 Contracts and Durable Ledger

**Files:**
- Create: \`src/oilfield_chemical_copilot/corpus_v2/__init__.py\`
- Create: \`src/oilfield_chemical_copilot/corpus_v2/models.py\`
- Create: \`src/oilfield_chemical_copilot/corpus_v2/canonical.py\`
- Create: \`src/oilfield_chemical_copilot/corpus_v2/ledger.py\`
- Create: \`tests/corpus_v2/test_models.py\`
- Create: \`tests/corpus_v2/test_canonical.py\`
- Create: \`tests/corpus_v2/test_ledger.py\`

**Interfaces:**
- Consumes: a private release root and the user-provided \`release-config.v1.json\`.
- Produces: \`ReleaseConfig\`, \`SourceDisposition\`, \`Stage\`, \`CorpusV2Ledger\`, and \`sha256_canonical_jsonl(records)\`.

- [ ] **Step 1: Write failing model and canonicalization tests**

\`\`\`python
def test_release_config_rejects_unknown_field_and_legacy_database() -> None:
    payload = valid_release_config(candidate_database_name="oilfield_copilot")
    with pytest.raises(CorpusV2ContractError, match="C2_CONFIG_INVALID"):
        ReleaseConfig.from_mapping(payload)

def test_canonical_jsonl_is_stable_and_newline_terminated() -> None:
    assert canonical_jsonl([{"b": 2, "a": 1}]) == b'{"a":1,"b":2}\\n'
\`\`\`

- [ ] **Step 2: Run the tests to verify failure**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_models.py tests/corpus_v2/test_canonical.py\`
Expected: FAIL because the Corpus V2 contract module does not exist.

- [ ] **Step 3: Implement strict records and canonical helpers**

Create frozen dataclasses for \`ReleaseConfig\`, \`ApprovedSource\`,
\`AcquisitionRecord\`, \`ExtractionRecord\`, \`CorpusV2Chunk\`,
\`EmbeddingRecord\`, and \`ReleaseBinding\`. Make \`SourceDisposition\` exactly
\`INDEXED, DUPLICATE, NON_TEXT, EMPTY, UNSUPPORTED, EXTRACTION_FAILED,
INTENTIONALLY_EXCLUDED, UNRESOLVED\`. Validate all SHA-256 values as lowercase
64-character hex, forbid booleans as counts, and require a candidate database name
different from both configured and supplied legacy names.

Implement canonical JSON with sorted keys, compact separators, UTF-8, a trailing
newline for JSONL, and no floats, paths, source text, or credentials in public
aggregate records.

- [ ] **Step 4: Write failing ledger tests**

\`\`\`python
def test_ledger_rejects_stage_skip_and_preserves_decision_history(tmp_path: Path) -> None:
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config)
    with pytest.raises(CorpusV2LedgerError, match="C2_STAGE_PREREQUISITE"):
        ledger.complete_stage(Stage.CHUNKED)
    ledger.record_disposition(source_id="doc-1", disposition=SourceDisposition.EMPTY,
                              reviewer_id="reviewer-a", reason_code="NO_EXTRACTED_TEXT")
    assert ledger.current_disposition("doc-1").reviewer_id == "reviewer-a"
\`\`\`

- [ ] **Step 5: Implement the SQLite state machine**

Create tables for \`release\`, \`source_register\`, \`acquisitions\`,
\`extractions\`, \`disposition_decisions\`, \`stage_checkpoints\`, and
\`event_history\`. Enable foreign keys and WAL mode. Each mutating method uses one
transaction, records a timestamped event, validates legal transitions, and survives a
process restart. Never store raw file content, extracted text, vectors, or credentials
in SQLite. Enforce the state order \`REGISTERED -> ACQUIRED -> PARSED -> REVIEWED ->
CHUNKED -> EMBEDDED -> INDEX_VALIDATED -> EVALUATED -> PROMOTION_READY -> PROMOTED\`.

- [ ] **Step 6: Run focused tests and Ruff**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_models.py tests/corpus_v2/test_canonical.py tests/corpus_v2/test_ledger.py\`
Run: \`uv run --extra dev ruff check src/oilfield_chemical_copilot/corpus_v2 tests/corpus_v2\`
Expected: PASS.

- [ ] **Step 7: Commit**

\`\`\`bash
git add src/oilfield_chemical_copilot/corpus_v2 tests/corpus_v2
git commit -m "feat: add corpus v2 release contracts"
\`\`\`

### Task 2: Seal the Steward-Approved Source and Critical Registers

**Files:**
- Create: \`src/oilfield_chemical_copilot/corpus_v2/registers.py\`
- Create: \`tests/corpus_v2/test_registers.py\`
- Modify: \`ingestion/corpus_v2.py\`

**Interfaces:**
- Consumes: canonical private \`approved-source-register.v1.jsonl\`, critical-source register, and release config.
- Produces: validated 385-member ledger initialization and sealed register manifests.

- [ ] **Step 1: Write failing register tests**

\`\`\`python
def test_register_requires_exact_approved_count_and_unique_drive_ids(tmp_path: Path) -> None:
    with pytest.raises(CorpusV2RegisterError, match="C2_SOURCE_REGISTER_INVALID"):
        load_approved_register(write_register(tmp_path, sources=make_sources(384)))

def test_critical_register_cannot_reference_unapproved_source(tmp_path: Path) -> None:
    with pytest.raises(CorpusV2RegisterError, match="C2_CRITICAL_REGISTER_INVALID"):
        validate_critical_register(approved_sources, [CriticalSource("outside-register")])
\`\`\`

- [ ] **Step 2: Run tests to verify failure**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_registers.py\`
Expected: FAIL because register validation is absent.

- [ ] **Step 3: Implement registers and initialization command**

Require each approved entry to contain only an opaque source identity, Drive file ID,
declared MIME type, pinned revision token, and steward approval binding. Require exactly
385 unique source identities and unique Drive file IDs. Require a critical register that
is sorted, nonempty, a subset of the approved source set, and signed by the same release
configuration binding. The initializer command reads both paths only under the approved
private root, verifies their digests equal the release config, persists them to the
ledger, and writes canonical manifests atomically. It must not call Drive or parse files.

- [ ] **Step 4: Add adversarial tests**

Cover duplicate IDs, unknown keys, wrong count, path traversal, digest mismatch,
noncanonical JSONL, an unapproved critical source, duplicate critical entries, and
attempted reinitialization of an existing release ID.

- [ ] **Step 5: Run tests and commit**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_registers.py tests/corpus_v2/test_ledger.py\`
Run: \`uv run --extra dev ruff check ingestion/corpus_v2.py src/oilfield_chemical_copilot/corpus_v2 tests/corpus_v2\`
Run: \`git add src/oilfield_chemical_copilot/corpus_v2/registers.py ingestion/corpus_v2.py tests/corpus_v2/test_registers.py\`
Commit: \`git commit -m "feat: seal corpus v2 source registers"\`

### Task 3: Acquire an Immutable Drive Snapshot Without Membership Inference

**Files:**
- Modify: \`pyproject.toml\`
- Create: \`src/oilfield_chemical_copilot/corpus_v2/drive.py\`
- Create: \`tests/corpus_v2/test_drive.py\`
- Modify: \`ingestion/corpus_v2.py\`

**Interfaces:**
- Consumes: initialized ledger, approved register, and a \`DriveSourceClient\`.
- Produces: one acquisition record and one private byte file per approved identity.

- [ ] **Step 1: Write failing Drive-adapter tests using a fake client**

\`\`\`python
def test_acquire_blocks_when_drive_revision_changes(tmp_path: Path) -> None:
    client = FakeDriveClient(metadata=metadata(file_id="drive-1", revision="new"))
    with pytest.raises(CorpusV2AcquisitionError, match="C2_DRIVE_REVISION_CHANGED"):
        acquire_source(approved_source(revision="old"), client, snapshot_root=tmp_path)

def test_acquisition_hashes_downloaded_bytes_and_never_uses_filename_as_identity(tmp_path: Path) -> None:
    record = acquire_source(approved_source(), FakeDriveClient(bytes=b"approved"), snapshot_root=tmp_path)
    assert record.byte_sha256 == hashlib.sha256(b"approved").hexdigest()
    assert record.source_id == approved_source().source_id
\`\`\`

- [ ] **Step 2: Run tests to verify failure**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_drive.py\`
Expected: FAIL because acquisition code is absent.

- [ ] **Step 3: Implement the narrow Drive client**

Add the Google Drive API dependency and a \`DriveSourceClient\` protocol with
\`metadata(file_id)\` and \`download(file_id, export_mime_type)\`. Implement
\`GoogleDriveSourceClient\` using a service-account or user credential path supplied
through an environment variable, never through the ledger or manifests. Read current
metadata before download; require ID, MIME type, and pinned revision equality. For
native Google files, require the register's explicit approved export MIME type. For
stored files, use raw authenticated download. Do not use names, parent folders, or
search to identify sources.

- [ ] **Step 4: Implement atomic snapshot acquisition**

Write source bytes to a same-directory temporary filename, fsync, calculate SHA-256
from written bytes, verify expected revision again when supported, then atomically
rename to an opaque release/source location. Record only the release-relative snapshot
path, byte hash, MIME type, revision, and status in the ledger. Reject duplicate final
paths, changed revisions, missing metadata, unexpected MIME type, nonregular files,
partial output, and repeat acquisition whose observed facts differ from the ledger.

- [ ] **Step 5: Add resume and privacy tests**

Test 385-source loop resume after an injected failure, zero duplicate downloads after a
restart, no credentials in exceptions/log output, native-export policy rejection,
wrong-byte hash detection, and exact register/acquisition set equality.

- [ ] **Step 6: Run tests and commit**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_drive.py tests/corpus_v2/test_registers.py\`
Run: \`uv run --extra dev ruff check src/oilfield_chemical_copilot/corpus_v2 ingestion/corpus_v2.py tests/corpus_v2\`
Run: \`git add pyproject.toml src/oilfield_chemical_copilot/corpus_v2/drive.py ingestion/corpus_v2.py tests/corpus_v2/test_drive.py\`
Commit: \`git commit -m "feat: acquire immutable corpus v2 snapshots"\`

### Task 4: Produce Provenance-Bound Extraction, Review Candidates, and Chunks

**Files:**
- Create: \`src/oilfield_chemical_copilot/corpus_v2/processing.py\`
- Create: \`tests/corpus_v2/test_processing.py\`
- Modify: \`ingestion/corpus_v2.py\`

**Interfaces:**
- Consumes: acquired private snapshots and the parser/chunk policy versions in release config.
- Produces: extraction manifest, mechanical disposition candidates, and canonical V2 chunks.

- [ ] **Step 1: Write failing processing tests**

\`\`\`python
def test_v2_chunk_id_binds_release_source_hash_location_and_text() -> None:
    first = build_v2_chunk(release_id="r1", source_id="doc-1", byte_sha256="a" * 64,
                           location="page:1", ordinal=0, text="evidence")
    changed_source = build_v2_chunk(release_id="r1", source_id="doc-2", byte_sha256="a" * 64,
                                    location="page:1", ordinal=0, text="evidence")
    assert first.chunk_id != changed_source.chunk_id

def test_failed_critical_source_blocks_review_completion() -> None:
    assert review_gate_status(critical_failure_fixture()) == "BLOCKED"
\`\`\`

- [ ] **Step 2: Run tests to verify failure**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_processing.py\`
Expected: FAIL because V2 processing does not exist.

- [ ] **Step 3: Implement deterministic extraction**

Wrap existing \`parse_document\` implementations but pass only the private snapshot
path. Capture parser type, parser-policy version, source-byte hash, valid page/sheet
locations, chunk count, and a fixed outcome code. Convert no output to an \`EMPTY\`
candidate, unsupported extensions to \`UNSUPPORTED\`, and exceptions to
\`EXTRACTION_FAILED\`; retain a safe exception class/code only. Do not persist text in
the ledger or public reports. Put private extracted content only in the approved release
tree and reference it by its manifest digest.

- [ ] **Step 4: Implement V2 chunk identity and candidate review**

Create chunk IDs as SHA-256 over release ID, approved source ID, source-byte hash,
parser policy version, chunk policy version, validated location, ordinal, and chunk
text digest. Store source ID and byte hash in \`ChunkMetadata.extra\` for V2 only.
Write candidate dispositions to the ledger, but require an explicit reviewer decision
for DUPLICATE, NON_TEXT, EMPTY, UNSUPPORTED, EXTRACTION_FAILED, and
INTENTIONALLY_EXCLUDED. Reject review completion if a source lacks a final disposition,
a source is both indexed and excluded, a duplicate lacks a representative source, or a
critical source lacks usable indexed evidence.

- [ ] **Step 5: Add adversarial tests**

Cover malformed locations, duplicate IDs, source IDs outside the register, changed
snapshot bytes, path-derived legacy chunk IDs, invalid reviewer IDs, duplicate
dispositions, empty ordinary documents with reviewed disposition, and every critical
failure outcome.

- [ ] **Step 6: Run tests and commit**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_processing.py tests/ingest/test_parsers.py tests/ingest/test_chunking.py\`
Run: \`uv run --extra dev ruff check src/oilfield_chemical_copilot/corpus_v2 tests/corpus_v2\`
Run: \`git add src/oilfield_chemical_copilot/corpus_v2/processing.py ingestion/corpus_v2.py tests/corpus_v2/test_processing.py\`
Commit: \`git commit -m "feat: bind corpus v2 chunks to source provenance"\`

### Task 5: Create and Validate the Isolated Corpus V2 PGVector Database

**Files:**
- Create: \`db/corpus_v2_migrations/0001_corpus_v2_release.sql\`
- Create: \`src/oilfield_chemical_copilot/corpus_v2/store.py\`
- Create: \`ingestion/corpus_v2_apply_migrations.py\`
- Create: \`tests/corpus_v2/test_store.py\`
- Create: \`tests/corpus_v2/test_v2_migration.py\`
- Modify: \`ingestion/corpus_v2.py\`

**Interfaces:**
- Consumes: reviewed V2 chunk manifest and embedding records.
- Produces: separate V2 database rows and an exact index contract.

- [ ] **Step 1: Write failing schema and store tests**

\`\`\`python
def test_v2_store_refuses_legacy_database_name() -> None:
    with pytest.raises(CorpusV2StoreError, match="C2_DATABASE_TARGET_INVALID"):
        CorpusV2Store(database_url=legacy_url, release_config=config)

def test_validate_release_rejects_extra_or_orphan_database_chunk(fake_store) -> None:
    fake_store.insert_orphan_chunk()
    with pytest.raises(CorpusV2StoreError, match="C2_INDEX_EXACT_SET_MISMATCH"):
        validate_v2_index(fake_store, release_binding)
\`\`\`

- [ ] **Step 2: Run tests to verify failure**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_store.py tests/corpus_v2/test_v2_migration.py\`
Expected: FAIL because V2 storage code and migration are absent.

- [ ] **Step 3: Define the V2-only schema**

Create a V2 migrations directory that is never discovered by
\`ingestion/apply_migrations.py\`. The schema includes \`corpus_release\` with one
immutable release row and a V2 \`chunks\` table compatible with existing retrieval reads
plus non-null \`release_id\`, \`source_id\`, \`source_sha256\`, and
\`manifest_sha256\` columns. Add foreign keys and unique constraints that prevent a
chunk from belonging to another release. Create the required PGVector indexes only in
the candidate database.

- [ ] **Step 4: Implement V2 migration and indexing boundaries**

The V2 migration CLI requires both a candidate URL and a separately supplied legacy URL;
it parses database names and rejects equality before connecting for writes. It verifies
that \`corpus_release\` is absent or has the exact configured release ID. The V2 store
accepts only \`CorpusV2Chunk\` and embedding records whose source hash and release ID
match the sealed chunk manifest. It validates finite vectors and the exact configured
dimension before batch upsert. It cannot call the legacy \`PgVectorStore.upsert_chunks\`.

- [ ] **Step 5: Implement exact index validation**

Read V2 metadata in a read-only transaction and compare all canonical chunk IDs, source
IDs, source hashes, locations, model identity, dimensions, and release manifest digest
against the intended manifest. Reject missing, extra, duplicate, null, mixed-model, or
foreign-release rows. Emit a private index contract and an aggregate-only report.

- [ ] **Step 6: Add integration tests guarded by the existing integration marker**

Provision only a database whose name ends in \`_test\`. Test empty V2 migration,
candidate upsert, exact validation, rejection of legacy URL equality, failed
dimension/model changes, and rejection of one injected extra row. Never run these tests
against a developer's legacy database.

- [ ] **Step 7: Run tests and commit**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_store.py tests/corpus_v2/test_v2_migration.py\`
Run: \`uv run --extra dev pytest -m integration tests/corpus_v2/test_v2_migration.py\` only when an isolated \`_test\` database is available.
Run: \`uv run --extra dev ruff check src/oilfield_chemical_copilot/corpus_v2 ingestion/corpus_v2_apply_migrations.py tests/corpus_v2\`
Run: \`git add db/corpus_v2_migrations src/oilfield_chemical_copilot/corpus_v2/store.py ingestion/corpus_v2_apply_migrations.py tests/corpus_v2/test_store.py tests/corpus_v2/test_v2_migration.py ingestion/corpus_v2.py\`
Commit: \`git commit -m "feat: add isolated corpus v2 vector store"\`

### Task 6: Guard Legacy, Seal the Release Binding, and Publish Aggregate Status

**Files:**
- Create: \`src/oilfield_chemical_copilot/corpus_v2/release.py\`
- Create: \`ingestion/corpus_v2_legacy_guard.py\`
- Create: \`tests/corpus_v2/test_release.py\`
- Create: \`docs/superpowers/reports/2026-09-07-corpus-v2-rebuild-status.md\`
- Modify: \`ingestion/corpus_v2.py\`

**Interfaces:**
- Consumes: legacy URL, V2 release config, private manifests, and V2 index contract.
- Produces: legacy read-only fingerprint, sealed release binding, and public aggregates.

- [ ] **Step 1: Write failing release tests**

\`\`\`python
def test_release_binding_rejects_missing_stage_digest(tmp_path: Path) -> None:
    with pytest.raises(CorpusV2ReleaseError, match="C2_RELEASE_BINDING_INVALID"):
        seal_release_binding(tmp_path, incomplete_stage_digests())

def test_public_status_never_contains_source_identifiers() -> None:
    report = build_public_status(aggregate_status)
    assert "drive-" not in report
    assert "C:\\\\" not in report
\`\`\`

- [ ] **Step 2: Run tests to verify failure**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_release.py\`
Expected: FAIL because release sealing and legacy guard code do not exist.

- [ ] **Step 3: Implement legacy preservation preflight**

Use a default-transaction-read-only PostgreSQL connection to capture the legacy
fingerprint using the existing index-preflight pattern. Require the expected 198-source /
4,797-chunk contract before an operational V2 build starts. Define the restore rehearsal
as an operator-only command contract: a backup is captured with the platform's approved
PostgreSQL backup tool, restored into a disposable database with a distinct name, and
re-fingerprinted. The code records only backup artifact digest, restore-test result, and
aggregate counts; it never deletes or changes legacy data.

- [ ] **Step 4: Implement atomic release binding**

Bind canonical digests for source register, acquisition, extraction, current final
dispositions, chunks, embeddings, index contract, evaluation specification/result, and
promotion state. Use the existing authenticated private-artifact publication capability
for staged, locked, no-replace publication. Verification must reject extra members,
partial trees, a stale ledger projection, digest mismatch, or any source disposition
other than one final state.

- [ ] **Step 5: Write aggregate closure status**

Create the tracked report with only the verified starting findings, release phase,
aggregate disposition counts, candidate index aggregate counts, gate status, and next
approval gate. It must state that reconciliation closed with findings and that no V2
build has occurred until Gate 2 is actually executed.

- [ ] **Step 6: Run tests and commit**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_release.py tests/evaluation/test_private_artifact_publication.py\`
Run: \`uv run --extra dev ruff check src/oilfield_chemical_copilot/corpus_v2 ingestion/corpus_v2_legacy_guard.py tests/corpus_v2\`
Run: \`git add src/oilfield_chemical_copilot/corpus_v2/release.py ingestion/corpus_v2_legacy_guard.py ingestion/corpus_v2.py tests/corpus_v2/test_release.py docs/superpowers/reports/2026-09-07-corpus-v2-rebuild-status.md\`
Commit: \`git commit -m "feat: seal corpus v2 release bindings"\`

### Task 7: Make Application Release Selection Explicit and Cache-Safe

**Files:**
- Modify: \`app/streamlit_app.py\`
- Modify: \`tests/app/test_streamlit_app.py\`
- Create: \`src/oilfield_chemical_copilot/corpus_v2/runtime.py\`
- Create: \`tests/corpus_v2/test_runtime.py\`

**Interfaces:**
- Consumes: explicit runtime release configuration and verified private release binding.
- Produces: one verified store/retriever/cache instance per selected release.

- [ ] **Step 1: Write failing runtime selection tests**

\`\`\`python
def test_runtime_rejects_database_or_manifest_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("CORPUS_RELEASE_ID", "corpus-v2-r1")
    monkeypatch.setenv("CORPUS_RELEASE_MANIFEST_SHA256", "a" * 64)
    with pytest.raises(CorpusV2RuntimeError, match="C2_RUNTIME_RELEASE_INVALID"):
        load_runtime_release()

def test_rag_cache_is_scoped_by_release_and_retrieval_mode(monkeypatch) -> None:
    first = _build_rag_service("corpus-v2-r1", "hybrid")
    second = _build_rag_service("legacy-r1", "hybrid")
    assert first is not second
\`\`\`

- [ ] **Step 2: Run tests to verify failure**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_runtime.py tests/app/test_streamlit_app.py\`
Expected: FAIL because the current cache key is retrieval mode only.

- [ ] **Step 3: Implement runtime contract verification**

Require explicit environment values for release ID, database URL, embedding provider/model/
dimension, and release-binding digest. Resolve only a private binding path supplied by the
operator; validate its canonical contents before constructing an embedding provider,
opening PostgreSQL, loading chunks, or initializing the keyword cache. Verify the
database's V2 release metadata and exact contract in a read-only transaction. Do not
allow a missing V2 value to fall back to the legacy URL. Legacy mode is an explicit,
separately verified release configuration.

- [ ] **Step 4: Update the Streamlit cache boundary**

Change \`_build_rag_service\` to accept a fully validated runtime release object or its
release ID plus retrieval mode. Make the cache key include release ID, database identity,
embedding model, dimension, manifest digest, and retrieval mode. Clear no global cache
during configuration parsing. Confirm monitoring remains on its independently configured
database.

- [ ] **Step 5: Add regression tests**

Cover V2/legacy cache separation, changed manifest digest, changed database URL, wrong
embedding model/dimension, unavailable database, no silent fallback, hybrid keyword
index construction from one release only, and absolute-path citation hiding.

- [ ] **Step 6: Run tests and commit**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_runtime.py tests/app/test_streamlit_app.py tests/storage/test_pgvector.py\`
Run: \`uv run --extra dev ruff check app/streamlit_app.py src/oilfield_chemical_copilot/corpus_v2 tests/app tests/corpus_v2\`
Run: \`git add app/streamlit_app.py src/oilfield_chemical_copilot/corpus_v2/runtime.py tests/app/test_streamlit_app.py tests/corpus_v2/test_runtime.py\`
Commit: \`git commit -m "feat: require explicit corpus release selection"\`

### Task 8: Freeze Candidate Evaluation and Promotion Controller

**Files:**
- Create: \`src/oilfield_chemical_copilot/corpus_v2/evaluation.py\`
- Create: \`tests/corpus_v2/test_evaluation.py\`
- Modify: \`ingestion/corpus_v2.py\`
- Modify: \`src/oilfield_chemical_copilot/corpus_v2/release.py\`

**Interfaces:**
- Consumes: validated V2 index contract and a steward-approved private evaluation spec.
- Produces: one sealed aggregate candidate result and a gated promotion-ready state.

- [ ] **Step 1: Write failing evaluation gate tests**

\`\`\`python
def test_promotion_rejects_unfrozen_thresholds_or_unseen_cohort_reuse() -> None:
    with pytest.raises(CorpusV2EvaluationError, match="C2_EVALUATION_SPEC_INVALID"):
        freeze_evaluation_spec(invalid_spec)

def test_promotion_requires_every_prior_gate_and_rollback_rehearsal() -> None:
    assert promotion_status(incomplete_gate_results()) == "BLOCKED"
\`\`\`

- [ ] **Step 2: Run tests to verify failure**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_evaluation.py\`
Expected: FAIL because the V2 evaluation/promotion contract is absent.

- [ ] **Step 3: Implement evaluation specification binding**

Require a sealed private spec before model or retrieval initialization. It names separate
development and unseen acceptance cohorts, fixed scoring protocol, numerical thresholds,
registered regression families, latency measurement, and a no-tuning rule. The controller
validates that cohort identifiers are disjoint from historical sealed E1a-3/E1a-4
artifacts and that no result exists before the spec is frozen.

- [ ] **Step 4: Implement promotion-state checks**

Promotion can enter \`PROMOTION_READY\` only when legacy guard, source accounting,
critical-source evidence, exact index contract, evaluation result, canary plan, and
rollback rehearsal all have valid sealed bindings. The controller writes an
aggregate-only promotion recommendation; only the release owner action may transition
it to \`PROMOTED\`. A failed canary or rollback rehearsal transitions to \`BLOCKED\`.

- [ ] **Step 5: Add safety tests**

Cover result publication with extra source fields, an evaluation run before contract
verification, invalid probability/metric values, missing no-answer safety result,
threshold relaxation after results, failed canary, failed rollback, and repeated
promotion action.

- [ ] **Step 6: Run tests and commit**

Run: \`uv run --extra dev pytest -q tests/corpus_v2/test_evaluation.py tests/corpus_v2/test_release.py\`
Run: \`uv run --extra dev ruff check src/oilfield_chemical_copilot/corpus_v2 ingestion/corpus_v2.py tests/corpus_v2\`
Run: \`git add src/oilfield_chemical_copilot/corpus_v2/evaluation.py src/oilfield_chemical_copilot/corpus_v2/release.py ingestion/corpus_v2.py tests/corpus_v2/test_evaluation.py\`
Commit: \`git commit -m "feat: gate corpus v2 evaluation and promotion"\`

### Task 9: Review, Integrate, and Conduct Separately Authorized Operations

**Files:**
- Modify: \`docs/superpowers/reports/2026-09-07-corpus-v2-rebuild-status.md\`
- Modify: \`README.md\`
- Test: all affected test modules and the full suite.

**Interfaces:**
- Consumes: merged controller code, a user-approved private release configuration, and separately approved operational credentials.
- Produces: only the release stage explicitly authorized by the user.

- [ ] **Step 1: Request independent code review**

Request review of contract boundaries, private-data leakage, legacy-write prevention,
Drive revision enforcement, exact index validation, runtime cache keys, and promotion
authorization. Resolve findings before any operational authorization.

- [ ] **Step 2: Verify implementation without private operations**

Run: \`uv run --extra dev pytest -q tests/corpus_v2 tests/app/test_streamlit_app.py tests/storage/test_pgvector.py\`
Run: \`uv run --extra dev ruff check .\`
Run: \`uv run --extra dev pytest -q\`
Expected: all tests pass; no Drive, parser, embedding, candidate database, or production operation is run.

- [ ] **Step 3: Create a reviewed pull request**

Include the design, implementation plan, code, tests, and aggregate-only status report.
Explicitly state that no 385-document acquisition or Corpus V2 index build has been
performed by the code-change PR.

- [ ] **Step 4: Obtain a new operational authorization**

Before any runtime operation, record the release owner's approval of:
the exact private release config, Drive credential/identity access, candidate database
URL/role, legacy backup/restore environment, critical register, parser/OCR policy, and
evaluation thresholds. If any is absent, stop after code integration.

- [ ] **Step 5: Execute one authorized stage at a time**

Run only the current approved stage, verify its sealed manifest and aggregate result,
update the aggregate status report, and stop for the next approval gate. The order is:
legacy guard; source-register initialization; snapshot acquisition; extraction/review;
candidate indexing; index validation; evaluation; canary; promotion. Never combine
stages or continue after a BLOCKED state.

- [ ] **Step 6: Commit and hand off**

Commit aggregate-only documents after each authorized stage. For promotion, record the
explicit release configuration and rollback evidence; for rejection, retain the
candidate private artifacts and leave production on legacy.
