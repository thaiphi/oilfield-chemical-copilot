# Private Corpus Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable metadata-only reconciliation that explains where Drive, local ingestion, the verified 4,797-chunk/198-source index, and E1a-4 locator eligibility diverge.

**Architecture:** A focused evaluation module owns strict records, a private SQLite checkpoint store, exact matching, capacity calculation, and canonical JSONL sealing. A narrow CLI imports provider pages through stdin, inventories approved local files, reads index metadata only after contract verification, runs reconciliation, and emits aggregate-only status. Google Drive and PostgreSQL remain read-only; SQLite is authoritative working state and sealed JSONL is the portable audit record.

**Tech Stack:** Python 3.11, standard-library `sqlite3`, `hashlib`, `json`, existing `psycopg`, existing E1 index and E1a-3 sampling contracts, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-private-corpus-reconciliation-design.md`

## Global Constraints

- Keep all state beneath `.private/corpus-reconciliation/v1/`; no private file may be tracked.
- Do not persist document text, chunks, tables, images, questions, claims, labels, answers, credentials, bearer URLs, permission principals, or raw provider errors.
- Do not download, parse, ingest, embed, reindex, retrieve, rerank, or call a model.
- Verify the exact index contract and sealed E1a-3 allocation before index or locator import.
- Use SHA-256 for sealing; provider checksums are identity hints only when compared using the same algorithm.
- Filename and size may create a review candidate but never an automatic accepted match.
- Do not add Qdrant, another vector store, a workflow framework, or a second ingestion pipeline.
- Every CLI failure emits only a closed error code and aggregate status.
- Every production behavior is introduced through a failing focused test first.

---

### Task 1: Strict Records And Private SQLite Run State

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py`
- Create: `tests/evaluation/test_corpus_reconciliation.py`

**Interfaces:**
- Produces: `CorpusReconciliationError`, `DriveFileRecord`, `LocalFileRecord`, `IndexSourceRecord`, `IndexLocatorRecord`, `ReconciliationStore`, and `require_private_reconciliation_root(path: Path, expected_root: Path) -> Path`.
- Consumes: Python standard-library `sqlite3`, `dataclasses`, `Path`, and strict JSON-compatible mappings.

- [x] **Step 1: Write failing tests for strict records, private paths, schema creation, and resumable runs**

```python
def test_store_resumes_same_contract_bound_run(tmp_path: Path) -> None:
    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    store = ReconciliationStore.create(
        root=root,
        run_id="run-001",
        index_contract_sha256="a" * 64,
        e1a3_allocation_sha256="b" * 64,
    )
    store.set_checkpoint(stage="drive_inventory", status="IN_PROGRESS", committed_records=5)
    store.close()

    resumed = ReconciliationStore.open(root=root, run_id="run-001")
    assert resumed.checkpoint("drive_inventory").committed_records == 5
```

Also test unsupported schema versions, booleans where integers are required, unknown mapping keys, conflicting unfinished runs for the same contract pair, `PRAGMA foreign_keys=ON`, `journal_mode=WAL`, and rejection of paths outside the exact expected private root.

- [x] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py`

Expected: collection or import failure because `corpus_reconciliation` does not exist.

- [x] **Step 3: Implement strict records and `ReconciliationStore`**

Use this public surface:

```python
class ReconciliationStore:
    @classmethod
    def create(cls, *, root: Path, run_id: str, index_contract_sha256: str, e1a3_allocation_sha256: str) -> ReconciliationStore: ...
    @classmethod
    def open(cls, *, root: Path, run_id: str) -> ReconciliationStore: ...
    def set_checkpoint(self, *, stage: str, status: str, committed_records: int, page_token: str | None = None, error_code: str | None = None) -> None: ...
    def checkpoint(self, stage: str) -> CheckpointRecord: ...
    def close(self) -> None: ...
```

Create normalized tables `runs`, `checkpoints`, `drive_files`, `local_files`, `index_sources`, `index_locators`, `document_matches`, and `review_decisions` with foreign keys and uniqueness constraints from the spec. Use `INSERT ... ON CONFLICT DO UPDATE` only where the immutable identity fields are unchanged; otherwise raise a closed conflict code.

- [x] **Step 4: Run focused tests and Ruff**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py`

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py`

- [x] **Step 5: Commit Task 1**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py
git commit -m "feat: add resumable corpus reconciliation state"
```

### Task 2: Idempotent Metadata Imports And Restart Recovery

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py`
- Modify: `tests/evaluation/test_corpus_reconciliation.py`

**Interfaces:**
- Consumes: Task 1 records and `ReconciliationStore`.
- Produces: `import_drive_page(store, records, page_token, next_page_token)`, `inventory_local_files(store, roots)`, `import_index_inventory(store, sources, locators)`, and aggregate `StageResult`.

- [x] **Step 1: Write failing tests for page commits, expired-token rescan, local hashing, index conflicts, and simulated disconnect recovery**

```python
def test_drive_rescan_upserts_stable_ids_without_duplication(store: ReconciliationStore) -> None:
    page = (DriveFileRecord(drive_file_id="drive-1", name="private.pdf", mime_type="application/pdf", size_bytes=10, checksum_algorithm=None, checksum=None, modified_time="2026-08-23T00:00:00Z", parent_ids=("folder-1",)),)
    import_drive_page(store=store, records=page, next_page_token="token-2")
    import_drive_page(store=store, records=page, page_token="token-2", next_page_token=None)
    assert store.count("drive_files") == 1
    assert store.checkpoint("drive_inventory").status == "COMPLETE"
```

Simulate failure after one committed page, reopen the store, import the next page, and prove the first page remains. Verify local SHA-256 without storing bytes, exact index-contract binding, duplicate locator rejection, and no raw exception text in checkpoint errors.

- [x] **Step 2: Run the new tests and verify RED**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py -k "drive or local or index or restart"`

Expected: failures because the import functions do not exist.

- [x] **Step 3: Implement idempotent import functions**

Commit every Drive page and bounded local/index batch independently. Treat provider page tokens as hints. Normalize paths and locators before identity comparison. Calculate local SHA-256 in streaming blocks. Accept pre-existing parser metadata as input; do not invoke parsers.

- [x] **Step 4: Run focused tests and Ruff**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py`

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py`

- [x] **Step 5: Commit Task 2**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py
git commit -m "feat: import resumable corpus metadata"
```

### Task 3: Exact Matching, Duplicate Aliases, And Review Queue

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py`
- Modify: `tests/evaluation/test_corpus_reconciliation.py`

**Interfaces:**
- Consumes: imported Drive, local, and index records.
- Produces: `reconcile_document_matches(store) -> MatchSummary` and closed statuses `EXACT_MATCH`, `DUPLICATE_ALIAS`, `DRIVE_ONLY`, `LOCAL_ONLY`, `PARSED_NOT_INDEXED`, `INDEX_ONLY`, `AMBIGUOUS_REVIEW_REQUIRED`, and `INELIGIBLE`.

- [x] **Step 1: Write failing tests for match precedence and conflicts**

```python
def test_filename_and_size_only_requires_review(store_with_three_layers: ReconciliationStore) -> None:
    summary = reconcile_document_matches(store_with_three_layers)
    assert summary.ambiguous_review_required == 1
    assert store_with_three_layers.match_status("drive-1") == "AMBIGUOUS_REVIEW_REQUIRED"
```

Test exact Drive-ID provenance, same-algorithm checksum equality, exact ingestion provenance, conflicting higher-precedence evidence, one canonical content identity for duplicate aliases, and rejection when one indexed source maps to conflicting hashes.

- [x] **Step 2: Run matching tests and verify RED**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py -k "match or duplicate or ambiguous"`

- [x] **Step 3: Implement deterministic matching**

Apply the exact precedence from the spec. Do not calculate confidence scores. Persist a match only when all higher-precedence evidence is absent or consistent. Use closed reason codes for review candidates.

- [x] **Step 4: Run focused tests and Ruff**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py`

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py`

- [x] **Step 5: Commit Task 3**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py
git commit -m "feat: reconcile exact corpus identities"
```

### Task 4: Locator Eligibility And E1a-4 Capacity Dry Run

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py`
- Modify: `tests/evaluation/test_corpus_reconciliation.py`

**Interfaces:**
- Consumes: `IndexLocatorRecord`, the verified E1a-3 allocation locator set, and existing `build_sampling_slots()` / `allocate_sampling_slots()`.
- Produces: `calculate_locator_capacity(store, prior_locator_keys) -> CapacityReport` and `dry_run_e1a4_allocation(store, prior_locator_keys) -> DryRunResult`.

- [x] **Step 1: Write failing tests for eight-stratum coverage and exact allocation**

```python
def test_capacity_requires_twelve_fresh_locators_in_every_topic_role_cell(store: ReconciliationStore) -> None:
    report = calculate_locator_capacity(store=store, prior_locator_keys=())
    assert len(report.strata) == 8
    assert all(item.required_locators == 12 for item in report.strata)
```

Add tests for exact E1a-3 source/locator exclusion, locator-level topic overrides, title-only exclusion, duplicate locator rejection, one unavailable stratum, exact 96-slot success, and no sampling-frame writes.

- [x] **Step 2: Run capacity tests and verify RED**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py -k "capacity or locator or dry_run"`

- [x] **Step 3: Implement capacity and dry-run functions**

Return private per-stratum counts in `CapacityReport`, but expose only `sufficient: bool` per public stratum through public serialization. Reuse the existing deterministic allocator without changing its ordering or eligibility rules.

- [x] **Step 4: Run focused tests and Ruff**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py`

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py`

- [x] **Step 5: Commit Task 4**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py
git commit -m "feat: diagnose e1a4 locator capacity"
```

### Task 5: Atomic Canonical JSONL Snapshots

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py`
- Modify: `tests/evaluation/test_corpus_reconciliation.py`

**Interfaces:**
- Consumes: completed SQLite stages.
- Produces: `seal_reconciliation_snapshots(store, root) -> SnapshotSet` and `verify_reconciliation_snapshots(root) -> SnapshotSet`.

- [x] **Step 1: Write failing tests for canonical bytes, atomic publication, rollback, and privacy schema**

```python
def test_snapshot_seal_rolls_back_complete_set_after_mid_publish_failure(store: ReconciliationStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_SNAPSHOT_WRITE_FAILED"):
        seal_reconciliation_snapshots(store=store, root=tmp_path)
    assert not tuple((tmp_path / "snapshots").glob("*.jsonl"))
```

Test all six JSONL files and manifests, sorted deterministic records, trailing newline, strict keys, existing-set verification without rewrite, partial-set rejection, and rejection of content/credential fields.

- [x] **Step 2: Run snapshot tests and verify RED**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py -k "snapshot or manifest or privacy"`

- [x] **Step 3: Implement atomic sealing and verification**

Build every payload from committed SQLite rows. Validate all destinations before publication. Write and fsync same-directory temporary files, publish with `os.replace`, and remove every newly published file after a later failure. SHA-256 manifests contain lowercase 64-character digests plus newline.

- [x] **Step 4: Run focused tests and Ruff**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py`

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py`

- [x] **Step 5: Commit Task 5**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py tests/evaluation/test_corpus_reconciliation.py
git commit -m "feat: seal corpus reconciliation snapshots"
```

### Task 6: Aggregate-Only CLI And Provider Boundaries

**Files:**
- Create: `eval/reconcile_private_corpus.py`
- Modify: `tests/evaluation/test_corpus_reconciliation.py`

**Interfaces:**
- Consumes: Tasks 1-5 APIs, existing `verify_e1_index_contract`, existing E1a-3 sealed allocation, JSON Drive pages on stdin, approved local roots, and read-only PostgreSQL metadata.
- Produces CLI commands `init`, `import-drive-page`, `inventory-local`, `import-index`, `reconcile`, `capacity`, `seal`, and `status`.

- [x] **Step 1: Write failing CLI tests for safe status, stdin imports, read-only index import, and sanitized failure**

```python
def test_cli_rejects_missing_prerequisites_without_traceback(monkeypatch, capsys) -> None:
    runner = _runner_module()
    monkeypatch.setattr(sys, "argv", ["reconcile_private_corpus.py", "status"])
    assert runner.cli() == 1
    error = json.loads(capsys.readouterr().err)
    assert error == {"status": "CORPUS_RECONCILIATION_BLOCKED", "error_code": "CORPUS_RECONCILIATION_PREREQUISITES_MISSING"}
```

Test exact private root, no raw paths/IDs/hashes in stdout or stderr, invalid stdin schema, no model initialization, PostgreSQL `default_transaction_read_only=on`, exact contract verification before the first index query, and safe aggregate status keys.

- [x] **Step 2: Run CLI tests and verify RED**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py -k "cli"`

- [x] **Step 3: Implement the CLI**

`import-drive-page` reads exactly one JSON object from stdin with keys `records`, `page_token`, and `next_page_token`; the current page token must match the saved checkpoint before the page is committed. `inventory-local` accepts only explicitly provided roots and never follows symlinks outside them. `import-index` selects only source metadata and locators from `chunks` after contract verification. Every command opens the existing run, executes one stage, commits, and emits aggregate JSON.

- [x] **Step 4: Run focused tests, Ruff, and CLI preflight**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py`

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py eval/reconcile_private_corpus.py tests/evaluation/test_corpus_reconciliation.py`

Run: `uv run python eval/reconcile_private_corpus.py status`

Expected before private initialization: one sanitized blocked object and exit code 1.

- [x] **Step 5: Commit Task 6**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/corpus_reconciliation.py eval/reconcile_private_corpus.py tests/evaluation/test_corpus_reconciliation.py
git commit -m "feat: add private corpus reconciliation cli"
```

### Task 7: Execute Reconciliation And Update The Public Gate

**Files:**
- Modify: `docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md`
- Modify: `docs/CURRICULUM_REMEDIATION_BACKLOG.md`
- Create locally only: `.private/corpus-reconciliation/v1/`

**Interfaces:**
- Consumes: verified CLI, connected Google Drive metadata, approved local roots, configured read-only PostgreSQL URL, index contract, and E1a-3 allocation.
- Produces: private SQLite state, six sealed JSONL snapshots with manifests, aggregate blocker classification, and updated public plan status.

- [x] **Step 1: Run presence-only preflight and initialize the bound run**

Verify Git ignore status, zero tracked private files, exact index contract, exact E1a-3 allocation manifest, and absent E1a-4 sampling frame. Initialize `run-2026-08-23-v1` only after all checks pass.

- [x] **Step 2: Import Google Drive metadata in resumable pages**

Use the connected Drive metadata search/list actions. Pass each provider page and the token used to request it directly to `import-drive-page` through stdin without writing intermediary tracked files or printing private records. Continue until `next_page_token` is absent; if interrupted, resume only from the exact token saved in the existing SQLite run.

- [x] **Step 3: Inventory approved local roots and import the verified index**

Use only roots established by exact anchor provenance. If no approved local root can be derived without guessing, mark `DRIVE_TO_LOCAL_GAP` and stop that branch. Start PostgreSQL only for the read-only import and stop the service afterward if this task started it.

- [x] **Step 4: Run matching, eligibility reconciliation, capacity, and dry-run allocation**

Do not make review decisions automatically. Seal a closed blocker when ambiguous matches remain. If the allocator succeeds, do not run the E1a-4 sampling-frame sealer in this task; report `READY_FOR_E1A4_SAMPLING_APPROVAL`.

- [x] **Step 5: Seal private snapshots and update aggregate-only tracked plans**

Write only counts, safe status, blocker classes, verification evidence, and the next approval gate. Do not include reconstructive small-cell values or private identifiers.

- [x] **Step 6: Run final verification**

Run: `uv run pytest -q tests/evaluation/test_corpus_reconciliation.py`

Run: `uv run pytest -q`

Run: `uv run ruff check .`

Run: `git diff --check`

Verify the private root is ignored, `git ls-files .private` is empty, no private status entry is visible, all snapshot manifests validate, and tracked reports contain no private identifiers or paths.

- [x] **Step 7: Commit tracked Task 7 updates**

```powershell
git add -- docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md docs/CURRICULUM_REMEDIATION_BACKLOG.md
git commit -m "docs: record private corpus reconciliation gate"
```

## Execution Result — 2026-08-23

- Status: `CORPUS_RECONCILIATION_COMPLETE`; the E1a-4 no-write dry run is intentionally `BLOCKED`.
- Sealed private inventory: 385 topic-scoped Drive candidates, 232 local files, 198 contract-verified index sources, 1,874 locator records, and 815 closed match rows.
- Identity result: 117 filename-and-size candidates require human review; no ambiguous candidate was promoted to an exact match.
- Capacity result: all four supporting strata are sufficient and all four foundational strata are insufficient after exact E1a-3 locator exclusion.
- Artifact result: six canonical private JSONL snapshots and six SHA-256 manifests verified, including no-rewrite restart verification.
- Verification: 35 focused reconciliation tests passed; the full suite passed 810 tests with 2 skipped; full Ruff passed; whitespace, Git-ignore, tracked-private-file, and public privacy scans passed.
- Decision: do not run the E1a-4 sampling-frame sealer, ingest/reindex documents, reuse E1a-3 locators, or weaken the exact grid. The next approval gate is a narrow, separately reviewed foundational-locator evidence audit of the already approved corpus.
