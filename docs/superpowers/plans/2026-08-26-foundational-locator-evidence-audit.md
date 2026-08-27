# Foundational Locator Evidence Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Review every unreviewed locator from the already approved foundational source, persist each page-level decision durably, seal a versioned correction proposal, and rerun capacity without changing the index or publishing an E1a-4 sampling frame.

**Architecture:** A focused evaluation module extends the existing private reconciliation SQLite database with bound audit-run, candidate, and append-only decision tables. A narrow private CLI verifies the active reconciliation seal, binds the exact Drive PDF bytes and exact candidate set, extracts only requested PDF pages, records one controller-validated review decision at a time, seals a correction proposal, and calculates hypothetical capacity from current decisions without mutating `index_locators`.

**Tech Stack:** Python 3.12, SQLite, pypdf, existing reconciliation contracts, pytest, Ruff, Google Drive raw-file connector.

**Spec:** `docs/superpowers/specs/2026-08-23-private-corpus-reconciliation-design.md`, plus the foundational-audit gate in `docs/CURRICULUM_REMEDIATION_BACKLOG.md`.

## Global Constraints

- Keep all raw PDF bytes, page text, source identifiers, locator identifiers, decisions, hashes, and correction artifacts beneath the ignored private reconciliation root.
- Do not run retrieval, Qdrant, generation, prompt/model tuning, ingestion, reindexing, or the E1a-4 sampling-frame sealer.
- Do not mutate `index_sources`, `index_locators`, the source PDF, or the active reconciliation seal.
- Bind every audit run to the exact active reconciliation binding digest, raw Drive PDF SHA-256, candidate-set SHA-256, and reconciliation run ID.
- Persist one append-only decision transaction at a time; connection loss may lose only an unsubmitted decision.
- Require exact candidate parity before closure; unresolved or second-review decisions keep the audit blocked.
- Public documentation may contain aggregate counts and blocker classes only.

---

### Task 1: Durable Foundational Audit Contracts

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/foundational_locator_audit.py`
- Create: `tests/evaluation/test_foundational_locator_audit.py`

**Interfaces:**
- Consumes: `ReconciliationStore`, the verified active snapshot binding digest, and current `index_locators` rows.
- Produces: `FoundationalAuditStore.open(database_path: Path, run_id: str, audit_id: str)`, `initialize_audit(store: ReconciliationStore, audit_id: str, snapshot_binding_sha256: str, source_drive_file_id: str, source_file_sha256: str) -> FoundationalAuditStore`, `record_locator_decision(audit: FoundationalAuditStore, record: LocatorAuditDecision) -> AuditStatus`, `audit_status(audit: FoundationalAuditStore) -> AuditStatus`, and strict immutable record types.

- [x] **Step 1: Write failing tests for exact candidate binding and resumable storage**

```python
def test_initialize_audit_binds_exact_candidate_set(tmp_path: Path) -> None:
    store = _reconciliation_with_foundational_candidates(tmp_path)
    audit = initialize_audit(
        store=store,
        audit_id="foundational-locator-audit-v1",
        snapshot_binding_sha256="a" * 64,
        source_drive_file_id="drive-source-1",
        source_file_sha256="b" * 64,
    )
    status = audit_status(audit)
    assert status.candidate_count == 3
    assert status.status == "IN_PROGRESS"


def test_record_locator_decision_commits_one_append_only_transaction(
    tmp_path: Path,
) -> None:
    audit = _initialized_audit(tmp_path)
    result = record_locator_decision(
        audit=audit,
        record=_decision(locator="page:1", decision="KEEP_INELIGIBLE"),
    )
    assert result.current_decision_count == 1
    audit.close()
    reopened = FoundationalAuditStore.open(
        database_path=audit.database_path,
        run_id=audit.run_id,
        audit_id=audit.audit_id,
    )
    assert audit_status(reopened).current_decision_count == 1
```

- [x] **Step 2: Run the contract tests and verify RED**

Run: `python -m pytest -q tests/evaluation/test_foundational_locator_audit.py -k "initialize or record"`

Expected: FAIL because the module and APIs do not exist.

- [x] **Step 3: Implement strict schema and record validation**

Create three SQLite tables under the existing reconciliation database:

```sql
create table foundational_audit_runs (
    run_id text not null,
    audit_id text not null,
    snapshot_binding_sha256 text not null,
    source_id text not null,
    source_drive_file_id text not null,
    source_file_sha256 text not null,
    candidate_set_sha256 text not null,
    candidate_count integer not null,
    status text not null,
    primary key (run_id, audit_id)
);
create table foundational_audit_candidates (
    run_id text not null,
    audit_id text not null,
    source_id text not null,
    locator text not null,
    page_number integer not null,
    primary key (run_id, audit_id, source_id, locator)
);
create table foundational_audit_decisions (
    run_id text not null,
    audit_id text not null,
    decision_id text not null,
    source_id text not null,
    locator text not null,
    decision text not null,
    proposed_topic text,
    reason_code text not null,
    page_text_sha256 text not null,
    reviewer_id text not null,
    supersedes_decision_id text,
    decided_at text not null,
    primary key (run_id, audit_id, decision_id)
);
```

Allow only these exact decision contracts:

```python
AUDIT_DECISION_CONTRACTS = {
    ("PROMOTE_FOUNDATIONAL", "SUBSTANTIVE_TARGET_EVIDENCE"),
    ("KEEP_INELIGIBLE", "NO_TARGET_TOPIC"),
    ("KEEP_INELIGIBLE", "TITLE_OR_INDEX_ONLY"),
    ("KEEP_INELIGIBLE", "INSUFFICIENT_CONTEXT"),
    ("KEEP_INELIGIBLE", "SUPPORTING_ONLY"),
    ("KEEP_INELIGIBLE", "DUPLICATE_PAGE_CONTENT"),
    ("NEEDS_SECOND_REVIEW", "AMBIGUOUS_OR_NONEXTRACTABLE"),
}
```

Require one approved topic only for `PROMOTE_FOUNDATIONAL`; require null topic for every other decision. Derive candidates only from current rows with `source_role='foundational'` and `substantive_status='INELIGIBLE'`. Parse PDF locators only from exact `page:<positive integer>` values.

- [x] **Step 4: Run contract tests and verify GREEN**

Run: `python -m pytest -q tests/evaluation/test_foundational_locator_audit.py -k "initialize or record"`

Expected: PASS.

- [x] **Step 5: Commit Task 1**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/foundational_locator_audit.py tests/evaluation/test_foundational_locator_audit.py
git commit -m "feat: add durable foundational locator audit contracts"
```

### Task 2: Verified PDF Page Boundary

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/foundational_locator_audit.py`
- Create: `eval/audit_foundational_locators.py`
- Modify: `tests/evaluation/test_foundational_locator_audit.py`

**Interfaces:**
- Consumes: raw PDF path, expected PDF SHA-256, and exact bound candidates.
- Produces: `verify_source_pdf(audit: FoundationalAuditStore, pdf_path: Path) -> VerifiedSourcePdf`, `extract_candidate_page(audit: FoundationalAuditStore, pdf_path: Path, locator: str) -> LocatorReviewPacket`, and CLI commands `init`, `next`, `record`, `status`, `seal`, and `capacity` with aggregate-only stdout/stderr.

- [x] **Step 1: Write failing tests for byte identity, page provenance, and safe output**

```python
def test_extract_candidate_page_requires_bound_pdf_and_exact_locator(
    tmp_path: Path,
) -> None:
    pdf = _three_page_pdf(tmp_path)
    packet = extract_candidate_page(
        audit=_initialized_audit(tmp_path, expected_pdf=pdf),
        pdf_path=pdf,
        locator="page:2",
    )
    assert packet.page_number == 2
    assert packet.locator == "page:2"
    assert len(packet.page_text_sha256) == 64


def test_cli_status_never_prints_private_identifiers(capsys) -> None:
    assert cli(["status", "--private-root", str(_root())]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "status", "candidate_count", "current_decision_count", "remaining_count"
    }
```

- [x] **Step 2: Run PDF/CLI tests and verify RED**

Run: `python -m pytest -q tests/evaluation/test_foundational_locator_audit.py -k "pdf or page or cli"`

Expected: FAIL because PDF verification, extraction, and CLI do not exist.

- [x] **Step 3: Implement fail-closed PDF extraction and aggregate-only CLI**

Use `pypdf.PdfReader` only after SHA-256 matches the audit binding. Convert the 1-based locator page to a 0-based reader index, reject out-of-range pages, extract the exact page text, normalize line endings only, and hash the exact normalized UTF-8 text stored in the private packet. `next` returns one private packet only to the controller process; normal stdout reports counts only.

- [x] **Step 4: Run PDF/CLI tests and verify GREEN**

Run: `python -m pytest -q tests/evaluation/test_foundational_locator_audit.py -k "pdf or page or cli"`

Expected: PASS.

- [x] **Step 5: Commit Task 2**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/foundational_locator_audit.py eval/audit_foundational_locators.py tests/evaluation/test_foundational_locator_audit.py
git commit -m "feat: verify foundational audit pdf pages"
```

### Task 3: Versioned Correction Proposal And Hypothetical Capacity

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/foundational_locator_audit.py`
- Modify: `eval/audit_foundational_locators.py`
- Modify: `tests/evaluation/test_foundational_locator_audit.py`

**Interfaces:**
- Consumes: exact current decisions for every candidate and the unchanged reconciliation inventory.
- Produces: `seal_correction_proposal(audit: FoundationalAuditStore) -> AuditSeal`, `verify_correction_proposal(audit: FoundationalAuditStore, expected_binding_sha256: str) -> AuditSeal`, and `calculate_hypothetical_capacity(audit: FoundationalAuditStore) -> HypotheticalCapacityReport` without updating inventory rows.

- [x] **Step 1: Write failing tests for exact closure, atomic sealing, and no-write capacity**

```python
def test_seal_requires_one_current_closed_decision_per_candidate(tmp_path: Path) -> None:
    audit = _initialized_audit(tmp_path)
    with pytest.raises(FoundationalLocatorAuditError, match="AUDIT_INCOMPLETE"):
        seal_correction_proposal(audit=audit)


def test_hypothetical_capacity_does_not_update_index_locators(tmp_path: Path) -> None:
    audit = _completed_audit(tmp_path)
    before = _locator_rows(audit)
    report = calculate_hypothetical_capacity(audit=audit)
    assert _locator_rows(audit) == before
    assert len(report.strata) == 8
```

- [x] **Step 2: Run seal/capacity tests and verify RED**

Run: `python -m pytest -q tests/evaluation/test_foundational_locator_audit.py -k "seal or hypothetical"`

Expected: FAIL because proposal sealing and hypothetical capacity do not exist.

- [x] **Step 3: Implement atomic private proposal sealing and no-write calculation**

Write `foundational-locator-corrections.v1.jsonl`, its SHA-256 manifest, and `audit-binding.v1.json` plus manifest under `.private/corpus-reconciliation/v1/foundational-locator-audit/v1/sealed/`. Include only current decisions and exact binding fields. Require zero unresolved/second-review decisions. Calculate capacity from a temporary in-memory projection that treats current promotions as substantive, foundational, unused locators; call the existing deterministic allocator against that projection and never update SQLite inventory tables.

- [x] **Step 4: Run seal/capacity tests and verify GREEN**

Run: `python -m pytest -q tests/evaluation/test_foundational_locator_audit.py -k "seal or hypothetical"`

Expected: PASS.

- [x] **Step 5: Commit Task 3**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/foundational_locator_audit.py eval/audit_foundational_locators.py tests/evaluation/test_foundational_locator_audit.py
git commit -m "feat: seal foundational locator correction proposal"
```

### Task 4: Execute The 133-Locator Private Review

**Files:**
- Create locally only: `.private/corpus-reconciliation/v1/foundational-locator-audit/v1/`

**Interfaces:**
- Consumes: verified raw Drive PDF bytes and Tasks 1-3 controller.
- Produces: 133 durable current decisions, a verified private correction proposal, and an aggregate hypothetical-capacity result.

- [x] **Step 1: Verify exact Drive PDF identity and initialize the bound audit**

Fetch the exact Drive file as raw bytes, verify its SHA-256, verify its page count covers every bound locator, and initialize only if the active reconciliation binding and candidate-set digest match.

- [x] **Step 2: Review and persist every candidate page**

For each locator in controller order, inspect the exact page text and page rendering when extraction is insufficient. Record `PROMOTE_FOUNDATIONAL`, `KEEP_INELIGIBLE`, or `NEEDS_SECOND_REVIEW` immediately. Do not batch uncommitted decisions.

- [x] **Step 3: Resolve every second-review decision**

Use append-only supersession records. Do not seal while any current decision is `NEEDS_SECOND_REVIEW`.

- [x] **Step 4: Seal and verify the correction proposal**

Run the `seal` command, then perform a separate no-write `verify` call using the sealed audit-binding digest as its direct trust anchor while also requiring the embedded active reconciliation binding to match current audit state.

- [x] **Step 5: Run hypothetical capacity and allocator**

Report only whether each of the eight public strata is sufficient, whether exact 96-slot allocation is available, and aggregate promoted/retained counts. Do not write an E1a-4 population or sampling frame.

### Task 5: Public Closure, Verification, And Review Gate

**Files:**
- Modify: `docs/superpowers/plans/2026-08-23-private-corpus-reconciliation.md`
- Modify: `docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md`
- Modify: `docs/CURRICULUM_REMEDIATION_BACKLOG.md`

**Interfaces:**
- Consumes: verified private aggregate results only.
- Produces: synchronized public gate status and a reviewable branch.

- [x] **Step 1: Update public documents with aggregates only**

Record candidate count, promoted/retained/second-review totals, eight-stratum sufficient/insufficient statuses, allocator available/unavailable status, seal verification, and the next approval gate. Do not include private identifiers, page text, filenames, locators, or hashes.

- [x] **Step 2: Run focused and full verification**

Run:

```powershell
python -m pytest -q tests/evaluation/test_foundational_locator_audit.py
python -m pytest -q tests/evaluation/test_corpus_reconciliation.py
python -m pytest -q
python -m ruff check .
git diff --check
git ls-files .private
git status --short
```

Expected: all tests pass, Ruff and diff checks are clean, `git ls-files .private` is empty, and no private artifact appears in status.

- [x] **Step 3: Request independent review before any mapping application**

Review only the controller code, tests, aggregate public updates, and private artifact contract/digest verification. Do not expose private PDF content or locator decisions. Mapping application remains unauthorized until this review approves the versioned proposal.

- [x] **Step 4: Commit the tracked audit implementation and public result**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/foundational_locator_audit.py eval/audit_foundational_locators.py tests/evaluation/test_foundational_locator_audit.py docs/superpowers/plans/2026-08-26-foundational-locator-evidence-audit.md docs/superpowers/plans/2026-08-23-private-corpus-reconciliation.md docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md docs/CURRICULUM_REMEDIATION_BACKLOG.md
git commit -m "feat: audit foundational locator evidence"
```

## Execution Result — 2026-08-26

- Identity and scope: the exact Drive PDF bytes matched the mounted copy; the bound audit covered all 133 candidates and closed with 133 current decisions and zero second-review items.
- Decision aggregates: 92 candidates were promoted as substantive foundational evidence and 41 remained ineligible. Promotions were iron sulfide 5, scale 26, corrosion 33, and paraffin 28. Retained reasons were 20 title/index-only, 13 no-target-topic, 5 supporting-only, and 3 insufficient-context.
- Artifact result: the stricter private v2 correction seal contains two artifacts and two manifests and was published by one atomic directory rename. It binds the active verified reconciliation seal, exact Drive provenance, exact PDF bytes, candidate set, and all 133 page-text digests. A separate no-write verification passed; the earlier v1 seal is preserved as superseded history.
- Capacity result: all four supporting strata remain sufficient. Foundational scale, corrosion, and paraffin are sufficient; foundational iron sulfide remains insufficient at 5 of 12 fresh locators. The exact 96-slot allocation remains unavailable and no E1a-4 sampling frame was written.
- Verification result: 17 foundational-audit tests, 62 reconciliation tests, and the full 666-test suite passed with 2 skips; Ruff and diff checks passed, and no private file is tracked or visible in Git status.
- Independent review: approved with no Critical, Important, or Minor findings after the controller derived PDF identity from authenticated reconciliation evidence and rejected unrelated bytes.
- Decision at this checkpoint: do not apply the proposed locator mappings, run E1a-4 sampling, ingest/reindex material, change retrieval, reuse E1a-3 evidence, or weaken the grid. The then-next gate was a narrow Iron Sulfide follow-up; its completed result is recorded below.

## Supplement Follow-Up — 2026-08-27

- The user clarified that the approved Iron Sulfide topic folder supplements the core training PDF, so a separate bounded audit tested existing supporting-role pages before any new-source acquisition.
- The authenticated supplement audit froze 252 eligible pages across 17 PDFs and reached its registered stop rule after a 10-page prefix: seven proposed foundational promotions, three retained supporting pages, and zero unresolved decisions.
- Its stricter private v2 atomic seal and separate no-write verification passed, including exact source-root identity and an authenticated same-database binding to this audit's v2 proposal. Combined with this audit's verified proposal, all eight topic/role strata become sufficient and the exact 96-slot allocator succeeds without changing the inventory. The earlier supplement v1 seal is preserved as superseded history.
- Both proposals remain unapplied. Independent review approved the stricter supplement v2 corrections and aggregate result with no findings. The next explicit authorization is whether to apply both verified proposals and seal the deterministic private E1a-4 population; ingestion, reindexing, retrieval, and Qdrant changes remain unauthorized.
