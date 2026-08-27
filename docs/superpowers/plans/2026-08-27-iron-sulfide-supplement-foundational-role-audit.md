# Iron Sulfide Supplement Foundational Role Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the already approved Iron Sulfide supplement contains seven additional fresh, substantive pages that legitimately qualify as foundational evidence, while preserving every decision durably and leaving the corpus and index unchanged.

**Architecture:** A focused multi-PDF audit controller extends the existing private reconciliation SQLite database with a source-bound candidate inventory, page-text bindings, append-only review decisions, and a deterministic early-stop rule. It authenticates the active reconciliation seal, admits only sources with current accepted Drive-to-local identity and byte-identical mounted PDFs, reviews a fixed prefix of the 252 eligible pages, seals a no-write correction proposal atomically, and combines it with the independently approved core-PDF proposal for a hypothetical capacity calculation.

**Tech Stack:** Python 3.12, SQLite, pypdf, existing reconciliation and foundational-audit contracts, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-private-corpus-reconciliation-design.md`, `docs/superpowers/plans/2026-08-26-foundational-locator-evidence-audit.md`, and the user-approved 2026-08-27 clarification that the topic folders supplement the core training PDF.

## Global Constraints

- Keep raw PDF paths, file bytes, extracted page text, source and Drive identifiers, locator identifiers, decisions, and hashes beneath the ignored private reconciliation boundary or inside the private SQLite database.
- Do not run retrieval, Qdrant, generation, prompt/model tuning, ingestion, reindexing, mapping application, or the E1a-4 sampling-frame sealer.
- Do not mutate `drive_files`, `local_files`, `index_sources`, `index_locators`, `document_matches`, `review_decisions`, the active reconciliation seal, or the approved core-PDF correction proposal.
- Candidate rows must be topic `iron_sulfide`, current role `supporting`, substantively eligible, unused by E1a-3, available to E1a-4, and exact `page:<positive integer>` locators beneath the approved supplement folder.
- Admit a source only when the authenticated reconciliation has a unique Drive and local filename mapping, a current `ACCEPT` decision, a local SHA-256, and the mounted PDF bytes equal that SHA-256. Exclude every source that cannot meet the full identity contract.
- Freeze the exact candidate set and deterministic order before reading page content. Review only the next candidate; do not cherry-pick.
- Stop when seven current `PROMOTE_FOUNDATIONAL` decisions exist with no unresolved decision, or after the complete eligible set is reviewed. Do not require review of the unused suffix after the target is met.
- A promoted page must teach a generalizable iron-sulfide principle and contain enough self-contained evidence for at least one technical claim. Case-, product-, vendor-, or job-specific evidence remains supporting.
- Persist each append-only decision in one SQLite transaction. Connection loss may lose only an unsubmitted decision.
- Public documentation may contain aggregate counts and gate outcomes only.

---

### Task 1: Multi-Source Candidate And Provenance Contracts

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/iron_sulfide_supplement_audit.py`
- Create: `tests/evaluation/test_iron_sulfide_supplement_audit.py`

**Interfaces:**
- Consumes: `ReconciliationStore`, the active reconciliation binding SHA-256, the approved supplement root, and the existing index-locator inventory.
- Produces: `IronSulfideSupplementAuditStore`, `initialize_supplement_audit(...)`, `bind_supplement_pages(...)`, `supplement_audit_status(...)`, and immutable source/candidate/status records.

- [x] **Step 1: Write failing tests for exact scope and authenticated source identity**

```python
def test_initialize_freezes_only_fresh_substantive_iron_sulfide_supporting_pages(
    tmp_path: Path,
) -> None:
    store, binding, source_root = _sealed_reconciliation(tmp_path)
    audit = initialize_supplement_audit(
        store=store,
        audit_id="iron-sulfide-supplement-audit-v1",
        snapshot_binding_sha256=binding,
        source_root=source_root,
        promotion_target=7,
    )
    assert supplement_audit_status(audit).candidate_count == 3


def test_initialize_rejects_same_name_pdf_with_untrusted_bytes(tmp_path: Path) -> None:
    store, binding, source_root = _sealed_reconciliation(tmp_path)
    (source_root / "supplement.pdf").write_bytes(b"unrelated")
    with pytest.raises(
        IronSulfideSupplementAuditError,
        match="IRON_SULFIDE_SUPPLEMENT_SOURCE_PROVENANCE_INVALID",
    ):
        initialize_supplement_audit(
            store=store,
            audit_id="iron-sulfide-supplement-audit-v1",
            snapshot_binding_sha256=binding,
            source_root=source_root,
            promotion_target=7,
        )
```

- [x] **Step 2: Run the scope tests and verify RED**

Run: `python -m pytest -q tests/evaluation/test_iron_sulfide_supplement_audit.py -k "initialize or provenance"`

Expected: FAIL because the supplement audit module does not exist.

- [x] **Step 3: Implement strict private schema, source admission, and page binding**

Create private tables with composite keys anchored by `(run_id, audit_id)`:

```sql
create table iron_sulfide_supplement_audit_runs (
    run_id text not null,
    audit_id text not null,
    snapshot_binding_sha256 text not null,
    source_set_sha256 text not null,
    candidate_set_sha256 text not null,
    source_count integer not null,
    candidate_count integer not null,
    promotion_target integer not null,
    status text not null,
    primary key (run_id, audit_id)
);
create table iron_sulfide_supplement_audit_sources (
    run_id text not null,
    audit_id text not null,
    source_id text not null,
    drive_file_id text not null,
    relative_path text not null,
    file_sha256 text not null,
    page_count integer not null,
    primary key (run_id, audit_id, source_id)
);
create table iron_sulfide_supplement_audit_candidates (
    run_id text not null,
    audit_id text not null,
    source_id text not null,
    locator text not null,
    page_number integer not null,
    review_order integer not null,
    page_text_sha256 text,
    primary key (run_id, audit_id, source_id, locator),
    unique (run_id, audit_id, review_order)
);
create table iron_sulfide_supplement_audit_decisions (
    run_id text not null,
    audit_id text not null,
    decision_id text not null,
    source_id text not null,
    locator text not null,
    decision text not null,
    reason_code text not null,
    page_text_sha256 text not null,
    reviewer_id text not null,
    supersedes_decision_id text,
    decided_at text not null,
    primary key (run_id, audit_id, decision_id)
);
```

Authenticate `snapshot_binding_sha256` with `verify_reconciliation_snapshots` before creating audit tables. Resolve source basenames only inside `source_root`, require one mounted PDF, one Drive record, one local record, one matching ambiguous document-match row, and one current `ACCEPT` decision. Require mounted SHA-256, local SHA-256, Drive size, local size, and PDF page range to agree. Freeze candidates ordered by normalized indexed source path and page number; hash the canonical source and candidate records. Read each admitted PDF into memory once, hash the same bytes given to `PdfReader(BytesIO(...))`, normalize only line endings, and transactionally bind every candidate to its page-text SHA-256.

- [x] **Step 4: Run scope tests and verify GREEN**

Run: `python -m pytest -q tests/evaluation/test_iron_sulfide_supplement_audit.py -k "initialize or provenance"`

Expected: PASS.

- [x] **Step 5: Commit Task 1**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/iron_sulfide_supplement_audit.py tests/evaluation/test_iron_sulfide_supplement_audit.py docs/superpowers/plans/2026-08-27-iron-sulfide-supplement-foundational-role-audit.md
git commit -m "feat: add iron sulfide supplement audit contracts"
```

### Task 2: Deterministic Review And Durable Decisions

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/iron_sulfide_supplement_audit.py`
- Create: `eval/audit_iron_sulfide_supplement.py`
- Modify: `tests/evaluation/test_iron_sulfide_supplement_audit.py`

**Interfaces:**
- Consumes: one bound next candidate and its exact mounted PDF.
- Produces: `next_supplement_candidate(...)`, `extract_supplement_page(...)`, `record_supplement_decision(...)`, and aggregate-only CLI commands `init`, `next`, `record`, and `status`.

- [x] **Step 1: Write failing tests for strict prefix order, decision contracts, and early stop**

```python
def test_decisions_must_follow_frozen_candidate_order(tmp_path: Path) -> None:
    audit = _initialized_audit(tmp_path)
    with pytest.raises(
        IronSulfideSupplementAuditError,
        match="IRON_SULFIDE_SUPPLEMENT_DECISION_OUT_OF_ORDER",
    ):
        record_supplement_decision(audit=audit, record=_decision(locator="page:2"))


def test_seventh_promotion_closes_review_without_reviewing_unused_suffix(
    tmp_path: Path,
) -> None:
    audit = _initialized_audit(tmp_path, candidate_count=9, promotion_target=7)
    for index in range(1, 8):
        record_supplement_decision(
            audit=audit,
            record=_promotion(locator=f"page:{index}"),
        )
    status = supplement_audit_status(audit)
    assert status.status == "TARGET_MET"
    assert status.reviewed_count == 7
    assert status.remaining_count == 2
```

- [x] **Step 2: Run review tests and verify RED**

Run: `python -m pytest -q tests/evaluation/test_iron_sulfide_supplement_audit.py -k "decision or target or cli"`

Expected: FAIL because review APIs and CLI do not exist.

- [x] **Step 3: Implement exact review contracts and aggregate-only CLI**

Allow only:

```python
SUPPLEMENT_DECISION_CONTRACTS = {
    ("PROMOTE_FOUNDATIONAL", "GENERALIZABLE_FOUNDATIONAL_EVIDENCE"),
    ("KEEP_SUPPORTING", "CASE_OR_APPLICATION_SPECIFIC"),
    ("KEEP_SUPPORTING", "PRODUCT_OR_VENDOR_SPECIFIC"),
    ("KEEP_SUPPORTING", "PROCEDURAL_OR_JOB_SPECIFIC"),
    ("KEEP_SUPPORTING", "DATA_OR_EXAMPLE_WITHOUT_GENERAL_PRINCIPLE"),
    ("KEEP_SUPPORTING", "TITLE_INDEX_OR_REFERENCE_ONLY"),
    ("KEEP_SUPPORTING", "INSUFFICIENT_STANDALONE_CONTEXT"),
    ("KEEP_SUPPORTING", "DUPLICATE_PAGE_CONTENT"),
    ("KEEP_SUPPORTING", "WRONG_TOPIC"),
    ("NEEDS_SECOND_REVIEW", "AMBIGUOUS_FOUNDATIONAL_ROLE"),
}
```

Require the decision to target the exact next `review_order` and match its bound page-text digest. A `NEEDS_SECOND_REVIEW` row blocks advancement until one append-only superseding decision resolves that same candidate. After seven current promotions, expose `TARGET_MET` and no next candidate. If every candidate is reviewed with fewer than seven promotions, expose `EXHAUSTED_INSUFFICIENT`. Write private review packets atomically beneath the audit root; stdout and stderr contain aggregate status/error codes only.

- [x] **Step 4: Run review tests and verify GREEN**

Run: `python -m pytest -q tests/evaluation/test_iron_sulfide_supplement_audit.py -k "decision or target or cli"`

Expected: PASS.

- [x] **Step 5: Commit Task 2**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/iron_sulfide_supplement_audit.py eval/audit_iron_sulfide_supplement.py tests/evaluation/test_iron_sulfide_supplement_audit.py
git commit -m "feat: add deterministic supplement role review"
```

### Task 3: Atomic Seal And Combined No-Write Capacity

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/iron_sulfide_supplement_audit.py`
- Modify: `eval/audit_iron_sulfide_supplement.py`
- Modify: `tests/evaluation/test_iron_sulfide_supplement_audit.py`

**Interfaces:**
- Consumes: a closed supplement audit, the verified core-PDF v2 proposal, and unchanged reconciliation inventory.
- Produces: `seal_supplement_proposal(...)`, `verify_supplement_proposal(...)`, and `calculate_combined_hypothetical_capacity(...)`.

- [ ] **Step 1: Write failing tests for prefix closure, four-file atomic sealing, and combined no-write capacity**

```python
def test_seal_rejects_gap_in_reviewed_prefix(tmp_path: Path) -> None:
    audit = _audit_with_noncontiguous_decisions(tmp_path)
    with pytest.raises(
        IronSulfideSupplementAuditError,
        match="IRON_SULFIDE_SUPPLEMENT_AUDIT_INCOMPLETE",
    ):
        seal_supplement_proposal(audit=audit)


def test_combined_capacity_never_updates_index_locators(tmp_path: Path) -> None:
    core_audit, supplement_audit = _closed_audits(tmp_path)
    before = _locator_rows(supplement_audit)
    report = calculate_combined_hypothetical_capacity(
        core_audit=core_audit,
        supplement_audit=supplement_audit,
    )
    assert _locator_rows(supplement_audit) == before
    assert report.allocation_available is True
```

- [ ] **Step 2: Run seal/capacity tests and verify RED**

Run: `python -m pytest -q tests/evaluation/test_iron_sulfide_supplement_audit.py -k "seal or combined"`

Expected: FAIL because supplement sealing and combined capacity do not exist.

- [ ] **Step 3: Implement immutable v1 seal and combined projection**

Publish exactly four files under the ignored supplement audit root using one atomic directory rename: canonical current-prefix decisions JSONL, its SHA-256 manifest, an audit binding JSON, and its manifest. Bind schema version, reconciliation binding, source-set digest, candidate-set digest, page-binding digest, stop rule, reviewed-prefix length, decision digest, and the verified core-PDF v2 binding digest. Reject extra files, partial publication, stale manifests, decision gaps, unresolved decisions, or a seal that differs from current SQLite state.

For combined capacity, verify both private seals, reject overlapping locator keys, project core and supplement promotions into a temporary in-memory inventory, call the existing capacity and exact allocator contracts, and never update reconciliation rows or write an E1a-4 sampling frame.

- [ ] **Step 4: Run seal/capacity tests and verify GREEN**

Run: `python -m pytest -q tests/evaluation/test_iron_sulfide_supplement_audit.py -k "seal or combined"`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/iron_sulfide_supplement_audit.py eval/audit_iron_sulfide_supplement.py tests/evaluation/test_iron_sulfide_supplement_audit.py
git commit -m "feat: seal iron sulfide supplement proposal"
```

### Task 4: Execute The Private Supplement Review

**Files:**
- Create locally only: ignored supplement-audit tables and private packet/seal directory beneath the existing reconciliation root.

**Interfaces:**
- Consumes: the 252 frozen provenance-eligible pages and Tasks 1-3 controller.
- Produces: durable prefix decisions, a verified supplement proposal, and aggregate combined-capacity status.

- [ ] **Step 1: Initialize and bind the exact supplement candidate set**

Authenticate the active reconciliation seal, resolve only the approved supplement folder, verify all admitted mounted PDFs against sealed local hashes, bind every candidate page digest, and report aggregate counts only.

- [ ] **Step 2: Review and persist the deterministic prefix**

For each next page, judge foundational role using the frozen definition. Persist the decision immediately. Render and inspect a page when extracted text does not resolve diagrams, tables, or page context. Do not skip ahead or batch uncommitted decisions.

- [ ] **Step 3: Resolve every second-review decision**

Use append-only supersession. Do not advance or seal while the current prefix ends in `NEEDS_SECOND_REVIEW`.

- [ ] **Step 4: Stop only at the registered terminal condition**

Stop when seven promotions are current or the full eligible set is exhausted. Do not review or use the suffix after `TARGET_MET`.

- [ ] **Step 5: Seal, independently verify, and calculate combined capacity**

Publish the private atomic seal, verify it without writing, verify the core-PDF v2 seal, and run the combined no-write capacity and allocator calculation. Do not apply either proposal.

### Task 5: Public Closure And Independent Review

**Files:**
- Modify: `docs/superpowers/plans/2026-08-27-iron-sulfide-supplement-foundational-role-audit.md`
- Modify: `docs/superpowers/plans/2026-08-26-foundational-locator-evidence-audit.md`
- Modify: `docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md`
- Modify: `docs/CURRICULUM_REMEDIATION_BACKLOG.md`

**Interfaces:**
- Consumes: verified private aggregate results only.
- Produces: synchronized public gate status and a reviewable tracked commit.

- [ ] **Step 1: Update public documents with aggregates only**

Record eligible source/page counts, reviewed-prefix and decision totals when safe to disclose, terminal status, seal verification, eight-stratum capacity status, allocator availability, and the next approval gate. Never include private identifiers, filenames, locators, page text, paths, or hashes.

- [ ] **Step 2: Run focused and full verification**

```powershell
python -m pytest -q tests/evaluation/test_iron_sulfide_supplement_audit.py
python -m pytest -q tests/evaluation/test_foundational_locator_audit.py
python -m pytest -q tests/evaluation/test_corpus_reconciliation.py
python -m pytest -q
python -m ruff check .
git diff --check
git ls-files .private
git status --short
```

- [ ] **Step 3: Request independent review before mapping application or E1a-4 sampling**

Review only controller code, tests, aggregate public status, and private artifact contract/digest verification. The reviewer must not inspect private paths, files, page text, identifiers, decisions, or hashes.

- [ ] **Step 4: Commit the tracked implementation and aggregate result**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/iron_sulfide_supplement_audit.py eval/audit_iron_sulfide_supplement.py tests/evaluation/test_iron_sulfide_supplement_audit.py docs/superpowers/plans/2026-08-27-iron-sulfide-supplement-foundational-role-audit.md docs/superpowers/plans/2026-08-26-foundational-locator-evidence-audit.md docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md docs/CURRICULUM_REMEDIATION_BACKLOG.md
git commit -m "feat: audit iron sulfide supplement evidence"
```
