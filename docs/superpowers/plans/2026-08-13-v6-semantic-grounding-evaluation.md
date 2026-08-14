# V6 Semantic-Grounding Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently measure the current production formatter with one fresh, sealed private V6 holdout after V5's public correction.

**Architecture:** First harden recursive aggregate-report privacy validation and add a closed V6 evaluator approval scope; then freeze that public boundary. A private author and independent reviewer prepare a fresh balanced fixture; the evaluator seals, preflights, and consumes it once. Only aggregate-safe evidence is public.

**Tech Stack:** Python 3.11 standard library, production RAG formatter/models, semantic-grounding evaluator, pytest, Ruff, Node test runner, Git-ignore checks.

## Global Constraints

- Never inspect, modify, reuse, or rerun V1-V5 private case content.
- Keep V6 private data only under `.private/evaluation/semantic_grounding_v6/`.
- Keep V6 wording distinct from prior holdouts and public regression tests.
- Use exactly 36 cases: six categories, three allow and three fallback outcomes per category.
- Freeze public formatter/evaluator behavior before V6 approval and scoring.
- The immediate score approval is a user conversation gate tied to the immediately preflighted sealed and code digests; it creates no second approval artifact.
- V6 has one score only and requires a separate immediate approval before that score.
- A V6 failure requires V7; no V6 rerun is permitted.
- No external model, database, retrieval, Docker, or live RAG call is in scope.

### Task 1: Harden Aggregate-Report Privacy Before V6 Authoring

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/semantic_grounding.py`
- Modify: `tests/evaluation/test_semantic_grounding.py`

**Interfaces:**
- Consumes: the `payload` constructed by `_write_aggregate_report()`.
- Produces: recursive rejection of unsafe aggregate-payload keys and string values before report creation.

- [x] **Step 1: Write focused failing privacy tests**

Add direct unit tests for a new `_validate_aggregate_payload(payload: object) -> None` helper. Supply a nested mapping containing an unsafe key such as `"source"`, then a separate nested string value containing an unsafe fragment such as `"private path"`. Each test must raise `EvaluationError` with `AGGREGATE_REPORT_PRIVACY_VIOLATION` before any report file is written.

- [x] **Step 2: Run the focused tests and confirm the expected failure**

```powershell
uv run pytest tests/evaluation/test_semantic_grounding.py -q
```

Expected: both tests fail because current validation checks only failure-class names.

- [x] **Step 3: Implement the recursive validator and use it before writing**

Implement `_validate_aggregate_payload()` to recurse through mappings and sequences, reject unsafe key fragments and unsafe string values using the existing unsafe-fragment policy, and reject unsupported value types. Call it on the complete `payload` in `_write_aggregate_report()` before opening the report path.

- [x] **Step 4: Verify the privacy boundary**

```powershell
uv run pytest tests/evaluation/test_semantic_grounding.py tests/rag/test_formatter.py -q
uv run ruff check src/oilfield_chemical_copilot/evaluation/semantic_grounding.py tests/evaluation/test_semantic_grounding.py src/oilfield_chemical_copilot/rag/formatter.py tests/rag/test_formatter.py
```

Expected: privacy tests, existing evaluator tests, formatter tests, and Ruff pass before any V6 private approval is created.

### Task 2: Add And Verify The V6 Approval Scope

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/semantic_grounding.py`
- Modify: `tests/evaluation/test_semantic_grounding.py`

**Interfaces:**
- Consumes: `Approval.for_current_artifacts(..., scope=...)`, `preflight(..., expected_scope=...)`, and `evaluate_once(..., expected_scope=...)`.
- Produces: explicit acceptance of `semantic-grounding-v6` without weakening V4 or V5 scope checks.

- [x] **Step 1: Write a focused failing scope test**

Create a reviewed and sealed public test fixture with a V6 approval. Assert `preflight(..., expected_scope="semantic-grounding-v6")` fails with `APPROVAL_INVALID` before the extension, and assert no state file is created.

- [x] **Step 2: Implement the minimal closed-scope extension**

Add `semantic-grounding-v6` to `EvaluationScope` and `_load_approval()` only. Preserve the existing explicit `expected_scope` comparisons in `preflight()` and `evaluate_once()`.

- [x] **Step 3: Verify V6 acceptance and wrong-scope rejection**

Extend the same test to prove V6 preflight passes while `expected_scope="semantic-grounding-v5"` raises `APPROVAL_SCOPE_MISMATCH` before one-shot consumption.

- [x] **Step 4: Run focused evaluator and formatter validation**

```powershell
uv run pytest tests/evaluation/test_semantic_grounding.py tests/rag/test_formatter.py -q
uv run ruff check src/oilfield_chemical_copilot/evaluation/semantic_grounding.py tests/evaluation/test_semantic_grounding.py src/oilfield_chemical_copilot/rag/formatter.py tests/rag/test_formatter.py
```

Expected: all focused tests and Ruff pass before any V6 private approval is created.

### Task 3: Author, Review, And Validate The Private V6 Matrix

**Files:**
- Create locally only: `.private/evaluation/semantic_grounding_v6/draft/v6.jsonl`
- Create locally only: `.private/evaluation/semantic_grounding_v6/review/v6.jsonl`

**Interfaces:**
- Consumes: `SemanticCase` and `ReviewDecision` JSONL schemas from `semantic_grounding.py`.
- Produces: a 36-case independently reviewed V6 matrix ready to seal.

- [x] **Step 1: Create the private V6 directory layout**

Create `draft`, `review`, `sealed`, `approval`, `state`, and `results` beneath `.private/evaluation/semantic_grounding_v6/`. Confirm `git check-ignore -v` identifies the root as ignored.

- [x] **Step 2: Author fresh, balanced cases**

Create six synthetic cases each for `exact_value`, `range_bound`, `unit`, `qualifier_condition`, `conflicting_evidence`, and `no_established_threshold`. Each category must contain three `allow` and three `fallback` cases. Use `author_id: "author-v6"`; do not copy V1-V5 or public regression text.

- [x] **Step 3: Independently review every case**

Create exactly one approved review decision per case with `reviewer_id: "reviewer-v6"`. The reviewer verifies the intended category and outcome without reading formatter implementation details.

- [x] **Step 4: Validate private structure without exposing content**

Run `validate_cases()` and `verify_no_prior_overlap()` against V1-V5 sealed identities. Inspect only case/category/outcome counts, review status, and sanitized overlap status.

Expected: 36 cases, six per category, 18 allows, 18 fallbacks, all reviewed, and no normalized-question overlap.

### Task 4: Seal And Preflight V6

**Files:**
- Create locally only: `.private/evaluation/semantic_grounding_v6/sealed/v6.jsonl`
- Create locally only: `.private/evaluation/semantic_grounding_v6/sealed/v6.sha256`
- Create locally only: `.private/evaluation/semantic_grounding_v6/approval/v6.json`

**Interfaces:**
- Consumes: reviewed V6 draft, V1-V5 identity paths, and frozen public formatter/evaluator files.
- Produces: a sealed digest-bound V6 fixture with a passing preflight summary.

- [x] **Step 1: Seal the reviewed fixture**

Use `seal_cases(draft_path, review_path, sealed_path, digest_path)`. Inspect only the canonical record count and whether the SHA-256 digest has 64 hexadecimal characters.

- [x] **Step 2: Create a V6 digest-bound approval**

Use `Approval.for_current_artifacts(sealed_path, formatter_path, scope="semantic-grounding-v6")` and save its JSON mapping under the V6 private approval directory.

- [x] **Step 3: Run V6 preflight without scoring**

Call `preflight(..., expected_scope="semantic-grounding-v6")` with the sealed fixture, V1-V5 prior paths, private root, V6 state path, private diagnostics path, and the public V6 report destination.

Expected: valid seal, no prior overlap, matching approval digests, available state, valid private/public paths, and 36 cases. Do not call `evaluate_once`.

### Task 5: Obtain Immediate Approval And Run The One-Shot V6 Score

**Files:**
- Create locally only: `.private/evaluation/semantic_grounding_v6/state/consumed.json`
- Create locally only: `.private/evaluation/semantic_grounding_v6/results/diagnostics.json`
- Create: `docs/superpowers/reports/2026-08-13-semantic-grounding-v6.json`

**Interfaces:**
- Consumes: passing preflight, digest-bound V6 approval, and frozen formatter/evaluator.
- Produces: one private diagnostic result and one public aggregate-only report.

- [x] **Step 1: Request immediate one-shot approval**

Report only seal validity, digest match, prior-overlap status, private-ignore status, available state, and the frozen public file digests. Wait for an explicit user approval immediately before scoring. This user approval is not stored as a second artifact; it authorizes only the artifact state verified by this preflight. Any code or fixture change invalidates it and requires a new preflight and approval.

- [x] **Step 2: Score exactly once**

Call `evaluate_once(..., expected_scope="semantic-grounding-v6")` with V1-V5 prior paths. Inspect only aggregate counts, category totals, failure-class totals, and integrity gates.

- [x] **Step 3: Verify report safety**

Confirm the public report has exactly `status`, `counts`, `categories`, `failure_classes`, and `gates`, with no sensitive payload keys. Confirm V6 private files remain ignored and absent from `git status --short`.

- [x] **Step 4: Apply the no-rerun result protocol**

Record aggregate counts in `docs/PROJECT_STATUS.md`. A zero-false-allow result is a safety pass; zero false fallbacks is required for full acceptance. Any miss preserves V6 unchanged, adds only distinct public regression coverage when justified, and requires V7.

V6 passed full acceptance: 36/36 cases, zero false allows, and zero false fallbacks. Its fixture is consumed and preserved under the ignored private root; it will not be rerun.

### Task 6: Final Verification And Readiness Decision

**Files:**
- Modify: `docs/PROJECT_STATUS.md`
- Read: `docs/superpowers/reports/2026-08-13-semantic-grounding-v6.json`

**Interfaces:**
- Consumes: aggregate-safe V6 evidence and public correction tests.
- Produces: Terra's evidence-based readiness decision without waiting for Sol unless the aggregate result is ambiguous.

- [x] **Step 1: Run full public verification**

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
git diff --check
git check-ignore -v .private/evaluation/semantic_grounding_v6/sealed/v6.jsonl
git status --short
```

- [x] **Step 2: Make the Terra readiness decision**

Use the aggregate report and verification evidence. A clear false allow produces `CHANGES REQUIRED` without Sol. An ambiguous result may receive a bounded Sol second opinion using only aggregate/public evidence.

Terra verdict: `APPROVE`. V6 passed full acceptance with zero false allows and zero false fallbacks. The aggregate report has only the approved top-level keys, the V6 private fixture is ignored, workflow tests pass, the public Python suite passes, Ruff is clean, and `git diff --check` is clean. The evidence is unambiguous, so Sol was not needed for a second opinion.

- [ ] **Step 3: Commit only after explicit user request**

Stage only public code, tests, aggregate report, project status, and durable V6 design/plan. Never stage `.private/`.

## Plan Self-Review

- Coverage: recursive report privacy, V6 scope isolation, private fresh authoring, independent review, V1-V5 non-overlap, sealing, preflight, immediate one-shot approval, aggregate safety, no-rerun handling, public verification, and Git privacy all have a task.
- Scope: recursive report privacy and one evaluator literal extension are the only public behavior changes before V6 authoring; formatter behavior is otherwise frozen until a V6 result provides evidence.
- Privacy: the plan never requires printed case content and confines all private artifacts to the ignored V6 root.
