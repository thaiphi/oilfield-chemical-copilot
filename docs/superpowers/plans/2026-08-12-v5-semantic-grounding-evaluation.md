# V5 Semantic-Grounding Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score one fresh, private, sealed 36-case holdout against the corrected production `format_answer()` boundary without reusing or rerunning V4.

**Architecture:** Reuse the hardened semantic-grounding evaluator and production formatter unchanged before the score. A private author creates fresh cases, a separate private reviewer approves each case, and the evaluator seals, preflights, and atomically consumes one digest-bound approval before scoring. Only an aggregate-safe V5 report enters the repository.

**Tech Stack:** Python 3.11 standard library, existing `semantic_grounding` evaluator, production RAG formatter/models, pytest, Ruff, Git-ignore checks.

## Global Constraints

- Do not inspect, modify, or rerun V4 private artifacts.
- Keep V5 draft, review, sealed fixture, approval, state, and diagnostics under `.private/evaluation/semantic_grounding_v5/`.
- Private V5 questions, excerpts, answers, and failure-class details never enter Git, reports, commit messages, or user-facing summaries.
- Use exactly 36 synthetic cases: six categories, with three allow and three fallback cases in each category.
- Use fresh V5 wording and compare normalized V5 question identities against V1-V4 before sealing.
- Do not modify `format_answer()` or evaluator behavior before the V5 score.
- `false_allows == 0` is the V5 pass gate; report false fallbacks separately.
- V5 permits exactly one score. A failed V5 is preserved and never rerun; a later measurement requires a new V6 fixture.
- No Docker, Ollama, OpenAI, database, retrieval, generation, or live-RAG call is used.

---

### Task 1: Verify The Reusable Evaluator Contract

**Files:**
- Read: `src/oilfield_chemical_copilot/evaluation/semantic_grounding.py`
- Read: `tests/evaluation/test_semantic_grounding.py`
- Read: `src/oilfield_chemical_copilot/rag/formatter.py`
- Read: `tests/rag/test_formatter.py`
- Test: `tests/evaluation/test_semantic_grounding.py`
- Test: `tests/rag/test_formatter.py`

**Interfaces:**
- Consumes: `preflight(...)` and `evaluate_once(..., cases, prior_paths, private_root)`.
- Produces: proof that V5 can reuse the hardened production-boundary evaluator without code changes.

- [x] **Step 1: Run the focused reusable-contract tests**

Run:

```powershell
uv run pytest tests/evaluation/test_semantic_grounding.py tests/rag/test_formatter.py -q
uv run ruff check src/oilfield_chemical_copilot/evaluation/semantic_grounding.py tests/evaluation/test_semantic_grounding.py src/oilfield_chemical_copilot/rag/formatter.py tests/rag/test_formatter.py
```

Expected: tests and Ruff pass. A failure stops this task before private V5 authoring.

- [x] **Step 2: Confirm the V5 evaluator call contract**

Use this call shape only after private authoring and approval:

```python
summary = evaluate_once(
    sealed_path,
    digest_path,
    approval_path,
    state_path,
    private_diagnostics_path,
    public_report_path,
    cases=v5_cases,
    prior_paths=v1_to_v4_paths,
    private_root=private_root,
)
```

Expected: `evaluate_once` invokes preflight before consuming state and calls the real formatter once per sealed case.

### Task 2: Generalize The Evaluator Approval Scope Before Private Authoring

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/semantic_grounding.py`
- Modify: `tests/evaluation/test_semantic_grounding.py`

**Interfaces:**
- Consumes: the current V4-only approval model.
- Produces: an evaluator that accepts an explicit `semantic-grounding-v5` approval scope without weakening V4 validation.

- [x] **Step 1: Write the failing public scope test**

```python
def test_v5_approval_scope_is_accepted_only_for_a_v5_run() -> None:
    approval = Approval.for_current_artifacts(sealed_path, formatter_path, scope="semantic-grounding-v5")
    assert approval.scope == "semantic-grounding-v5"
```

- [x] **Step 2: Verify the test fails before implementation**

Run:

```powershell
uv run pytest tests/evaluation/test_semantic_grounding.py -q
```

Expected: FAIL because the current approval model permits only `semantic-grounding-v4`.

- [x] **Step 3: Implement minimal explicit scope support**

Keep approval scopes as a closed literal set and require the expected scope as an explicit evaluator argument. Do not infer scope from a filesystem path.

```python
evaluate_once(..., expected_scope="semantic-grounding-v5")
```

V4 callers must continue to require `semantic-grounding-v4`.

- [x] **Step 4: Run focused evaluator tests and Ruff**

Run:

```powershell
uv run pytest tests/evaluation/test_semantic_grounding.py -q
uv run ruff check src/oilfield_chemical_copilot/evaluation/semantic_grounding.py tests/evaluation/test_semantic_grounding.py
```

Expected: PASS. Freeze the public evaluator code before any V5 private approval is created.

### Task 3: Author And Independently Review Fresh Private V5 Cases

**Files:**
- Create locally only: `.private/evaluation/semantic_grounding_v5/draft/v5.jsonl`
- Create locally only: `.private/evaluation/semantic_grounding_v5/review/v5.jsonl`
- Read locally only: V1-V4 private fixture identities required for normalized-question overlap comparison

**Interfaces:**
- Consumes: `SemanticCase` JSONL records and `ReviewDecision` JSONL records.
- Produces: 36 fresh reviewed V5 records ready for sealing.

- [x] **Step 1: Create the private directory structure**

Create only these directories:

```text
.private/evaluation/semantic_grounding_v5/
  draft/
  review/
  sealed/
  approval/
  state/
  results/
```

- [x] **Step 2: Author the balanced private case matrix**

Create exactly six cases in each category: `exact_value`, `range_bound`, `unit`, `qualifier_condition`, `conflicting_evidence`, and `no_established_threshold`. In each category, assign three records `expected_outcome: "allow"` and three `expected_outcome: "fallback"`.

Every record uses the evaluator's complete private schema:

```json
{
  "case_id": "v5-unit-01",
  "category": "unit",
  "question": "Fresh synthetic V5 question.",
  "excerpts": ["Fresh synthetic V5 evidence."],
  "answer": "Fresh synthetic V5 answer.",
  "expected_outcome": "fallback",
  "failure_class": "unsupported_unit_conversion",
  "author_id": "author-v5"
}
```

Do not reuse V4 text, values, phrasing, or public regression wording.

- [x] **Step 3: Perform the independent private review**

For every authored case, create one review record using a reviewer identity different from `author-v5`:

```json
{"case_id":"v5-unit-01","reviewer_id":"reviewer-v5","verdict":"approved"}
```

The reviewer checks category, intended allow/fallback outcome, clarity, and claimed failure class without reading formatter rules.

- [x] **Step 4: Validate the private matrix before sealing**

Run a private helper invocation that loads V5 cases and reviews and calls:

```python
validate_cases(v5_cases, v5_reviews)
verify_no_prior_overlap(v5_cases, v1_to_v4_paths)
```

Expected: 36 total cases, six cases per category, 18 allow/18 fallback, approved independent reviews, and no V1-V4 normalized-question overlap. Inspect only counts and sanitized status codes.

### Task 4: Seal And Preflight V5 Without Scoring

**Files:**
- Create locally only: `.private/evaluation/semantic_grounding_v5/sealed/v5.jsonl`
- Create locally only: `.private/evaluation/semantic_grounding_v5/sealed/v5.sha256`
- Create locally only: `.private/evaluation/semantic_grounding_v5/approval/v5.json`
- Create locally only: `.private/evaluation/semantic_grounding_v5/state/consumed.json`
- Create later, public aggregate only: `docs/superpowers/reports/2026-08-13-semantic-grounding-v5.json`

**Interfaces:**
- Consumes: approved draft/review JSONL plus V1-V4 identity paths.
- Produces: sealed V5 payload, digest-bound approval, and a passing preflight summary before any score.

- [x] **Step 1: Confirm the private root is ignored**

Run:

```powershell
git check-ignore -v .private/evaluation/semantic_grounding_v5/sealed/
git status --short
```

Expected: `.private/` is ignored and no V5 private file appears in Git status.

- [x] **Step 2: Seal the approved private fixture**

Run a private helper invocation:

```python
seal_cases(draft_path, review_path, sealed_path, digest_path)
```

Expected: canonical sealed JSONL and a 64-character SHA-256 digest. Do not print record content.

- [x] **Step 3: Create digest-bound approval**

Run a private helper invocation:

```python
approval = Approval.for_current_artifacts(sealed_path, formatter_path)
approval_path.write_text(json.dumps(approval.to_mapping(), sort_keys=True) + "\n", encoding="utf-8")
```

Expected: approval scope is `semantic-grounding-v5`, and all approval digests are created only after the public evaluator code is frozen. A scope mismatch stops the task before preflight or scoring; never alter the sealed V5 fixture to fit a stale scope.

- [x] **Step 4: Run preflight and inspect only safe outputs**

Run a private helper invocation using V5 paths, V1-V4 prior paths, and a public report destination:

```python
preflight(
    sealed_path,
    digest_path,
    approval_path,
    state_path,
    formatter_path,
    v5_cases,
    v1_to_v4_paths,
    private_root,
    private_diagnostics_path,
    public_report_path,
)
```

Expected: valid seal, no overlap, matching approval digests, available state, valid private paths, and a public report destination. Inspect only booleans and counts.

### Task 5: One-Shot V5 Score And Aggregate-Safe Evidence

**Files:**
- Create locally only: `.private/evaluation/semantic_grounding_v5/results/diagnostics.json`
- Create locally only: `.private/evaluation/semantic_grounding_v5/state/consumed.json`
- Create: `docs/superpowers/reports/2026-08-13-semantic-grounding-v5.json`
- Modify: `docs/PROJECT_STATUS.md`

**Interfaces:**
- Consumes: passing V5 preflight and a frozen V5 approval.
- Produces: one private score and one public aggregate-only result.

- [x] **Step 1: Obtain approval for the irreversible one-shot score**

Confirm the sealed digest, formatter digest, evaluator digest, prior-overlap status, ignored private root, and available state. Request explicit approval immediately before executing `evaluate_once`.

- [x] **Step 2: Execute exactly one V5 score**

Run `evaluate_once` with sealed V5 paths, V1-V4 prior paths, `private_root`, and an aggregate-report destination under `docs/superpowers/reports/`.

Expected: the state is created exclusively before evaluation and exactly one `SemanticGroundingSummary` is returned. Inspect only aggregate counts and failure-class counts.

- [x] **Step 3: Verify the aggregate report is safe and durable**

Run:

```powershell
git diff -- docs/superpowers/reports/2026-08-13-semantic-grounding-v5.json
git status --short
```

Confirm the report has only `status`, `counts`, `categories`, `failure_classes`, and `gates`; it must contain no question, answer, excerpt, source, path, URL, credential, identifier, or raw error field.

- [x] **Step 4: Apply the result protocol**

If `false_allows == 0`, record the aggregate pass in `docs/PROJECT_STATUS.md` and proceed to review.

V5 failed with seven false allows and one false fallback. Its private artifacts are preserved unchanged and will not be rerun. Public synthetic regressions, distinct from V5 wording, now cover the aggregate-safe failure classes; the formatter correction broadens comparison and unit checks, preserves explicit conditions, recognizes a conflict within one or many sources, and permits an explicitly grounded absent-threshold statement. A fresh V6 design is required before another measurement.

### Task 6: Final Verification And Review

**Files:**
- Modify if needed: `docs/PROJECT_STATUS.md`
- Read: `docs/superpowers/reports/2026-08-13-semantic-grounding-v5.json`

- [x] **Step 1: Run all public validation**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
git diff --check
git check-ignore -v .private/evaluation/semantic_grounding_v5/sealed/
git status --short
```

Expected: public tests and Ruff pass, no whitespace errors, the V5 private root is ignored and unstaged, and only aggregate-safe V5 evidence is Git-visible.

- [x] **Step 2: Make the Terra readiness decision**

Terra reviewed the aggregate result and validation evidence. The public suite, lint, whitespace check, aggregate-report allowlist, and private-ignore check pass. The V5 safety gate failed with seven false allows, so the measurement-readiness verdict is `CHANGES REQUIRED`: preserve V5 and require fresh V6 measurement after this public correction. The evidence is clear; Sol was not needed for a second opinion.

- [ ] **Step 3: Commit only after explicit user request**

Stage only public evaluator changes, tests, aggregate report, status, and durable V5 plan/spec. Never stage `.private/`.

## Plan Self-Review

- Coverage: freshness, balanced categories, independent review, V1-V4 non-overlap, private-only storage, hardened preflight, one-shot scoring, aggregate reporting, pass/fail protocol, and full validation all map to a task.
- Scope: evaluator-scope generalization is the first implementation task because the current public approval type is V4-specific; no private V5 authoring, sealing, or scoring proceeds with an ambiguous scope.
- Privacy: every private artifact path is explicit and every public output is aggregate-only.
- Approval: private authoring and public scope changes begin only after approval of this plan; the one-shot score requires a separate immediate approval.
