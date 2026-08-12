# V4 Semantic-Grounding Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and score one fresh, private, sealed 36-case holdout against the actual `format_answer()` production boundary to measure preservation of technical claim meaning.

**Architecture:** A small evaluation-only package reads a private synthetic fixture, constructs the real `RagDraft` and `SourceEvidence` inputs, then calls the production `format_answer()` unchanged. The evaluator validates, canonicalizes, seals, and hashes the fixture before a one-shot score; it writes detailed diagnostics only under `.private/` and writes one aggregate-only durable report after scoring.

**Tech Stack:** Python 3.11 standard library, existing RAG models and formatter, pytest, Ruff, Git-ignore checks.

## Global Constraints

- The complete v4 fixture, review records, digest, approval, one-shot state, and detailed diagnostics live only below `.private/evaluation/semantic_grounding_v4/` and remain ignored.
- Use 36 fresh synthetic cases: six each for exact values, ranges/bounds, units, qualifiers/conditions, conflicting evidence, and absent-established-threshold statements.
- Each category has both grounded-pass and unsupported-fallback cases. The evaluator records both false allows and false fallbacks.
- The evaluator must call the production `oilfield_chemical_copilot.rag.formatter.format_answer()`; it must not duplicate the grounding rule.
- Before sealing, compare normalized private v4 case identities against v1-v3 identities without emitting any question, excerpt, filename, source content, or diagnostic detail.
- Before the v4 score, do not modify `format_answer()`, its production guards, prompts, retrieval, models, sources, or existing evaluation fixtures.
- Bind the sealed fixture SHA-256, formatter SHA-256, and evaluator SHA-256 to a private approval record. Consume that approval atomically before the one permitted score.
- Public/durable output is aggregate-only. It may contain totals, category labels, confusion-matrix counts, hashes, booleans, and sanitized failure-class counts only.
- If v4 fails, preserve it unchanged; add only a public synthetic regression test and minimum production correction afterward. Do not rerun v4.
- No Docker, Ollama, OpenAI, database, retrieval, or live RAG call is needed.

---

### Task 1: Evaluation Contract and Aggregate Reporter

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/semantic_grounding.py`
- Create: `tests/evaluation/test_semantic_grounding.py`

**Interfaces:**
- `SemanticCase(case_id, category, question, excerpts, answer, expected_outcome, failure_class, author_id)` stores private evaluation inputs only.
- `ReviewDecision(case_id, reviewer_id, verdict)` records a logical independent review for every case.
- `validate_cases(cases, reviews) -> ValidationSummary` requires exactly 36 unique cases, six balanced categories with both outcomes represented, and an approved reviewer distinct from each author.
- `seal_cases(cases_path, reviews_path, sealed_path, digest_path) -> ValidationSummary` writes canonical private JSONL and a SHA-256 digest.
- `verify_no_prior_overlap(cases, prior_paths) -> None` compares normalized private record digests and raises only `PRIOR_CASE_OVERLAP`.
- `evaluate_case(case) -> CaseObservation` calls the real `format_answer()` and records only expected/observed outcome plus a sanitized failure class.
- `aggregate(observations) -> SemanticGroundingSummary` exposes only category totals, pass counts, false allows, false fallbacks, and failure-class counts.
- `preflight(...) -> PreflightSummary` rejects an invalid seal, prior overlap, mismatched approval digest, consumed state, private artifacts outside `private_root`, or a report inside `private_root` before a case is evaluated.
- `evaluate_once(..., cases, prior_paths, private_root) -> SemanticGroundingSummary` calls `preflight`, atomically consumes private approval, evaluates exactly 36 cases once, writes a private diagnostic file, and writes an aggregate-only report.

- [x] **Step 1: Write failing contract tests**

Add toy-only tests that reject a 35-case fixture, duplicate IDs, an unknown category, a category with only one expected outcome, and an overlap digest. Assert a valid 36-case toy matrix seals deterministically and has a 64-character lowercase SHA-256 digest.

- [x] **Step 2: Write failing production-boundary tests**

Use only synthetic public test excerpts. Assert `evaluate_case()` marks a normal grounded answer as allowed only because the real formatter returns `weak_evidence=False`, and marks an unsupported answer as fallback only because that same call returns `weak_evidence=True`.

- [x] **Step 3: Implement the minimal contract**

Use dataclasses and standard-library `hashlib`, `hmac`, `json`, and `pathlib`. Build one `SourceEvidence` per case excerpt with synthetic metadata; construct a valid `RagDraft`; call `format_answer()` exactly once per case. Store no evaluator-side similarity or grounding decision rule.

- [x] **Step 4: Run focused tests and lint**

Run:

```powershell
uv run pytest tests/evaluation/test_semantic_grounding.py -v
uv run ruff check src/oilfield_chemical_copilot/evaluation/semantic_grounding.py tests/evaluation/test_semantic_grounding.py
```

Expected: focused tests and Ruff pass without a production formatter change.

### Task 2: One-Shot Integrity and Private V4 Authoring

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/semantic_grounding.py`
- Modify: `tests/evaluation/test_semantic_grounding.py`
- Create locally only: `.private/evaluation/semantic_grounding_v4/draft/`
- Create locally only: `.private/evaluation/semantic_grounding_v4/review/`
- Create locally only: `.private/evaluation/semantic_grounding_v4/sealed/`
- Create locally only: `.private/evaluation/semantic_grounding_v4/results/`
- Create locally only: `.private/evaluation/semantic_grounding_v4/approval/`
- Create locally only: `.private/evaluation/semantic_grounding_v4/state/`

**Interfaces:**
- `preflight(...) -> PreflightSummary` rejects a sealed-v4 mismatch, v1-v3 overlap, invalid artifact digest, consumed state, private path violation, or report-under-private-root before evaluating a case. Git-ignore status is separately verified by the repository audit command.
- `evaluate_once(..., cases, prior_paths, private_root)` invokes `preflight` itself, then accepts only an approval whose three digests exactly match and creates its state file with exclusive creation before calling `evaluate_case()`.
- The private reviewer approves/rejects every case’s category, expected outcome, and claimed semantic failure class without inspecting formatter code.

- [x] **Step 1: Add failing one-shot and privacy tests**

Use toy private paths supplied by `tmp_path`. Assert mismatched approval hashes fail before evaluation, an existing state file prevents a second evaluation, generated public reports reject fields containing question/excerpt/source/path/error content, and a valid score emits aggregate-only keys.

- [x] **Step 2: Implement approval and atomic consumption**

Create a private approval JSON with `scope`, `approved`, `fixture_sha256`, `formatter_sha256`, `evaluator_sha256`, and nonempty `nonce`. Use `Path.open("x")` to consume it before evaluating; sanitize every failure to a stable code.

- [x] **Step 3: Author and review v4 locally**

Create 36 fresh synthetic records from handout-derived ambiguity patterns without copying handout text. Balance six positive and six negative cases across the six categories. A separate logical reviewer checks category, expected outcome, wording clarity, and no-v1-v3 reuse; record only private approval/rejection metadata.

- [x] **Step 4: Seal and preflight without scoring**

Validate all cases, verify private-only Git-ignore status, verify no v1-v3 identity overlap, seal canonical JSONL, write SHA-256, hash the frozen formatter/evaluator, and create the digest-bound approval. Run preflight and inspect only counts, booleans, and hashes.

- [x] **Step 5: Run focused tests and lint**

Run the Task 1 commands again. Expected: test coverage includes no-second-run and aggregate-only safeguards; no call to Docker, Ollama, or RAG service occurs.

### Task 3: Single Score, Public Evidence, and Failure Protocol

**Files:**
- Create: `docs/superpowers/reports/2026-08-11-semantic-grounding-v4.json`
- Modify: `docs/PROJECT_STATUS.md` only after a successful/private v4 run, using aggregate-only wording.
- Modify only on a v4 failure: `tests/rag/test_formatter.py` and `src/oilfield_chemical_copilot/rag/formatter.py`.

**Interfaces:**
- The durable report has only `status`, `counts`, `categories`, `failure_classes`, and `gates`, all aggregate numeric/boolean/string values.
- A v4 failure does not mutate any v4 file. The public regression test must use a newly written synthetic example, not a v4 record.

- [x] **Step 1: Execute v4 once**

Run `evaluate_once(...)` against the sealed fixture and its digest-bound private approval. Do not retry on a failing score. Inspect only the aggregate summary and private diagnostic class counts.

- [x] **Step 2: Write aggregate-only durable evidence**

Write the report with: total cases, total passes/rate, each category total/pass/false-allow/false-fallback, grounded-cases-correctly-allowed, unsupported-cases-correctly-rejected, aggregate failure-class counts, and integrity gates. Do not include fixture contents, source metadata, private paths, or raw errors.

- [x] **Step 3: Apply the failure protocol only if required**

Preserve the sealed v4 digest and state. Add one new public test for each discovered failure class using different synthetic wording, implement the smallest formatter correction, run focused formatter tests, and explicitly record that v4 was not rerun. If v4 passes, make no formatter change.

- [x] **Step 4: Run full verification and privacy audit**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
git diff --check
git check-ignore -v .private/evaluation/semantic_grounding_v4/sealed/
git status --short
```

Expected: all tracked tests and lint pass, the private fixture is ignored and unstaged, and the durable report is aggregate-only.

## Plan Self-Review

- Spec coverage: all required categories, positive/negative cases, real-boundary scoring, sealing, hashes, v1-v3 nonreuse, one-shot execution, aggregate reporting, failure protocol, privacy, and full validation map to Tasks 1-3.
- Placeholder scan: no deferred implementation step is required to establish the v4 contract, score, or report.
- Boundary check: the evaluator owns integrity and aggregation; `format_answer()` remains the only production grounding decision point until the sealed score identifies a defect.

## Post-Score Review Correction

- The v4 review found that the original public contract described `preflight` as a required gate but `evaluate_once` could bypass it. The implementation now invokes preflight directly and enforces private/public path boundaries before one-shot consumption.
- Fixture validation rejects sensitive key fragments in failure-class labels before sealing; aggregate reporting retains the same check as defense in depth. Public tests cover approval mismatch, private-path rejection, prior-overlap rejection before state consumption, and the pre-seal report-key denylist.
- These corrections do not alter the sealed v4 fixture, its consumed state, or its 24/36 result. V4 remains unrepeated; any new measurement requires a fresh v5 fixture.
