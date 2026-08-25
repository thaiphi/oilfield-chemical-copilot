# Production Hardening Holdout and Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sealed 36-case synthetic policy holdout, a provider-free one-shot offline evaluator, and an unwired future service boundary without changing current application behavior.

**Architecture:** New code lives in a standalone `production_hardening` package and remains unwired from the current application; the evaluator's sole frozen-project import is `from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision, classify_claim_scope`. Private authoring, review, approval, seal, digest, and consumed-attempt state remain under a repository-local path excluded through `.git/info/exclude`; tracked task reports contain fixed aggregate counts and booleans only. Preflight binds the holdout, frozen policy source, and evaluator source digests without classifying; after separate approval, scoring atomically consumes an irreversible lock, calls `classify_claim_scope(question)` exactly once per sealed case in memory, reads `.action` and `.category`, and serializes aggregates only.

**Tech Stack:** Python 3 standard library (`dataclasses`, `enum`, `hashlib`, `hmac`, `json`, `pathlib`, `typing`, `unicodedata`), the frozen claim_scope v1 policy interface in `oilfield_chemical_copilot.evaluation.abstention_policy`, pytest, and Git local exclude metadata; local Ollama with Granite remains the default application provider, while OpenAI remains optional and is neither installed nor required.

## Global Constraints

- Project: Oilfield Chemical Troubleshooting Copilot.
- This is approved Option 2 only: 36 independently authored and independently reviewed sealed synthetic cases, a provider-free offline evaluator, approval-gated one-shot scoring, and an unwired service-boundary design.
- Existing policy, 12-case fixture, prompts, retrieval, models, index, live RAG, service, production tests, datasets, and app behavior are frozen: do not create, edit, regenerate, or reformat them. The only permitted frozen-project import is `from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision, classify_claim_scope`, executed lazily after Task 4 lock consumption; call unchanged `classify_claim_scope(question)` exactly once for each sealed case.
- All implementation is additive except the repository-local `.git/info/exclude` entry used to keep private holdout artifacts untracked.
- Local Ollama/Granite remains the default future provider. OpenAI is optional but not required; do not add an OpenAI dependency, credential requirement, or network requirement.
- Never run Docker, the live RAG pipeline, Ollama, OpenAI, a provider SDK, or the application service while implementing or running this plan.
- Do not generate, inspect, or score the 36 holdout cases outside Task 3 and the single approved execution in Task 4.
- Do not put questions, expected or predicted labels, source content, case-level results, filesystem paths, URLs, credentials, exception messages, stack traces, or raw errors in any tracked report or terminal output.
- Every tracked report is aggregate-only: integer counts, booleans, a fixed task number, and a fixed status enum. A failing operation emits only a sanitized error-code count.
- Do not commit, push, open a pull request, stage files, or instruct another worker to do so in any task.
- Strict success criterion: **36/36 exact action matches and 36/36 exact category matches, with 0 false allows, 0 false abstains, and 0 stratum failures.** No partial-pass threshold exists.
- Holdout distribution is fixed: exactly 18 `general_review` cases labeled `allow`; exactly 6 `site_specific_determination`, 6 `field_ready_prescription`, and 6 `complete_input_substitution` cases, each labeled `abstain`.
- The holdout has exactly three strata, `S01`-`S03`, with 12 cases per stratum; every stratum has exactly 6 `allow` and 6 `abstain` cases.

## File Map

- `production_hardening/__init__.py`: marks the isolated package; exports no runtime integration.
- `production_hardening/aggregate_report.py`: enforces the aggregate-only tracked-report schema.
- `production_hardening/holdout_contract.py`: private holdout record types, validation, canonical sealing, hashing, and non-disclosing CLI output.
- `production_hardening/offline_evaluator.py`: provider-free digest preflight and approval-gated one-shot evaluation of the frozen claim_scope v1 classifier.
- `production_hardening/service_boundary.py`: question-only answerability types and protocol for a future, currently unwired service boundary.
- `tests/production_hardening/`: tests only the new isolated package with synthetic toy records that are not holdout cases.
- `.private/production-hardening/holdout/`: git-excluded authoring, review, sealing, approval, and one-shot state.
- `docs/superpowers/reports/2026-08-09-production-hardening-task-*.json`: aggregate-only task reports.

## Acceptance Criteria

- The holdout contains exactly 36 unique synthetic cases in strata `S01`-`S03`, exactly 12 cases per stratum, with distinct author and reviewer identities and an approval for every case.
- The corpus distribution is exactly 18 `general_review` -> `allow`, 6 `site_specific_determination` -> `abstain`, 6 `field_ready_prescription` -> `abstain`, and 6 `complete_input_substitution` -> `abstain`; every stratum contains exactly 6 allow and 6 abstain cases.
- The author and reviewer operate in separate clean contexts and are not exposed to the frozen fixture, prompts, retrieval, models, index, live RAG, service, production tests, datasets, app outputs, or previous evaluation outputs.
- The canonical sealed holdout and SHA-256 digest exist only in the git-excluded local workspace and pass the contract validator without printing private content.
- The evaluator imports only Python standard-library modules, `production_hardening` modules, and exactly `oilfield_chemical_copilot.evaluation.abstention_policy` with names `AbstentionPolicyDecision` and `classify_claim_scope`; it contains no other application, network, provider, Docker, RAG, or service import/execution path.
- Preflight does not call `classify_claim_scope` or compare labels. Scoring cannot start without separate explicit user approval bound to the sealed-holdout SHA-256, frozen policy source SHA-256, and evaluator source SHA-256.
- The first accepted scoring attempt atomically consumes the approval before comparison. A completed or interrupted attempt cannot be rerun with the same approval.
- After lock consumption, `classify_claim_scope(question)` is called exactly once for each of the 36 sealed questions; its `AbstentionPolicyDecision.action` and `.category` attributes remain in memory and only aggregate scores are serialized.
- The sole score is a strict pass only for 36/36 action and category, zero false allows, zero false abstains, and zero stratum failures.
- The service boundary accepts only normalized question text and returns a typed answerability decision. It has no field or parameter for retrieved context, topic, input labels, prompts, citations, provider state, or answer content, and remains unwired and dependency-free.
- A possible future allow adapter may delegate the unchanged normalized question to the unchanged `BasicRagService` exactly once only after an `allow` decision, but that adapter and all wiring are explicit non-goals.
- The final changed-file audit contains only files explicitly named by this plan plus the local exclude entry and private excluded artifacts; all new tests pass.

## Explicit Non-Goals

- No changes to current policy or policy taxonomy.
- No changes to or reuse of the existing 12-case fixture.
- No prompt, retrieval, model, embedding, index, live RAG, service, production-test, dataset, UI, API, or app-behavior changes.
- No provider benchmarking, model tuning, prompt tuning, retrieval tuning, threshold tuning, retries, rescoring, or case replacement after results are known.
- No Docker, Ollama, Granite, OpenAI, internet, external API, or provider SDK call.
- No prediction input file, prediction serialization, or case-level classifier-output serialization.
- No `BasicRagService` adapter, production adapter, route, dependency injection, feature flag, import, or runtime wiring for the service boundary.
- No publication, commit, push, pull request, or case-level report.

---

### Task 1: Holdout Contract and Validator

**Project:** Oilfield Chemical Troubleshooting Copilot

**Task class:** Isolated implementation and unit testing

**Brief:** Define a deterministic private-data contract, aggregate-report guard, validator, canonical sealer, and digest writer for exactly 36 independently authored/reviewed synthetic cases.

**Scope:** Add only the contract/report modules and their tests. Use toy records named `T01` and `T02` in tests; do not author or read any real holdout case and do not inspect frozen project artifacts.

**Model routing:** Use a coding-capable worker in an isolated context. Local Granite is sufficient; no provider call is needed because implementation and tests are local.

**Files:**

- Create: `production_hardening/__init__.py`
- Create: `production_hardening/aggregate_report.py`
- Create: `production_hardening/holdout_contract.py`
- Create: `tests/production_hardening/test_aggregate_report.py`
- Create: `tests/production_hardening/test_holdout_contract.py`
- Create: `docs/superpowers/reports/2026-08-09-production-hardening-task-1.json`
- Modify: none

**Interfaces:**

- `AggregateReport(task: int, status: Literal["pass", "fail", "blocked"], counts: Mapping[str, int], gates: Mapping[str, bool])`
- `write_aggregate_report(destination: Path, report: AggregateReport) -> None`; reject string values outside the fixed `status` enum, negative counts, and keys containing `question`, `label`, `source`, `path`, `url`, `credential`, `secret`, `error`, `exception`, or `trace`.
- `Action = Literal["allow", "abstain"]` and `Category = Literal["general_review", "site_specific_determination", "field_ready_prescription", "complete_input_substitution"]`.
- `RequiredPair(action: Action, category: Category, count: int)`.
- `HoldoutContract(strata: tuple[str, ...], cases_per_stratum: int, allow_per_stratum: int, abstain_per_stratum: int, required_pairs: tuple[RequiredPair, ...])`; require strata exactly `("S01", "S02", "S03")`, `cases_per_stratum == 12`, `allow_per_stratum == 6`, `abstain_per_stratum == 6`, and required pairs exactly `allow/general_review: 18`, `abstain/site_specific_determination: 6`, `abstain/field_ready_prescription: 6`, and `abstain/complete_input_substitution: 6`.
- `AuthoredCase(case_id: str, stratum_id: str, question: str, expected_action: str, expected_category: str, author_id: str, synthetic: bool)`.
- `ReviewDecision(case_id: str, reviewer_id: str, verdict: Literal["approved", "rejected"])`.
- `SealedCase(case_id: str, stratum_id: str, question: str, expected_action: str, expected_category: str, author_id: str, reviewer_id: str, synthetic: Literal[True])`.
- `ValidationSummary(case_count: int, stratum_count: int, approved_count: int, distribution_valid: bool, strata_balance_valid: bool, violation_counts: Mapping[str, int])`; it must contain no case IDs, labels, or content.
- `load_contract(path: Path) -> HoldoutContract`, `load_authored_cases(path: Path) -> tuple[AuthoredCase, ...]`, and `load_reviews(path: Path) -> tuple[ReviewDecision, ...]`.
- `validate_for_sealing(cases: Sequence[AuthoredCase], reviews: Sequence[ReviewDecision], contract: HoldoutContract) -> ValidationSummary`; reject wrong count, duplicate IDs, missing/extra strata, any stratum other than 12 cases with exactly 6 allow and 6 abstain, any action/category pair outside the four required pairs, any corpus pair count other than 18/6/6/6, false `synthetic`, blank question, missing/rejected review, and identical author/reviewer identity.
- `seal_holdout(cases_path: Path, reviews_path: Path, contract_path: Path, sealed_path: Path, digest_path: Path) -> ValidationSummary`; write sorted canonical UTF-8 JSONL with `\n` endings by `case_id`, then write the lowercase SHA-256 hex digest. Never print records or raw exceptions.
- `verify_seal(sealed_path: Path, digest_path: Path, contract: HoldoutContract) -> ValidationSummary`; compare digest with `hmac.compare_digest` and revalidate all records.

**TDD steps:**

- [ ] Write aggregate-report tests proving a 36-count report is accepted and reports containing `question`, `expected_label`, `source_content`, `file_path`, `url`, `credential`, or `raw_error` keys are rejected with the sanitized code `REPORT_SCHEMA_VIOLATION`.
- [ ] Write validator tests using a generated toy matrix of 36 nonsensitive records: three 12-case strata, each with 6 `general_review`/`allow` records and 6 abstain records, and corpus totals of 6 for each abstain category. Cover: the valid matrix; 35 records; duplicate ID; 13 records in `S01`; missing `S03`; a 7-allow/5-abstain `S01` paired with a 5-allow/7-abstain `S02` while corpus totals stay unchanged; 17 `general_review` plus 7 `site_specific_determination`; 5 `field_ready_prescription` plus 7 `complete_input_substitution`; `general_review` paired with `abstain`; an abstain category paired with `allow`; blank question; `synthetic == false`; unknown action; unknown category; missing review; rejected review; and equal author/reviewer IDs.
- [ ] Write sealing tests proving canonical output is invariant to input order, the digest is 64 lowercase hexadecimal characters, tampering causes `SEAL_DIGEST_MISMATCH`, and exception text never contains a question or label.
- [ ] Run `python -m pytest tests/production_hardening/test_aggregate_report.py tests/production_hardening/test_holdout_contract.py -q`; expect failures because the new modules do not exist.
- [ ] Implement the exact interfaces above using only standard-library imports and sanitized exception codes.
- [ ] Run `python -m pytest tests/production_hardening/test_aggregate_report.py tests/production_hardening/test_holdout_contract.py -q`; expect all tests to pass.
- [ ] Run `python -m production_hardening.holdout_contract --self-test`; expect exactly one line, `status=pass checks_failed=0`, and no private values.
- [ ] Create the task report through `write_aggregate_report` with task `1`, status `pass`, integer test/check totals, and boolean `privacy_guard_passed`; inspect keys only with `python -c "import json; d=json.load(open(r'docs/superpowers/reports/2026-08-09-production-hardening-task-1.json')); print(sorted(d))"`.

**Validation:** Both test files pass; import inspection confirms only standard-library dependencies; report-schema negative tests pass; CLI output has one aggregate line; no holdout workspace or real case content exists yet.

**Privacy boundary:** Unit tests use invented toy strings unrelated to oilfield cases. Tests, exceptions, stdout, and the task report must never echo record content, labels, case IDs, paths, URLs, credentials, or raw errors.

**Report:** `docs/superpowers/reports/2026-08-09-production-hardening-task-1.json`, containing only task/status, aggregate counts, and privacy booleans.

**Return:** Return only the report path, `pass`/`fail`/`blocked`, aggregate test counts, and whether the privacy guard passed.

**No commit:** Do not stage, commit, push, or open a pull request.

---

### Task 2: Provider-Free Offline Evaluator and Aggregate Report

**Project:** Oilfield Chemical Troubleshooting Copilot

**Task class:** Isolated implementation and unit testing

**Brief:** Build a deterministic offline evaluator that directly invokes unchanged `classify_claim_scope(question) -> AbstentionPolicyDecision` from `oilfield_chemical_copilot.evaluation.abstention_policy` after digest-bound user approval, with atomic one-shot consumption, strict metrics, and aggregate-only reporting.

**Scope:** Add only the evaluator and tests. Exercise it with toy records and a spy classifier returning attribute-based decisions. The evaluator contains the one exact static lazy frozen-project import, executed only during approved Task 4 scoring; preflight and Task 2 tests hash `src/oilfield_chemical_copilot/evaluation/abstention_policy.py` directly without importing it. Do not change policy code, import any other frozen application module, invoke providers/RAG/service code, or serialize per-case classifier decisions.

**Model routing:** Use a coding-capable worker for implementation. Run evaluation logic as deterministic Python only; do not route scoring to any model.

**Files:**

- Create: `production_hardening/offline_evaluator.py`
- Create: `production_hardening/frozen_scope_audit.py`
- Create: `production_hardening/frozen_scope_manifest.json`
- Create: `tests/production_hardening/test_offline_evaluator.py`
- Create: `tests/production_hardening/test_frozen_scope_audit.py`
- Create: `docs/superpowers/reports/2026-08-09-production-hardening-task-2.json`
- Read only: `src/oilfield_chemical_copilot/evaluation/abstention_policy.py`
- Modify: none

**Interfaces:**

- Under `TYPE_CHECKING`, permit `from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision`; define `PolicyClassifier = Callable[[str], "AbstentionPolicyDecision"]`.
- The exact frozen interface is `classify_claim_scope(question: str) -> AbstentionPolicyDecision`; evaluator extraction must use `decision.action` and `decision.category` attributes. Mapping access such as `decision["action"]`, `decision.get(...)`, or conversion through `dict(...)` is forbidden.
- `locate_policy_source() -> Path`; resolve exactly `Path("src/oilfield_chemical_copilot/evaluation/abstention_policy.py")`, require that frozen `.py` file to exist, and expose no path in stdout/reports. Do not import or execute policy code during source hashing.
- `FrozenScopeManifest(policy_sha256: str, public_fixture_sha256: str)` records only the SHA-256 values for `src/oilfield_chemical_copilot/evaluation/abstention_policy.py` and `eval/public_answer_evaluation.jsonl`; `FrozenScopeAuditSummary(policy_digest_verified: bool, public_fixture_digest_verified: bool, frozen_scope_preserved: bool)` compares their current bytes to the tracked manifest without consulting Git status.
- `load_frozen_classifier() -> PolicyClassifier`; contain exactly `from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision, classify_claim_scope` inside this function and return `classify_claim_scope` under the declared exact interface. Call this loader only after approval verification and atomic lock creation.
- `InMemoryDecision(case_id: str, action: Action, category: Category)`; instances may exist only in local process memory during the approved scoring call and must never be serialized.
- `Approval(scope: Literal["holdout-36-one-shot"], approved: Literal[True], holdout_sha256: str, policy_source_sha256: str, evaluator_source_sha256: str, nonce: str)`.
- `PreflightSummary(case_count: int, seal_valid: bool, holdout_sha256: str, policy_source_sha256: str, evaluator_source_sha256: str, approval_present: bool, approval_digest_matches: bool, attempt_available: bool, classifier_calls: Literal[0])`; preflight hashes the canonical holdout bytes, exact frozen source file `src/oilfield_chemical_copilot/evaluation/abstention_policy.py`, and `production_hardening/offline_evaluator.py`. It must not import/call the classifier or compare labels.
- `ScoreSummary(case_count: int, action_exact: int, category_exact: int, false_allows: int, false_abstains: int, strata_total: int, stratum_failures: int, strict_pass: bool)`.
- `preflight(sealed_path: Path, digest_path: Path, approval_path: Path | None, state_path: Path, contract: HoldoutContract) -> PreflightSummary`.
- `classify_in_memory(cases: Sequence[SealedCase], classifier: PolicyClassifier) -> tuple[InMemoryDecision, ...]`; iterate the 36 canonical cases once, call `decision = classifier(case.question)` exactly once per case, read `decision.action` and `decision.category` exactly once, and retain only those two attribute values in memory until aggregate scoring finishes.
- `score(cases: Sequence[SealedCase], decisions: Sequence[InMemoryDecision], contract: HoldoutContract) -> ScoreSummary`; `false_allows` means policy action is `allow` while expected action is not; `false_abstains` means policy action is `abstain` while expected action is not; a stratum fails if any case in it misses action or category or is a false allow/abstain.
- `evaluate_once(sealed_path: Path, digest_path: Path, approval_path: Path, state_path: Path, report_path: Path, contract_path: Path, classifier: PolicyClassifier | None = None) -> ScoreSummary`.
- `evaluate_once` must recompute and verify all three approved digests, complete structural preflight, atomically create `state_path` with exclusive-create mode, and record only the three digests/nonce/fixed state before importing policy code, making the first classifier call, or comparing labels. If `classifier is None`, it then calls `load_frozen_classifier()`; tests pass a spy. It calls the selected classifier exactly 36 times, scores in memory, discards case-level decisions, and writes a fixed aggregate report. If import/classification/scoring fails after lock acquisition, write sanitized status `fail` and increment `evaluation_failure_count`; never remove or reuse the lock.
- CLI modes: `--preflight` performs no label comparison and prints only fixed aggregate fields; `--score-once` calls `evaluate_once`; there is no retry or force flag.

**TDD steps:**

- [ ] Write toy scoring tests using the fixed 18/6/6/6 distribution and three balanced strata: 36/36 action and category gives strict pass; one action miss fails its stratum; one category miss fails its stratum; an incorrect allow increments `false_allows`; an incorrect abstain increments `false_abstains`; any nonzero error makes `strict_pass == false`; `strata_total` is exactly `3`.
- [ ] Write preflight tests proving missing approval is reported only as `approval_present=false`; a holdout, policy source, or evaluator source digest mismatch is sanitized; preflight creates no state file; direct source hashing does not add `oilfield_chemical_copilot.evaluation.abstention_policy` to `sys.modules`; a spy `load_frozen_classifier` receives zero calls; and `classifier_calls == 0`.
- [ ] Write exactly-once tests with a 36-case spy classifier returning toy objects with `.action` and `.category` attributes. Prove: no approval causes zero calls; valid approval bound to all three digests creates the lock before the first call; each question is passed once and only once; each returned attribute is read once; total calls equal 36; no decision is written to disk; a second evaluator call is rejected with zero additional calls; and an injected failure on call 17 leaves the lock in place with no retry.
- [ ] Write return-shape tests whose attribute-based toy `AbstentionPolicyDecision` passes and whose mapping-only object fails with sanitized code `INVALID_POLICY_DECISION`; assert evaluator source contains no subscript/get/dict conversion for policy results.
- [ ] Write an AST/import test allowing standard-library modules, `production_hardening`, and only the exact frozen-project module `oilfield_chemical_copilot.evaluation.abstention_policy`. Require the runtime import inside `load_frozen_classifier` to import exactly `AbstentionPolicyDecision` and `classify_claim_scope`; permit the same module's type-only `AbstentionPolicyDecision` import under `TYPE_CHECKING`; reject every other `oilfield_chemical_copilot` import, top-level runtime policy import, `docker`, `ollama`, `openai`, `requests`, `httpx`, `urllib.request`, `socket`, `subprocess`, RAG/service imports, and dynamic imports. Assert `src/oilfield_chemical_copilot/evaluation/abstention_policy.py` is never opened for write and is absent from the changed-file list.
- [ ] Write privacy tests that inject a unique question, label, path, URL, credential-shaped string, and exception message and assert none appears in stdout, stderr, exception text, state, or report.
- [ ] Write frozen-scope audit tests proving the tracked manifest contains only the two fixed digest keys and a changed policy or public fixture makes `frozen_scope_preserved=false` even when no Git-tracked file changes exist.
- [ ] Run `python -m pytest tests/production_hardening/test_offline_evaluator.py -q`; expect failure because the evaluator does not exist.
- [ ] Implement the exact interfaces and fixed report schema above.
- [ ] Run `python -m pytest tests/production_hardening/test_offline_evaluator.py tests/production_hardening/test_aggregate_report.py tests/production_hardening/test_holdout_contract.py -q`; expect all tests to pass.
- [ ] Create the Task 2 aggregate report with task `2`, status `pass`, test/check counts, and booleans `preflight_digest_binding_verified`, `policy_digest_verified`, `frozen_scope_manifest_verified`, `classifier_exactly_once`, `provider_free`, `one_shot_guard_passed`, and `privacy_guard_passed`.
- [ ] Create `production_hardening/frozen_scope_manifest.json` through `write_frozen_scope_manifest`; do not print its values or include paths/content in any report.

**Validation:** All new isolated-package tests pass; source-digest, exact-import, attribute-return, and frozen-scope-manifest tests pass; the spy proves exactly 36 calls after lock consumption and zero calls before approval or on retry; the tracked manifest verifies the current frozen policy and public fixture digests; strict-pass truth table exactly matches the global criterion.

**Privacy boundary:** The evaluator may read private records only inside process memory during the approved Task 4 score. All outward surfaces are fixed aggregates; no case-level value or raw error may be serialized or printed.

**Report:** `docs/superpowers/reports/2026-08-09-production-hardening-task-2.json`, aggregate-only.

**Return:** Return only the report path, status, aggregate test counts, and the five safety booleans.

**No commit:** Do not stage, commit, push, or open a pull request.

---

### Task 3: Independent Authoring, Review, and Sealing

**Project:** Oilfield Chemical Troubleshooting Copilot

**Task class:** Controlled private data authoring and independent review

**Brief:** Produce exactly 36 synthetic holdout cases, hand them from author to reviewer through a local shared staging directory outside every worktree, independently review every case, move only the approved artifacts into the controlling worktree's git-excluded `.private` location, seal the corpus there, and delete the staging handoff without exposing private content to implementation or scoring workers.

**Scope:** This task is performed by two different workers in separate fresh contexts and may use isolated linked worktrees. The author creates cases; the reviewer may approve or reject but may not rewrite them. Their only cross-worktree handoff is a run-unique directory beneath the absolute Git common directory, never a worktree path, tracked file, report, terminal transcript, or chat message. The sealing operator validates the staged artifacts, copies them into the controlling worktree's ignored `.private` location, seals there, verifies the seal, and then deletes the exact staging run directory. No worker in this task runs the application or views evaluation results.

**Model routing:** Use two distinct workers and sessions: Author A and Reviewer B, each receiving a role-specific assignment packet with `Project`, `Task class`, `Brief`, `Scope`, `Validation`, `Report`, and `Return`. Local Granite is the default for either role; OpenAI may be used only if the user separately chooses it, but is not required. Never use the same worker/session for both roles, never pass hidden conversation state between them, and use deterministic local Python/PowerShell only for staging, validation, sealing, status checks, and cleanup.

**Files:**

- Modify locally, never stage: `.git/info/exclude` by adding exactly `/.private/production-hardening/holdout/` if absent
- Create transiently, never stage: `<absolute-git-common-dir>/codex-private/production-hardening/holdout/task-3/<32-lowercase-hex-run-id>/contract.json`
- Create transiently, never stage: `<absolute-git-common-dir>/codex-private/production-hardening/holdout/task-3/<32-lowercase-hex-run-id>/author/holdout-authored.jsonl`
- Create transiently, never stage: `<absolute-git-common-dir>/codex-private/production-hardening/holdout/task-3/<32-lowercase-hex-run-id>/reviewer/holdout-reviews.jsonl`
- Create transiently when needed, never stage: `<absolute-git-common-dir>/codex-private/production-hardening/holdout/task-3/<32-lowercase-hex-run-id>/feedback/review-rejections.jsonl`
- Create locally after staged validation: `.private/production-hardening/holdout/contract.json`
- Create locally after staged validation: `.private/production-hardening/holdout/author/holdout-authored.jsonl`
- Create locally after staged validation: `.private/production-hardening/holdout/reviewer/holdout-reviews.jsonl`
- Create locally: `.private/production-hardening/holdout/sealed/holdout-36.jsonl`
- Create locally: `.private/production-hardening/holdout/sealed/holdout-36.sha256`
- Create: `docs/superpowers/reports/2026-08-09-production-hardening-task-3.json`
- Modify tracked files: none

**Interfaces:**

- The controlling worker resolves `git rev-parse --path-format=absolute --git-common-dir` into a variable without echoing it, generates one cryptographically random 16-byte lowercase hexadecimal run ID, and computes the staging root as the exact normalized join `<absolute-git-common-dir>/codex-private/production-hardening/holdout/task-3/<run-id>`. Before creating, opening, copying from, or deleting that directory, it must verify all of: the run ID matches `^[0-9a-f]{32}$`; the normalized staging parent is exactly the normalized join beneath the resolved Git common directory; the normalized staging root is a strict descendant of that parent; the root is not a filesystem root, worktree root, symlink, junction, or other reparse point; and the run directory did not previously exist. A failed check stops with a sanitized code and no path output.
- Every author, reviewer, revision, and sealing assignment independently resolves its own absolute Git common directory and recomputes the staging root from the supplied run ID and fixed suffix. No assignment accepts a caller-supplied arbitrary absolute path. The run ID and path template may appear in the private role packet, but no private file content may be pasted into the packet, chat, return, report, or terminal.
- The controlling worktree pins its final root as its own normalized `.private/production-hardening/holdout/` before dispatch. That final root is not shared between linked worktrees. Only the sealing operator writes there, and only after exclusion, tracked-file, staged-file, and staged-content validation pass.
- No Git porcelain or plumbing command receives a staged handoff file as an input, and no staged handoff byte may be added to the index, object database, stash, patch, commit, or report. Git commands are limited to resolving common/worktree metadata and checking the final `.private` prefix through raw-output-suppressed aggregate guards.
- Reviewer rejection is the only recoverable pre-seal outcome. On any other terminal failure, worker interruption, malformed output, privacy-guard failure, or context-boundary breach, the controller stops dispatch, removes only exact partial final Task 3 files, revalidates the staging path guards, deletes only the exact staging run directory, verifies absence, and returns sanitized aggregate failure fields. It must not inspect, print, salvage, relocate, archive, or reuse staged content. If cleanup cannot be verified, it reports `staging_cleanup_verified=false`, blocks Task 4, and requires manual local cleanup without disclosing the path or content.
- `contract.json` contains the fixed contract only: actions `allow`/`abstain`; categories `general_review`, `site_specific_determination`, `field_ready_prescription`, and `complete_input_substitution`; required pairs/counts 18/6/6/6; strata `S01`-`S03`; 12 cases, 6 allow, and 6 abstain per stratum. It contains no examples from the frozen fixture or application.
- Author output is UTF-8 JSONL matching `AuthoredCase`; IDs are `H001`-`H036`; every record sets `synthetic: true` and `author_id: "author-a"`. Author A must allocate exactly 12 cases to each stratum, with 6 `general_review`/`allow` and 6 abstain cases per stratum, while corpus abstain-category totals are exactly 6 `site_specific_determination`, 6 `field_ready_prescription`, and 6 `complete_input_substitution`.
- Reviewer output is UTF-8 JSONL matching `ReviewDecision`; all 36 IDs occur once, `reviewer_id` is `reviewer-b`, and only `approved` cases may seal.
- Rejections and fixed reason codes are written only to the staging feedback file. The controller returns only an aggregate rejection count to Author A's context; Author A reads the feedback file directly and writes revisions to the author file. Reviewer B then re-reads the staged draft in its independent context. Case text, labels, decisions, IDs, and reason codes are never relayed through chat, reports, returns, or terminal output.
- After all 36 approvals, the sealer validates the three staged inputs without printing them, copies the exact validated bytes to the corresponding final `.private` contract/author/reviewer paths, and produces canonical `SealedCase` JSONL plus its SHA-256 digest through `seal_holdout` at the final `.private` paths. Staging is never used as the final sealed location.

**Context restrictions:**

- Author A may read only its role packet, the staged private contract vocabulary/stratum matrix, its own staged author file, staged rejection feedback when present, and general domain knowledge needed to write synthetic questions. Its packet names no policy or fixture path, and Author A must not run repository discovery or read the existing 12-case fixture, policy source, prompts, retrieval content, models, index, live RAG, service, production tests, datasets, app outputs, policy classifier outputs, reviewer file, or score reports.
- Reviewer B may read only its role packet, the staged private contract, and Author A's staged 36-case draft. Its packet names no policy or fixture path, and Reviewer B must not run repository discovery or read any frozen artifact, policy source, existing fixture, prompt, provider output, application output, or evaluation result. Reviewer B independently checks realism, ambiguity, action/category correctness, exact distribution, stratum balance, synthetic provenance, and leakage.
- The reviewer records case-level decisions only in the private review file. Any tracked review report is counts-only. Rejections return to Author A as private case IDs plus fixed reason codes; revised cases are reviewed again by Reviewer B before sealing.
- The sealing operator may run only path guards, Git exclusion/status guards, byte copies, the contract validator/sealer, seal verification, and cleanup; it may not display file contents. Implementation and scoring workers receive only aggregate validation and the seal digest.

**TDD steps:**

- [ ] In the controlling worktree, add the local exclude line and run `git check-ignore -q .private/production-hardening/holdout/probe` with stdout/stderr suppressed; record only `git_excluded=true` for exit code `0`. Capture `git status --porcelain=v1 --untracked-files=all` and `git ls-files --stage` in memory, inspect them for the final holdout prefix, and emit only booleans `git_status_private_clean` and `git_index_private_clean`. If any guard is false, stop before creating the staging root.
- [ ] Resolve the Git common directory without echoing it, generate the 32-character run ID, compute and validate the exact staging root described above, require it not to exist, create only `contract`, `author`, `reviewer`, and `feedback` children needed by this task, and deny access to other OS accounts while retaining access for the current account used by all linked-worktree workers. Emit only `staging_ready=true`; do not print the resolved path or run ID in terminal output or tracked reports.
- [ ] Create the staged private contract and run `python -m production_hardening.holdout_contract validate-contract --contract <staging-root>/contract.json`; expect `status=pass checks_failed=0` only. Commands must suppress raw stderr and convert failures to fixed sanitized codes.
- [ ] Before authoring, run the validator against the empty staged author/review locations; expect sanitized failure counts for missing 36 records, with no paths or content printed.
- [ ] Author A writes 36 independently created synthetic cases under the stated context restriction: 18 `general_review`/`allow`; 6 each of `site_specific_determination`/`abstain`, `field_ready_prescription`/`abstain`, and `complete_input_substitution`/`abstain`; three 12-case strata with exactly 6 allow and 6 abstain in each. Do not derive, paraphrase, mutate, or inspect the frozen 12-case fixture.
- [ ] Run `python -m production_hardening.holdout_contract validate-draft --contract <staging-root>/contract.json --cases <staging-root>/author/holdout-authored.jsonl`; expect aggregate counts `case_count=36 stratum_count=3 distribution_valid=true strata_balance_valid=true checks_failed=0`.
- [ ] Reviewer B independently reviews all 36 cases and writes one decision per ID to the staged reviewer file. Rejected cases and fixed reason codes go only to the staged feedback file; Author A reads that file directly, revises the staged author file, and returns aggregate counts only. Repeat only staged author/reviewer validation, never scoring.
- [ ] Run `python -m production_hardening.holdout_contract validate-review --contract <staging-root>/contract.json --cases <staging-root>/author/holdout-authored.jsonl --reviews <staging-root>/reviewer/holdout-reviews.jsonl`; expect `case_count=36 stratum_count=3 approved_count=36 distribution_valid=true strata_balance_valid=true checks_failed=0`.
- [ ] Re-run the in-memory Git status/index scans in the controlling, author, and reviewer worktrees. Require no tracked, staged, or status-visible entry containing the final holdout prefix or fixed staging suffix; print only aggregate booleans. Confirm structurally that staging remains beneath Git common metadata and outside all `git worktree list --porcelain` roots, without printing those roots.
- [ ] Copy the validated staged contract, author file, and reviewer file byte-for-byte into their final ignored `.private` paths in the controlling worktree; do not copy the feedback file. Immediately repeat targeted ignore, index, and status checks with raw output suppressed. On any copy or Git-guard failure, remove only exact partial final Task 3 files, securely delete the validated staging run directory, and fail with sanitized booleans/counts.
- [ ] Seal once with `python -m production_hardening.holdout_contract seal --contract .private/production-hardening/holdout/contract.json --cases .private/production-hardening/holdout/author/holdout-authored.jsonl --reviews .private/production-hardening/holdout/reviewer/holdout-reviews.jsonl --sealed .private/production-hardening/holdout/sealed/holdout-36.jsonl --digest .private/production-hardening/holdout/sealed/holdout-36.sha256`; expect only aggregate success output. On seal failure, remove exact partial sealed outputs, delete the validated staging run directory, and return a fixed failure code; never retry sealing with changed content.
- [ ] Run `python -m production_hardening.holdout_contract verify-seal --contract .private/production-hardening/holdout/contract.json --sealed .private/production-hardening/holdout/sealed/holdout-36.jsonl --digest .private/production-hardening/holdout/sealed/holdout-36.sha256`; expect `case_count=36 stratum_count=3 approved_count=36 distribution_valid=true strata_balance_valid=true checks_failed=0`.
- [ ] After successful seal verification, revalidate the exact normalized staging root and suffix, reject any reparse point or path escape, recursively delete only that run directory with a literal-path API, and verify it no longer exists. Do not delete the staging parent or any sibling run. Filesystem overwrites are not a reliable SSD/journal secure-erase mechanism, so do not claim cryptographic erasure; `staging_cleanup_verified=true` means the directory entry and files are inaccessible through the live filesystem after deletion. If deletion or absence verification fails, retain the valid final `.private` seal but mark Task 3 failed, return no digest, and require manual local cleanup before any later task.
- [ ] Run final raw-output-suppressed Git checks: `git check-ignore` on the final sealed file must exit `0`; in-memory `git status --porcelain=v1 --untracked-files=all` and `git ls-files --stage` scans must find no final holdout or staging entry; and the staging run must remain absent. Emit only `git_excluded`, `git_status_private_clean`, `git_index_private_clean`, and `staging_cleanup_verified`.
- [ ] Create the Task 3 report with task `3`, status `pass`, counts `case_count=36`, `stratum_count=3`, `approved_count=36`, `authors=1`, `reviewers=1`, and booleans `distribution_valid`, `strata_balance_valid`, `identities_distinct`, `seal_valid`, `git_excluded`, `git_status_private_clean`, `git_index_private_clean`, `staging_outside_worktrees`, `staging_cleanup_verified`, and `privacy_guard_passed`. Do not put category names, labels, IDs, reason codes, local paths, the run ID, or the digest in the report.

**Validation:** Exactly 36 valid approved cases; exact 18/6/6/6 action/category-pair distribution; exactly three strata with 12 cases, 6 allow, and 6 abstain each; author/reviewer IDs differ; staged validation passes; staging is structurally under Git common metadata and outside every worktree; final bytes and seal digest verify only under the controlling worktree's ignored `.private` location; no holdout artifact is tracked, staged, status-visible, printed, reported, or passed through chat; and the exact staging run directory is absent after sealing.

**Privacy boundary:** During authoring and review, questions, labels, review decisions, reason codes, and source content exist only in the run-unique Git-common staging directory and are exchanged by direct local file reads. After validation they exist only under `.private/production-hardening/holdout/`, and the staging run is deleted after seal verification. No private file or content is tracked, staged, status-visible, copied into docs/tests, pasted into chat or a task return, printed in terminal output, placed in a tracked report, or shown to any implementation worker. Author A cannot read policy, fixture, reviewer output, or evaluation artifacts; Reviewer B cannot read policy, fixture, provider/application output, or evaluation artifacts.

**Report:** `docs/superpowers/reports/2026-08-09-production-hardening-task-3.json`, aggregate-only; it includes staging-location and cleanup booleans but no local path, run ID, digest, category, label, case ID, reason code, or content.

**Return:** Return only the report path, status, aggregate counts/booleans, and, only when seal verification and staging cleanup both succeeded, the final sealed holdout SHA-256 digest value. Do not return staging/final local paths, the run ID, case-level content, IDs, labels, decisions, or reason codes. On any terminal failure, return only the fixed status, sanitized failure-code count, cleanup boolean, and report path if a report was safely written.

**No commit:** Do not stage, commit, push, or open a pull request; the local exclude and private workspace remain local and uncommitted.

---

### Task 4: User-Approved One-Shot Evaluation

**Project:** Oilfield Chemical Troubleshooting Copilot

**Task class:** Approval-gated controlled offline operation

**Brief:** Preflight and bind the sealed holdout, frozen policy source, and evaluator source; stop for separate user approval; then consume that approval in exactly one provider-free execution of `classify_claim_scope(question) -> AbstentionPolicyDecision` from `oilfield_chemical_copilot.evaluation.abstention_policy` over the sealed cases.

**Scope:** This task does not run the service, RAG, Docker, or any model/provider. It imports only the frozen policy module through the evaluator, calls the unchanged classifier exactly once per sealed question after approval/lock consumption, keeps all case-level decisions in memory, and writes aggregates only.

**Model routing:** No model. Use deterministic local Python only. Docker, RAG, Ollama, Granite inference, OpenAI, provider SDKs, and the service are prohibited in every step.

**Files:**

- Read only: `src/oilfield_chemical_copilot/evaluation/abstention_policy.py`
- Read only: `production_hardening/offline_evaluator.py`
- Create after separate approval: `.private/production-hardening/holdout/approval/score-once.json`
- Create atomically at scoring start: `.private/production-hardening/holdout/state/score-once-consumed.json`
- Create: `docs/superpowers/reports/2026-08-09-production-hardening-task-4.json`
- Modify: none

**Interfaces:**

- Preflight computes the sealed holdout SHA-256 from canonical bytes, the policy source SHA-256 from exact file `src/oilfield_chemical_copilot/evaluation/abstention_policy.py` without importing the module, and the evaluator source SHA-256 from `production_hardening/offline_evaluator.py`; it invokes the classifier zero times.
- The approval artifact matches `Approval`, is created only after the user sees aggregate preflight results, and binds those exact three SHA-256 values, a fresh nonce, and scope `holdout-36-one-shot`.
- The consumed-state artifact is created with exclusive mode before the first classifier call or label comparison and contains only the three digests, nonce, and fixed state `consumed`; its existence permanently blocks a second attempt.
- After lock consumption, the evaluator executes the exact lazy import `from oilfield_chemical_copilot.evaluation.abstention_policy import AbstentionPolicyDecision, classify_claim_scope`, calls `classify_claim_scope` once for each canonical sealed case, reads each returned decision's `.action` and `.category` attributes, scores the 36 in-memory decisions, and then discards them without serialization.
- The tracked score report is exactly the aggregate `ScoreSummary` plus task/status, aggregate classifier-call count, and booleans `approved_digests_bound`, `policy_digest_verified`, `evaluator_digest_verified`, and `holdout_digest_verified`. It contains no labels, IDs, questions, classifier decisions, content, paths, URLs, credentials, digest values, or raw errors.

**TDD steps:**

- [ ] Run provider-free tests first: `python -m pytest tests/production_hardening/test_offline_evaluator.py -q`; expect all tests to pass.
- [ ] Run preflight only: `python -m production_hardening.offline_evaluator --preflight --contract .private/production-hardening/holdout/contract.json --sealed .private/production-hardening/holdout/sealed/holdout-36.jsonl --digest .private/production-hardening/holdout/sealed/holdout-36.sha256 --state .private/production-hardening/holdout/state/score-once-consumed.json`; expect aggregate confirmation of 36 sealed cases, a valid seal, three 64-character SHA-256 values, an available attempt, and `classifier_calls=0`. This command must not compare actions/categories.
- [ ] **STOP. Return the aggregate preflight result and request separate explicit user approval to perform the one-shot score. Do not create the approval artifact and do not continue in the same autonomous run.**
- [ ] After a new explicit user approval, create `score-once.json` with the displayed holdout, policy-source, and evaluator-source SHA-256 values plus a fresh nonce. Do not accept implied, prior, standing, or blanket approval.
- [ ] Execute exactly once: `python -m production_hardening.offline_evaluator --score-once --contract .private/production-hardening/holdout/contract.json --sealed .private/production-hardening/holdout/sealed/holdout-36.jsonl --digest .private/production-hardening/holdout/sealed/holdout-36.sha256 --approval .private/production-hardening/holdout/approval/score-once.json --state .private/production-hardening/holdout/state/score-once-consumed.json --report docs/superpowers/reports/2026-08-09-production-hardening-task-4.json`.
- [ ] Do not retry for any result, including strict failure, interruption after lock acquisition, sanitized scoring failure, or report-write failure. Do not delete or edit the consumed-state artifact.
- [ ] Inspect only report keys and value types with `python -c "import json; d=json.load(open(r'docs/superpowers/reports/2026-08-09-production-hardening-task-4.json')); print(sorted(d), {k:type(v).__name__ for k,v in d.items()})"`; never print nested/private data.

**Validation:** Before approval, only preflight has run, classifier-call count is zero, and no consumed state exists. After approval, exactly one consumed state exists; policy and evaluator digests still match approval; classifier-call count is exactly 36; no case-level decision file exists. The report contains `case_count`, `classifier_calls`, `action_exact`, `category_exact`, `false_allows`, `false_abstains`, `strata_total`, `stratum_failures`, and `strict_pass`; strict pass is true only for the exact global criterion.

**Privacy boundary:** This task never prints, returns, or reports questions, labels, case IDs, source content, paths, URLs, credentials, in-memory decisions, case-level outcomes, or raw errors. It never calls Docker, RAG, Ollama, Granite inference, OpenAI, any provider, `BasicRagService`, or the application service.

**Report:** `docs/superpowers/reports/2026-08-09-production-hardening-task-4.json`, aggregate-only and generated by the one scoring attempt.

**Return:** Before approval, return only aggregate preflight fields and the approval request. After scoring, return only the report path, aggregate `ScoreSummary`, aggregate classifier-call count, and the two binding/freeze booleans; never include case-level diagnostics or remediation advice based on holdout results.

**No commit:** Do not stage, commit, push, or open a pull request.

---

### Task 5: Unwired Service-Boundary Design

**Project:** Oilfield Chemical Troubleshooting Copilot

**Task class:** Additive interface design and unit testing

**Brief:** Define a pure, dependency-free, question-only answerability request/decision protocol for a possible future service integration while leaving `BasicRagService` and the current application completely untouched.

**Scope:** Add normalization, request/decision types, protocol, and tests only. The boundary receives one normalized question string and nothing else. Do not create an adapter, accept generation/retrieval/provider data, import `BasicRagService`, or add any inbound import from existing code.

**Model routing:** Use a coding-capable worker for type design. No provider is invoked and no provider choice appears in the boundary.

**Files:**

- Create: `production_hardening/service_boundary.py`
- Create: `tests/production_hardening/test_service_boundary.py`
- Create: `docs/superpowers/reports/2026-08-09-production-hardening-task-5.json`
- Modify: none

**Interfaces:**

- `normalize_question(raw: str) -> str`; apply Unicode NFKC normalization, collapse every whitespace run to one ASCII space, and trim leading/trailing whitespace. Reject an empty result with sanitized code `EMPTY_NORMALIZED_QUESTION`.
- `AnswerabilityRequest(normalized_question: str)`; this is the only request field. Construction rejects a value that is blank or differs from `normalize_question(value)`, so the boundary can receive normalized text only.
- `AnswerabilityDecision(action: Action, category: Category)`; this typed output permits only the four action/category pairs fixed by `HoldoutContract`.
- `AnswerabilityBoundary(Protocol)` with `def decide(self, request: AnswerabilityRequest) -> AnswerabilityDecision`.
- Neither request nor protocol accepts retrieved context, topic, input labels, prompts, citations, provider state/configuration, model names, credentials, request IDs, answer text, generated content, or any second input field.
- The module contains no concrete `AnswerabilityBoundary`, adapter, factory, singleton, network/client import, `BasicRagService` import, or application import.
- Future sequencing constraint, not implemented here: a separately approved allow adapter would call `AnswerabilityBoundary.decide(...)` once and, only for `allow`, delegate the same unchanged normalized question to unchanged `BasicRagService` exactly once. That adapter remains unwired and is an explicit non-goal.

**TDD steps:**

- [ ] Write normalization tests for leading/trailing whitespace, tabs/newlines, repeated spaces, and Unicode compatibility characters; assert deterministic NFKC/collapsed output and `EMPTY_NORMALIZED_QUESTION` for whitespace-only input.
- [ ] Write dataclass/protocol tests proving `fields(AnswerabilityRequest)` is exactly `{"normalized_question"}`, non-normalized construction is rejected, `fields(AnswerabilityDecision)` is exactly `{"action", "category"}`, all four permitted pairs are accepted, and mismatched/unknown pairs are rejected.
- [ ] Write negative signature tests proving no request/protocol parameter or annotation accepts `retrieved_context`, `topic`, `labels`, `prompts`, `citations`, `provider_state`, `provider`, `model`, `answer`, `content`, or a second request field.
- [ ] Write a structural protocol test with a toy in-test fake and verify the module exposes no adapter, client, factory, execution, logging, persistence, or serialization function.
- [ ] Write an AST/import test that permits only `dataclasses`, `typing`, `unicodedata`, `re`, and `production_hardening.holdout_contract`; rejects provider, network, RAG, `BasicRagService`, and other application imports; and rejects top-level calls or side effects.
- [ ] Run `python -m pytest tests/production_hardening/test_service_boundary.py -q`; expect failure because the module does not exist.
- [ ] Implement only the exact normalizer, dataclasses, pair validation, and protocol above.
- [ ] Run `python -m pytest tests/production_hardening/test_service_boundary.py -q`; expect all tests to pass.
- [ ] Run `git diff --name-only`; verify no frozen file is listed and no existing service file has changed.
- [ ] Create the Task 5 report with task `5`, status `pass`, test/check counts, and booleans `single_input_field`, `normalized_input`, `typed_decision`, `forbidden_inputs_absent`, `dependency_free`, and `unwired`.

**Validation:** Normalization, exact-field, typed-pair, negative-signature, protocol, and AST tests pass; changed-file output includes only planned additive files/reports plus local Git exclude metadata; `BasicRagService`, current service, and app files are unchanged and do not import the new boundary.

**Privacy boundary:** The boundary receives only normalized question text and returns only typed action/category data. It cannot receive or retain context, topic, input labels, prompts, citations, provider state, or answer content; it has no logging, persistence, transmission, or reporting helper. Tests use nonsensitive toy values and the task report contains aggregate booleans/counts only.

**Report:** `docs/superpowers/reports/2026-08-09-production-hardening-task-5.json`, aggregate-only.

**Return:** Return only the report path, status, aggregate tests/checks, and six boundary booleans.

**No commit:** Do not stage, commit, push, or open a pull request.

---

### Task 6: Final Review

**Project:** Oilfield Chemical Troubleshooting Copilot

**Task class:** Read-only conformance and privacy review

**Brief:** Verify plan conformance, frozen-scope preservation, privacy controls, one-shot state, strict aggregate result, and unwired boundary without rerunning scoring or exposing private content.

**Scope:** Read code/tests and aggregate report keys; run only the new unit-test suite and non-content metadata checks. Do not reopen or print the sealed holdout, approval, or consumed state, and do not rerun Task 4.

**Model routing:** Use a fresh reviewer context for code and conformance review. No model is used for scoring; local Granite may assist code review but receives no private holdout content.

**Files:**

- Create: `docs/superpowers/reports/2026-08-09-production-hardening-task-6.json`
- Modify: none

**Interfaces:**

- Consume Task 1-5 aggregate reports by schema only.
- Produce `AggregateReport(task=6, status, counts, gates)` with gates `all_new_tests_passed`, `frozen_scope_preserved`, `holdout_git_excluded`, `one_shot_consumed`, `approved_digests_bound`, `policy_digest_verified`, `preflight_digest_binding_verified`, `classifier_exactly_once`, `strict_success`, `boundary_unwired`, and `privacy_review_passed`.
- `strict_success` is true only if Task 4 reports exactly: `case_count=36`, `classifier_calls=36`, `action_exact=36`, `category_exact=36`, `false_allows=0`, `false_abstains=0`, `strata_total=3`, `stratum_failures=0`, and `strict_pass=true`.

**TDD steps:**

- [ ] Run `python -m pytest tests/production_hardening -q`; expect all new tests to pass. Do not run existing production tests because they are frozen.
- [ ] Run `python -m production_hardening.holdout_contract verify-seal --contract .private/production-hardening/holdout/contract.json --sealed .private/production-hardening/holdout/sealed/holdout-36.jsonl --digest .private/production-hardening/holdout/sealed/holdout-36.sha256`; inspect aggregate output only and expect 36 cases, 3 strata, 36 approvals, valid distribution/balance, and zero failures.
- [ ] Run `git check-ignore -q .private/production-hardening/holdout/sealed/holdout-36.jsonl`; expect exit code `0`.
- [ ] Confirm consumed state existence without reading it: `python -c "from pathlib import Path; print('consumed=' + str(Path(r'.private/production-hardening/holdout/state/score-once-consumed.json').is_file()).lower())"`; expect `consumed=true`.
- [ ] Inspect Task 4 aggregate fields only; require `approved_digests_bound=true`, `policy_digest_verified=true`, and `classifier_calls=36`, then apply the exact `strict_success` predicate above. Do not rerun the evaluator under any outcome.
- [ ] Run `python -c "from production_hardening.frozen_scope_audit import audit_frozen_scope; s=audit_frozen_scope(); print('policy_digest_verified=' + str(s.policy_digest_verified).lower() + ' public_fixture_digest_verified=' + str(s.public_fixture_digest_verified).lower() + ' frozen_scope_preserved=' + str(s.frozen_scope_preserved).lower())"`; set `frozen_scope_preserved=false` if either digest differs, regardless of whether Git reports a tracked or untracked file change. Do not print paths, file content, or digest values.
- [ ] Run `git diff --name-only` only as a supplementary scope check; include `production_hardening/frozen_scope_audit.py`, `production_hardening/frozen_scope_manifest.json`, and `tests/production_hardening/test_frozen_scope_audit.py` in the permitted changed-file list, but do not use Git status to determine `frozen_scope_preserved`.
- [ ] Review `production_hardening/service_boundary.py` and its tests; confirm the request has exactly one normalized-text field, the decision is typed, restricted inputs are absent, and there is no adapter, provider client/config, `BasicRagService` import, runtime factory, or application wiring.
- [ ] Scan tracked task-report key names with `python -c "import glob,json; banned=('question','label','source','path','url','credential','secret','error','exception','trace'); docs=[json.load(open(p)) for p in glob.glob(r'docs/superpowers/reports/2026-08-09-production-hardening-task-*.json')]; print('privacy_keys_pass=' + str(not any(any(x in str(k).lower() for x in banned) for d in docs for k in d)).lower())"`; expect `privacy_keys_pass=true`. Do not print report values.
- [ ] Create Task 6's aggregate report. Set status `pass` only when every gate is true; otherwise set `fail` with aggregate failed-gate count only.

**Validation:** All new tests pass; seal metadata validates the exact distribution/balance; private artifacts are ignored; one-shot state exists; all three approved digests were bound; the manifest audit verifies the frozen policy and public fixture digests even if a file is untracked; exactly 36 classifier calls occurred through the attribute-based `AbstentionPolicyDecision` interface; Task 4 meets the exact strict criterion; the question-only boundary remains unwired; all report schemas pass privacy review.

**Privacy boundary:** Review metadata and aggregate keys only. Never read private files into the terminal, copy their content, reveal report paths inside report content, or expose raw failures. A failed gate is reported as a count and boolean only.

**Report:** `docs/superpowers/reports/2026-08-09-production-hardening-task-6.json`, aggregate-only.

**Return:** Return only the Task 6 report path, overall status, aggregate gate/test counts, and the ten gate booleans. Do not provide case-level findings or propose tuning from the holdout.

**No commit:** Do not stage, commit, push, or open a pull request.
