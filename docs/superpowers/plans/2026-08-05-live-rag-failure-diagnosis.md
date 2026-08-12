# Live Public RAG Failure Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the existing live public RAG citation and abstention failures separately for `vector` and `hybrid`, using the same public-only database and local Granite baseline, before approving any retrieval or prompt change.

**Architecture:** Add evaluation-only, runtime instrumentation around the existing retriever and Granite generator. A pure classifier converts each in-memory observation into one citation-failure category and/or one abstention-failure category, and a strict writer persists only aggregate category counts by mode plus the already approved safe provenance. The runner must pass the writer an explicit safe `verified_preflight` status, supplied only after canonical-dataset, public-manifest, and local-configuration checks pass; the writer may not infer preflight success from aggregate counts.

**Tech Stack:** Python 3.11, existing `BasicRagService`, PGVector, MinSearch, local Ollama `granite4.1:8b`, `granite-embedding:latest`, pytest, Ruff.

## Approval Gates

1. **Plan gate:** Do not implement any task until the user explicitly approves this plan.
2. **Live-run gate:** After Tasks 1-3 pass focused and full validation, present the exact command and privacy-test evidence before running against the local public database and Ollama.
3. **Change gate:** After the aggregate diagnosis is reviewed, do not change retrieval, prompts, models, or evaluation cases until the user approves the next experiment.

## What the User Learns

This milestone shows whether each mode's failures occur because no qualifying evidence reaches the RAG service, expected public evidence is absent from the retrieved set, retrieved allowed evidence is not selected for citation, generation fails safely, or the system answers when the case requires abstention. It also shows whether `vector` and `hybrid` have different failure distributions even though their current pass/fail totals are identical.

The categories identify the next layer to investigate; they do not prove the root cause. In particular, `no_qualifying_retrieval` can reflect ranking, thresholding, or context-budget behavior and does not by itself justify lowering a threshold.

## Design Decision

Use runtime-only diagnostic observations and aggregate them immediately. Extending only the current pass/fail summary would not distinguish retrieval-stage, citation-selection, and generation-stage behavior. Persisting case-level records, even with public case IDs, is rejected because the requested boundary is aggregate-only and excludes questions, answers, evidence, chunk IDs, paths, URLs, and raw errors.

## Global Constraints

- Use only `eval/public_answer_evaluation.jsonl` and a database that exactly matches the derived `data/sample` public chunk manifest.
- Reuse local Ollama `granite4.1:8b` with `{"temperature": 0}` and `granite-embedding:latest`; OpenAI and all remote providers are out of scope.
- Run the same 12 questions in `vector` and `hybrid`; vary only `retrieval_mode`, keep all other settings identical, and call `service.answer(question)` with no topic filter.
- Keep questions, answer text, evidence text, retrieved/cited chunk IDs, filenames, paths, database/Ollama URLs, credentials, provider/model strings, and raw exceptions in memory only. Model identifiers remain hashed in safe provenance.
- Persist only whitelisted provenance, question count, baseline-reproduction status, and per-mode aggregate failure-category count maps. Do not persist case IDs or per-case outcomes.
- Do not add logging, tracing, debug dumps, temporary JSONL captures, snapshots, or report fields that can retain runtime material.
- Do not modify retrieval, ranking, thresholds, chunking, embeddings, prompts, RAG production behavior, public questions, or evidence allowlists in this milestone.
- Do not claim a winning retrieval mode, chemistry correctness, operational safety, private-corpus quality, statistical significance, or production readiness.
- Do not commit or push unless the user explicitly requests it.
- Run tasks sequentially; never use more than two concurrent subagents and use `fork_context: false`.

## File Map

- Create `src/oilfield_chemical_copilot/evaluation/live_rag_diagnosis.py`: runtime diagnostic types, exhaustive category classifier, aggregate-only schema validation, and JSON/Markdown writer.
- Create `tests/evaluation/test_live_rag_diagnosis.py`: category coverage, count reconciliation, unsafe-field rejection, and serialization sentinel tests.
- Modify `src/oilfield_chemical_copilot/evaluation/live_rag.py`: evaluation-only recording retriever/generator state and a runtime capture that never retains exceptions.
- Modify `tests/evaluation/test_live_rag.py`: recording/capture lifecycle and data-minimization tests.
- Modify `eval/live_rag_answer_eval.py`: feed in-memory captures to both the existing evaluator and new diagnosis writer without a second RAG call.
- Modify `tests/eval/test_live_rag_answer_eval.py`: mode isolation, baseline reconciliation, aggregate-only output, and preflight tests.
- Modify `README.md`, `docs/COURSE_ALIGNED_PLAN.md`, and `docs/PROJECT_STATUS.md` only after the approved live run: record sanitized aggregate findings and the next decision gate.
- Do not modify `src/oilfield_chemical_copilot/rag/service.py`, `src/oilfield_chemical_copilot/rag/prompt_builder.py`, retrieval modules, the public fixture, or their production tests.

## Expected Diagnostic Categories

Each failed deterministic check contributes to exactly one category in its own dimension. The citation and abstention literal sets below are closed allowlists: categories with zero observations may be omitted from serialized count maps, but every serialized key must be a member of its dimension's allowlist. Unknown literals, contradictory observations, and deterministic failures that cannot be classified must fail closed instead of becoming an `other` bucket.

Classification must use an explicit, deterministic partition. For citation failures, classify abstentions first by no qualifying retrieval, recorded generation failure after qualifying retrieval, or abstention after qualifying retrieval without generation failure. For non-abstaining answers, classify an insufficient-evidence case with citations as unexpected citation; otherwise classify no citations, no allowed evidence retrieved, allowed evidence retrieved but not cited, or mixed allowed/disallowed citations in that order. Reject states that satisfy none or more than one terminal branch. Apply the same exclusive-branch rule to abstention failures.

### Citation failures

- `expected_citation_missing_no_qualifying_retrieval`: a citation-required case abstained and the post-filter retriever returned no IDs.
- `expected_citation_missing_generation_failure`: qualifying retrieval existed, but Granite generation failed safely and no citation was emitted.
- `expected_citation_missing_abstained_after_qualifying_retrieval`: a citation-required case abstained after qualifying retrieval without a recorded generation failure.
- `expected_citation_allowed_evidence_not_retrieved`: the system answered, but none of the case's allowed public evidence IDs was in the qualifying retrieved set.
- `expected_citation_allowed_retrieved_not_cited`: allowed evidence was retrieved, but the answer cited only other retrieved evidence.
- `expected_citation_mixed_with_disallowed`: at least one allowed and at least one non-allowed retrieved item were cited.
- `expected_citation_missing_after_answer`: a non-abstaining answer emitted no citations; this should be unreachable under the current draft contract and therefore highlights contract drift.
- `unexpected_citation_when_abstention_expected`: an insufficient-evidence case answered with one or more citations.

### Abstention failures

- `over_abstention_no_qualifying_retrieval`: a sufficient-evidence case abstained before generation because the post-filter retrieved set was empty.
- `over_abstention_generation_failure`: a sufficient-evidence case had qualifying retrieval but Granite generation failed safely.
- `over_abstention_after_qualifying_retrieval`: a sufficient-evidence case abstained after qualifying retrieval without a recorded generation failure; this is treated as contract drift requiring review.
- `under_abstention_answered_on_insufficient_case`: an insufficient-evidence case returned a generated, cited answer.

## Aggregate Report Contract

The new local, gitignored files are `live_rag_failure_diagnosis.json` and `live_rag_failure_diagnosis.md`. Their schema contains no case list:

```json
{
  "public": true,
  "provenance": "the existing validated safe provenance object",
  "baseline_reproduced": true,
  "modes": {
    "vector": {
      "question_count": 12,
      "citation_failures": {"category_literal": 0},
      "abstention_failures": {"category_literal": 0}
    },
    "hybrid": {
      "question_count": 12,
      "citation_failures": {"category_literal": 0},
      "abstention_failures": {"category_literal": 0}
    }
  }
}
```

`baseline_reproduced` is computed by the writer and is true only when the runner passes `verified_preflight=True`, each mode reruns 12 cases with citation `4 pass / 8 fail` and abstention `6 pass / 6 fail`, every per-case diagnosis reconciles with its paired deterministic result, and the aggregate category sums reconcile to those fail totals. The writer must not infer safe preflight success from counts or provenance alone. If it is false, the writer may record aggregate observed counts for drift review, but the diagnosis cannot support a retrieval or prompt decision.

---

### Task 1: Pure Failure Taxonomy and Aggregate-Only Writer

**Recommended model:** Luna (`gpt-5.6-luna`) because this is isolated two-file work with exact interfaces and deterministic tests.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** targeted implementation.
- **Brief:** `docs/superpowers/plans/2026-08-05-live-rag-failure-diagnosis.md`, Task 1.
- **Scope:** Create `src/oilfield_chemical_copilot/evaluation/live_rag_diagnosis.py`; create `tests/evaluation/test_live_rag_diagnosis.py`. Do not touch other files.
- **Validation:** `uv run pytest tests/evaluation/test_live_rag_diagnosis.py -v` and `uv run ruff check src/oilfield_chemical_copilot/evaluation/live_rag_diagnosis.py tests/evaluation/test_live_rag_diagnosis.py`.
- **Report:** `.codex/reports/live-rag-failure-diagnosis-task-1.md`; include no runtime values, IDs, URLs, paths, or raw errors.
- **Return:** status, changed files, validation results, category-count reconciliation, and concerns.

**Interfaces**

- Define `GenerationOutcome = Literal["not_called", "succeeded", "failed"]`.
- Define runtime-only `LiveDiagnosticObservation` with expectation booleans, `retrieved_evidence_ids`, `cited_evidence_ids`, `abstained`, and `generation_outcome`.
- Define report-safe `LiveFailureDiagnosis` with only `citation_failure: CitationFailureCategory | None` and `abstention_failure: AbstentionFailureCategory | None`; it must not contain a case/question ID.
- Define `classify_live_failure(observation, deterministic_result) -> LiveFailureDiagnosis`; validate that cited IDs are a subset of retrieved IDs and that retrieval, abstention, citation, and generation states select exactly one valid branch for each failed deterministic dimension. Reject contradictory or unclassified states with a generic `ValueError` that contains no runtime values.
- Define closed `CitationFailureCategory` and `AbstentionFailureCategory` allowlists from the literals in **Expected Diagnostic Categories**. Validate every diagnosis and every serialized count-map key against the appropriate allowlist; reject unknown or cross-dimension literals.
- Define `write_live_failure_diagnosis(diagnoses_by_mode, deterministic_results_by_mode, output_dir, provenance, *, verified_preflight: bool) -> tuple[Path, Path]`; accept exactly `vector` and `hybrid`, require equal ordered diagnosis/result lengths, reconcile every paired case before aggregation, validate every category key against its closed dimension allowlist, reject contradictory states, reuse the existing safe provenance validator rather than weakening it, and serialize only the approved schema. Compute `baseline_reproduced` from the explicit safe `verified_preflight` boolean plus the exact baseline and reconciliation conditions; never infer preflight success from counts.

- [ ] Write parameterized failing tests covering every allowlisted category literal, including `expected_citation_missing_abstained_after_qualifying_retrieval`, passing checks producing `None`, and mutually exclusive/exhaustive classification for every failed deterministic dimension.
- [ ] Write failing tests for unknown and cross-dimension category literals, cited-not-retrieved and incompatible generation/answer states, unequal paired lengths, per-case diagnosis/result disagreement even when aggregate totals would match, and category sums that do not match deterministic failure totals. Every rejection must use a generic message without runtime values.
- [ ] Write writer tests proving `baseline_reproduced` remains false when `safe_preflight_passed=False` even if all counts match, and becomes true only when the explicit status, exact baseline totals, per-case reconciliation, and aggregate reconciliation all pass.
- [ ] Write report tests that inject unique privacy sentinels for question, answer, evidence, case/question ID, chunk/source ID, filename/path, URL, credential, provider/model, and raw error into every applicable runtime-only input, then prove byte-for-byte that no sentinel appears in JSON, Markdown, or validation errors.
- [ ] Explicitly validate both closed category allowlists, mutually exclusive/exhaustive failed-dimension branches, contradictory-state rejection, one-to-one per-case deterministic reconciliation before aggregation, and injected sentinel privacy absence.
- [ ] Implement the minimum immutable types, classifier, schema whitelist, count reconciliation, and writer needed to pass.
- [ ] Run the focused pytest and Ruff commands in the assignment packet.

### Task 2: Evaluation-Only Runtime Observation Capture

**Recommended model:** Luna (`gpt-5.6-luna`) because the change is limited to one existing evaluation module and its focused test.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** targeted implementation.
- **Brief:** `docs/superpowers/plans/2026-08-05-live-rag-failure-diagnosis.md`, Task 2.
- **Scope:** Modify `src/oilfield_chemical_copilot/evaluation/live_rag.py` and `tests/evaluation/test_live_rag.py`. Do not modify the RAG service, prompt builder, retrievers, or production tests.
- **Validation:** `uv run pytest tests/evaluation/test_live_rag.py -v` and `uv run ruff check src/oilfield_chemical_copilot/evaluation/live_rag.py tests/evaluation/test_live_rag.py`.
- **Report:** `.codex/reports/live-rag-failure-diagnosis-task-2.md`; report only code/test status and coarse concerns.
- **Return:** status, changed files, validation results, confirmation that runtime exceptions/text/paths are not retained, and concerns.

**Interfaces**

- Add `RecordingRetriever(delegate)` that clears before each call, delegates `retrieve(question, topic=None)`, and retains only the returned chunk-ID tuple for the current call—not hits, text, scores, paths, or query text.
- Extend `RecordingAnswerGenerator` to expose only `generation_outcome`; set it to `not_called` on reset, `succeeded` after a valid draft, and `failed` before re-raising `RagGenerationError`. Never retain the exception or message.
- Return `LiveAnswerCapture(answer: GeneratedAnswer, retrieved_evidence_ids: tuple[str, ...], generation_outcome: GenerationOutcome)` from `capture_live_answer(...)` so the runner can judge and classify the same single service call.
- Keep answer/evidence/cited IDs runtime-only; no write method and no serialization helper may accept `LiveAnswerCapture` directly.

- [ ] Write failing tests for recorder reset between questions, `topic=None` delegation, ID-only retrieval state, generation success/failure/not-called states, and no stale state after safe fallback.
- [ ] Update existing capture tests to access `capture.answer` and verify the existing draft/citation/abstention behavior remains unchanged.
- [ ] Implement the minimum wrappers and capture contract without changing production RAG behavior.
- [ ] Run the focused pytest and Ruff commands in the assignment packet.

### Task 3: Integrate Diagnosis into the Existing Vector-versus-Hybrid Runner

**Recommended model:** Terra (`gpt-5.6-terra`) because this joins the runtime capture, deterministic evaluator, report writer, preflight, and two retrieval pipelines.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** integration implementation.
- **Brief:** `docs/superpowers/plans/2026-08-05-live-rag-failure-diagnosis.md`, Task 3.
- **Scope:** Modify `eval/live_rag_answer_eval.py` and `tests/eval/test_live_rag_answer_eval.py`; consume Task 1 and Task 2 interfaces without modifying them unless a reviewed interface defect blocks integration.
- **Validation:** `uv run pytest tests/eval/test_live_rag_answer_eval.py tests/evaluation/test_live_rag.py tests/evaluation/test_live_rag_diagnosis.py -v`, then focused Ruff on the three implementation modules and three tests.
- **Report:** `.codex/reports/live-rag-failure-diagnosis-task-3.md`; include aggregate fake-test counts only and no sentinel payload values.
- **Return:** status, changed files, validation results, output filenames/schema, privacy assertions, and concerns.

**Interfaces and behavior**

- Wrap each mode-specific pipeline in `RecordingRetriever`; keep all non-mode settings identical and preserve the single public-manifest preflight.
- For each case, call `capture_live_answer` exactly once, pass `capture.answer` to the existing deterministic evaluator/judge path, classify from the same in-memory capture, and discard per-case runtime material after aggregation.
- Continue writing the existing baseline comparison report unchanged; additionally write the aggregate-only diagnosis report. Do not add diagnosis fields to the existing report-safe per-case model.
- Pair each case's diagnosis with the deterministic result from that same single call and require per-case reconciliation before aggregation; do not rely only on final count equality.
- Preserve canonical-dataset, local-provider/model, public-manifest, temperature-zero, no-topic-filter, and unsafe-provenance rejections before runtime construction. Produce `verified_preflight=True` only after the canonical dataset, public manifest, and local configuration checks all succeed, and pass that explicit status to `write_live_failure_diagnosis`; never derive or substitute it from counts.
- Require the writer's closed category allowlist validation, mutually exclusive/exhaustive classification, contradictory-state checks, one-to-one per-case reconciliation, aggregate reconciliation, and injected-sentinel privacy tests to pass before accepting either output file.

- [ ] Extend fake-runner tests to cover one retrieval-stage failure, one generation-stage failure, `expected_citation_missing_abstained_after_qualifying_retrieval`, one citation-selection failure, and one under-abstention failure in each mode.
- [ ] Prove each mode receives identical questions/settings, only `retrieval_mode` differs, and each case triggers one service call rather than a diagnostic rerun.
- [ ] Prove the runner calls the writer with `verified_preflight=True` only after every safe preflight check succeeds; prove failed preflight rejects before runtime construction/output, and matching counts alone cannot mark the baseline reproduced.
- [ ] Add deterministic fake cases whose aggregate totals can match while case pairings are wrong and assert rejection, plus fake contradictory capture/result states and unknown category literals that must fail closed.
- [ ] Assert output filenames, exact schemas, and allowlisted category keys; inject unique privacy sentinels into all applicable fake runtime fields and scan both diagnosis files and surfaced validation errors for byte-for-byte absence, including case/question IDs and chunk/source IDs.
- [ ] Implement the minimum runner integration and run the focused validation.
- [ ] Run `uv run pytest -q` and `uv run ruff check .`; stop for review if unrelated pre-existing failures occur rather than broadening scope.

### Task 4: Approved Live Public Run and Learning Record

**Recommended model:** Terra (`gpt-5.6-terra`) because this task owns environment validation, the live integration run, privacy review, and coordinated source-of-truth updates.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** live validation and documentation.
- **Brief:** `docs/superpowers/plans/2026-08-05-live-rag-failure-diagnosis.md`, Task 4, only after the live-run gate is approved.
- **Scope:** Generate local gitignored reports under `data/processed/evaluation/live_rag`; modify only `README.md`, `docs/COURSE_ALIGNED_PLAN.md`, and `docs/PROJECT_STATUS.md` with sanitized aggregates and the approved interpretation.
- **Validation:** exact public-manifest preflight; exact live command below; JSON/Markdown schema and privacy scan; focused tests; `uv run pytest -q`; `uv run ruff check .`; `git status --short` to confirm no generated/private artifacts are staged.
- **Report:** `.codex/reports/live-rag-failure-diagnosis-task-4.md`; include category counts by mode, baseline-reproduction status, validation results, and risks only.
- **Return:** status, changed documentation files, aggregate categories by mode, baseline reconciliation, validation/privacy evidence, and the recommended next decision gate.

- [ ] Confirm the database exactly matches `public_sample_chunk_ids()` and the canonical dataset/hash and baseline environment match the completed run.
- [ ] Run the approved local command:

```powershell
$env:LLM_PROVIDER = "ollama"
$env:ANSWER_EVAL_JUDGE_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "granite4.1:8b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:EMBEDDING_PROVIDER = "ollama"
$env:OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"
uv run python eval/live_rag_answer_eval.py --dataset eval/public_answer_evaluation.jsonl --output-dir data/processed/evaluation/live_rag --database-url $env:DATABASE_URL
```

- [ ] Verify each mode ran 12 questions, category sums equal observed deterministic fail totals, and `baseline_reproduced` is true only if the recorded `4/8` citation and `6/6` abstention counts recur.
- [ ] Parse the JSON against the exact whitelist and scan JSON/Markdown for forbidden keys and recognizable question, answer, evidence, ID, path, URL, credential, model/provider, and raw-error material. Do not print forbidden values during the scan.
- [ ] If the baseline drifted or any count is unclassified, stop: record only safe aggregate drift status and do not update a retrieval/prompt recommendation.
- [ ] If valid, update the three source-of-truth documents with only aggregate category counts, bounded interpretation, learning outcome, risks, and the next approval gate.
- [ ] Run all validation in the assignment packet and confirm generated reports remain gitignored and unstaged.

### Task 5: Final Milestone Review

**Recommended model:** Sol (`gpt-5.6-sol`) because Sol is reserved for final milestone review and should not implement fixes.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** final milestone review.
- **Brief:** Review this plan, Tasks 1-4 reports, scoped diffs, and aggregate diagnosis only.
- **Scope:** Read-only review of `src/oilfield_chemical_copilot/evaluation/live_rag_diagnosis.py`, `src/oilfield_chemical_copilot/evaluation/live_rag.py`, `eval/live_rag_answer_eval.py`, their focused tests, the three updated docs, and `.codex/reports/live-rag-failure-diagnosis-task-1.md` through `task-4.md`.
- **Validation:** Reconcile requirements to tests/results; verify report schemas and category sums; confirm forbidden data is absent and no retrieval/prompt behavior changed.
- **Report:** `.codex/reports/live-rag-failure-diagnosis-final-review.md`; aggregate findings only.
- **Return:** approve/block status, requirement gaps, privacy findings, validation evidence, risks, and the proposed next gate. Do not edit files.

- [ ] Confirm all acceptance criteria below have direct evidence.
- [ ] Confirm no out-of-scope file or behavior change entered the milestone.
- [ ] Return the review to the user for the change gate; do not start the next experiment.

## Acceptance Criteria

- The canonical 12-case public dataset runs once per mode through the existing `BasicRagService`, public-only database, local Granite generation/judge, temperature `0`, and no topic filter.
- The exact public-manifest and baseline-configuration checks run before retrieval/generation construction and reject mixed, incomplete, private, remote-provider, or nonbaseline inputs.
- Retrieval and prompt implementation remain unchanged; only evaluation modules, the live runner, focused tests, local gitignored reports, and post-run source-of-truth docs change.
- Every citation and abstention failure is assigned to exactly one approved category in its dimension; category totals reconcile to deterministic fail totals per mode, with no `other` bucket.
- The citation allowlist includes `expected_citation_missing_abstained_after_qualifying_retrieval`; both category dimensions are closed, mutually exclusive, and exhaustive for failed deterministic checks. Unknown/cross-dimension literals and contradictory or unclassified states are rejected with generic, data-free errors, and allowlist validation is required in Task 1 and Task 3.
- Every diagnosis reconciles with its paired per-case deterministic result before aggregation; aggregate equality cannot conceal swapped or contradictory case outcomes.
- The diagnosis report contains only safe provenance, baseline-reproduction status, question counts, modes, approved category literals, and integer counts. It contains no case IDs, questions, answers, evidence, chunk/source IDs, filenames, paths, URLs, credentials, raw errors, unhashed model identifiers, or per-case outcomes.
- Privacy tests inject unique sentinels into every applicable runtime-only field and prove byte-for-byte absence from JSON, Markdown, and validation errors; the live schema/privacy scan passes; generated reports remain gitignored and unstaged.
- `baseline_reproduced` is false unless the runner explicitly passes verified safe preflight status and the prior 12-question, citation `4 pass / 8 fail`, abstention `6 pass / 6 fail` totals recur in both modes with successful per-case and aggregate reconciliation. Matching counts or provenance alone are insufficient.
- Focused tests, the full pytest suite, and Ruff pass, or unrelated pre-existing failures are reported without expanding scope.
- Documentation describes what the categories support, their limits, and the next approval gate without selecting a winner or claiming chemistry/operational readiness.

## Recommended Next Decision Gate

After final review, the user chooses one bounded follow-up based on the dominant validated category distribution:

- If `allowed_evidence_not_retrieved` or `no_qualifying_retrieval` dominates and differs materially by mode, approve one retrieval-focused experiment tied to that category; do not assume reranking is the answer.
- If `allowed_retrieved_not_cited`, `mixed_with_disallowed`, or generation-failure categories dominate, keep retrieval frozen and plan a separate generator/citation-contract diagnosis before any prompt change.
- If `under_abstention_answered_on_insufficient_case` dominates, keep retrieval and prompt frozen until an abstention-policy experiment and safety guardrails are planned.
- If the modes remain category-identical, the baseline drifts, or any failures are unclassified, do not select a mode or modify retrieval/prompt behavior; first approve a validity or dataset-contract investigation.

The next task begins only after the user reviews the aggregate report and explicitly approves one branch.

## Risk Notes

- Temperature `0` does not guarantee identical output across Ollama/model/runtime changes; the baseline-reproduction gate prevents silent comparison drift.
- The 12 public cases and reviewed allowlists are a learning fixture, not chemistry ground truth; categories describe contract behavior only.
- Post-filter retrieval observations cannot alone separate ranking, threshold, and context-budget causes; any retrieval experiment needs a narrower follow-up measurement.
- Aggregate-only reporting reduces debuggability by design. If a category cannot be explained without case-level persistence, inspect it transiently in memory under a separately approved privacy plan rather than weakening this report boundary.
