# E1a-4 Requirements-Aware Evidence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a requirements-aware, provenance-bound evaluator can classify frozen C1 evidence as sufficient, partially sufficient, or insufficient more safely than a paired context-only control on a fresh private population.

**Architecture:** A question-only requirement extractor freezes up to three atomic user-request obligations before C1 context is loaded. A separate support judge then maps every frozen requirement to only supplied C1 passage ranks, and a deterministic controller derives the evidence state and answer-boundary decision. Canonical claims and blind support labels remain evaluator-only and independently determine the gold state; neither candidate evaluator receives them.

**Tech Stack:** Python, existing private SHA-256 artifact contracts, Ollama-compatible chat adapters, existing C1 snapshot and index-preflight contracts, pytest, Ruff.

**Spec:** This plan implements the approved next experiment described in the active Codex task. It supersedes no existing experiment and must not alter the closed E1a-3 result.

## Progress Ledger

**Last reconciled:** 2026-08-23

| Task | Status | Evidence and next gate |
| --- | --- | --- |
| 1. Core requirements-aware contracts | `COMPLETE` | Contract, provenance, aggregation, and empty-evidence behavior are implemented and covered by focused tests. |
| 2. Fresh E1a-4 population contracts and private population | `BLOCKED` | Reconciliation identity review is closed and its active seven-artifact seal verifies. A bound 133-candidate foundational evidence audit then proposed 92 promotions and retained 41 candidates with zero unresolved items; its two-artifact correction proposal and manifests verify without applying mappings. All supporting strata and foundational scale, corrosion, and paraffin are sufficient, but foundational iron sulfide remains at 5 of 12 fresh locators. The deterministic 96-slot allocation remains unavailable and no E1a-4 sampling-frame file was written. |
| 3. Question-only requirements and blind gold labels | `NOT_STARTED` | Blocked on the complete sealed Task 2 population. |
| 4. Frozen C1 snapshot wrapper | `NOT_STARTED` | Blocked on Tasks 2-3 and its separate execution approval. |
| 5. Paired evaluator runner | `NOT_STARTED` | Blocked on Tasks 1-4. |
| 6. One-shot paired evaluation and report | `NOT_STARTED` | Blocked on Tasks 1-5 and explicit one-shot approval. |

**Verification recorded at reconciliation:** the combined Task 1-2 focused suite passed 47 tests and Ruff passed for the Task 1-2 files. This verifies implementation files only; it is not evidence that a private population exists or that Task 2 is complete. All 11 Task 1-2 implementation, test, runner, and plan files are currently untracked and must be brought under repository control before project finalization.

**Private corpus reconciliation evidence:** the contract-bound reconciliation run completed every inventory, matching, and capacity stage; the no-write E1a-4 allocation stage blocked as designed; and all six private JSONL snapshots plus manifests verified. The blocker is now classified as foundational-locator freshness across all four topics, not general supporting-source capacity and not evidence that bulk ingestion is required. The next allowed action is a separately reviewed metadata/evidence audit for additional substantive foundational locators in the already approved corpus. If that audit cannot establish twelve fresh foundational locators per topic without reusing E1a-3 evidence, E1a-4 remains rejected under its current design and any new foundational-source acquisition requires a new approval.

**Foundational-locator audit evidence:** the approved audit reviewed all 133 bound candidates, proposed 92 promotions, retained 41 as ineligible, and closed with zero unresolved items. The stricter v2 correction seal and manifests passed separate no-write verification and have not been applied; the earlier v1 seal is preserved as superseded history. Independent review approved the v2 contract with no findings. The projection establishes sufficient foundational capacity for scale, corrosion, and paraffin, but only 5 of 12 required fresh foundational iron-sulfide locators. The exact allocation therefore remains blocked. The next allowed decision is either a narrow new-source acquisition plan for the seven-locator iron-sulfide deficit or closure of E1a-4 under the current grid.

**Audit alignment:** E1a-4 addresses the answer-evaluation and evidence-sufficiency gap recorded by the curriculum and Zoomcamp audits. It does not establish bulk-PDF ingestion scale, select chunk size/overlap, add section or authority metadata, or add Kestra recovery policy. Those are separate measured gaps in `docs/superpowers/reports/2026-08-16-project-vs-zoomcamp-audit.md` and must not be silently folded into this experiment.

## Global Constraints

- E1a-3 is closed as `E1A3_CONTEXT_CLASSIFIER_COMPLETED_REJECTED`; its 30 cases are observed regression evidence only and must never tune models, prompts, thresholds, schemas, or heuristics.
- All case content, source IDs, locators, claims, requirements, passage text, outputs, and per-case results stay below `.private/retrieval-evaluation/v1/e1a4/` and remain Git-ignored.
- Do not access the sealed holdout.
- Do not change production retrieval, reranking, C1, corpus, index, or answer generation.
- No fallback model, database, fixture, or retry that changes the frozen contract is allowed.
- A malformed, missing, duplicate, or provenance-invalid model output rejects the entire run; no case may be dropped.
- The classifier must never receive canonical claims, gold states, source identity, retrieval scores, C1 selection diagnostics, or prior E1a/E1a-3 outputs.
- A support result is `SUPPORTED` only with at least one cited delivered passage rank. `UNCLEAR` and `UNSUPPORTED` are non-support.

---

## Hypothesis And Decision Rule

**Hypothesis:** Separating question requirements from requirement-to-passage support prevents topical-but-incomplete C1 contexts from being labeled `SUFFICIENT`, while preserving useful recognition of genuinely sufficient contexts.

**Paired candidates:**

- **A, control:** question plus frozen C1-delivered passages, using the existing context-only contract shape.
- **B, candidate:** the same question and C1 passages plus sealed question-only requirements; its state is controller-derived from requirement support labels.

**Pre-registered acceptance criteria:** Candidate B is accepted for later E1b design only when all conditions hold on one fresh 30-case, 10/10/10 state-balanced population:

1. Every artifact, set equality, SHA-256 binding, model specification, structured output, and cited passage rank validates.
2. B makes zero `SUFFICIENT` predictions among the 20 gold non-sufficient cases.
3. B correctly predicts `SUFFICIENT` for at least 7 of 10 gold-sufficient cases. This is a reject-only diagnostic floor, not a production-calibration claim.
4. B's sufficient recall is at least A's sufficient recall.
5. B has at least one more exact three-state classification than A.
6. No model, prompt, threshold, schema, model version, temperature, invalid-output rule, or aggregation rule changes after the frozen input contract is sealed.

Any failed condition is `E1A4_REQUIREMENTS_GATE_REJECTED`. Passing does not authorize E1b, E1c, production wiring, or the sealed holdout.

## Private Experiment Sequence

```text
new sealed metadata-only source register and 96-slot allocation
  -> fresh private questions with at most three atomic material obligations
  -> canonical claims and canonical-evidence verification sealed
  -> freshness check against development, E1a-2, and all E1a-3 questions and locators
  -> question-only requirements sealed
  -> approved index preflight
  -> normal no-oracle retrieval, frozen reranker, frozen C1 snapshot once
  -> independent blind canonical-claim support labels sealed
  -> deterministic 10/10/10 state-balanced selection or fail-closed stop
  -> paired evaluator input contract sealed
  -> A and B each run once against the same sealed inputs
  -> controller validates, scores, and writes a private result
  -> privacy-safe aggregate tracked report
```

The independent gold path is mandatory: the blind reviewer sees canonical claim IDs/text plus blinded rank 1-5 passage text, but never requirements, C1 output, evaluator prompts, sources, scores, or candidate outputs. Gold-state selection and scoring use only that sealed path.

## Evaluator Contract

### 1. Requirement Extraction

Input: `question_id`, user question.

Output: one to three ordered atomic requirements:

```json
{"requirements":[{"requirement_id":"r1","requirement":"..."}]}
```

The extractor is question-only. It must not answer the question, add technical facts, name sources, see evidence, or produce more than three obligations. Population authoring must likewise cap every question at three material obligations; questions needing more are ineligible.

### 2. Requirement-To-Evidence Support Judgment

Input: user question, exact sealed requirements, and exact C1-delivered `{rank, passage_text}` entries.

Output has exactly one record per supplied requirement:

```json
{"requirement_support":[
  {"requirement_id":"r1","status":"SUPPORTED","supporting_ranks":[2]}
]}
```

`status` is one of `SUPPORTED`, `UNSUPPORTED`, or `UNCLEAR`. `SUPPORTED` requires direct support for the full requirement in one or more cited delivered ranks. `UNSUPPORTED` and `UNCLEAR` must cite no ranks. A relevant passage that does not state the requested fact is non-support. The judge cannot add, remove, rewrite, or reorder requirements and cannot emit technical prose.

### 3. Deterministic Evidence-State Aggregation

- every requirement `SUPPORTED` -> `SUFFICIENT`;
- at least one `SUPPORTED` and at least one `UNSUPPORTED` or `UNCLEAR` -> `PARTIALLY_SUFFICIENT`;
- no requirement `SUPPORTED` -> `INSUFFICIENT`.

The controller validates cited ranks against the exact delivered context, requirement-set equality, and each sealed digest before calculating the state. It never trusts an evaluator-provided final state.

### 4. Deterministic Answer Boundary

- `SUFFICIENT` -> `FULL_ANSWER_PERMITTED`;
- `PARTIALLY_SUFFICIENT` -> `SUPPORTED_ONLY_RESPONSE_REQUIRED`;
- `INSUFFICIENT` -> `ABSTENTION_REQUIRED`.

E1a-4 produces this decision only. It must not initialize a generator or create an answer; testing whether a generator obeys the decision remains E1b.

## Model-Swappable Boundary

Define two narrow protocols, `RequirementExtractor` and `RequirementSupportJudge`, rather than a new agent framework. Each run's private contract pins provider adapter, model ID, model version when available, system prompt SHA-256, response-schema SHA-256, temperature, and invalid-output behavior. An Ollama adapter is the first implementation; later local or OpenAI adapters implement the same protocols without changing artifact, provenance, aggregation, or scoring code.

Do not use LangChain, a generic workflow engine, a new vector store, or a second retrieval pipeline. They do not help this narrow classification experiment.

## Freshness And Contamination Boundary

- Create a new `e1a4` private root; do not copy or extend E1a-3 artifacts.
- The 96-question population must have whitespace-normalized question wording distinct from development, E1a-2, and every E1a-3 case.
- Assigned canonical locators must be disjoint from E1a-3 locators. If that is impossible for a required stratum, stop before authoring; do not silently reuse a locator. A later plan may define a narrowly approved exception policy.
- Requirements must be generated and sealed before retrieval/C1 snapshot creation.
- Requirement extraction and support judgment may share a model only if their distinct prompt/schema digests and inputs are sealed before either call. They are still logically separate operations.
- Do not score or compare the E1a-3 30 cases during candidate selection. They may be retained privately as non-tuning regression evidence after E1a-4 is closed.

## Task 1: Define Core Requirements-Aware Contracts

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/requirements_evidence_gate.py`
- Create: `tests/evaluation/test_requirements_evidence_gate.py`

**Interfaces:**
- Produces `RequirementExtractor`, `RequirementSupportJudge`, `RequirementFixture`, `RequirementSupport`, `RequirementsGateObservation`, `derive_evidence_state`, and `derive_answer_boundary`.
- Consumes `DeliveredEvidence` and `EvidenceState` from `evaluation/evidence_state.py`.

- [x] Write failing tests for exact requirement parity, one-to-three atomic requirements, required rank provenance for `SUPPORTED`, empty ranks for non-support, controller-derived states, and controller-derived boundaries.
- [x] Run the focused test file and confirm it fails because the contracts do not exist.
- [x] Implement only the typed mappings, strict parsers, protocol definitions, deterministic aggregation, and safe error codes.
- [x] Run the focused test file and Ruff for the changed files.

## Task 2: Build Fresh E1a-4 Private Population Contracts

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/e1a4_population.py`
- Create: `src/oilfield_chemical_copilot/evaluation/e1a4_selection.py`
- Create: `eval/seal_e1a4_sampling_frame.py`
- Create: `eval/seal_e1a4_population.py`
- Create: `tests/evaluation/test_e1a4_population.py`
- Create: `tests/evaluation/test_e1a4_selection.py`
- Create locally only: `.private/retrieval-evaluation/v1/e1a4/`

**Interfaces:**
- Produces the complete 96-slot source allocation, fresh questions, canonical claims, SHA-256 manifests, and deterministic 10/10/10 selected IDs.
- Consumes the approved read-only index metadata and prior private question/locator inventories only for eligibility and novelty checks.

- [x] Write failing tests for exact 96-slot stratification, at-most-three canonical claims, question uniqueness, prior-question rejection, mandatory E1a-3 locator-disjointness, and fail-closed balanced selection.
- [x] Run the focused tests and confirm the E1a-4 population API is absent.
- [x] Implement private-only artifact validation and atomic sealing code. Do not retrieve, rerank, or call a model in this task.
- [x] Run focused tests and Ruff for the Task 2 contract and sealer files.
- [x] Add focused integration tests plus a presence-only, sanitized, fail-closed preflight to the sampling-frame sealer before any real sampling-frame run.
- [x] Run tracked-diff, untracked-file whitespace, and ignore checks for the private root after the sampling-frame correction.
- [ ] Seal the metadata-only E1a-4 source register and exact 96-slot allocation after database and index prerequisites pass.
  - **Blocked 2026-08-23 after sealed reconciliation:** the approved 4,797-chunk/198-source index contract and all reconciliation snapshots passed. All supporting topic strata meet fresh capacity, but all foundational topic strata remain below the required twelve fresh locators after exact E1a-3 exclusion. A no-write deterministic allocation therefore returned `CORPUS_RECONCILIATION_E1A4_ALLOCATION_UNAVAILABLE`; no sampling-frame file was written. Do not reuse E1a-3 locators, promote ambiguous document matches, ingest/reindex material, or weaken the 96-slot grid. The next gate is a separately reviewed foundational-locator evidence audit against the already approved corpus.
  - **Updated 2026-08-26 after the foundational-locator audit:** the independently approved proposal would make foundational scale, corrosion, and paraffin sufficient, but foundational iron sulfide remains at 5 of 12 fresh locators. The exact allocation is still unavailable. Do not apply the proposal or write a sampling frame; require either a separately approved acquisition of at least seven fresh substantive foundational iron-sulfide locators or closure under the current grid.
- [ ] Author and validate the fresh 96-question population and canonical-claim drafts without retrieval or model calls.
- [ ] Seal the population and claims atomically, verify manifests, and rerun the aggregate Task 2 readiness check.

## Task 3: Freeze Question-Only Requirements And Independent Gold Labels

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/e1a4_requirements.py`
- Create: `src/oilfield_chemical_copilot/evaluation/e1a4_blind_support.py`
- Create: `eval/generate_e1a4_question_requirements.py`
- Create: `eval/build_e1a4_blind_support_packets.py`
- Create: `eval/seal_e1a4_blind_support_labels.py`
- Create: `tests/evaluation/test_e1a4_requirements.py`
- Create: `tests/evaluation/test_e1a4_blind_support.py`

**Interfaces:**
- Produces question-only requirement fixtures and an independent canonical-claim blind-label artifact.
- Consumes the sealed E1a-4 population and, for blind labels only, the later frozen rank 1-5 evidence snapshot.

- [ ] Write failing tests proving requirement input exposes only question ID/text and blind packets exclude requirements, sources, scores, delivery decisions, and candidate outputs.
- [ ] Write failing tests that pin model/prompt/schema/temperature digests and reject malformed outputs without retry or case removal.
- [ ] Implement strict artifact contracts only; freeze the run configuration before any model call.
- [ ] Run focused tests and privacy-path checks.

## Task 4: Reuse Frozen C1 Snapshot Mechanics Without Changing C1

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/e1a4_snapshot.py`
- Create: `eval/freeze_e1a4_c1_snapshot.py`
- Create: `tests/evaluation/test_e1a4_snapshot.py`

**Interfaces:**
- Produces a metadata-bound, no-oracle C1 snapshot for the fresh E1a-4 question set.
- Consumes `IndexFingerprint`, the current evaluation retrieval configuration, frozen MiniLM reranking, and the unchanged C1 policy implementation.

- [ ] Write failing tests that require preflight before fixture load/model initialization and bind exact C1 delivered ranks/text through hashes.
- [ ] Implement a thin versioned wrapper around existing `index_preflight` and `e1a3_snapshot` mechanics. Do not fork retrieval, reranking, or delivery logic.
- [ ] Run focused tests. Do not run the snapshot until its separate approval gate.

## Task 5: Implement Paired Model-Swappable Evaluation Runner

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/e1a4_runner.py`
- Create: `eval/run_e1a4_paired_evidence_gate.py`
- Create: `tests/evaluation/test_e1a4_runner.py`

**Interfaces:**
- Produces one sealed paired result with private observations and public-safe aggregate metrics.
- Consumes the selected fresh input contract, frozen control specification, frozen candidate specification, independent gold states, and the Task 1 contracts.

- [ ] Write failing tests for pairwise identical question/C1 evidence, complete selected ID set, zero silent fallback, no evaluator-visible gold fields, and acceptance-rule calculation.
- [ ] Implement a single one-shot runner that runs A then B once per sealed input and validates both before writing a result. It must reject the complete experiment on any bad output.
- [ ] Run focused tests and Ruff. Do not make a model call until Task 6 approval.

## Task 6: Execute One Fresh Paired Evaluation And Report

**Files:**
- Create: `eval/render_e1a4_aggregate_report.py`
- Create: `docs/superpowers/reports/2026-08-19-e1a4-requirements-aware-evidence-gate.md`
- Create locally only: `.private/retrieval-evaluation/v1/e1a4/results/paired-evidence-gate.v1.json`

**Interfaces:**
- Produces a privacy-safe aggregate report and one private, manifest-verified result.

- [ ] Verify all Task 1-5 contracts, index preflight, exact input hashes, selected case IDs, model specifications, and public report destination before initializing either model adapter.
- [ ] Request explicit approval to consume the fresh E1a-4 set once.
- [ ] Run the paired one-shot evaluation once; write no case-level tracked output.
- [ ] Render aggregate metrics only, apply the acceptance criteria, run focused tests, Ruff, `git diff --check`, private ignore checks, and a scan for private content in the report.
- [ ] Send the aggregate evidence packet to Sol for an optional final recommendation; Terra owns the resulting action decision.

## Execution Gates

1. Approve Tasks 1-5 contract and fixture infrastructure.
2. Approve private E1a-4 population authoring after Task 2 tests pass.
3. Approve requirements, snapshot, blind-label, and deterministic selection stages after their contracts are reviewed.
4. Approve the one-shot Task 6 paired model run only after the exact fresh input contract and acceptance criteria are sealed.

No task authorizes E1b, E1c, production integration, retrieval tuning, a C1 change, model switching after a result, or sealed-holdout access.
