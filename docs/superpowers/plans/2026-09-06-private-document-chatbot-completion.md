# Private Document Chatbot Completion Plan

**Goal:** Deliver a local chatbot that answers questions from the user's approved private handout collection, using the core handout and supplements, with verifiable citations and honest limits.

**Architecture:** Retain the existing application, local model adapter, retrieval pipeline, index, and reconciliation records. Establish document coverage, evaluate the full answer path, correct demonstrated failures, and verify persistent operation. E1a-4 remains a bounded supporting experiment.

**Tech Stack:** Existing Python/Streamlit application, Ollama, PostgreSQL/PGVector, SQLite reconciliation checkpoints, private evaluation artifacts, pytest, and Docker Compose.

**Spec:** [Capstone Focus](../../CAPSTONE_FOCUS.md), clarified by the user's September 6 instruction that the capstone must answer questions from the approved private handout collection.

**Status:** Planning complete; product acceptance OPEN. This document authorizes no private execution, new experiment, corpus change, or deployment by itself. Existing execution approvals and experiment boundaries remain in force.

## Authority And Evidence

This is the controlling completion sequence. Course locks remain learning milestones. Submission readiness and private chatbot acceptance must be reported separately.

The current starting evidence is the public record at `0863f99`, incorporated into PR #2, whose merge was verified in the prior session as `d5049199384250ae5d4283887e5fc3d0427b2376`. No new private verification was performed for this plan. The original checkout was still at `acf10fc` during planning; its older statuses must not override the reviewed branch record.

| Work | Recorded state | Meaning for completion |
| --- | --- | --- |
| Course Modules 1-5 and 7 | Learning milestones locked | Public sample and implementation evidence exist; private product acceptance remains open. |
| Reconciliation and 117 ambiguous decisions | Closed in the later public record | Reuse committed decisions and the verified seven-artifact seal; do not repeat the review. |
| Foundational audits and corrections | Closed/applied in later mapping closure | The historical locator deficit is resolved for experiment allocation. |
| E1a-4 mapping and sampling frame | 170 mapped sources; all eight strata sufficient; 96 unique source-locator assignments | Sufficient experiment capacity; not proof of complete collection coverage or correct answers. |
| Publication correction | Merged; recorded 847 tests passed, 14 skipped | Software verification, not answer-quality evidence. |
| E1a-4 | Task 1 complete; Task 2 needs questions/claims; Tasks 3-6 not started | Preserve the existing experiment and its acceptance criteria. |
| Private chatbot acceptance | OPEN | Full answer quality, coverage, latency, restart, and user demonstration remain to be established. |

Evidence sources: [E1a-4 plan](2026-08-19-e1a4-requirements-aware-evidence-gate.md), [learning roadmap](../../LEARNING_ROADMAP.md), and the August 27 correction and August 28 publication plans at the reviewed revision. The first milestone reconciles missing files and stale links in this checkout before execution relies on them.

## Global Constraints

- Use the already approved private collection. Keep its exact root, document names, identifiers, passages, case records, and results in existing private records; public reports contain safe aggregates only.
- The core handout and its supplements define collection scope. Experiment topic/role eligibility does not exclude other approved material from the product scope.
- Reuse SQLite decisions and sealed artifacts. A disconnected session must resume from durable verified state rather than silently re-create completed work.
- No private payload, Drive file, live database, model, retrieval, or holdout was accessed in preparing this plan.
- Preserve E1a-3 as rejected observed evidence. Do not tune on its cases. Preserve E1a-4 freshness, blindness, invalid-output handling, and one-shot rules.
- Verify the approved index contract before any separately authorized private runtime evaluation. Never substitute a sample or different index on failure.
- Keep experiment and application configurations distinct. A passing experimental gate does not silently change the application.
- Persistent application monitoring remains aggregate-only. Private evaluation records are controlled evaluation artifacts, not general chat logging.
- No new framework, vector store, broad ingestion, or generalized infrastructure is a completion prerequisite. Each change must address a demonstrated acceptance blocker.
- User decisions and existing authorization prevail; routine checks do not create a new approval ceremony. Separate restricted execution gates remain explicit.

## Milestone 1: Reconcile The Maintained Project Record

**Status:** COMPLETE — integrated and clean-checkout verified at `726f9df`. **Deliverable:** one reproducible code revision with an accurate public evidence index.

**Files:** `docs/LEARNING_ROADMAP.md`, `docs/CURRICULUM_REMEDIATION_BACKLOG.md`, `docs/CAPSTONE_FOCUS.md`, `docs/CAPSTONE_EVIDENCE.md`, `README.md`, and this plan. Inspect untracked public-intended reports/code individually before deciding whether to retain them; do not bulk-stage them.

**Consumes:** merged public history and preserved local changes. **Produces:** revision, evidence links, and a disposition for each relevant local-only dependency.

- [x] Record the consolidated objective, known completed work, and remaining acceptance milestones.
- [x] Reconcile the stale original checkout with the merged revision in an isolated workspace; preserve unrelated edits and local-only work. Do not reset or overwrite the original checkout.
- [x] Verify every report/code dependency cited by the active plans exists at the chosen revision. Mark unavailable evidence explicitly; inspect public-intended documents for disclosure before adding them.
- [x] Correct stale historical statuses and links in the maintained documents. Record the selected application configuration separately from rejected experimental configurations.
- [x] Run the existing documentation checks and affected tests after any retained implementation changes. Record revision and verification in this ledger.
- [x] Resolve normal code review and integrate the recovered revision. Keep deferred historical reports outside the maintained evidence set unless their individual provenance and disclosure review succeeds.
- [x] Verify the integrated revision from a clean checkout and record its active evidence map before closing M1.

**Exit:** all active links resolve, necessary code is reproducible, the integrated revision has clean-checkout verification and an active evidence map, and progress is not inferred from an untracked file or an old checkout. M1 does not close on merge alone. **Stop:** missing evidence is recorded as missing; completed private audits are not rerun to repair documentation.

**Recovery checkpoint completed — 2026-09-06:** An isolated branch based on `0863f99` recovered the local-only public E1a-4 dependency closure in `1cb45fe`: five evaluation modules, one CLI, and three test modules. The recovered focused suite passed 43 tests and Ruff was clean; the combined E1a-4 mapping, sampling, and reviewer-document suite passed 192 tests with 10 documented skips. The original checkout remains unchanged. Historical public-intended reports still require individual provenance and disclosure review before they may enter the maintained evidence set; the recovery status is recorded in [the maintained-record reconciliation report](../reports/2026-09-06-maintained-record-reconciliation.md).

**M1 closeout — 2026-09-07:** PR #3 merged as `726f9df`. Its isolated clean-checkout verification recorded `897 passed, 14 skipped`, clean Ruff and whitespace checks, and the active public evidence map in the reconciliation report. No private runtime or artifact was accessed.

## Milestone 2: Establish Collection Coverage And A Baseline Contract

**Status:** IN PROGRESS. **Deliverable:** a private coverage register and a frozen evaluation specification for the current application.

**Files/responsibility:** reuse reconciliation storage and inventory readers; publish only aggregate evidence in `docs/superpowers/reports/2026-09-06-private-chatbot-completion.md` when executed. Keep per-document dispositions and cases in approved private storage.

**Execution update — 2026-09-07:** the verified private coverage register now binds 385 sealed document identities. The first bounded review recorded 117 identities as blocked solely because their mapping to the runtime index is not yet verified; it did not infer exclusion, extraction failure, or usable coverage. 268 identities remain. No corpus, index, retrieval, model, or private payload changed.

- [ ] Read the existing sealed inventory/decisions under the applicable private-review authority. Give every approved document one recorded disposition: indexed and usable, duplicate linked to its representative, intentionally excluded with reason, or blocked by a specific extraction/indexing failure.
- [ ] Confirm the full core handout and approved supplemental groups are represented; distinguish document identity, readable substantive content, and runtime index membership. Account for tables/charts and image-only pages. Use existing extraction evidence first and inspect only unresolved coverage gaps.
- [ ] Confirm what the application actually queries and how its citations identify document/page. Record index freshness and whether Drive updates are manual; do not claim live synchronization without evidence.
- [ ] Separate whole-collection coverage from E1a-4's four-topic grid. List additional approved question families, including water analysis/reference/chart questions, and require acceptance cases for them if they remain in product scope.
- [ ] Freeze a baseline specification using independently verified expected evidence, full-path answer/citation grading, latency measurement, and failure categories. Use dedicated development questions; reserve a separate unseen acceptance cohort before tuning. Do not consume E1a-4 or historical holdout cases for this baseline.

**Exit:** 100% of approved document identities have a disposition, every usable claim has substantive source evidence, and all missing product coverage is visible. An exclusion is not evidence of support. Core or required supplement gaps must be resolved or explicitly accepted as a scope limit before final completion.

**Stop:** a failed runtime/index preflight produces an unavailable result. A coverage gap permits a scoped correction proposal; it does not authorize bulk ingestion. Retrieval relevance alone cannot establish correct answers.

## Milestone 3: Measure Failures And Finish The Bounded Evidence Decision

**Status:** PENDING M2 and existing execution gates. **Deliverable:** baseline failure report and an E1a-4 accept/reject decision.

**Files/responsibility:** reuse the existing live answer evaluator and E1a-4 modules/runners from the reconciled revision. The [E1a-4 plan](2026-08-19-e1a4-requirements-aware-evidence-gate.md) owns its exact contracts; this plan does not replace them.

- [ ] Under the baseline execution authority, measure the actual application from question through delivered evidence to answer and citations. Classify failures as missing/unreadable source, retrieval miss, delivery loss, evidence judgment, unsupported generation, citation mismatch, or runtime failure. Report useful-answer rate as well as abstention.
- [ ] Complete private question and canonical-claim authoring for the existing 96-slot E1a-4 frame; independently verify canonical support before accepting ground truth.
- [ ] Follow E1a-4's sealed requirements, frozen C1 snapshot, blind labels, deterministic selection, and paired one-shot evaluation in their existing order and gates.
- [ ] Preserve the exact E1a-4 conditions: valid contracts; zero false-SUFFICIENT among 20 nonsufficient cases; at least 7/10 sufficient recognized; recall at least control; at least one more exact-state match than control; frozen configuration with no fallback or tuning.
- [ ] Record acceptance or rejection and whether the observed application failure is addressed. An experimental pass is evidence for a later integration proposal only.

**Exit:** a measured baseline and a closed experiment decision. **Stop:** rejection stops work on this candidate and observed cohort. Any successor needs a distinct measured hypothesis and bounded decision, not an automatic new experiment.

## Milestone 4: Correct The Answer Path And Validate On Unseen Questions

**Status:** PENDING M3 decision. **Deliverable:** one frozen application candidate with independently assessed answer-quality evidence.

**Files/responsibility:** changes, if justified, belong to existing `src/oilfield_chemical_copilot/rag/`, `src/oilfield_chemical_copilot/retrieval/`, their tests, and `app/streamlit_app.py` only where the measured failure requires it. A scoped implementation plan must name exact files and regression cases after the cause is established.

- [ ] Fix the demonstrated limiting failure using development evidence. If E1a-4 passes and addresses that failure, define and verify its answer-generation integration under the later E1b boundary. If it fails, leave it out and record the product blocker or an explicitly narrower supported product scope.
- [ ] Test answers to direct lookups, explanations, paraphrases, cross-document questions, and conflicting evidence. The core/supplement relationship guides interpretation; do not blindly prefer the core over more directly relevant evidence or inject gold source identities into retrieval.
- [ ] Freeze application revision, model, prompt, index, answer policy, test cohort, grading rubric, and numerical thresholds before unseen evaluation. Obtain subject-matter grading against the cited material; a model judge is advisory.
- [ ] Run the accepted candidate through the real application path on the fresh cohort. Preserve every invalid output and runtime failure in the denominator. Do not retune on that cohort or reuse it as new validation evidence.
- [ ] Record an acceptance decision with rates, denominators, per-family coverage, uncertainty, and remaining limitations. A small test can reject unsafe behavior but cannot establish broad operational safety.

**Proposed product acceptance contract, to finalize before any acceptance run:**

| Dimension | Proposed minimum and scoring definition |
| --- | --- |
| Cohort | At least 60 fresh cases: at least 40 answerable and 20 unanswerable/unsupported. Cover all four experiment topics and every additional approved question family from M2; increase cohort size where needed rather than silently omit a family. Include core-only, supplement-only, combined-source, and paraphrase cases. |
| Useful answers | At least 80% of answerable cases correct and sufficiently complete against independently authored essential claims. Blanket abstention fails this criterion. |
| Citation support | Every emitted citation resolves to the approved document and page/other stable locator; at least 95% of material answer claims directly supported by their cited evidence. Report counts and unsupported claims explicitly. |
| Unsupported questions | Zero unqualified substantive answers on the 20 or more unsupported cases; state the missing evidence or ask a relevant clarification. Any dangerous unsupported operational prescription rejects the candidate across the entire cohort. |
| Runtime | Zero silent index/model fallbacks and no unhandled application crash; failed requests remain evaluation failures. |
| Response time | Proposed p95 complete-response latency at most 90 seconds on the recorded local hardware/model. Separate cold-start time. Confirm the user-facing target before freezing; never relax it after seeing acceptance results. |

These numbers are explicit planning proposals, not achieved scores, historical E1a-4 amendments, or a claim of production safety. Finalizing the contract is a concrete M2/M4 deliverable; it must happen before observing acceptance outcomes.

## Milestone 5: Verify Daily Use And Deliver The Private Chatbot

**Status:** PENDING M4. **Deliverable:** demonstrated local private chatbot with persistent data and a user runbook.

**Files/responsibility:** existing application/Compose configuration where a demonstrated defect requires a change, `docs/CAPSTONE_REVIEWER_GUIDE.md`, `docs/CAPSTONE_EVIDENCE.md`, `README.md`, and the aggregate completion report. Private setup details belong in a private companion runbook.

- [ ] Restart the application and database without deleting volumes. Verify index identity/counts and the ability to query already indexed documents without reparsing/re-embedding the corpus. Reuse observed demonstration cases, not the sealed acceptance cohort.
- [ ] Verify interruption/reconnection preserves completed reconciliation/evaluation checkpoints and does not restart a consumed one-shot run. This does not require storing raw chat history.
- [ ] Document the current new/changed/deleted document process, including whether updates are manual. Demonstrate checkpointed recovery for that process on synthetic files; if no update implementation exists, disclose that limit and specify a bounded measured follow-up instead of claiming automatic Drive sync.
- [ ] Demonstrate private questions in the user-facing app with usable document/page citations, limitations, and clarification behavior. Record user acceptance of usefulness and any exclusions.
- [ ] Verify the public reviewer path separately from a clean checkout, link only public-safe evidence, and prepare submission artifacts. Public sample success does not close private acceptance.

**Exit:** M1-M5 evidence is recorded; the user can use and restart the private chatbot; numerical acceptance and disclosed scope are satisfied. **Stop:** any failed required acceptance dimension leaves product completion open. Hosted or multi-user production readiness is a later scope.

## Execution And Progress Discipline

- Every session starts by reading this ledger and verifying the checkout revision. Update status after each completed milestone with revision, evidence, verification, blocker, and next action.
- A milestone completes through its deliverable and acceptance evidence, not through number of tests, documents, reviews, or elapsed time.
- One correction addresses one measured failure. After a failed candidate, make a recorded continue, narrow-scope, or stop decision before further experimentation.
- Reuse the existing seals, validators, checkpoints, and evaluators. Add infrastructure only when its absence demonstrably blocks a milestone.
- Do not repeat completed private reviews to compensate for stale documentation. Do not equate metadata allocation, retrieval hit rate, or classifier accuracy with answer correctness.
- Immediate execution checkpoint: M2 coverage evidence review. The existing E1a-4 question/claim authoring gate remains separate and follows its own execution boundary; neither gate changes retrieval or the corpus.
