# Curriculum And Capstone Remediation Backlog

**Basis:** [2026 Curriculum And Capstone Audit](CURRICULUM_CAPSTONE_AUDIT_2026.md)

**Rule:** Items below are a backlog, not approved implementation work.

## Active Experiment Status - 2026-08-28

- E1a-4 Task 1 contracts are complete.
- E1a-4 Task 2 mapping and metadata-only sampling frame are sealed and independently no-write verified. The verified mapping has 170 source records and all eight topic/role strata sufficient. The frame has exactly 96 slots, 96 unique slot identities, and 96 unique source-locator assignments across all four topics and both roles, with zero E1a-3 locator reuse.
- The reconciliation seven-artifact snapshot set and both correction proposals were reverified before application, and mutation-boundary checks are clean. Qdrant and application services were not run; retrieval, ingestion, reindexing, model execution, question authoring, and claim authoring did not occur.
- E1a-4 Tasks 3-6 have not started. The next checkpoint is private question and canonical-claim authoring. Do not weaken locator freshness, the exact grid, or the one-shot gates; do not retrieve, ingest, or reindex to bypass this checkpoint.

## P0 - Preserve Submission Integrity

1. **Publish a fixed public reviewer commit.**
   - Outcome: the public remote contains the reviewer package and the two
     latest local commits; the submission provides one immutable commit hash.
   - Why: peer reviewers receive a public repository URL and commit hash.
   - Boundary: external action; requires explicit approval to push.

2. **Perform a fresh public-clone verification at that commit.**
   - Outcome: an independent clone can follow the public path with only the
     tracked sample corpus and documented prerequisites.
   - Why: it is the missing evidence behind the 1/2 reproducibility score.
   - Boundary: do not use private corpus, private fixtures, local paths, or
     developer database state.

## P1 - Make The Capstone Rubric Defensible

1. **Define and evaluate two final-answer approaches.**
   - Outcome: a documented comparison and justified selected answer boundary,
     or a clear reason why no second approach is appropriate.
   - Why: the present LLM-evaluation evidence supports 1/2, not 2/2, under the
     published rubric.
   - Boundary: new evaluation design, public fixture, and approval before
     changing any production behavior.

2. **Make the selected retrieval mode explicit in public documentation.**
   - Outcome: one documented default, evidence for its selection, and a stated
     residual-risk boundary.
   - Why: vector and hybrid have both been measured, but the mode-selection
     decision is not yet a concise reviewer-facing argument.
   - Boundary: documentation first; changing default behavior requires an
     approved experiment.

3. **Prepare the peer-review submission packet.**
   - Outcome: public repository URL, fixed commit hash, brief demo, and the
     completed three-peer-review process if certificate eligibility applies.
   - Why: peer-review points and certificate workflow cannot be inferred from
     local tests.
   - Boundary: external course submission action.

## P2 - Resolve The Black Water Evaluation Gap

1. **Design a new retrieval experiment before changing RAG.**
   - Outcome: an approved, privacy-safe experiment question that separates
     source availability, parsing/OCR, vector recall, keyword recall, RRF
     ranking, top-k, and evidence-threshold effects.
   - Why: the observed safe fallback is consistent with a ranking/threshold
     issue, but the public evaluation pack has no corresponding case.
   - Boundary: no reindexing, lowering threshold, query rewrite, reranking, or
     direct production tuning before the design is approved.

2. **Add a public analogue only when one can be released safely.**
   - Outcome: a synthetic or public `Black Water`-like retrieval case with an
     expected source identifier and no private source text.
   - Why: this creates a regression guard without exposing proprietary
     material.
   - Boundary: retain private handouts and sealed evaluation details locally.

3. **Run one fresh sealed evaluation after any correction.**
   - Outcome: aggregate score and failure classes for the changed boundary.
   - Why: a rerun of an already seen holdout would bias the result.
   - Boundary: fixture, one-shot state, and detailed diagnostics remain
     Git-ignored.

4. **Reconcile private-corpus coverage, then validate bulk-PDF ingestion readiness.**
   - Outcome: first reconcile metadata-only counts across Drive inventory,
     downloaded/ingested files, indexed sources and chunks, duplicate groups,
     E1a-4 topic/role eligibility, substantive locators, and E1a-3 exclusions.
     Only if that reconciliation identifies an ingestion gap, run a
     pre-registered scale test that reports parser success/failure counts,
     chunk/index reconciliation, elapsed time, restart behavior, and
     privacy-safe failure classes for a representative multi-PDF batch.
   - Why: the approved private index already contains 4,797 chunks from 198
     sources, so `E1A4_ALLOCATION_UNAVAILABLE` is not evidence that the Drive
     lacks material. Current evidence still does not validate hundreds-of-PDF
     ingestion scale, experimentally select the 1,200/150 chunking defaults,
     preserve section or authority hierarchy, or prove retry/recovery behavior.
   - Boundary: do not ingest or reindex private material, change chunking, add
     semantic metadata, or weaken E1a-4 freshness until the reconciliation
     identifies the actual capacity loss and any follow-on experiment is
     reviewed and approved.

## P3 - Complete Optional Learning Material Deliberately

1. **Create a compact homework evidence index.**
   - Outcome: for each official Module 1-5 homework, record whether it was
     completed separately, adapted, or intentionally skipped.
   - Why: current module locks prove project learning but not course homework
     completion.

2. **Decide whether to take the dlt workshop homework.**
   - Outcome: either a workshop artifact or an explicit decision not to pursue
     it.
   - Why: dlt is used in the project, but workshop compliance is not evidenced
     and the official mandatory status is ambiguous.

3. **Evaluate optional Module 6 upgrades only against a measured gap.**
   - Candidates: document reranking, query rewriting, Elasticsearch, or
     LangChain.
   - Why: these add rubric points only when they improve a measured result and
     remain understandable to a reviewer.

4. **Consider cloud deployment only after privacy and operating boundaries are
   approved.**
   - Outcome: an explicit hosting design, secret handling, data boundary, and
     cost plan.
   - Why: the project intentionally keeps Grafana localhost-only today; cloud
     deployment is optional bonus work, not a prerequisite for correcting RAG.

## Recommended Next Approval

For E1a-4, the authenticated mapping and metadata-only 96-slot frame are sealed and independently no-write verified. The next approval is limited to private question and canonical-claim authoring. Do not retrieve, tune models, ingest, or reindex documents as part of that checkpoint. New-source acquisition is not justified by the current capacity evidence.

Approve **P0.1 plus P0.2** when ready to publish: push the two verified local
commits, then validate an independent public clone at the resulting fixed hash.
That is the shortest path from a strong local capstone to a reviewer-accessible
submission. It does not require changing the application.
