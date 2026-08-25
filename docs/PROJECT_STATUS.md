# Project Status

**Project:** Oilfield Chemical Troubleshooting Copilot<br>
**Last updated:** 2026-08-15

## Completed Milestones

1. **Inventory**: source inventory is complete.
2. **Parsing and chunking**: deterministic ingestion and chunking are complete.
3. **Storage, keyword, and vector retrieval**: storage plus keyword and vector retrieval are complete.
4. **Source-grounded local Ollama RAG**: local, source-grounded answer generation is complete.
5. **Hybrid RRF retrieval**: hybrid keyword-plus-vector retrieval with Reciprocal Rank Fusion is complete.
6. **Public retrieval evaluation**: the fixed-`k` public retrieval baseline is complete.
7. **Public synthetic-answer evaluation**: deterministic checks and structured-judge evaluation for the committed answer fixture are complete.
8. **Live public vector-versus-hybrid RAG baseline**: both modes ran 12 public questions. Each had deterministic citations of 4 pass/8 fail and abstention of 6 pass/6 fail; all 12 judge results were available per mode.
9. **Approved live RAG failure diagnosis**: baseline reproduction is true. Both modes ran 12 questions and had category-identical failures: citations had allowed-retrieved-not-cited (1), mixed-with-disallowed (1), and unexpected-citation-when-abstention-expected (6); abstention had under-abstention (6).
10. **Claim-scope policy investigation**: the approved public local shadow run reproduced the control baseline in both modes. Each mode had 12 paired questions and six allow/six abstain decisions; control was citation 4 pass/8 fail and abstention 6 pass/6 fail, while shadow was citation 10 pass/2 fail and abstention 12 pass/0 fail.
11. **Production-hardening holdout and policy integration**: the sealed local v2 holdout scored 36/36 exact actions and categories, with zero false allows, zero false abstains, and zero stratum failures. The claim-scope policy now runs before RAG; focused tests prove abstain makes zero retriever/generator calls and general-review questions continue through the normal RAG path.
12. **Citation selection**: a deterministic source selector now requires a question match against a source filename or declared topic before using answer-content overlap. A local ID-only rerun kept retrieval and the claim-scope policy unchanged; both vector and hybrid modes retrieved allowed evidence and cited allowed-only evidence for all six evidence-sufficient public cases.
13. **Chemical-dose tool boundary**: the explicit `Product dose:` contract validates product-ppm water-basis inputs and calculates product gallons per day with the `42 gal/bbl` conversion. Recognized requests pass the production claim-scope gate before parsing; closed requests make zero calculator, retriever, and generator calls, while non-tool questions retain the RAG path.
14. **Aggregate-safe monitoring**: process-local monitoring records six closed response and routing outcomes plus count/minimum/average/maximum latency. It accepts no payloads and retains no prompts, answers, excerpts, source paths, tool inputs, identifiers, or raw errors; existing raw-content database tables remain unused.
15. **V4 semantic-grounding evaluation**: a fresh sealed private 36-case pre-fix holdout scored the real formatter once. It passed 24/36 (66.7%), with 12 false allows and zero false fallbacks across numeric values, ranges, units, qualifiers, conflicting evidence, and absent-threshold statements. Public synthetic regressions now guard the discovered failure classes. Post-score review also made future evaluator runs require preflight, enforce private/public artifact paths, and reject sensitive aggregate-report key fragments before sealing. V4 was not rerun, so a fresh v5 is required to measure the correction.
16. **V5 semantic-grounding evaluation**: a fresh sealed private 36-case holdout scored the real formatter exactly once. It passed 28/36 (77.8%), with seven false allows and one false fallback. Aggregate failure classes were comparator change (2), condition omission (2), conflict erasure (2), unit substitution (1), and grounded claim (1). The fixture and diagnostics remain private and ignored; V5 will not be rerun. Public synthetic regressions now cover those aggregate-safe classes, and a fresh V6 is required to measure the correction.
17. **V6 semantic-grounding evaluation**: a fresh sealed private 36-case holdout scored the real formatter exactly once after recursive aggregate-report privacy hardening and the V5 formatter correction. It passed 36/36 with zero false allows and zero false fallbacks across numeric values, ranges, units, conditions, conflicts, and absent-threshold statements. The V6 fixture and diagnostics remain private and ignored; V6 will not be rerun. This is a controlled formatter-boundary result, not proof of chemistry correctness, retrieval quality, LLM-answer quality, or production readiness.
18. **Module 1 bounded agentic-routing implementation**: the opt-in local route has a one-decision Ollama planner and exactly two controller-owned tools: knowledge search and deterministic product-dose calculation. Claim-scope abstention occurs before planning; unknown, malformed, multi-tool, and planner-error outcomes execute no tool and fall back to the existing RAG path. The public suite passed 518 tests (2 integration tests skipped), the workflow-contract suite passed 22 checks, and Ruff plus `git diff --check` passed. The local Granite smoke test passed on 2026-08-14: the service was reachable, `granite4.1:8b` was present, and the planner returned exactly one tool call. Only aggregate smoke status was recorded.
19. **Module 3 public Kestra orchestration**: a five-stage public-sample flow completed inventory, parse/chunk, local Granite embed/load, count validation, and dlt aggregate publication. The verified run recorded 10 source files, 11 chunks, 11 expected and actual indexed chunks, and one aggregate publication using `granite-embedding:latest`. The worker image excludes private build-context material, and the durable report contains no source text, source names, paths, credentials, or execution identifiers.
20. **Module 4 public dual evaluation**: the active vector and hybrid RAG boundaries completed against the committed public answer pack with the claim-scope policy enabled. Across six answerable cases, vector recorded Hit Rate@5/MRR@5 `1.000/1.000`, citation `9/3`, and abstention `12/0`; hybrid recorded `1.000/0.917`, citation `10/2`, and abstention `10/2`. This public measurement demonstrates that successful retrieval does not alone prove correct citation or abstention behavior. It does not establish chemistry correctness, operational safety, private-corpus quality, or a mode winner.
21. **Module 4 sealed local evaluation attempt**: the approved 12-case handout fixture was sealed and its one-shot state was consumed before RAG initialization. Runtime evaluation returned the fixed `RUNTIME_UNAVAILABLE` category, so an aggregate-only `unavailable` report with zero metrics was recorded. No score was produced, no fixture values entered durable project material, and the sealed hash will not be replayed. The sanitized category does not retain enough information to diagnose the underlying generation failure retrospectively.
22. **Module 4 sealed local v2 evaluation**: after a separate non-scoring synthetic runtime smoke check succeeded, a fresh non-overlapping 12-case fixture was sealed and scored once. Across six answerable cases, vector recorded Hit Rate@5/MRR@5 `0.500/0.500`, citation `7/5`, and abstention `8/4`; hybrid recorded `0.833/0.722`, citation `8/4`, and abstention `10/2`. Hybrid performed better on all recorded aggregates, but neither mode is ready for operational use. The v2 fixture, detailed statuses, and one-shot state remain private and will not be replayed.
23. **Module 5 privacy-safe local monitoring**: locked on 2026-08-16 after the live dashboard and teaching review. Streamlit records only closed hourly outcome, retrieval-mode, latency, and aggregate-feedback metrics. Dedicated migrations, a least-privilege Grafana reader role, a six-panel locally bound dashboard with panel information descriptions, and an explicit fixed synthetic seed were verified end-to-end. No raw question, answer, citation, source, identifier, credential, or private material entered the durable evidence.
24. **Module 7 capstone reviewer package**: locked on 2026-08-16 after Module 5 approval. The public reviewer guide, rubric evidence map, and documentation contract are complete. The tracked sample path parsed/indexed 11 chunks with local Granite embeddings; Streamlit and Grafana returned HTTP 200; the explicit synthetic seed completed; and full verification passed 579 Python tests, 22 workflow-contract tests, and Ruff. The aggregate-only report records the additive demo-seed boundary. The package is local and reproducible, not hosted or production-ready.

## Learning Progress

- **Module 1: Agentic RAG** - locked on 2026-08-14. The bounded implementation has public and local Granite evidence, and the practical teaching review is complete.
- **Module 2: Vector Search** - locked on 2026-08-15. The teaching review covered embedding-model coordinate systems, semantic ranking, topic filtering, and PGVector storage. Focused public checks passed 35 tests; the count-only live Granite-to-PGVector smoke check returned one result under the `scale` topic filter.
- **Module 3: Orchestration** - locked on 2026-08-15. The practical teaching review covered task boundaries, artifact handoff, independent count validation, and aggregate-only dlt publication.
- **Module 4: Evaluation** - locked on 2026-08-15. Public and fresh sealed-local v2 evaluation are complete, and the teaching review connects their different results to ground truth, Hit Rate@5, MRR@5, citations, and abstention. Any RAG change needs a separate approved experiment and a fresh fixture.
- **Module 5: Monitoring** - locked on 2026-08-16. The teaching review covered outcome versus feedback, latency, privacy-safe hourly aggregation, Grafana's read-only role, synthetic telemetry, and local reviewer traffic.
- **Module 6: Best Practices** - hybrid RRF is implemented early; reranking and query rewriting remain deferred.
- **Module 7: End-to-End Project** - locked on 2026-08-16. The public reviewer package and live local path are verified with Module 5 as the dashboard/monitoring component.

## Later Milestones

- Alerting and production monitoring operations
- Orchestration
- Tool validation
- Capstone readiness

The completed baselines, diagnosis, and policy boundary do not select a retrieval winner or establish chemistry correctness, operational safety, private-corpus quality, or production readiness. The policy result is a bounded claim-scope control, not a substitute for field-specific engineering review.

## Immediate Next Task

Audit the locked Module 5 and Module 7 changes, then create intentional commits. Future capability work requires a separate approved scope.

## Course Alignment

The course-aligned sequence and its project-specific quality gates are maintained in [COURSE_ALIGNED_PLAN.md](COURSE_ALIGNED_PLAN.md). Official course ordering introduces function calling earlier; this project intentionally evaluates the already built baseline before expanding capability.

## Source of Truth

This file records implementation status. [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) is the source of truth for learning sequence; [COURSE_ALIGNED_PLAN.md](COURSE_ALIGNED_PLAN.md) is the detailed milestone ledger. Task-specific implementation briefs and reports record approval-gated evidence.
