# Project Status

**Project:** Oilfield Chemical Troubleshooting Copilot<br>
**Last updated:** 2026-08-10

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

## Learning Progress

- **Module 1: Core RAG fundamentals** - locked.
- **Module 2: Evaluation** - locked.
- **Module 3: Safety and claim scope** - locked.
- **Module 4: Tool calling** - next lesson; the code remains preserved as implemented-early work.

## Later Milestones

- Online monitoring
- Orchestration
- Tool validation
- Capstone readiness

The completed baselines, diagnosis, and policy boundary do not select a retrieval winner or establish chemistry correctness, operational safety, private-corpus quality, or production readiness. The policy result is a bounded claim-scope control, not a substitute for field-specific engineering review.

## Immediate Next Task

Resume the **Module 4 tool-calling lesson** from the explicit product-ppm water-basis contract, validation, deterministic calculation, and scope-first route. No new implementation is required until the lesson exposes a verified gap.

## Course Alignment

The course-aligned sequence and its project-specific quality gates are maintained in [COURSE_ALIGNED_PLAN.md](COURSE_ALIGNED_PLAN.md). Official course ordering introduces function calling earlier; this project intentionally evaluates the already built baseline before expanding capability.

## Source of Truth

This file records implementation status. [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) is the source of truth for learning sequence; [COURSE_ALIGNED_PLAN.md](COURSE_ALIGNED_PLAN.md) is the detailed milestone ledger. Task-specific implementation briefs and reports record approval-gated evidence.
