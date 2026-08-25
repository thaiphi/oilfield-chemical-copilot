# Live RAG Answer Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Compare the real public-corpus `vector` and `hybrid` RAG paths using the same local Granite generator and grounded-answer evaluator.

**Architecture:** A live evaluator runs `BasicRagService` without an oracle topic filter, captures its structured draft plus cited public source excerpts only in memory, and evaluates each mode with deterministic checks and the existing structured judge. A separate comparison writer stores aggregate-only, per-mode reports; no answer or evidence text persists.

**Tech Stack:** Python 3.11, PGVector, MinSearch, local Ollama Granite, existing RAG service, pytest, Ruff.

## Global Constraints

- Public synthetic cases and the exact `data/sample` manifest only; reject a mixed or incomplete database before retrieval.
- Compare `vector` and `hybrid` under identical public questions, embeddings, Granite model, and non-mode settings; vary only `retrieval_mode`.
- Call `BasicRagService.answer(question)` with no `topic`; the live comparison must not use `oracle_gold_topic`.
- Use local Ollama `granite4.1:8b` with generation option `{"temperature": 0}`; OpenAI is out of scope for this comparison.
- Runtime answer text, evidence excerpts, source IDs, paths, database URLs, credentials, and raw errors must never serialize to reports.
- Judge failures stay `unavailable`; do not create substitute scores or a false winner.
- Do not change retrieval behavior or claim chemistry correctness, production readiness, or a selected best mode from this small baseline.
- Commit only after explicit user instruction.

---

### Task 1: Report-Safe Evaluation Primitives and Public Case Review

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/answers.py`
- Modify: `eval/answer_eval.py`
- Modify: `eval/public_answer_evaluation.jsonl`
- Modify: `tests/evaluation/test_answers.py`
- Modify: `tests/eval/test_answer_eval.py`

**Interfaces:**
- Move runtime-only `GeneratedAnswer`, report-safe `AnswerEvaluationResult`, `evaluate_cases`, and report-summary helpers from the CLI into the evaluation package.
- Add `write_mode_comparison_report(results_by_mode, output_dir, provenance)` producing only per-mode question counts, deterministic count maps, judge count maps, aggregate scores, safe mode/config provenance, and no case IDs.
- Retain the existing single-fixture CLI behavior through package imports.
- Expand sufficient-case `allowed_evidence_ids` only with human-reviewed public chunk IDs; do not derive allowable IDs from live retrieval output.

- [ ] Write failing tests for per-mode aggregate reports, source/answer/path sentinel exclusion, and broader reviewed public allowable IDs.
- [ ] Run the focused tests and confirm the new API/imports are absent or reports have the old unsafe shape.
- [ ] Implement the smallest package refactor and public case update.
- [ ] Run focused tests and Ruff.

### Task 2: Deterministic Granite Generation and Live Capture

**Files:**
- Modify: `src/oilfield_chemical_copilot/rag/ollama_client.py`
- Modify: `tests/rag/test_ollama_client.py`
- Create: `src/oilfield_chemical_copilot/evaluation/live_rag.py`
- Create: `tests/evaluation/test_live_rag.py`

**Interfaces:**
- Extend `OllamaAnswerClient` and `LazyOllamaAnswerClient` with optional `generation_options: dict[str, object] | None`; default remains `None` for app behavior.
- Define `capture_live_answer(case, service, recording_generator) -> GeneratedAnswer`.
- The recording generator delegates to the real Ollama generator and retains the `RagDraft` only for the current call. A successful capture supplies the draft fields as runtime answer material, cited `SourceEvidence.chunk_id` values, public source excerpts as runtime evidence, and `RagAnswer.weak_evidence`. A fallback supplies no citations/evidence and records abstention.

- [ ] Write failing tests proving temperature `0` reaches the delegated Ollama client, successful captures use draft/cited chunk IDs, and weak/failing generation captures abstention without source text in result models.
- [ ] Run focused tests to establish RED.
- [ ] Implement the optional generation options and live-capture module.
- [ ] Run focused tests and Ruff.

### Task 3: Public-Only Vector versus Hybrid Runner

**Files:**
- Create: `eval/live_rag_answer_eval.py`
- Create: `tests/eval/test_live_rag_answer_eval.py`
- Modify: `src/oilfield_chemical_copilot/evaluation/live_rag.py`

**Interfaces:**
- CLI flags: `--dataset` defaulting to `eval/public_answer_evaluation.jsonl`, `--output-dir` defaulting to `data/processed/evaluation/live_rag`, and `--database-url` defaulting from `DATABASE_URL`.
- Load the public cases, derive `public_sample_chunk_ids()`, call `validate_public_stored_chunk_ids`, build the shared embedding provider/store/keyword index once, then build mode-specific pipelines with `replace(settings, retrieval_mode=mode)`.
- For each mode and case, call `service.answer(case.question)` with no topic; run the deterministic evaluator and shared `AnswerJudge`; write a report keyed by `vector` and `hybrid`.
- Provenance contains only public dataset/corpus hashes, safe provider labels, hashed generation/judge model identifiers, mode settings, fixed temperature, and `topic_filter: "none"`.

- [ ] Write failing fake-service/store tests for exact public-manifest preflight, `topic=None`, identical non-mode settings, mode separation, all-unavailable judge handling, and report sentinel exclusion.
- [ ] Run the focused CLI tests to establish RED.
- [ ] Implement the runner and aggregate-only comparison report.
- [ ] Run focused tests and Ruff.

### Task 4: Documentation and Public Baseline

**Files:**
- Modify: `README.md`
- Modify: `docs/COURSE_ALIGNED_PLAN.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `.codex/prompts/new_task.md`

- [ ] Document the exact Docker/Postgres/Ollama prerequisites and the live command.
- [ ] Explain the difference between the synthetic evaluator fixture and live RAG capture; explain why no oracle topic filter is used.
- [ ] Update stale project-status claims and repair the startup prompt's references to missing project-plan/current-inventory files.
- [ ] Run `uv run pytest -q`, focused Ruff, then the live public baseline; scan JSON and Markdown reports for text, source-ID, path, URL, credential, and error sentinels.
- [ ] Record per-mode aggregate results without claiming a winner or operational readiness.

## Acceptance Criteria

- Both modes run the identical public case set through actual `BasicRagService` and local Granite with temperature `0`.
- The live runner refuses a mixed/incomplete database and never applies a topic filter.
- Reports are aggregate-only and contain no runtime text or sensitive metadata.
- Unit tests cover capture, mode separation, preflight, unavailable judge, and privacy boundaries; full suite passes.
- The final baseline is clearly labeled as a small public comparison, not chemistry validation or a retrieval-change decision.
