# Grounded Answer Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven development task-by-task.

**Goal:** Add a public, privacy-safe baseline for grounded RAG answer evaluation.

**Architecture:** Public JSONL cases drive deterministic answer checks and an optional structured LLM judge. Local Granite/Ollama is default; OpenAI remains optional.

## Constraints

- Public synthetic data only; no private fixtures or answer text in reports.
- No production-readiness or chemistry-truth claims.
- Judge failures are visible as unavailable; never converted into passing scores.
- Commit only after explicit user instruction.

### Task 1: Public Cases and Deterministic Checks

**Files:** Create `eval/public_answer_evaluation.jsonl`, `src/oilfield_chemical_copilot/evaluation/answers.py`, `tests/evaluation/test_answers.py`.

- [x] Write failing tests for valid citation IDs, missing citations, invalid IDs, sufficient-evidence answers, and required abstention on insufficient evidence.
- [x] Implement frozen `AnswerEvaluationCase` and `DeterministicAnswerResult`; validate 12 public cases and retain only IDs/statuses.
- [x] Run `uv run pytest tests/evaluation/test_answers.py -v` and Ruff.

### Task 2: Structured Local/Optional Judge

**Files:** Create `src/oilfield_chemical_copilot/evaluation/judge.py`; modify `eval/answer_eval.py`; create `tests/eval/test_answer_eval.py`.

- [x] Write failing fake-provider tests for strict JSON rubric parsing, scores 1-5, safe provider/model identity, and unavailable judge behavior.
- [x] Implement `AnswerJudge` with Ollama default and optional OpenAI provider; require scores for groundedness, relevance, limitation awareness, and operational certainty.
- [x] Implement aggregate-only report writer with deterministic counts, average rubric scores, and judge status; assert prompt/answer/evidence/path sentinels never serialize.
- [x] Run focused pytest and Ruff.

### Task 3: Public Baseline and Documentation

**Files:** Modify `README.md`; test full suite and local public judge run.

- [x] Document what deterministic checks and a judge each measure, local Granite default, optional OpenAI, and same-model bias.
- [x] Run full pytest, then the public baseline. Confirm reports contain aggregate scores/status only and no path/text sentinels.
- [x] Report teaching checkpoint: deterministic failures identify contract breaks; rubric scores compare answer behavior but do not certify chemistry advice.

## Self-Review

Tasks cover public data, deterministic grounding, structured judge behavior, safe reporting, local/optional provider configuration, and learning limits. No private content, tool calling, or production claims are in scope.
