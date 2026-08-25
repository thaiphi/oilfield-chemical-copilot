# Module 7 Capstone Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a public, reproducible reviewer path and a privacy-safe capstone evidence package without changing application behavior.

**Architecture:** Keep all existing runtime boundaries unchanged. Add a reviewer guide and rubric evidence map that point only to the public sample corpus and tracked evidence, then use focused documentation tests plus an existing public-sample run to verify the claimed path.

**Tech Stack:** Python 3.11, `uv`, pytest, Ruff, Docker Compose, PostgreSQL/PGVector, local Ollama, Streamlit, Grafana, Kestra.

## Global Constraints

- Use `data/sample` only for all Module 7 live checks; do not read, copy, index, or describe `.private`, `data/private`, or `eval/private` material.
- Do not change RAG, retrieval, claim-scope, tool, evaluation, or monitoring behavior.
- Keep Ollama local; OpenAI is optional and not required for the public reviewer path.
- Do not add reranking or query rewriting without a separately approved measured-gap experiment.
- The reviewer-facing path must not require a private path, credential, or untracked artifact.
- Do not lock Module 5, lock Module 7, commit, or push without separate user approval.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `docs/CAPSTONE_REVIEWER_GUIDE.md` | Exact public-clone, setup, public ingest/index, Streamlit, evaluation, and Grafana sequence. |
| `docs/CAPSTONE_EVIDENCE.md` | Rubric-to-evidence map and explicit limitations. |
| `tests/capstone/test_reviewer_docs.py` | Static public-only, reproducibility, and rubric documentation contract. |
| `README.md` | Concise entry links to the guide and evidence map. |
| `docs/superpowers/reports/2026-08-15-module-7-capstone-readiness.md` | Sanitized live verification result and residual gate. |
| `docs/PROJECT_STATUS.md` | Module 7 implementation progress without premature lock claim. |
| `docs/LEARNING_ROADMAP.md` | Active Module 7 state and remaining Module 5 approval gate. |
| `docs/COURSE_ALIGNED_PLAN.md` | Detailed capstone evidence ledger. |

## Task 1: Public Reviewer Documentation Contract

**Files:**
- Create: `tests/capstone/test_reviewer_docs.py`
- Create: `docs/CAPSTONE_REVIEWER_GUIDE.md`
- Create: `docs/CAPSTONE_EVIDENCE.md`

- [x] Write static tests requiring the reviewer guide to contain `uv sync`, public-only ingestion/indexing, Docker Compose, Streamlit, the explicit Grafana demo seed, full test verification, and Git privacy checks.
- [x] Write static tests rejecting `.private`, `data/private`, `eval/private`, `OPENAI_API_KEY=`, and a private local path in reviewer-facing documents.
- [x] Write the guide with Windows PowerShell commands, local Ollama model prerequisites, public `data/sample` ingestion/indexing, app startup, a public-safe query example, Grafana demo viewing, evaluation evidence locations, and teardown.
- [x] Write the evidence map linking every course criterion to tracked code, reports, and reviewer actions. State that no production, chemistry, or private-corpus claim is made.
- [x] Run `uv run pytest tests/capstone/test_reviewer_docs.py -q` and `uv run ruff check tests/capstone`.

## Task 2: Integrate The Reviewer Surface

**Files:**
- Modify: `README.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `docs/LEARNING_ROADMAP.md`
- Modify: `docs/COURSE_ALIGNED_PLAN.md`

- [x] Add a concise README capstone-review entry that links to the guide and evidence map.
- [x] Mark Module 7 as active implementation with a reviewer package in progress; keep Module 5 active and unsealed.
- [x] Record the rubric/evidence relationship and the rule that reranking/query rewriting remain excluded without a measured-gap experiment.
- [x] Run `uv run pytest tests/capstone/test_reviewer_docs.py -q` and `git diff --check`.

## Task 3: Run The Public Reviewer Path

**Files:**
- Create: `docs/superpowers/reports/2026-08-15-module-7-capstone-readiness.md`

- [x] Validate `docker compose config --quiet` and the documented public-only environment values.
- [x] Run public sample parsing and public indexing against the local public database with local Ollama embeddings; record only file/chunk/index aggregate counts.
- [x] Start the app and monitoring services, run the explicit synthetic Grafana seed, and verify the Streamlit and Grafana local endpoints without capturing private or raw answer content.
- [x] Run `uv run pytest`, `node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs`, `uv run ruff check .`, and `git diff --check`.
- [x] Write the report with command status, aggregate public counts, six-panel verification, privacy scan result, and the residual Module 5 lock gate only.

## Task 4: Release-Candidate Privacy Audit

**Files:**
- Modify: `tests/capstone/test_reviewer_docs.py`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `docs/LEARNING_ROADMAP.md`
- Modify: `docs/COURSE_ALIGNED_PLAN.md`

- [x] Add static checks that the evidence map names feedback plus at least five dashboard charts and explains the no-reranking/no-query-rewriting decision.
- [x] Run `git status --short`, `git diff --name-only`, `git ls-files --others --exclude-standard`, `git diff --cached --name-only`, and targeted `git check-ignore` checks for `.private`, `data/private`, `eval/private`, and `.env`.
- [x] Update Module 7 as evidence-complete but not locked; state that the user must review/lock Module 5 before a final capstone readiness or commit decision.
- [x] Run `uv run pytest tests/capstone -q`, `uv run ruff check .`, and `git diff --check`.

## Plan Self-Review

- Spec coverage: Tasks 1-2 create and integrate the reviewer contract; Task 3 proves it against public data and existing services; Task 4 audits Git/privacy and records the remaining human approval gate.
- Placeholder scan: every task names its files, commands, and prohibited data classes.
- Scope: no task alters the RAG pipeline or processes private corpus material.
