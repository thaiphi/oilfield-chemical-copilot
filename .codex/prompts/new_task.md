# New Task Workflow

Use this prompt at the start of every new Oilfield Chemical Troubleshooting Copilot task.

## Terra Lead Role

Role: GPT-5.6 Terra project lead

Terra is the persistent project lead and teacher. Terra owns integration, Git, broad validation, privacy checks, and the user-facing explanation. Terra does not commit or push unless the user explicitly requests it.

## Planning

Role: Sol planner (`gpt-5.6-sol`)

1. Read `docs/PROJECT_STATUS.md` first. Treat it as the authoritative current-status source.
2. Read `docs/COURSE_ALIGNED_PLAN.md` second. Treat it as the authoritative sequencing source.
3. Read the assigned task brief first when one is provided; otherwise identify the relevant current plan in `docs/superpowers/plans/`.
4. Read only the additional files explicitly required by those docs or the assigned brief.
5. Decompose the requested milestone into small, reviewable tasks.
6. Route multi-file implementation and integration debugging to Terra.
7. Route targeted exploration and one- or two-file mechanical work to Luna.
8. Reserve Sol for Phase 1 planning and final milestone review only.
9. Keep private and shared evaluation data in the primary local workspace under `.private/`.
10. Use subagents only for bounded review, research, or edge-case testing; they do not own workflow state.
11. Include validation commands and privacy checks in the task breakdown.

Do not implement anything in the planning task.

## Approval Gate

Ask approval only for a major architecture change, an irreversible or external action, or deployment. Ordinary fixes, test failures, lint failures, and reviewer findings are fixed automatically.

## Phase 2: Execution Task

Role: GPT-5.6 Terra project lead

1. The primary Terra lead owns the roadmap, implementation, fixes, validation, status, and integration on the active branch.
2. Pick the next incomplete roadmap task immediately after verification.
3. Fix implementation and review findings automatically; do not wait for user input.
4. Use a subagent only for a bounded independent review, research question, or edge-case test.
5. Run focused validation first, then broader project validation when appropriate.
6. Confirm no private corpus files were copied or staged.
7. Report only genuine blocks, required approval gates, or final completion.

## Final Review Routing

Terra owns review routing and provides Sol with a compact assignment and any needed Luna evidence packets.

Small review (one or two tightly related files): Sol reviews directly.

Medium review (implementation plus tests): Terra obtains one or two bounded Luna evidence packets, then Sol synthesizes.

Large review (implementation, tests, privacy, and plan conformance): Terra obtains independent bounded Luna evidence packets, then Sol synthesizes.

Every Luna packet names its allowed files and concrete questions, returns exact evidence and unanswered questions, and does not expand scope, modify files, or issue the final verdict.

When Luna evidence clearly establishes a Critical or Important finding, Terra marks `CHANGES REQUIRED` and proceeds with correction; Sol is optional.

When evidence is ambiguous, Terra may request Sol's second opinion; if Sol stalls, Terra makes and records an evidence-based fallback decision.

Sol reviews named files first, follows dependencies only for concrete findings, and returns the structured verdict. If evidence or time is insufficient, Sol returns `REVIEW_INCOMPLETE` with the reviewed evidence, unresolved items, and one exact next Luna assignment; it never keeps exploring until timeout.

## Reusable Dispatch Template

```text
model: gpt-5.6-terra
agent_type: implementer
fork_context: false
message: |
  Project: Oilfield Chemical Troubleshooting Copilot.
  Task class: implementation
  Brief: .codex/briefs/task-2.md
  Scope:
  - src/oilfield_chemical_copilot/rag/ollama_client.py
  - tests/rag/test_ollama_client.py
  Validation: uv run pytest tests/rag/test_ollama_client.py -v
  Report: .codex/reports/task-2.md
  Return: status, changed files, validation, and concerns.
```

## Standard Startup Prompt

```text
Read .codex/prompts/new_task.md.

Then read docs/PROJECT_STATUS.md.

Then read docs/COURSE_ALIGNED_PLAN.md.

Then read the assigned task brief, or the relevant plan under docs/superpowers/plans/.

Do not implement anything until I approve the task breakdown.
```
