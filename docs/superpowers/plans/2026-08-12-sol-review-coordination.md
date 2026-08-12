# Sol Review Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bounded Sol reviews synthesize scoped evidence and return a verdict rather than time out during broad exploration.

**Architecture:** Terra decides whether a review is small, medium, or large. Luna collects independent bounded evidence for medium and large reviews. Sol remains the read-only review lead and synthesizes a structured final verdict, including `REVIEW_INCOMPLETE` when evidence is incomplete.

**Tech Stack:** TOML agent profiles, Node.js hook validation, Node.js contract tests, Markdown workflow documentation.

## Global Constraints

- Do not change application or RAG behavior.
- Do not read, modify, or report private evaluation artifacts.
- Agent review tasks are read-only and never commit or push.
- Preserve Terra as workflow owner, Luna as bounded evidence collector, and Sol as review lead.

---

### Task 1: Encode the Sol review contract

**Files:**
- Modify: `.codex/agents/reviewer.toml`
- Modify: `.codex/prompts/new_task.md`
- Test: `tests/codex_hooks/workflow-contract.test.cjs`

**Interfaces:**
- Consumes: review assignment packets with named scope and optional Luna evidence packets.
- Produces: a structured Sol verdict with defined categories and an overall verdict.

- [x] **Step 1: Write failing contract tests**

```js
assert.match(reviewer, /Sol is the review lead/);
assert.match(reviewer, /REVIEW_INCOMPLETE/);
assert.match(prompt, /Small review/);
```

- [x] **Step 2: Run the focused contract test and verify it fails**

Run: `node --test tests/codex_hooks/workflow-contract.test.cjs`

Expected: FAIL because the existing profile prohibits review subagent coordination and lacks the incomplete-review contract.

- [x] **Step 3: Add the minimal reviewer and startup-prompt instructions**

```text
For medium or large reviews, synthesize controller-supplied Luna evidence packets.
When evidence is incomplete, return REVIEW_INCOMPLETE rather than continuing broad exploration.
```

- [x] **Step 4: Run the focused contract test and verify it passes**

Run: `node --test tests/codex_hooks/workflow-contract.test.cjs`

Expected: PASS.

### Task 2: Enforce bounded review dispatches

**Files:**
- Modify: `.codex/hooks/agent-policy.cjs`
- Modify: `tests/codex_hooks/agent-policy.test.cjs`

**Interfaces:**
- Consumes: `Agent` packets for `agent_type: reviewer`.
- Produces: denial for reviewer packets with no structured review return contract or broad unbounded scope.

- [x] **Step 1: Write failing hook tests**

```js
assertDenied(validateAgentDispatch(packet("reviewer", "review", ["entire repository"])), /named file paths/i);
assertDenied(validateAgentDispatch(reviewerPacketWithoutVerdict), /review return contract/i);
```

- [x] **Step 2: Run the focused hook test and verify it fails**

Run: `node --test tests/codex_hooks/agent-policy.test.cjs`

Expected: FAIL because reviewer-specific scope and return validation do not exist.

- [x] **Step 3: Add minimal reviewer dispatch validation**

```js
if (input.agent_type === "reviewer" && fields.scope.some((entry) => !FILE_LIKE_SCOPE.test(entry))) {
  return deny("Sol reviews require named file paths or supplied evidence packets.");
}
```

- [x] **Step 4: Run hook tests and verify they pass**

Run: `node --test tests/codex_hooks/agent-policy.test.cjs`

Expected: PASS.

### Task 3: Verify workflow-only scope

**Files:**
- Modify: `docs/superpowers/specs/2026-08-12-sol-review-coordination-design.md`
- Modify: `docs/superpowers/plans/2026-08-12-sol-review-coordination.md`

- [x] **Step 1: Run focused workflow validation**

Run: `node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs`

Expected: PASS.

- [x] **Step 2: Run normal validation**

Run: `uv run pytest`

Expected: PASS with existing skips only.

- [x] **Step 3: Inspect changed paths**

Run: `git diff --check; git status --short`

Expected: no whitespace errors and no application/RAG source changes introduced by this task.
