# Terra-Led Subagent Workflow and Dispatch Guard

## Purpose

Make GPT-5.6 Terra the persistent project lead and teaching partner for the
Oilfield Chemical Troubleshooting Copilot. Use GPT-5.6 Sol only for milestone
planning, difficult escalations, and final milestone review. Use GPT-5.6 Luna
for tightly scoped exploration and small mechanical changes.

The workflow must preserve the existing user approval gate: no milestone
implementation begins until the user approves Sol's task breakdown.

## Model Responsibilities

| Role | Model | Owns |
| --- | --- | --- |
| Main conversation and project lead | GPT-5.6 Terra | Project context, teaching, task coordination, integration, validation, and user-facing status |
| Milestone planner and final reviewer | GPT-5.6 Sol | Task decomposition, difficult architectural escalations, and final milestone review |
| Feature implementer | GPT-5.6 Terra | Bounded multi-file feature work, focused tests, documentation, and implementation fix rounds |
| Explorer and mechanical implementer | GPT-5.6 Luna | Read-only investigation, isolated one- or two-file changes, focused tests, and narrow documentation updates |

GPT-5.5 is removed from the default project routing. The main conversation
remains with Terra so the user has one consistent technical guide. Sol returns
plans and reviews to Terra; Terra explains and presents them to the user.

## Dispatch Contract

Each subagent dispatch must use `fork_context: false` and contain:

1. One sentence of project context.
2. One task-brief path as the requirements source.
3. Named file or module scope and required interfaces.
4. A report-file path and concise return contract.
5. Focused validation requirements.

The lead keeps Git, broad project validation, privacy checks, and user-facing
teaching in the main thread. The same implementer is reused for fix rounds when
possible. At most two subagents run concurrently.

## Hook Guardrails

The hook layer enforces only deterministic policy. It does not attempt to judge
whether a task is intellectually difficult enough for Sol.

### Session Guidance

A `SessionStart` hook adds project-lead context on startup, resume, and
compaction. It states that Terra is the persistent lead and reminds the model
to delegate planning and final review to Sol. If the active model is not Terra,
the hook emits a visible warning rather than blocking the session.

### Spawn Validation

A `PreToolUse` hook matching the local `Agent` tool validates subagent creation
before a worker starts. It denies dispatches that:

- omit `fork_context: false`;
- use an undefined agent profile;
- omit a task brief, report path, named scope, or validation instruction;
- attempt an implementation dispatch through the Sol planner or reviewer
  profiles; or
- describe broad, unspecified work for a Luna implementation profile.

The agent profiles themselves pin model and sandbox settings, so the hook
validates role selection and packet shape rather than duplicating model mapping
logic.

### Configuration

Project configuration sets Terra as the main model, enables multi-agent support
and hooks, and limits concurrently open subagents to two. The default spawned
agent remains Luna because every real dispatch must explicitly select its role.

## Files to Change During Implementation

- `.codex/config.toml`
- `.codex/hooks.json`
- `.codex/hooks/agent-policy.cjs`
- `.codex/agents/planner.toml`
- `.codex/agents/explorer.toml`
- `.codex/agents/implementer.toml`
- `.codex/agents/mechanical-implementer.toml`
- `.codex/agents/reviewer.toml`
- `.codex/prompts/new_task.md`
- `tests/codex_hooks/test_agent_policy.ps1` or an equivalent focused Node test

## Validation

1. Run the policy script against valid Terra, Luna, and Sol dispatch payloads.
2. Confirm malformed packets and invalid `fork_context` values are denied.
3. Confirm a Luna broad-scope packet is denied and a bounded packet passes.
4. Parse the hook configuration and validate every referenced local path.
5. Review `git diff` to ensure no private corpus paths or contents were added.
6. Start or resume a Codex session, trust the project-local hook, and confirm
   the session guidance is surfaced.

## Non-Goals

- Automatically deciding a task's intellectual difficulty.
- Replacing user approval of a milestone task breakdown.
- Adding a separate agent for tests, documentation, Git, or push operations.
- Changing the application code for the RAG milestone.
