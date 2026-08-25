# Terra-Led Subagent Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GPT-5.6 Terra the persistent project lead and add a project-local Codex hook that rejects malformed or incorrectly routed subagent dispatches.

**Architecture:** A small CommonJS policy module receives Codex hook JSON through stdin. A `SessionStart` invocation returns compact Terra-led workflow context, while a `PreToolUse` invocation validates the `Agent` tool's structured dispatch arguments before a subagent can start. Static workflow files define role-specific prompts and configuration; native Node tests verify both policy behavior and file contracts.

**Tech Stack:** Codex project configuration and hooks, Node.js 22 CommonJS, Node built-in `node:test`, PowerShell, Python 3.11 `tomllib` for TOML parse validation.

## Global Constraints

- The main conversation model is exactly `gpt-5.6-terra`.
- Sol is used only through the `planner` and `reviewer` profiles.
- Terra owns multi-file implementation, integration debugging, and user-facing teaching.
- Luna owns only targeted exploration and mechanical implementation scoped to one or two named files.
- Every dispatch must explicitly set `fork_context: false` and use a `Task class`, `Brief`, `Scope`, `Validation`, and `Report` field.
- Keep at most two subagents concurrently open.
- Do not add private corpus files or contents to the repository.
- Do not commit or push unless the user explicitly asks.

---

## File Structure

- `.codex/hooks/agent-policy.cjs`: Pure policy functions plus a stdin/stdout hook entrypoint.
- `.codex/hooks.json`: Connects Codex `SessionStart` and `PreToolUse` events to the policy module.
- `tests/codex_hooks/agent-policy.test.cjs`: Native Node tests for accepted and rejected hook payloads.
- `tests/codex_hooks/workflow-contract.test.cjs`: Static checks for hook configuration, model routing, and packet contract language.
- `.codex/config.toml`: Sets Terra as the project lead and enables multi-agent tools and hooks.
- `.codex/agents/*.toml`: Pins each worker profile to its approved model and role boundary.
- `.codex/prompts/new_task.md`: Documents the Terra-led, Sol-planned approval-gated workflow and exact dispatch packet format.

### Task 1: Create and Test the Dispatch Policy Module

**Files:**

- Create: `.codex/hooks/agent-policy.cjs`
- Create: `tests/codex_hooks/agent-policy.test.cjs`

**Interfaces:**

- Consumes: Codex `SessionStart` payloads with `model` and `source`, and `PreToolUse` payloads where `tool_name` is `Agent` and `tool_input` contains `agent_type`, `fork_context`, and `message`.
- Produces: `sessionStartOutput(payload)` and `validateAgentDispatch(payload)`, each returning either a valid Codex hook-output object or `null` for a permitted dispatch.

- [ ] **Step 1: Write the failing policy tests**

Create `tests/codex_hooks/agent-policy.test.cjs` with these fixtures and assertions:

```js
const test = require("node:test");
const assert = require("node:assert/strict");
const {
  sessionStartOutput,
  validateAgentDispatch,
} = require("../../.codex/hooks/agent-policy.cjs");

const packet = (agentType, taskClass, scopeLines) => ({
  tool_name: "Agent",
  tool_input: {
    agent_type: agentType,
    fork_context: false,
    message: [
      "Project: Oilfield Chemical Troubleshooting Copilot.",
      `Task class: ${taskClass}`,
      "Brief: .codex/briefs/task-1.md",
      "Scope:",
      ...scopeLines.map((path) => `- ${path}`),
      "Validation: node --test tests/codex_hooks/agent-policy.test.cjs",
      "Report: .codex/reports/task-1.md",
      "Return: status, changed files, validation, and concerns.",
    ].join("\\n"),
  },
});

test("allows a valid Terra implementation packet", () => {
  assert.equal(
    validateAgentDispatch(packet("implementer", "implementation", [
      "src/rag/provider.py",
      "tests/rag/test_provider.py",
    ])),
    null,
  );
});

test("rejects inherited or true fork_context", () => {
  const payload = packet("implementer", "implementation", ["src/rag/provider.py"]);
  delete payload.tool_input.fork_context;
  assert.match(validateAgentDispatch(payload).hookSpecificOutput.permissionDecisionReason, /fork_context/i);
});

test("rejects a planner packet marked as implementation", () => {
  const result = validateAgentDispatch(packet("planner", "implementation", ["src/rag/provider.py"]));
  assert.match(result.hookSpecificOutput.permissionDecisionReason, /planner/i);
});

test("rejects a Luna mechanical packet with more than two files", () => {
  const result = validateAgentDispatch(packet("mechanical-implementer", "mechanical", [
    "a.py", "b.py", "c.py",
  ]));
  assert.match(result.hookSpecificOutput.permissionDecisionReason, /one or two/i);
});

test("rejects an undefined agent profile", () => {
  const result = validateAgentDispatch(packet("default", "implementation", ["src/rag/provider.py"]));
  assert.match(result.hookSpecificOutput.permissionDecisionReason, /approved named agent profiles/i);
});

test("rejects a packet without a task brief", () => {
  const payload = packet("implementer", "implementation", ["src/rag/provider.py"]);
  payload.tool_input.message = payload.tool_input.message.replace(
    "Brief: .codex/briefs/task-1.md\\n",
    "",
  );
  assert.match(validateAgentDispatch(payload).hookSpecificOutput.permissionDecisionReason, /Brief/i);
});

test("adds Terra-led context and warns on an unexpected main model", () => {
  const output = sessionStartOutput({ model: "gpt-5.5", source: "resume" });
  assert.match(output.hookSpecificOutput.additionalContext, /Terra/i);
  assert.match(output.systemMessage, /gpt-5.6-terra/i);
});
```

- [ ] **Step 2: Run the policy test to verify it fails**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs
```

Expected: FAIL because `.codex/hooks/agent-policy.cjs` does not exist.

- [ ] **Step 3: Implement pure validation and the hook entrypoint**

Create `.codex/hooks/agent-policy.cjs` with this behavior:

```js
const fs = require("node:fs");

const ROLE_TASK_CLASSES = Object.freeze({
  planner: "planning",
  reviewer: "review",
  implementer: "implementation",
  explorer: "exploration",
  "mechanical-implementer": "mechanical",
});

function deny(reason) {
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: reason,
    },
  };
}

function packetFields(message) {
  const text = String(message || "");
  const taskClass = text.match(/^Task class:\s*(.+)$/im)?.[1]?.trim();
  const brief = text.match(/^Brief:\s*(\.codex\/briefs\/[^\s]+\.md)$/im)?.[1];
  const report = text.match(/^Report:\s*(\.codex\/reports\/[^\s]+\.md)$/im)?.[1];
  const validation = text.match(/^Validation:\s*\S.+$/im)?.[0];
  const scopeBlock = text.match(/^Scope:\s*\r?\n((?:\s*-\s+[^\r\n]+\r?\n?)+)/im)?.[1] || "";
  const scope = [...scopeBlock.matchAll(/^\s*-\s+(.+)$/gm)].map((match) => match[1].trim());
  return { taskClass, brief, report, validation, scope };
}

function validateAgentDispatch(payload) {
  if (payload?.tool_name !== "Agent") return null;
  const input = payload.tool_input || {};
  if (input.fork_context !== false) return deny("Subagents must explicitly set fork_context: false.");
  const expectedTaskClass = ROLE_TASK_CLASSES[input.agent_type];
  if (!expectedTaskClass) return deny("Use one of the approved named agent profiles.");

  const fields = packetFields(input.message);
  if (!fields.brief || !fields.report || !fields.validation || fields.scope.length === 0) {
    return deny("Dispatch packet requires Brief, Scope, Validation, and Report fields.");
  }
  if (fields.taskClass !== expectedTaskClass) {
    return deny(`${input.agent_type} accepts only Task class: ${expectedTaskClass}.`);
  }
  if (input.agent_type === "mechanical-implementer" && fields.scope.length > 2) {
    return deny("Luna mechanical work is limited to one or two named files.");
  }
  return null;
}

function sessionStartOutput(payload) {
  const expectedModel = "gpt-5.6-terra";
  const output = {
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext:
        "You are the Terra project lead and teaching partner. Keep the user approval gate. " +
        "Delegate milestone planning and final review to Sol; delegate bounded work to Terra or Luna.",
    },
  };
  if (payload?.model && payload.model !== expectedModel) {
    output.systemMessage = `Workflow warning: this project expects ${expectedModel} as the main lead model.`;
  }
  return output;
}

function run() {
  const mode = process.argv[2];
  const payload = JSON.parse(fs.readFileSync(0, "utf8"));
  const output = mode === "session-start" ? sessionStartOutput(payload) : validateAgentDispatch(payload);
  if (output) process.stdout.write(`${JSON.stringify(output)}\\n`);
}

if (require.main === module) run();

module.exports = { packetFields, sessionStartOutput, validateAgentDispatch };
```

- [ ] **Step 4: Run focused policy tests**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs
```

Expected: PASS with seven tests.

- [ ] **Step 5: Verify the stdin entrypoint directly**

Run:

```powershell
'{"tool_name":"Agent","tool_input":{"agent_type":"implementer","fork_context":true,"message":""}}' | node .codex/hooks/agent-policy.cjs agent-dispatch
```

Expected: JSON with `permissionDecision` set to `deny` and a reason mentioning `fork_context`.

### Task 2: Register and Test the Project-Local Hooks

**Files:**

- Create: `.codex/hooks.json`
- Modify: `tests/codex_hooks/agent-policy.test.cjs`
- Create: `tests/codex_hooks/workflow-contract.test.cjs`

**Interfaces:**

- Consumes: `.codex/hooks/agent-policy.cjs` from Task 1.
- Produces: Hook registration for `SessionStart` and `PreToolUse` events, with a Windows command override that resolves the repository root before running Node.

- [ ] **Step 1: Write failing hook-configuration checks**

Create `tests/codex_hooks/workflow-contract.test.cjs`:

```js
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..", "..");
const read = (...parts) => fs.readFileSync(path.join(root, ...parts), "utf8");

test("hooks register session guidance and Agent dispatch validation", () => {
  const hooks = JSON.parse(read(".codex", "hooks.json"));
  assert.equal(hooks.hooks.SessionStart[0].matcher, "startup|resume|compact");
  assert.equal(hooks.hooks.PreToolUse[0].matcher, "^Agent$");
  assert.match(hooks.hooks.PreToolUse[0].hooks[0].command, /agent-policy\.cjs agent-dispatch/);
  assert.match(hooks.hooks.PreToolUse[0].hooks[0].commandWindows, /agent-policy\.cjs/);
});
```

- [ ] **Step 2: Run the hook-configuration check to verify it fails**

Run:

```powershell
node --test tests/codex_hooks/workflow-contract.test.cjs
```

Expected: FAIL because `.codex/hooks.json` does not exist.

- [ ] **Step 3: Add hook registration**

Create `.codex/hooks.json` with this exact structure. Keep context short because it is injected on every startup, resume, and compaction.

```json
{
  "description": "Terra-led subagent workflow guardrails.",
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|compact",
        "hooks": [
          {
            "type": "command",
            "command": "node \"$(git rev-parse --show-toplevel)/.codex/hooks/agent-policy.cjs\" session-start",
            "commandWindows": "powershell -NoProfile -Command \"$root = git rev-parse --show-toplevel; & node (Join-Path $root '.codex\\hooks\\agent-policy.cjs') session-start\"",
            "statusMessage": "Loading Terra workflow guidance",
            "additionalContextLimit": 300
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "^Agent$",
        "hooks": [
          {
            "type": "command",
            "command": "node \"$(git rev-parse --show-toplevel)/.codex/hooks/agent-policy.cjs\" agent-dispatch",
            "commandWindows": "powershell -NoProfile -Command \"$root = git rev-parse --show-toplevel; & node (Join-Path $root '.codex\\hooks\\agent-policy.cjs') agent-dispatch\"",
            "statusMessage": "Checking subagent packet",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 4: Expand contract checks for local paths**

Add this test to `tests/codex_hooks/workflow-contract.test.cjs`:

```js
test("hook module and referenced script path exist", () => {
  assert.equal(fs.existsSync(path.join(root, ".codex", "hooks", "agent-policy.cjs")), true);
  const hooks = JSON.parse(read(".codex", "hooks.json"));
  for (const event of ["SessionStart", "PreToolUse"]) {
    assert.match(hooks.hooks[event][0].hooks[0].command, /\.codex\/hooks\/agent-policy\.cjs/);
  }
});
```

- [ ] **Step 5: Run all Node hook tests**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
```

Expected: PASS.

### Task 3: Update Routing Configuration, Profiles, and Startup Workflow

**Files:**

- Modify: `.codex/config.toml`
- Modify: `.codex/agents/planner.toml`
- Modify: `.codex/agents/explorer.toml`
- Modify: `.codex/agents/implementer.toml`
- Modify: `.codex/agents/mechanical-implementer.toml`
- Modify: `.codex/agents/reviewer.toml`
- Modify: `.codex/prompts/new_task.md`
- Modify: `tests/codex_hooks/workflow-contract.test.cjs`

**Interfaces:**

- Consumes: The packet fields enforced by Task 1.
- Produces: A Terra-led default configuration and instructions that make every allowed profile emit a packet the hook accepts.

- [ ] **Step 1: Write failing static routing checks**

Append these tests to `tests/codex_hooks/workflow-contract.test.cjs`:

```js
test("project configuration uses Terra and enables hooks and multi-agent tools", () => {
  const config = read(".codex", "config.toml");
  assert.match(config, /^model = "gpt-5\.6-terra"$/m);
  assert.match(config, /^multi_agent = true$/m);
  assert.match(config, /^hooks = true$/m);
  assert.match(config, /^max_concurrent_threads_per_session = 2$/m);
});

test("agent profiles and startup prompt use the approved role mapping", () => {
  assert.match(read(".codex", "agents", "planner.toml"), /model = "gpt-5\.6-sol"/);
  assert.match(read(".codex", "agents", "reviewer.toml"), /model = "gpt-5\.6-sol"/);
  assert.match(read(".codex", "agents", "implementer.toml"), /model = "gpt-5\.6-terra"/);
  assert.match(read(".codex", "agents", "mechanical-implementer.toml"), /model = "gpt-5\.6-luna"/);
  assert.match(read(".codex", "prompts", "new_task.md"), /Role: GPT-5\.6 Terra project lead/);
  assert.match(read(".codex", "prompts", "new_task.md"), /Task class:/);
  assert.match(read(".codex", "prompts", "new_task.md"), /fork_context: false/);
});
```

- [ ] **Step 2: Run static routing checks to verify they fail**

Run:

```powershell
node --test tests/codex_hooks/workflow-contract.test.cjs
```

Expected: FAIL because `.codex/config.toml` still selects GPT-5.5 and does not enable hooks or `multi_agent`.

- [ ] **Step 3: Update `.codex/config.toml`**

Replace its routing section with:

```toml
model = "gpt-5.6-terra"
model_reasoning_effort = "medium"

[features]
multi_agent = true
hooks = true

[agents]
enabled = true
max_concurrent_threads_per_session = 2
default_subagent_model = "gpt-5.6-luna"
default_subagent_reasoning_effort = "low"
```

- [ ] **Step 4: Make agent-profile boundaries explicit**

Update profile descriptions and `developer_instructions` so the packet class matches the hook mapping:

```text
planner -> Task class: planning; read-only; do not implement.
reviewer -> Task class: review; read-only; final milestone review only.
implementer -> Task class: implementation; bounded multi-file work and fix rounds.
explorer -> Task class: exploration; read-only, targeted file/interface investigation.
mechanical-implementer -> Task class: mechanical; one or two named files only.
```

For every profile, require the standard packet fields: `Project`, `Task class`, `Brief`, `Scope`, `Validation`, `Report`, and `Return`. Keep the existing “do not commit or push” restriction for writable workers.

- [ ] **Step 5: Rewrite the new-task workflow around Terra**

In `.codex/prompts/new_task.md`:

1. Name Terra as the persistent project lead and teacher.
2. Keep Sol as Phase 1 planner and final reviewer only.
3. Route multi-file implementation and integration debugging to Terra.
4. Route targeted exploration and one- or two-file mechanical work to Luna.
5. Keep Git, broad validation, privacy checks, and user-facing explanation in Terra's lead role.
6. Preserve the explicit user approval gate.
7. Add this reusable dispatch template:

```text
model: gpt-5.6-terra
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

- [ ] **Step 6: Run all workflow contract checks and parse TOML**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run python -c "import pathlib, tomllib; tomllib.loads(pathlib.Path('.codex/config.toml').read_text(encoding='utf-8'))"
```

Expected: both commands exit 0.

### Task 4: Verify the Hook in Codex and Perform the Privacy Gate

**Files:**

- Modify only if a validation defect requires it: files from Tasks 1-3.

**Interfaces:**

- Consumes: Completed configuration, hook JSON, policy module, and tests from Tasks 1-3.
- Produces: A trusted project-local hook that surfaces Terra guidance and blocks a malformed subagent spawn before execution.

- [ ] **Step 1: Run focused and project-wide automated validation**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
```

Expected: all Node tests, Python tests, and Ruff checks pass.

- [ ] **Step 2: Validate hook JSON and Windows command shape**

Run:

```powershell
Get-Content -Raw .codex/hooks.json | ConvertFrom-Json | Out-Null
Get-Content -Raw .codex/hooks.json | Select-String -Pattern 'commandWindows|agent-policy.cjs|startup\|resume\|compact|\^Agent\$'
```

Expected: JSON parsing succeeds and all four expected hook markers are present.

- [ ] **Step 3: Trust and observe the project hook**

Start or resume a Codex session in this project. Open `/hooks`, review the project-local `.codex/hooks.json`, and trust its current hash. Confirm that a startup or resume surfaces the concise Terra-led workflow context.

Expected: Codex reports the hook as trusted and the main thread receives the Terra lead reminder.

- [ ] **Step 4: Confirm a malformed dispatch is blocked**

Attempt a disposable `Agent` dispatch using a valid named profile but omit `fork_context: false` or omit `Brief:`. Do not request code changes from that worker.

Expected: the `PreToolUse` hook rejects the dispatch before the subagent starts and explains the missing field.

- [ ] **Step 5: Run the privacy and scope review**

Run:

```powershell
git diff --check
git status --short
git diff -- .codex docs tests/codex_hooks
```

Expected: only workflow, hook, test, and documentation files are changed; no private corpus content or unrelated RAG implementation file is added by this work.

- [ ] **Step 6: Report results without committing**

Report the changed files, Node test result, Python test result, Ruff result, hook-trust result, malformed-dispatch result, and any remaining limitation. Do not stage, commit, or push unless the user explicitly requests it.

## Plan Self-Review

- Spec coverage: Task 1 enforces packet shape and role/task-class matching; Task 2 registers `SessionStart` and `PreToolUse`; Task 3 moves the lead to Terra and preserves Sol/Luna boundaries; Task 4 verifies hook behavior, privacy, and project validation.
- Placeholder scan: this plan contains no deferred implementation markers; every task names files, commands, and expected behavior.
- Interface consistency: the `Task class` values are defined once in Task 1 and used unchanged in Tasks 2 and 3.
