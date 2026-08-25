const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { validateAgentDispatch } = require("../../.codex/hooks/agent-policy.cjs");

const root = path.resolve(__dirname, "..", "..");
const read = (...parts) => fs.readFileSync(path.join(root, ...parts), "utf8");

test("hooks register session guidance and Agent dispatch validation", () => {
  const hooks = JSON.parse(read(".codex", "hooks.json"));
  const sessionStart = hooks.hooks.SessionStart[0].hooks[0];
  const agentDispatch = hooks.hooks.PreToolUse[0].hooks[0];

  assert.equal(hooks.hooks.SessionStart[0].matcher, "startup|resume|compact");
  assert.equal(hooks.hooks.PreToolUse[0].matcher, "^Agent$");
  assert.match(agentDispatch.command, /agent-policy\.cjs"\s+agent-dispatch/);

  assert.match(sessionStart.commandWindows, /^powershell -NoProfile\b/);
  assert.match(sessionStart.commandWindows, /git rev-parse --show-toplevel/);
  assert.match(sessionStart.commandWindows, /\bnode\b/);
  assert.match(sessionStart.commandWindows, /\.codex\\hooks\\agent-policy\.cjs/);
  assert.match(sessionStart.commandWindows, /session-start"$/);

  assert.match(agentDispatch.commandWindows, /^powershell -NoProfile\b/);
  assert.match(agentDispatch.commandWindows, /git rev-parse --show-toplevel/);
  assert.match(agentDispatch.commandWindows, /\bnode\b/);
  assert.match(agentDispatch.commandWindows, /\.codex\\hooks\\agent-policy\.cjs/);
  assert.match(agentDispatch.commandWindows, /agent-dispatch"$/);
});

test("hook module and referenced script path exist", () => {
  assert.equal(fs.existsSync(path.join(root, ".codex", "hooks", "agent-policy.cjs")), true);
  const hooks = JSON.parse(read(".codex", "hooks.json"));
  for (const event of ["SessionStart", "PreToolUse"]) {
    const hook = hooks.hooks[event][0].hooks[0];
    assert.match(hook.command, /\.codex\/hooks\/agent-policy\.cjs/);
    assert.match(hook.commandWindows, /\.codex\\hooks\\agent-policy\.cjs/);
  }
});

test("project configuration uses Terra and enables hooks and multi-agent tools", () => {
  const config = read(".codex", "config.toml");
  assert.match(config, /^model = "gpt-5\.6-terra"$/m);
  assert.match(config, /^multi_agent = true$/m);
  assert.match(config, /^hooks = true$/m);
  assert.match(config, /^max_concurrent_threads_per_session = 2$/m);
});

test("agent profiles and startup prompt enforce the approved routing and dispatch contract", () => {
  const planner = read(".codex", "agents", "planner.toml");
  const explorer = read(".codex", "agents", "explorer.toml");
  const implementer = read(".codex", "agents", "implementer.toml");
  const mechanicalImplementer = read(".codex", "agents", "mechanical-implementer.toml");
  const reviewer = read(".codex", "agents", "reviewer.toml");

  assert.match(planner, /^model = "gpt-5\.6-sol"$/m);
  assert.match(planner, /^sandbox_mode = "read-only"$/m);
  assert.match(planner, /^Task class: planning\. Read-only; do not implement\.$/m);
  assert.match(planner, /^Do not explore the repository\. Plan from the supplied brief and a controller-provided Luna reconnaissance packet only\.$/m);
  assert.match(planner, /^If the reconnaissance packet is missing or insufficient, return `RECON REQUIRED` with one exact Luna assignment: purpose, allowed file paths, questions to answer, and required return fields\. Do not inspect those files yourself\.$/m);
  assert.match(planner, /^The delegation matrix assigns Luna read-only reconnaissance and Terra post-approval implementation, but this role does not spawn, wait for, or coordinate agents\.$/m);

  assert.match(reviewer, /^model = "gpt-5\.6-sol"$/m);
  assert.match(reviewer, /^sandbox_mode = "read-only"$/m);
  assert.match(reviewer, /^Task class: review\. Read-only; final milestone review only; do not implement\.$/m);
  assert.match(reviewer, /^Sol is the review lead: synthesize a final verdict from the assignment and any controller-supplied Luna evidence packets\.$/m);
  assert.match(reviewer, /^Sol is an optional independent second opinion; Terra owns the final operational decision and does not wait on Sol when evidence establishes a Critical or Important finding\.$/m);
  assert.match(reviewer, /^For a small review of one or two tightly related files, review directly; do not delegate automatically\.$/m);
  assert.match(reviewer, /^For medium or large reviews with independent evidence streams, use only controller-supplied bounded Luna evidence packets; do not recursively explore the repository\.$/m);
  assert.match(reviewer, /^When a medium or large review lacks required evidence, return exact bounded Luna assignments for Terra to dispatch\.$/m);
  assert.match(reviewer, /^Reserve the final portion of the review window for judgment and return\. If the window or evidence is insufficient, return `REVIEW_INCOMPLETE` with reviewed evidence, unresolved items, and the reason; do not keep exploring until timeout\.$/m);
  assert.match(reviewer, /^Overall verdict: `APPROVE`, `APPROVE WITH MINOR ISSUES`, `CHANGES REQUIRED`, or `REVIEW_INCOMPLETE`\.$/m);
  assert.match(reviewer, /^Do not edit, commit, or push\.$/m);

  assert.match(implementer, /^model = "gpt-5\.6-terra"$/m);
  assert.match(implementer, /^Task class: implementation\. Handle bounded multi-file work and fix rounds\.$/m);
  assert.match(implementer, /^Start only after the controller supplies an approved Sol task item and any required Luna reconnaissance packet\.$/m);

  assert.match(explorer, /^model = "gpt-5\.6-luna"$/m);
  assert.match(explorer, /^sandbox_mode = "read-only"$/m);
  assert.match(explorer, /^Task class: exploration\. Read-only, targeted file\/interface investigation; do not implement\.$/m);
  assert.match(explorer, /^For review evidence, return a compact packet for Sol: inspected paths, exact line evidence, findings by severity, privacy or regression facts when assigned, and unanswered questions\.$/m);
  assert.match(explorer, /^Do not propose implementation, decide a final review verdict, expand scope, or wait for other agents\.$/m);

  assert.match(mechanicalImplementer, /^model = "gpt-5\.6-luna"$/m);
  assert.match(mechanicalImplementer, /^Task class: mechanical\. Work in one or two named files only\.$/m);

  const prompt = read(".codex", "prompts", "new_task.md");
  assert.match(prompt, /^Role: GPT-5\.6 Terra project lead$/m);
  assert.match(prompt, /^Terra is (?:the )?persistent project lead and teacher\./m);
  assert.match(prompt, /major architecture change, an irreversible or external action, or deployment/i);
  assert.match(prompt, /primary Terra lead owns the roadmap, implementation, fixes, validation, status, and integration/i);
  assert.match(prompt, /subagents only for bounded review, research, or edge-case testing/i);
  assert.match(prompt, /^Small review \(one or two tightly related files\): Sol reviews directly\.$/m);
  assert.match(prompt, /^Medium review \(implementation plus tests\): Terra obtains one or two bounded Luna evidence packets, then Sol synthesizes\.$/m);
  assert.match(prompt, /^Large review \(implementation, tests, privacy, and plan conformance\): Terra obtains independent bounded Luna evidence packets, then Sol synthesizes\.$/m);
  assert.match(prompt, /^When Luna evidence clearly establishes a Critical or Important finding, Terra marks `CHANGES REQUIRED` and proceeds with correction; Sol is optional\.$/m);
  assert.match(prompt, /^When evidence is ambiguous, Terra may request Sol's second opinion; if Sol stalls, Terra makes and records an evidence-based fallback decision\.$/m);
  assert.match(prompt, /`REVIEW_INCOMPLETE`/);

  const dispatch = prompt.match(/## Reusable Dispatch Template\r?\n\r?\n```text\r?\n([\s\S]*?)\r?\n```/);
  assert.ok(dispatch, "startup prompt includes a reusable dispatch template");
  assert.match(dispatch[1], /^model: gpt-5\.6-terra$/m);
  assert.match(dispatch[1], /^agent_type: implementer$/m);
  assert.match(dispatch[1], /^fork_context: false$/m);
  assert.match(dispatch[1], /^message: \|$/m);
  assert.match(dispatch[1], /^  Project: Oilfield Chemical Troubleshooting Copilot\.$/m);
  assert.match(dispatch[1], /^  Task class: implementation$/m);
  assert.match(dispatch[1], /^  Brief: \.codex\/briefs\/task-2\.md$/m);
  assert.match(dispatch[1], /^  Scope:$/m);
  assert.match(dispatch[1], /^  - src\/oilfield_chemical_copilot\/rag\/ollama_client\.py$/m);
  assert.match(dispatch[1], /^  - tests\/rag\/test_ollama_client\.py$/m);
  assert.match(dispatch[1], /^  Validation: uv run pytest tests\/rag\/test_ollama_client\.py -v$/m);
  assert.match(dispatch[1], /^  Report: \.codex\/reports\/task-2\.md$/m);
  assert.match(dispatch[1], /^  Return: status, changed files, validation, and concerns\.$/m);
  const message = dispatch[1]
    .match(/^message: \|\r?\n([\s\S]*)$/m)[1]
    .replace(/^  /gm, "");
  assert.equal(
    validateAgentDispatch({
      tool_name: "Agent",
      tool_input: {
        agent_type: "implementer",
        fork_context: false,
        message,
      },
    }),
    null,
  );
});
