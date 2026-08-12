const test = require("node:test");
const assert = require("node:assert/strict");
const {
  sessionStartOutput,
  validateAgentDispatch,
} = require("../../.codex/hooks/agent-policy.cjs");

const packet = (agentType, taskClass, scopeLines, {
  project = "Oilfield Chemical Troubleshooting Copilot.",
  returnField = "status, changed files, validation, and concerns.",
} = {}) => ({
  tool_name: "Agent",
  tool_input: {
    agent_type: agentType,
    fork_context: false,
    message: [
      `Project: ${project}`,
      `Task class: ${taskClass}`,
      "Brief: .codex/briefs/task-1.md",
      "Scope:",
      ...scopeLines.map((path) => `- ${path}`),
      "Validation: node --test tests/codex_hooks/agent-policy.test.cjs",
      "Report: .codex/reports/task-1.md",
      `Return: ${returnField}`,
    ].join("\n"),
  },
});

function assertDenied(result, reason) {
  assert.deepEqual(
    {
      hookEventName: result?.hookSpecificOutput?.hookEventName,
      permissionDecision: result?.hookSpecificOutput?.permissionDecision,
    },
    {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
    },
  );
  assert.match(result.hookSpecificOutput.permissionDecisionReason, reason);
}

test("allows a valid Terra implementation packet", () => {
  assert.equal(
    validateAgentDispatch(packet("implementer", "implementation", [
      "src/rag/provider.py",
      "tests/rag/test_provider.py",
    ])),
    null,
  );
});

test("rejects inherited fork_context", () => {
  const payload = packet("implementer", "implementation", ["src/rag/provider.py"]);
  delete payload.tool_input.fork_context;
  assertDenied(validateAgentDispatch(payload), /fork_context/i);
});

test("rejects true fork_context", () => {
  const payload = packet("implementer", "implementation", ["src/rag/provider.py"]);
  payload.tool_input.fork_context = true;
  assertDenied(validateAgentDispatch(payload), /fork_context/i);
});

test("rejects a planner packet marked as implementation", () => {
  const result = validateAgentDispatch(packet("planner", "implementation", ["src/rag/provider.py"]));
  assertDenied(result, /planner/i);
});

test("rejects a Luna mechanical packet with more than two files", () => {
  const result = validateAgentDispatch(packet("mechanical-implementer", "mechanical", [
    "a.py", "b.py", "c.py",
  ]));
  assertDenied(result, /one or two/i);
});

test("rejects an undefined agent profile", () => {
  const result = validateAgentDispatch(packet("default", "implementation", ["src/rag/provider.py"]));
  assertDenied(result, /approved named agent profiles/i);
});

test("allows a packet without a stale task brief", () => {
  const payload = packet("implementer", "implementation", ["src/rag/provider.py"]);
  payload.tool_input.message = payload.tool_input.message.replace(
    "Brief: .codex/briefs/task-1.md\n",
    "",
  );
  assert.equal(validateAgentDispatch(payload), null);
});

test("passes through non-Agent and missing payloads", () => {
  assert.equal(validateAgentDispatch({ tool_name: "Bash" }), null);
  assert.equal(validateAgentDispatch(), null);
});


test("rejects a packet without the project name", () => {
  const payload = packet("implementer", "implementation", ["src/rag/provider.py"]);
  payload.tool_input.message = payload.tool_input.message.replace(
    "Project: Oilfield Chemical Troubleshooting Copilot.\n",
    "",
  );
  assertDenied(validateAgentDispatch(payload), /Project/i);
});

test("rejects a packet for another project", () => {
  const result = validateAgentDispatch(packet("implementer", "implementation", [
    "src/rag/provider.py",
  ], { project: "Another project." }));
  assertDenied(result, /Oilfield Chemical Troubleshooting Copilot/i);
});

test("rejects a packet without a return contract", () => {
  const result = validateAgentDispatch(packet("implementer", "implementation", [
    "src/rag/provider.py",
  ], { returnField: "" }));
  assertDenied(result, /Return/i);
});

test("rejects broad Luna mechanical scope entries", () => {
  for (const scope of ["entire repository", "repository-wide refactor", "all source files"]) {
    const result = validateAgentDispatch(packet("mechanical-implementer", "mechanical", [scope]));
    assertDenied(result, /file paths/i);
  }
});

test("rejects a Sol reviewer packet with a broad scope entry", () => {
  const result = validateAgentDispatch(packet("reviewer", "review", ["entire repository"]));
  assertDenied(result, /Sol reviews require named file paths/i);
});

test("rejects a Sol reviewer packet without the structured review return contract", () => {
  const result = validateAgentDispatch(packet("reviewer", "review", ["src/rag/provider.py"]));
  assertDenied(result, /review return contract/i);
});

test("rejects a Sol reviewer packet whose return contract omits REVIEW_INCOMPLETE", () => {
  const result = validateAgentDispatch(packet("reviewer", "review", ["src/rag/provider.py"], {
    returnField: "Critical findings; Important findings; Minor findings; Evidence reviewed; Privacy verdict; Test coverage verdict; Plan/spec conformance; Overall verdict (APPROVE / APPROVE WITH MINOR ISSUES / CHANGES REQUIRED).",
  }));
  assertDenied(result, /REVIEW_INCOMPLETE/i);
});

test("allows a Sol reviewer packet with the structured review return contract", () => {
  const result = validateAgentDispatch(packet("reviewer", "review", ["src/rag/provider.py"], {
    returnField: "Critical findings; Important findings; Minor findings; Evidence reviewed; Privacy verdict; Test coverage verdict; Plan/spec conformance; Overall verdict (APPROVE / APPROVE WITH MINOR ISSUES / CHANGES REQUIRED / REVIEW_INCOMPLETE).",
  }));
  assert.equal(result, null);
});

test("adds Terra-led context and warns on an unexpected main model", () => {
  const output = sessionStartOutput({ model: "gpt-5.5", source: "resume" });
  assert.match(output.hookSpecificOutput.additionalContext, /Terra/i);
  assert.match(output.systemMessage, /gpt-5.6-terra/i);
});

test("adds Terra-led context without a warning for the expected main model", () => {
  const output = sessionStartOutput({ model: "gpt-5.6-terra", source: "startup" });
  assert.match(output.hookSpecificOutput.additionalContext, /Terra/i);
  assert.equal(output.systemMessage, undefined);
});
