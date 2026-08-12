const fs = require("node:fs");

const PROJECT_NAME = "Oilfield Chemical Troubleshooting Copilot.";
const FILE_LIKE_SCOPE = /^(?:(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)$/;
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
  const project = text.match(/^Project:\s*(.+)$/im)?.[1]?.trim();
  const taskClass = text.match(/^Task class:\s*(.+)$/im)?.[1]?.trim();
  const brief = text.match(/^Brief:\s*(\.codex\/briefs\/[^\s]+\.md)$/im)?.[1];
  const returnField = text.match(/^Return:\s*(\S.+)$/im)?.[1]?.trim();
  const report = text.match(/^Report:\s*(\.codex\/reports\/[^\s]+\.md)$/im)?.[1];
  const validation = text.match(/^Validation:\s*\S.+$/im)?.[0];
  const scopeBlock = text.match(/^Scope:\s*\r?\n((?:\s*-\s+[^\r\n]+\r?\n?)+)/im)?.[1] || "";
  const scope = [...scopeBlock.matchAll(/^\s*-\s+(.+)$/gm)].map((match) => match[1].trim());
  return { project, taskClass, brief, report, validation, returnField, scope };
}

function validateAgentDispatch(payload) {
  if (payload?.tool_name !== "Agent") return null;
  const input = payload.tool_input || {};
  if (input.fork_context !== false) return deny("Subagents must explicitly set fork_context: false.");
  const expectedTaskClass = ROLE_TASK_CLASSES[input.agent_type];
  if (!expectedTaskClass) return deny("Use one of the approved named agent profiles.");

  const fields = packetFields(input.message);
  if (fields.scope.length === 0) {
    return deny("Dispatch packet requires a named Scope.");
  }
  if (fields.project !== PROJECT_NAME) {
    return deny(`Dispatch packet requires Project: ${PROJECT_NAME}`);
  }
  if (!fields.returnField) {
    return deny("Dispatch packet requires a non-empty Return field.");
  }
  if (fields.taskClass !== expectedTaskClass) {
    return deny(`${input.agent_type} accepts only Task class: ${expectedTaskClass}.`);
  }
  if (input.agent_type === "mechanical-implementer" && fields.scope.length > 2) {
    return deny("Luna mechanical work is limited to one or two named files.");
  }
  if (input.agent_type === "mechanical-implementer" && fields.scope.some((entry) => !FILE_LIKE_SCOPE.test(entry))) {
    return deny("Luna mechanical work requires one or two named file paths.");
  }
  return null;
}

function sessionStartOutput(payload) {
  const expectedModel = "gpt-5.6-terra";
  const output = {
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext:
        "You are the Terra project lead and teaching partner. Own workflow state in this workspace, " +
        "continue through ordinary fixes and tests, and use subagents only for bounded review or research.",
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
  if (output) process.stdout.write(`${JSON.stringify(output)}\n`);
}

if (require.main === module) run();

module.exports = { packetFields, sessionStartOutput, validateAgentDispatch };
