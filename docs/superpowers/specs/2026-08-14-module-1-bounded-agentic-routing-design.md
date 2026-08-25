# Module 1 Bounded Agentic Routing Design

## Purpose

Demonstrate the Module 1 agentic-RAG concept with a local Granite/Ollama decision loop while preserving the existing claim-scope, citation, deterministic-calculation, and privacy boundaries.

## Current State

The application currently makes deterministic route choices: explicit `Product dose:` requests use the calculator, and other questions go through `BasicRagService`. The RAG service safely applies claim-scope abstention before retrieval and generation, but the LLM does not select a tool.

Ollama's native chat API supports tool schemas, assistant tool calls, and a follow-up message containing a tool result. The local client currently supports only a one-turn structured-answer call. The local Ollama service was unreachable during design, so model capability must be verified with a later live smoke test rather than assumed.

## Architecture

`AgenticRagService` wraps the existing `BasicRagService` and owns one bounded decision turn. It receives the original user question and returns the existing `RagAnswer` type.

1. Apply `classify_claim_scope(question)` before any agent, tool, retriever, or generator action. An abstention returns the existing scope-limited response.
2. When enabled, ask the local Ollama model to select zero or one tool from the fixed registry.
3. Validate the selected tool name, arguments, and call count in the controller.
4. Execute the approved tool through deterministic code.
5. Return the controller result through the existing safe answer path.

The controller, not the LLM, owns all side effects, input validation, citations, and safety decisions.

## Tool Registry

### `search_knowledge`

Schema: one non-blank `query` string with a bounded length.

The controller sends the validated query to the existing retriever, then generates an answer to the original user question using the existing prompt builder, formatter, source selection, and semantic-grounding boundary. The search query can improve retrieval wording, but it cannot replace the user question in the answer contract.

### `calculate_product_dose`

Schema: `water_bbl_per_day` and `product_ppm` numeric inputs.

The controller validates these values with the existing product-ppm water-basis calculator and returns its existing deterministic response. It does not ask the LLM to restate or alter the arithmetic. This preserves the distinction between a general calculation and a field-ready prescription.

### Excluded Tools

The water-analysis helper remains a sidebar-only starter utility. It has no citation-backed engineering-analysis contract, so it is not registered for LLM selection.

## Bounds And Fallbacks

- Tool-call limit: one per user request.
- Tool registry: exactly the two named tools.
- Unknown tool, malformed arguments, extra calls, or planner transport failure: execute no tool and use the existing RAG route safely.
- A closed claim scope executes no agent, tool, retrieval, or generation call.
- Agentic routing is disabled unless `AGENTIC_ROUTING_ENABLED=true`; the existing deterministic route remains available while local Granite tool calling is unverified.
- No request, source text, tool arguments, raw tool response, model thought, local path, or credential is persisted in a report or monitoring payload.

## Client Contract

Extend the local `OllamaClient` with a structured tool-call operation that sends a `tools` array to `/api/chat`, parses an assistant message with optional `tool_calls`, and sends the single tool result as a follow-up message only for a search action. The final search answer remains a schema-validated `RagDraft` and passes through the existing formatter.

The client must return sanitized errors. It must not expose provider bodies or raw model output in application messages.

## Validation

Public tests prove:

- a closed scope makes zero planner, tool, retriever, and generator calls;
- a valid search call uses its query for retrieval and the original question for answer generation;
- a valid calculator call returns the existing deterministic result;
- malformed, unknown, and multi-call outputs execute no tool and safely use the RAG fallback;
- existing explicit calculator and standard RAG paths remain compatible;
- tool messages and client errors do not reveal sensitive data.

The live Granite smoke test is an explicit later gate: it must verify local service reachability and one valid native tool-call exchange without recording prompt or response content. It is not required to unit-test the controller.

## Non-Goals

This is not a multi-step autonomous agent, field-treatment recommender, water-chemistry engine, persistent telemetry feature, or proof of local-model tool-call support. It is a one-decision, controller-bounded demonstration of the Module 1 concept.
