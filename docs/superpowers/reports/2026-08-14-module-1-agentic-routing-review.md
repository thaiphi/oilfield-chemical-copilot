# Module 1 Agentic Routing Implementation Review

**Date:** 2026-08-14  
**Status:** Locked after public implementation verification, local Granite capability evidence, and practical teaching review.

## Implemented Boundary

The opt-in `AGENTIC_ROUTING_ENABLED=true` route gives the local model one bounded decision: select zero or one controller-owned tool. The fixed registry contains only:

- `search_knowledge`, which may provide a retrieval query while the original user question remains the answer-generation question.
- `calculate_product_dose`, which accepts explicit water-rate and product-ppm values and returns deterministic calculator output.

Claim-scope abstention is evaluated before planning, tool execution, retrieval, or answer generation. Unknown, malformed, multi-tool, and planner-transport outcomes execute no tool and use the existing RAG route. The model never produces the final calculator result, and the sidebar-only water-analysis helper is not in the chat registry.

## Verification Evidence

- Native Ollama tool-message client, controller service, planner, and opt-in app route: 47 focused tests passed.
- Public Python suite: 518 passed, 2 integration tests skipped.
- Codex workflow-contract suite: 22 passed.
- Ruff and `git diff --check`: passed.

## Local Capability Evidence

The initial availability probe on 2026-08-14 found the local service unavailable. After the host service started, the approved synthetic public smoke test recorded only these aggregate fields: `service_reachable=true`, `model_present=true`, `tool_call_returned=true`, and `status=ok`. No prompt, response, tool arguments, source text, or local path was recorded.

## Teaching Review

This boundary separates an LLM's narrow role from deterministic system controls:

- The LLM selects a permitted action; it does not decide policy, execute arbitrary code, or author dose calculations.
- The controller validates the selected tool arguments before any execution.
- Retrieval can use a focused query, but the final answer is grounded against the user's original question and retrieved citations.
- Safety policy runs first, so a field-ready prescription never reaches the planner, calculator, retriever, or generator.

The practical Module 1 review was completed on 2026-08-14. Module 1 is locked.
