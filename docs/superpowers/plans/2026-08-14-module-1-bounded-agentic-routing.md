# Module 1 Bounded Agentic Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, one-decision Ollama tool-selection loop that can choose safe knowledge search or deterministic product-dose calculation without weakening existing safety controls.

**Architecture:** `AgenticRagService` applies claim-scope abstention before asking a planner to choose at most one fixed tool. The controller validates and executes that tool: search delegates to the existing grounded RAG answer path using the original user question, while product dose returns deterministic calculator output. Native Ollama tool messages are isolated in the local HTTP client; Streamlit enables this route only with an explicit environment flag.

**Tech Stack:** Python 3.11 standard library, local Ollama `/api/chat`, existing `BasicRagService`, claim-scope policy, product-dose calculator, pytest, Ruff, Streamlit.

## Global Constraints

- Claim-scope abstention occurs before all planner, tool, retrieval, and generator calls.
- The registry contains only `search_knowledge` and `calculate_product_dose`; the water-analysis helper remains sidebar-only.
- A request permits at most one tool call; unknown, malformed, or multi-call selections execute no tool.
- Search answers use the original user question for prompt construction and citations; only retrieval uses the agent-selected query.
- Product-dose results stay deterministic and are never restated by the model.
- Agentic routing is opt-in through `AGENTIC_ROUTING_ENABLED=true`; the current explicit routes remain available when disabled or when planning fails.
- No prompt, source text, tool arguments, model output, thought text, raw error, credential, or local path enters monitoring or public reports.
- The live Granite smoke test is not run until local Ollama becomes reachable.

---

### Task 1: Add Native Ollama Tool-Message Support

**Files:**
- Modify: `src/oilfield_chemical_copilot/ollama.py`
- Modify: `tests/test_ollama.py`

**Interfaces:**
- Produces `OllamaToolCall(name: str, arguments: dict[str, object])` and `OllamaToolResponse(content: str, tool_calls: tuple[OllamaToolCall, ...])`.
- Produces `OllamaClient.chat_with_tools(*, model: str, messages: list[dict[str, object]], tools: list[dict[str, object]], generation_options: dict[str, object] | None = None) -> OllamaToolResponse`.

- [x] **Step 1: Write failing client tests**

Add a fake `/api/chat` response containing one assistant tool call and assert the client returns its validated name and arguments. Assert the outgoing payload includes `model`, `stream: false`, the supplied `messages`, and the supplied `tools` schema. Add malformed payload cases for a non-string tool name, non-object arguments, and non-list `tool_calls`; each raises `OllamaClientError` without provider text.

```python
response = client.chat_with_tools(
    model="granite4.1:8b",
    messages=[{"role": "user", "content": "Find scale evidence."}],
    tools=[SEARCH_TOOL_SCHEMA],
)
assert response.tool_calls == (OllamaToolCall("search_knowledge", {"query": "scale evidence"}),)
```

- [x] **Step 2: Verify the new tests fail**

```powershell
uv run pytest tests/test_ollama.py -q
```

Expected: failure because `chat_with_tools`, `OllamaToolCall`, and `OllamaToolResponse` do not exist.

- [x] **Step 3: Implement safe request and response parsing**

Create the two frozen dataclasses. Reuse `_post("/api/chat", payload)` and add a parser that accepts only a mapping with a mapping `message`, a string `content` (empty allowed), and an absent or list `tool_calls`. Each call must have a mapping `function`, a non-blank string name, and mapping arguments. Raise only `OllamaClientError("Invalid Ollama tool response")` for invalid data.

- [x] **Step 4: Run focused client validation**

```powershell
uv run pytest tests/test_ollama.py -q
uv run ruff check src/oilfield_chemical_copilot/ollama.py tests/test_ollama.py
```

Expected: all client tests and Ruff pass.

### Task 2: Preserve Retrieval Query And Deterministic Dose Presentation

**Files:**
- Modify: `src/oilfield_chemical_copilot/rag/service.py`
- Modify: `src/oilfield_chemical_copilot/tools/chemical_dosage.py`
- Modify: `app/streamlit_app.py`
- Modify: `tests/rag/test_service.py`
- Modify: `tests/tools/test_chemical_dosage.py`
- Modify: `tests/app/test_streamlit_app.py`

**Interfaces:**
- Extends `BasicRagService.answer(question: str, topic: str | None = None, retrieval_query: str | None = None) -> RagAnswer`.
- Produces `product_dosage_answer(water_bbl_per_day: float, product_ppm: float) -> RagAnswer` in `chemical_dosage.py`.

- [x] **Step 1: Write failing behavior tests**

Add a service test proving `retrieval_query="scale deposits"` is passed to `retriever.retrieve()` while `generator.generate()` receives a prompt containing the original question. Add calculator presentation tests proving `product_dosage_answer(1000, 100)` contains `4.2 gallons/day`, has no sources, and uses the existing non-prescription label. Update the Streamlit sidebar/chat test to use this shared function.

- [x] **Step 2: Verify the focused tests fail**

```powershell
uv run pytest tests/rag/test_service.py tests/tools/test_chemical_dosage.py tests/app/test_streamlit_app.py -q
```

Expected: failures because `retrieval_query` and `product_dosage_answer` do not exist.

- [x] **Step 3: Implement the smallest shared boundaries**

In `BasicRagService.answer`, use `retrieval_query or question` only for `retriever.retrieve()` and keep `question` in `build_prompt()`. Move the existing `_dosage_answer` text construction into `product_dosage_answer()` beside `calculate_dosage()`. Make Streamlit import and call the shared function; do not change its sidebar behavior.

- [x] **Step 4: Run focused boundary validation**

```powershell
uv run pytest tests/rag/test_service.py tests/tools/test_chemical_dosage.py tests/app/test_streamlit_app.py -q
uv run ruff check src/oilfield_chemical_copilot/rag/service.py src/oilfield_chemical_copilot/tools/chemical_dosage.py app/streamlit_app.py tests/rag/test_service.py tests/tools/test_chemical_dosage.py tests/app/test_streamlit_app.py
```

Expected: focused tests and Ruff pass.

### Task 3: Implement The Controller-Owned Agentic Service

**Files:**
- Create: `src/oilfield_chemical_copilot/rag/agentic_service.py`
- Create: `tests/rag/test_agentic_service.py`

**Interfaces:**
- Produces `ToolSelection(name: str, arguments: dict[str, object])`.
- Produces `AgenticRagService(rag_service, planner, calculator=calculate_dosage)` with `answer(question: str, topic: str | None = None) -> RagAnswer`.
- Consumes a planner protocol: `select_tool(question: str) -> tuple[ToolSelection, ...]`.

- [x] **Step 1: Write failing agent-service tests**

Use fakes that record calls. Cover these exact scenarios:

```python
def test_closed_scope_skips_planner_tool_and_rag() -> None: ...
def test_search_uses_selected_query_and_original_question() -> None: ...
def test_valid_dose_selection_returns_deterministic_result() -> None: ...
def test_unknown_malformed_or_multiple_selection_executes_no_tool_and_falls_back_to_rag() -> None: ...
```

The closed-scope fake planner, calculator, and RAG service must each fail the test if called. The search fake must assert `rag_service.answer(original_question, retrieval_query=selected_query)`. The calculator test must assert both validated numeric arguments and the non-prescription response label.

- [x] **Step 2: Verify the agent-service tests fail**

```powershell
uv run pytest tests/rag/test_agentic_service.py -q
```

Expected: import failure because `agentic_service.py` does not exist.

- [x] **Step 3: Implement bounded selection and validation**

Define exactly two tool names and JSON-schema constants. `search_knowledge` accepts only one stripped string query of 1 through 500 characters. `calculate_product_dose` accepts exactly two non-boolean finite numeric fields and calls the existing calculator validation. For zero, unknown, malformed, or more than one selection, or an `OllamaClientError` from the planner, call `rag_service.answer(question, topic=topic)` and do not invoke a tool. For an allowed claim scope, the search branch calls `rag_service.answer(question, topic=topic, retrieval_query=query)`; the dose branch calls `product_dosage_answer()`.

- [x] **Step 4: Run focused agent-service validation**

```powershell
uv run pytest tests/rag/test_agentic_service.py tests/rag/test_service.py tests/tools/test_chemical_dosage.py -q
uv run ruff check src/oilfield_chemical_copilot/rag/agentic_service.py tests/rag/test_agentic_service.py
```

Expected: all agent, RAG, and calculator tests pass.

### Task 4: Add The Ollama Planner And Feature-Flagged App Route

**Files:**
- Modify: `src/oilfield_chemical_copilot/rag/agentic_service.py`
- Modify: `app/streamlit_app.py`
- Modify: `tests/rag/test_agentic_service.py`
- Modify: `tests/app/test_streamlit_app.py`

**Interfaces:**
- Produces `OllamaToolPlanner(model: str, client: OllamaClient)` implementing `select_tool(question: str) -> tuple[ToolSelection, ...]`.
- Adds `AGENTIC_ROUTING_ENABLED` parsing in Streamlit; only case-insensitive `"true"` enables agentic routing.

- [x] **Step 1: Write failing planner and app-route tests**

Add a fake `OllamaClient` that returns one `OllamaToolResponse` and assert `OllamaToolPlanner` sends the fixed two-tool registry with a system instruction that forbids unlisted tools and more than one call. Add app tests proving the flag disabled uses current `_route_prompt_with_outcome` behavior, the flag enabled uses `AgenticRagService`, and a planner transport failure returns the existing safe RAG route without exposing provider text.

- [x] **Step 2: Verify the focused tests fail**

```powershell
uv run pytest tests/rag/test_agentic_service.py tests/app/test_streamlit_app.py -q
```

Expected: failures because `OllamaToolPlanner` and feature-flagged route construction do not exist.

- [x] **Step 3: Implement the local planner and opt-in integration**

`OllamaToolPlanner` calls `chat_with_tools()` once with `temperature: 0`, the fixed registry, and the original question. It maps the returned calls to `ToolSelection` without executing them. In Streamlit, parse `AGENTIC_ROUTING_ENABLED`; when enabled, build `AgenticRagService` around the existing RAG service and `OllamaToolPlanner(model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL), client=OllamaClient(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)))`. When disabled, retain the current deterministic route; planner transport failure is handled inside `AgenticRagService`. Do not add a water-analysis chat tool or new raw-content monitoring fields.

- [x] **Step 4: Run app and planner validation**

```powershell
uv run pytest tests/test_ollama.py tests/rag/test_agentic_service.py tests/app/test_streamlit_app.py -q
uv run ruff check src/oilfield_chemical_copilot/ollama.py src/oilfield_chemical_copilot/rag/agentic_service.py app/streamlit_app.py tests/test_ollama.py tests/rag/test_agentic_service.py tests/app/test_streamlit_app.py
```

Expected: focused client, agent, and app tests pass.

### Task 5: Verify, Teach, And Validate Local Granite

**Files:**
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `docs/LEARNING_ROADMAP.md`
- Create: `docs/superpowers/reports/2026-08-14-module-1-agentic-routing-review.md`

**Interfaces:**
- Consumes: public unit-test evidence and a locally reachable Ollama service.
- Produces: a Module 1 teaching review that distinguishes deterministic controls from the LLM’s bounded selection role.

- [x] **Step 1: Run full public verification**

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
git diff --check
```

- [x] **Step 2: Run a local, content-free Granite capability smoke test**

Only after `http://localhost:11434/api/tags` confirms that `granite4.1:8b` is installed, send one synthetic public tool-selection prompt through `OllamaToolPlanner`. Record only `service_reachable`, `model_present`, `tool_call_returned`, and sanitized error status. Do not save the prompt, response, tool arguments, source content, local paths, or model thinking.

- [x] **Step 3: Update the Module 1 teaching review and status**

Record the routing flow, tool bounds, test counts, smoke-test aggregate status, and current limitation that this is a one-decision loop. Mark Module 1 as ready for its practical teaching review only if the public suite and live smoke test pass; do not lock Module 1 until the user completes that review.

- [ ] **Step 4: Request commit approval only after evidence review**

Stage only public code, tests, teaching report, and status documents after explicit user request. Never stage `.private/`.

## Plan Self-Review

- Coverage: native tool messages, one-call controller boundary, safety-first abstention, retrieval-query preservation, deterministic dose output, opt-in configuration, negative paths, no water-analysis agent tool, local capability evidence, teaching review, and Git privacy each have an explicit task.
- Scope: no persistent monitoring, autonomous loop, live retrieval evaluation, water-analysis expansion, or field-treatment recommendation is introduced.
- Type consistency: `OllamaToolCall` feeds `OllamaToolResponse`, which the planner maps to `ToolSelection`; the service consumes selections and returns the existing `RagAnswer` type.
