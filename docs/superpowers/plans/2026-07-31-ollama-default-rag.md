# Ollama-Default RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Ollama the default local Module 1 RAG provider with `granite-embedding:latest` for retrieval and `granite4.1:8b` for cited answer generation, while keeping OpenAI selectable.

**Architecture:** A small shared `OllamaClient` owns JSON HTTP calls to `/api/embed` and `/api/chat`. An Ollama embedding provider and answer adapter consume that client through their existing retrieval and RAG boundaries. Factory functions select local Ollama or the retained OpenAI client from environment configuration.

**Tech Stack:** Python 3.11+, standard-library HTTP JSON transport, Ollama HTTP API, PGVector, Streamlit, pytest, Docker Compose.

## Global Constraints

- Default provider values are `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=granite4.1:8b`, `EMBEDDING_PROVIDER=ollama`, and `OLLAMA_EMBEDDING_MODEL=granite-embedding:latest`.
- `granite-embedding:latest` must produce exactly 384 values for this repository's `vector(384)` storage.
- Keep `LLM_PROVIDER=openai` and the current OpenAI adapter working without behavior changes.
- Do not log API keys, prompts containing private source paths, raw provider responses, or provider URLs containing credentials.
- Local host execution uses `http://localhost:11434`; Docker Compose uses `http://host.docker.internal:11434`.
- Indexing and retrieval must use the same embedding provider model label.
- Do not add agentic tools, hybrid retrieval, monitoring, evaluation changes, or a database migration.

---

## File Structure

- Create `src/oilfield_chemical_copilot/ollama.py`: focused HTTP client with typed `embed` and `chat` methods.
- Create `tests/test_ollama.py`: transport contract, malformed-response, and failure tests.
- Modify `src/oilfield_chemical_copilot/retrieval/embeddings.py`: Ollama embedding settings, provider, and factory selection.
- Modify `ingestion/index_chunks.py`: expose Ollama embedding selection through existing CLI flags.
- Modify `tests/retrieval/test_embeddings.py` and `tests/ingest/test_index_chunks.py`: settings, provider, and CLI defaults.
- Create `src/oilfield_chemical_copilot/rag/ollama_client.py`: converts Ollama chat JSON into a validated `RagDraft`.
- Create `src/oilfield_chemical_copilot/rag/generator_factory.py`: lazy provider selection for OpenAI or Ollama.
- Modify `app/streamlit_app.py`: use the answer-generator factory instead of importing OpenAI directly.
- Create `tests/rag/test_ollama_client.py` and `tests/rag/test_generator_factory.py`: answer contract and provider-selection tests.
- Create `tests/rag/test_ollama_integration.py`: opt-in live sample-corpus smoke test.
- Modify `.env.example`, `docker-compose.yml`, `README.md`, and `tests/app/test_streamlit_app.py`: documented settings, Docker host routing, and app construction coverage.

### Task 1: Shared Ollama HTTP Client

**Files:**
- Create: `src/oilfield_chemical_copilot/ollama.py`
- Test: `tests/test_ollama.py`

**Interfaces:**
- Consumes: `base_url: str`, model names, strings, `Sequence[str]`, and standard-library `urllib.request`.
- Produces: `OllamaClient.embed(model: str, texts: Sequence[str]) -> list[list[float]]` and `OllamaClient.chat(model: str, system_prompt: str, user_prompt: str) -> str`.
- Produces: `OllamaClientError`, raised for transport errors and invalid Ollama JSON shapes without exposing response bodies.

- [ ] **Step 1: Write the failing transport contract tests**

```python
def test_embed_posts_a_batch_and_returns_float_vectors() -> None:
    client = OllamaClient("http://ollama.test", opener=_fake_opener({"embeddings": [[1, 2.5]]}))

    assert client.embed(model="granite-embedding:latest", texts=["scale"]) == [[1.0, 2.5]]


def test_chat_posts_non_streaming_json_request_and_returns_content() -> None:
    client = OllamaClient("http://ollama.test", opener=_fake_opener({"message": {"content": "{\\\"answer\\\": \\\"ok\\\"}"}}))

    assert client.chat(model="granite4.1:8b", system_prompt="system", user_prompt="user")
```

- [ ] **Step 2: Run the new test file to verify it fails**

Run: `uv run pytest tests/test_ollama.py -v`

Expected: FAIL because `oilfield_chemical_copilot.ollama` does not exist.

- [ ] **Step 3: Implement the minimal JSON client**

```python
class OllamaClientError(Exception):
    pass


class OllamaClient:
    def embed(self, *, model: str, texts: Sequence[str]) -> list[list[float]]:
        payload = self._post("/api/embed", {"model": model, "input": list(texts)})
        return _parse_embeddings(payload)

    def chat(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        payload = self._post(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        return _parse_chat_content(payload)
```

Use an injected `opener` only for tests. `_post` must set `Content-Type: application/json`, apply a finite timeout, and wrap `OSError`, `HTTPError`, JSON decode errors, and unexpected payload shapes as `OllamaClientError`.

- [ ] **Step 4: Extend the tests for malformed payloads and transport failures**

```python
@pytest.mark.parametrize("payload", [{}, {"embeddings": []}, {"message": {}}])
def test_client_rejects_invalid_ollama_payloads(payload) -> None:
    with pytest.raises(OllamaClientError):
        OllamaClient("http://ollama.test", opener=_fake_opener(payload)).embed(
            model="granite-embedding:latest", texts=["scale"]
        )
```

- [ ] **Step 5: Run focused checks**

Run: `uv run pytest tests/test_ollama.py -v`

Expected: PASS.

Run: `uv run ruff check src/oilfield_chemical_copilot/ollama.py tests/test_ollama.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit the task**

```powershell
git add src/oilfield_chemical_copilot/ollama.py tests/test_ollama.py
git commit -m "feat: add Ollama HTTP client"
```

### Task 2: Ollama Embedding Provider And Indexing

**Files:**
- Modify: `src/oilfield_chemical_copilot/retrieval/embeddings.py`
- Modify: `ingestion/index_chunks.py`
- Modify: `tests/retrieval/test_embeddings.py`
- Modify: `tests/ingest/test_index_chunks.py`

**Interfaces:**
- Consumes: `OllamaClient.embed`, `EmbeddingSettings`, `EMBEDDING_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL`, and `EMBEDDING_DIMENSION`.
- Produces: `OllamaEmbeddingProvider(model_name: str, dimension: int, client: OllamaClient)` implementing `EmbeddingProvider`.
- Produces: `EmbeddingSettings.ollama_base_url` and `EmbeddingSettings.ollama_embedding_model`.

- [ ] **Step 1: Write failing provider and configuration tests**

```python
def test_ollama_embedding_provider_uses_model_and_validates_dimension() -> None:
    provider = OllamaEmbeddingProvider(
        model_name="granite-embedding:latest",
        dimension=384,
        client=FakeOllamaClient([[0.0] * 384]),
    )

    assert provider.embed_query("scale") == [0.0] * 384


def test_ollama_embedding_provider_rejects_wrong_dimension() -> None:
    provider = OllamaEmbeddingProvider(
        model_name="granite-embedding:latest",
        dimension=384,
        client=FakeOllamaClient([[0.0] * 383]),
    )

    with pytest.raises(ValueError, match="expected 384, got 383"):
        provider.embed_query("scale")
```

Also assert `EmbeddingSettings.from_env()` and `index_chunks._build_argument_parser()` choose `ollama`, `granite-embedding:latest`, and the configured Ollama URL without changing deterministic defaults when explicitly requested.

- [ ] **Step 2: Run the focused tests to verify failure**

Run: `uv run pytest tests/retrieval/test_embeddings.py tests/ingest/test_index_chunks.py -v`

Expected: FAIL because the Ollama provider and CLI option do not exist.

- [ ] **Step 3: Implement provider selection and batch validation**

```python
if settings.provider == "ollama":
    return OllamaEmbeddingProvider(
        model_name=settings.ollama_embedding_model,
        dimension=settings.dimension,
        client=OllamaClient(settings.ollama_base_url),
    )
```

`embed_documents` must return `[]` for no texts, call `client.embed` once per requested batch, convert values to `float`, and reject any vector whose length differs from `dimension`. The indexing CLI choices must include `ollama`; its `--embedding-model` default must follow `OLLAMA_EMBEDDING_MODEL` when Ollama is selected.

- [ ] **Step 4: Run focused validation**

Run: `uv run pytest tests/retrieval/test_embeddings.py tests/ingest/test_index_chunks.py tests/retrieval/test_vector.py -v`

Expected: PASS, including model-label delegation through `VectorRetriever`.

Run: `uv run ruff check src/oilfield_chemical_copilot/retrieval/embeddings.py ingestion/index_chunks.py tests/retrieval/test_embeddings.py tests/ingest/test_index_chunks.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit the task**

```powershell
git add src/oilfield_chemical_copilot/retrieval/embeddings.py ingestion/index_chunks.py tests/retrieval/test_embeddings.py tests/ingest/test_index_chunks.py
git commit -m "feat: add Ollama embeddings"
```

### Task 3: Ollama Answer Adapter And Provider Factory

**Files:**
- Create: `src/oilfield_chemical_copilot/rag/ollama_client.py`
- Create: `src/oilfield_chemical_copilot/rag/generator_factory.py`
- Modify: `app/streamlit_app.py`
- Create: `tests/rag/test_ollama_client.py`
- Create: `tests/rag/test_generator_factory.py`
- Modify: `tests/app/test_streamlit_app.py`

**Interfaces:**
- Consumes: `OllamaClient.chat`, `RagDraft`, `RagConfigurationError`, `RagGenerationError`, `LLM_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
- Produces: `OllamaAnswerClient.generate(system_prompt: str, user_prompt: str, allowed_source_ids: set[str]) -> RagDraft`.
- Produces: `build_answer_generator() -> AnswerGenerator`, where `AnswerGenerator` defines `generate(system_prompt: str, user_prompt: str, allowed_source_ids: set[str]) -> RagDraft`.

- [ ] **Step 1: Write failing Ollama answer-contract tests**

```python
def test_ollama_adapter_parses_and_validates_cited_draft() -> None:
    adapter = OllamaAnswerClient(
        model="granite4.1:8b",
        client=FakeOllamaClient(_valid_draft_json()),
    )

    draft = adapter.generate(system_prompt="system", user_prompt="user", allowed_source_ids={"Source 1"})

    assert draft.cited_source_ids == ["Source 1"]


def test_ollama_adapter_hides_transport_details() -> None:
    adapter = OllamaAnswerClient(model="granite4.1:8b", client=FailingOllamaClient())

    with pytest.raises(RagGenerationError, match="Ollama answer generation failed"):
        adapter.generate(system_prompt="system", user_prompt="user", allowed_source_ids={"Source 1"})
```

Add factory tests proving `LLM_PROVIDER=ollama` returns a lazy Ollama generator without an OpenAI key, `LLM_PROVIDER=openai` returns the existing lazy OpenAI generator, and any other value raises `RagConfigurationError`.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `uv run pytest tests/rag/test_ollama_client.py tests/rag/test_generator_factory.py tests/app/test_streamlit_app.py -v`

Expected: FAIL because the adapter and factory do not exist.

- [ ] **Step 3: Implement the answer adapter and lazy factory**

```python
class OllamaAnswerClient:
    def generate(self, *, system_prompt: str, user_prompt: str, allowed_source_ids: set[str]) -> RagDraft:
        try:
            output_text = self.client.chat(
                model=self.model, system_prompt=system_prompt, user_prompt=user_prompt
            )
            return _parse_and_validate_draft(output_text, allowed_source_ids)
        except (OllamaClientError, ValueError, json.JSONDecodeError) as error:
            raise RagGenerationError("Ollama answer generation failed") from error
```

`build_answer_generator` must defer provider construction until `generate` is called so the Streamlit UI can start before a question is asked. Replace `LazyOpenAIAnswerClient()` in `_build_rag_service()` with `build_answer_generator()`.

- [ ] **Step 4: Run focused validation**

Run: `uv run pytest tests/rag/test_ollama_client.py tests/rag/test_generator_factory.py tests/rag/test_openai_client.py tests/rag/test_service.py tests/app/test_streamlit_app.py -v`

Expected: PASS. Existing OpenAI tests pass unchanged and Ollama failures become the current safe fallback through `BasicRagService`.

Run: `uv run ruff check src/oilfield_chemical_copilot/rag app/streamlit_app.py tests/rag tests/app/test_streamlit_app.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit the task**

```powershell
git add src/oilfield_chemical_copilot/rag/ollama_client.py src/oilfield_chemical_copilot/rag/generator_factory.py app/streamlit_app.py tests/rag/test_ollama_client.py tests/rag/test_generator_factory.py tests/app/test_streamlit_app.py
git commit -m "feat: add Ollama answer generation"
```

### Task 4: Runtime Configuration, Documentation, And Live Smoke Test

**Files:**
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Create: `tests/rag/test_ollama_integration.py`

**Interfaces:**
- Consumes: the factories from Tasks 2 and 3, `DATABASE_URL`, `TEST_DATABASE_URL`, and `RUN_OLLAMA_INTEGRATION=1`.
- Produces: reproducible host and Docker configuration plus an opt-in verification of the complete local RAG loop.

- [ ] **Step 1: Write the failing configuration and opt-in smoke tests**

```python
@pytest.mark.integration
def test_ollama_indexes_sample_chunks_and_returns_cited_answer(monkeypatch, tmp_path) -> None:
    if os.getenv("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("set RUN_OLLAMA_INTEGRATION=1 to run against local Ollama")

    # Generate sample chunks, embed with granite-embedding, use TEST_DATABASE_URL,
    # then assert the Granite-generated answer includes source evidence.
```

The test must reject a `TEST_DATABASE_URL` whose database name does not end in `_test`, use only generated public sample chunks, and clean up only its own test rows.

- [ ] **Step 2: Run the smoke test without the opt-in flag**

Run: `uv run pytest -m integration tests/rag/test_ollama_integration.py -v`

Expected: SKIPPED with the exact opt-in instruction, not a network call.

- [ ] **Step 3: Add environment, Docker, and README guidance**

```yaml
app:
  environment:
    LLM_PROVIDER: ${LLM_PROVIDER:-ollama}
    OLLAMA_BASE_URL: ${OLLAMA_DOCKER_BASE_URL:-http://host.docker.internal:11434}
    OLLAMA_MODEL: ${OLLAMA_MODEL:-granite4.1:8b}
    OLLAMA_EMBEDDING_MODEL: ${OLLAMA_EMBEDDING_MODEL:-granite-embedding:latest}
```

Document the host commands to start Ollama, parse the sample corpus, re-index it using `EMBEDDING_PROVIDER=ollama`, launch Streamlit, switch optionally to OpenAI, and run the integration smoke test. State that re-indexing is required after changing embedding providers because PGVector search is model-labelled.

- [ ] **Step 4: Run complete automated validation**

Run: `uv run pytest`

Expected: PASS with only opt-in integrations skipped when local services are not enabled.

Run: `uv run ruff check .`

Expected: `All checks passed!`

Run: `docker compose config`

Expected: valid Compose configuration with the container-specific Ollama URL.

- [ ] **Step 5: Run live local verification after services are ready**

Run: `docker compose up -d postgres`

Run: `$env:EMBEDDING_PROVIDER = "ollama"; $env:OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"; uv run python ingestion/index_chunks.py --input data/processed/chunks.jsonl --database-url "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"`

Run: `$env:RUN_OLLAMA_INTEGRATION = "1"; $env:TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot_test"; uv run pytest -m integration tests/rag/test_ollama_integration.py -v`

Expected: sample chunks are indexed with the Granite embedding label and the test receives a non-fallback cited answer.

- [ ] **Step 6: Commit the task**

```powershell
git add .env.example docker-compose.yml README.md tests/rag/test_ollama_integration.py
git commit -m "docs: configure local Ollama RAG"
```

## Final Verification

- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `docker compose config`.
- [ ] Run the opt-in Ollama integration test only with local Ollama and the disposable `_test` database running.
- [ ] Review `git diff --check` and confirm no private corpus paths, prompts, responses, or secrets were added.
