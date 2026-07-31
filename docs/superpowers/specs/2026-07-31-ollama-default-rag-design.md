# Ollama-Default RAG Design

## Goal

Run the Module 1 RAG path locally by default with Ollama, using `granite-embedding:latest` for retrieval and `granite4.1:8b` for answer generation. Preserve OpenAI as an explicit optional provider.

## Scope

- Add provider selection for answer generation and embeddings.
- Add Ollama adapters for `/api/embed` and `/api/chat`.
- Keep the current source-grounded `RagDraft` contract, weak-evidence fallback, and OpenAI adapter.
- Document local and Docker configuration, re-indexing, and validation.

## Non-Goals

- No agentic tool routing, hybrid retrieval, monitoring, evaluations, or UI redesign.
- No OpenAI API key requirement when Ollama is selected.
- No database migration. `granite-embedding:latest` emits 384-dimensional vectors, matching the existing PGVector schema.

## Configuration

Defaults in `.env.example`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=granite4.1:8b
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=granite-embedding:latest
EMBEDDING_DIMENSION=384
```

`LLM_PROVIDER=openai` continues to select `OPENAI_API_KEY` and `OPENAI_MODEL`. The existing deterministic and sentence-transformers embedding providers remain available for tests and offline development.

When the Streamlit app runs directly on the host, `OLLAMA_BASE_URL` uses `http://localhost:11434`. In Docker Compose, the app environment overrides it with `http://host.docker.internal:11434`; Ollama remains a host-managed service and is not added as a Compose container.

## Components And Data Flow

1. An answer-generator factory selects the existing lazy OpenAI client or a lazy Ollama client based on `LLM_PROVIDER`.
2. The Ollama chat client sends the existing system and user prompts to `/api/chat` with streaming disabled and requests JSON output. It parses the response into `RagDraft`, then applies the existing citation validation.
3. The embedding factory selects a new Ollama embedding provider when `EMBEDDING_PROVIDER=ollama`.
4. The embedding provider sends batches to `/api/embed`, validates that every returned vector is 384-dimensional, and labels stored rows with the configured Ollama embedding model.
5. Indexing and query retrieval use the same configured provider and model. The existing `embedding_model` database filter prevents mixed-model vector search.

```text
chunks -> Ollama /api/embed -> PGVector (model-labelled vectors)
question -> same /api/embed -> PGVector retrieval -> Ollama /api/chat -> cited answer
```

## Errors And Safety

- Invalid provider names, missing required OpenAI credentials, unreachable Ollama, malformed JSON, missing output, unexpected embedding counts, and dimension mismatches raise existing configuration or generation errors with no secrets in messages.
- Existing `BasicRagService` converts answer-generation failures to the weak-evidence response.
- Retrieval is not attempted against vectors from a different embedding model.

## Validation

- Unit tests mock Ollama HTTP responses for successful chat generation, embedding batches, malformed responses, transport failures, and vector-dimension mismatches.
- Existing OpenAI client tests remain unchanged and pass when OpenAI is selected.
- Environment and Docker tests verify the correct local versus container URL configuration.
- An opt-in live Ollama smoke test indexes the sample corpus with Granite embeddings, asks a sample troubleshooting question using `granite4.1:8b`, and verifies a cited answer.

## Learning Outcome

This demonstrates the two distinct models in a RAG system. The embedding model converts text and questions into comparable vectors for retrieval. The generation model reads retrieved evidence and writes a response. They may be different models, but stored content and questions must use the same embedding model and vector dimension.
