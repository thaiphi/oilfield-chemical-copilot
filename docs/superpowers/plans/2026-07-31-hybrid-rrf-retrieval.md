# Hybrid RRF Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine exact keyword retrieval and semantic PGVector retrieval with Reciprocal Rank Fusion (RRF), while retaining a selectable vector-only comparison path and source-level retrieval provenance.

**Architecture:** A pure RRF function fuses two ranked `RetrievalHit` lists by `chunk_id`; it does not compare keyword and vector scores directly. A hybrid pipeline obtains candidates independently from the existing `KeywordSearchIndex` and `VectorRetriever`, fuses the ranks, filters by a hybrid RRF threshold, and applies the existing context budget. The RAG service receives the correct evidence threshold for the selected mode, then preserves the existing bounded-prompt, structured-answer, and safe-fallback flow.

**Tech Stack:** Python 3.11, `minsearch`, PostgreSQL/PGVector, Ollama, Streamlit, pytest, ruff.

## Global Constraints

- Keep the default local RAG providers unchanged: Ollama `granite4.1:8b` for generation and `granite-embedding:latest` for embeddings.
- Add no database migration and no dependency; build the keyword index from rows already stored in PGVector.
- `RETRIEVAL_MODE` accepts exactly `hybrid` (default) and `vector`; vector mode must retain the current ranking and `RAG_MIN_SCORE` behavior.
- Hybrid ranking is `sum(1 / (rrf_k + rank))`, with one-based ranks and default `HYBRID_RRF_K=60`; it is not a similarity percentage.
- Hybrid mode uses only `HYBRID_MIN_RRF_SCORE`, default `0.015`; `RAG_MIN_SCORE` must not filter a fused RRF result.
- Empty questions must return no hits without embedding or keyword search. Either ranked candidate list may be empty without failing retrieval.
- Keep user-visible source output free of absolute paths. Preserve the existing safe fallback when evidence is weak or generation fails.
- Preserve the Module 1 boundary: no reranker, learned ranker, agentic tool selection, evaluation dataset, monitoring, or provider change.
- Keep tests deterministic; no live Ollama or Docker services are required for unit tests.
- Commit each completed task only after the user explicitly approves the commit.

---

## File Structure

- `src/oilfield_chemical_copilot/retrieval/hybrid.py`: pure, deterministic RRF rank fusion and provenance metadata.
- `src/oilfield_chemical_copilot/retrieval/keyword.py`: builds a `minsearch` index from persisted `RetrievalHit` records as well as parsed chunks.
- `src/oilfield_chemical_copilot/retrieval/pipeline.py`: validated hybrid settings, hybrid pipeline, selected-mode factory, and shared evidence-threshold contract.
- `src/oilfield_chemical_copilot/rag/service.py`: consumes the selected retrieval threshold instead of assuming every score is a vector similarity.
- `src/oilfield_chemical_copilot/rag/models.py` and `src/oilfield_chemical_copilot/rag/prompt_builder.py`: carry retrieval provenance safely from hits to cited evidence.
- `app/streamlit_app.py`: exposes hybrid/vector comparison in the sidebar, caches one service per selected mode, and shows source provenance.
- `.env.example` and `README.md`: document defaults, tuning knobs, the RRF formula, and the Module 1 learning objective.
- `tests/retrieval/test_hybrid.py`, `tests/retrieval/test_keyword.py`, `tests/retrieval/test_pipeline.py`, `tests/rag/test_service.py`, `tests/rag/test_prompt_builder.py`, `tests/rag/test_formatter.py`, and `tests/app/test_streamlit_app.py`: deterministic behavior, compatibility, provenance, and privacy coverage.

### Task 1: Deterministic RRF Fusion and Stored-Chunk Keyword Index

**Files:**
- Create: `tests/retrieval/test_hybrid.py`
- Modify: `src/oilfield_chemical_copilot/retrieval/hybrid.py`
- Modify: `src/oilfield_chemical_copilot/retrieval/keyword.py`
- Modify: `tests/retrieval/test_keyword.py`

**Interfaces:**
- Consumes: `RetrievalHit` from `src/oilfield_chemical_copilot/retrieval/models.py` and existing `KeywordSearchIndex.from_chunks(chunks)`.
- Produces: `fuse_ranked_hits(keyword_hits: list[RetrievalHit], vector_hits: list[RetrievalHit], *, rrf_k: int = 60, limit: int = 5) -> list[RetrievalHit]`.
- Produces: `KeywordSearchIndex.from_hits(hits: list[RetrievalHit]) -> KeywordSearchIndex`.
- Produces: each fused result has `retrieval_method == "hybrid"`; its `metadata` contains `rrf_methods: tuple[str, ...]`, `keyword_rank: int | None`, and `vector_rank: int | None`.

- [ ] **Step 1: Write the failing RRF unit tests**

Create `tests/retrieval/test_hybrid.py`. Use a local `_hit()` helper that creates complete `RetrievalHit` values and verify the exact RRF result for a shared top-ranked chunk:

```python
from pytest import approx

from oilfield_chemical_copilot.retrieval.hybrid import fuse_ranked_hits


def test_fuse_ranked_hits_sums_one_based_ranks_and_keeps_provenance() -> None:
    keyword = [_hit("shared", "keyword"), _hit("keyword-only", "keyword")]
    vector = [_hit("shared", "vector"), _hit("vector-only", "vector")]

    fused = fuse_ranked_hits(keyword, vector, rrf_k=60, limit=5)

    assert [hit.chunk_id for hit in fused] == ["shared", "keyword-only", "vector-only"]
    assert fused[0].score == approx(2 / 61)
    assert fused[0].retrieval_method == "hybrid"
    assert fused[0].metadata == {
        "rrf_methods": ("keyword", "vector"),
        "keyword_rank": 1,
        "vector_rank": 1,
    }
```

Add tests that assert: a chunk present in only one list receives only that list's reciprocal term; empty lists return `[]`; `limit=0` returns `[]`; `rrf_k=0` raises `ValueError` with `rrf_k must be at least 1`; and an equal-score tie is sorted by the best rank, then `chunk_id`.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_hybrid.py -v`

Expected: FAIL because `fuse_ranked_hits` does not exist.

- [ ] **Step 3: Replace the hybrid placeholder with pure fusion**

Replace `hybrid_search` in `src/oilfield_chemical_copilot/retrieval/hybrid.py` with `fuse_ranked_hits`. Record ranks with `enumerate(..., start=1)`, retain the first hit's immutable source fields for each `chunk_id`, and build a new `RetrievalHit` rather than mutating the inputs:

```python
def fuse_ranked_hits(
    keyword_hits: list[RetrievalHit],
    vector_hits: list[RetrievalHit],
    *,
    rrf_k: int = 60,
    limit: int = 5,
) -> list[RetrievalHit]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if limit < 1:
        return []
    # Collect {chunk_id: {"hit": RetrievalHit, "keyword_rank": int | None,
    # "vector_rank": int | None}} then create fused hits sorted by
    # (-rrf_score, best_present_rank, chunk_id).
```

For a chunk present in both lists, set `rrf_methods` to `("keyword", "vector")`; for one list, use a one-item tuple in that fixed order. Do not retain a source `metadata` dictionary in the fused result: use precisely the three RRF provenance keys above so user-visible provenance cannot inherit unexpected private fields.

- [ ] **Step 4: Add persisted-hit construction coverage**

Extend `tests/retrieval/test_keyword.py` with a `_retrieval_hit()` helper and this test:

```python
def test_keyword_index_can_be_built_from_stored_hits() -> None:
    index = KeywordSearchIndex.from_hits([
        _retrieval_hit("scale", "SCALE-X inhibitor compatibility", "scale"),
        _retrieval_hit("corrosion", "oxygen scavenger program", "corrosion"),
    ])

    hits = index.search("SCALE-X", limit=5, topic="scale")

    assert [hit.chunk_id for hit in hits] == ["scale"]
    assert hits[0].retrieval_method == "keyword"
```

- [ ] **Step 5: Implement `from_hits` without changing current chunk behavior**

Add a public class method and a private converter in `src/oilfield_chemical_copilot/retrieval/keyword.py`:

```python
@classmethod
def from_hits(cls, hits: list[RetrievalHit]) -> "KeywordSearchIndex":
    return cls([_document_for_hit(hit) for hit in hits])


def _document_for_hit(hit: RetrievalHit) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk_id,
        "content": hit.text,
        "source_file": hit.source_file,
        "source_path": hit.source_path,
        "topic": hit.topic,
        "parser_type": hit.parser_type,
        "page_or_sheet": hit.page_or_sheet,
        "chunk_index": hit.chunk_index,
        "metadata": dict(hit.metadata),
    }
```

Leave `from_chunks` and `_document_for_chunk` intact. `search()` continues to issue one-based keyword ranks and returns `retrieval_method="keyword"`.

- [ ] **Step 6: Run retrieval unit tests and lint**

Run: `uv run pytest tests/retrieval/test_hybrid.py tests/retrieval/test_keyword.py -v`

Expected: PASS, including exact shared-score, single-source, tie, blank/empty, invalid-`rrf_k`, and stored-hit tests.

Run: `uv run ruff check src/oilfield_chemical_copilot/retrieval/hybrid.py src/oilfield_chemical_copilot/retrieval/keyword.py tests/retrieval/test_hybrid.py tests/retrieval/test_keyword.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit after explicit user approval**

```powershell
git add src/oilfield_chemical_copilot/retrieval/hybrid.py src/oilfield_chemical_copilot/retrieval/keyword.py tests/retrieval/test_hybrid.py tests/retrieval/test_keyword.py
git commit -m "feat: add deterministic hybrid RRF fusion"
```

### Task 2: Mode-Aware Retrieval Pipeline and Correct Evidence Thresholds

**Files:**
- Modify: `src/oilfield_chemical_copilot/retrieval/pipeline.py`
- Modify: `src/oilfield_chemical_copilot/rag/service.py`
- Modify: `tests/retrieval/test_pipeline.py`
- Modify: `tests/rag/test_service.py`

**Interfaces:**
- Consumes: `fuse_ranked_hits(...)`, `KeywordSearchIndex`, `VectorRetriever`, and `VectorStore.search(...)`.
- Produces: `HybridRetrievalPipeline(store: VectorStore, embedding_provider: EmbeddingProvider, keyword_index: KeywordSearchIndex, settings: RetrievalSettings | None = None)` with `retrieve(question: str, topic: str | None = None) -> list[RetrievalHit]`.
- Produces: `build_retrieval_pipeline(*, store: VectorStore, embedding_provider: EmbeddingProvider, settings: RetrievalSettings, keyword_index: KeywordSearchIndex | None = None) -> BasicRetrievalPipeline | HybridRetrievalPipeline`.
- Produces: `RetrievalSettings.evidence_threshold: float`, equal to `min_score` in vector mode and `hybrid_min_rrf_score` in hybrid mode.

- [ ] **Step 1: Write failing setting and hybrid-pipeline tests**

Extend `tests/retrieval/test_pipeline.py` with a fake keyword index that records `search(query, limit, topic)` calls. Verify:

```python
def test_hybrid_pipeline_fuses_keyword_and_vector_candidates() -> None:
    settings = RetrievalSettings(
        retrieval_mode="hybrid",
        top_k=2,
        min_score=0.9,
        max_context_chars=1000,
        hybrid_candidate_limit=3,
        hybrid_rrf_k=60,
        hybrid_min_rrf_score=0.015,
    )
    pipeline = HybridRetrievalPipeline(
        store=FakeStore([_hit("shared", 0.2), _hit("semantic", 0.1)]),
        embedding_provider=DeterministicEmbeddingProvider(dimension=8),
        keyword_index=FakeKeywordIndex([_hit("shared", 1.0), _hit("exact", 0.5)]),
        settings=settings,
    )

    hits = pipeline.retrieve("SCALE-X compatibility", topic="scale")

    assert [hit.chunk_id for hit in hits] == ["shared", "exact", "semantic"]
    assert all(hit.retrieval_method == "hybrid" for hit in hits)
```

Also test that hybrid mode calls both methods with `hybrid_candidate_limit`, retains a keyword-only candidate even when vector similarity is below `RAG_MIN_SCORE`, filters only by `hybrid_min_rrf_score`, respects `top_k` and `max_context_chars`, and skips both calls for blank questions. Add `from_env` tests using `monkeypatch` for the exact defaults and invalid `RETRIEVAL_MODE="keyword"` error.

- [ ] **Step 2: Run the pipeline tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_pipeline.py -v`

Expected: FAIL because hybrid settings, the pipeline class, and the factory do not exist.

- [ ] **Step 3: Add validated hybrid settings and the selected-mode factory**

Extend `RetrievalSettings` in `src/oilfield_chemical_copilot/retrieval/pipeline.py` with these fields and defaults:

```python
retrieval_mode: str = "hybrid"
hybrid_candidate_limit: int = 10
hybrid_rrf_k: int = 60
hybrid_min_rrf_score: float = 0.015
```

In `from_env()`, read `RETRIEVAL_MODE`, `HYBRID_CANDIDATE_LIMIT`, `HYBRID_RRF_K`, and `HYBRID_MIN_RRF_SCORE`. Reject a mode outside `{"hybrid", "vector"}`, candidate/top-k/context values below one, `HYBRID_RRF_K` below one, and negative threshold values with a `ValueError` that names the invalid variable. Add:

```python
@property
def evidence_threshold(self) -> float:
    return self.min_score if self.retrieval_mode == "vector" else self.hybrid_min_rrf_score
```

Add `build_retrieval_pipeline`. It returns the unchanged `BasicRetrievalPipeline` when mode is `vector`; for `hybrid`, it requires a non-`None` keyword index and raises `ValueError("keyword_index is required for hybrid retrieval")` otherwise.

- [ ] **Step 4: Implement independent candidate retrieval and fusion**

Add `HybridRetrievalPipeline` to `pipeline.py`. Construct one `VectorRetriever` in `__init__`; retain the supplied keyword index. Its body must follow this order:

```python
def retrieve(self, question: str, topic: str | None = None) -> list[RetrievalHit]:
    if not question.strip():
        return []
    vector_hits = self.vector_retriever.search(
        question, limit=self.settings.hybrid_candidate_limit, topic=topic
    )
    keyword_hits = self.keyword_index.search(
        question, limit=self.settings.hybrid_candidate_limit, topic=topic
    )
    fused = fuse_ranked_hits(
        keyword_hits, vector_hits,
        rrf_k=self.settings.hybrid_rrf_k,
        limit=self.settings.top_k,
    )
    qualifying = [hit for hit in fused if hit.score >= self.settings.hybrid_min_rrf_score]
    return _fit_context_budget(qualifying, self.settings.max_context_chars)
```

Do not apply `min_score` in this class. Keep `BasicRetrievalPipeline.retrieve()` byte-for-byte behaviorally equivalent: vector search with `top_k`, then `score >= min_score`, then the shared context budget.

- [ ] **Step 5: Write and implement the RAG service threshold regression test**

Add this test to `tests/rag/test_service.py`:

```python
def test_service_accepts_qualified_hybrid_rrf_score() -> None:
    generator = FakeGenerator(draft=_draft())
    service = BasicRagService(
        retriever=FakeRetriever([_hit(score=2 / 61)]),
        generator=generator,
        min_score=0.015,
    )

    answer = service.answer("How should I assess scale risk?")

    assert answer.weak_evidence is False
    assert generator.calls == 1
```

Then change `BasicRagService.from_settings()` to pass `settings.evidence_threshold`. Retain the `max(hit.score) < self.min_score` service guard so fake retrievers and direct callers still receive a safe fallback; it now compares a score to the threshold from the corresponding mode.

- [ ] **Step 6: Run focused regression tests and lint**

Run: `uv run pytest tests/retrieval/test_pipeline.py tests/rag/test_service.py -v`

Expected: PASS. In particular, the existing vector weak-evidence test still avoids a generator call, and a valid hybrid score near `0.033` is accepted only with the hybrid threshold.

Run: `uv run ruff check src/oilfield_chemical_copilot/retrieval/pipeline.py src/oilfield_chemical_copilot/rag/service.py tests/retrieval/test_pipeline.py tests/rag/test_service.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit after explicit user approval**

```powershell
git add src/oilfield_chemical_copilot/retrieval/pipeline.py src/oilfield_chemical_copilot/rag/service.py tests/retrieval/test_pipeline.py tests/rag/test_service.py
git commit -m "feat: add selectable hybrid retrieval pipeline"
```

### Task 3: Source Provenance, Streamlit Comparison Controls, and Module 1 Documentation

**Files:**
- Modify: `src/oilfield_chemical_copilot/rag/models.py`
- Modify: `src/oilfield_chemical_copilot/rag/prompt_builder.py`
- Modify: `app/streamlit_app.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/rag/test_prompt_builder.py`
- Modify: `tests/rag/test_formatter.py`
- Modify: `tests/app/test_streamlit_app.py`

**Interfaces:**
- Consumes: fused `RetrievalHit.metadata["rrf_methods"]`, `RetrievalSettings.retrieval_mode`, `build_retrieval_pipeline(...)`, and `PgVectorStore.list_chunks()`.
- Produces: `SourceEvidence.retrieval_method: str = "vector"` and `SourceEvidence.retrieval_sources: tuple[str, ...] = ()`.
- Produces: `_build_rag_service(retrieval_mode: str) -> BasicRagService` and `_answer_question(prompt: str, retrieval_mode: str) -> RagAnswer`.
- Produces: `_citation_display(source: SourceEvidence) -> str` including a safe mode label such as `hybrid: keyword + vector`.

- [ ] **Step 1: Write the failing provenance tests**

Extend `tests/rag/test_prompt_builder.py` with a hybrid hit whose metadata is:

```python
metadata={
    "rrf_methods": ("keyword", "vector"),
    "keyword_rank": 1,
    "vector_rank": 2,
}
```

Assert that `prompt.sources[0].retrieval_method == "hybrid"`, `prompt.sources[0].retrieval_sources == ("keyword", "vector")`, and the prompt contains no absolute path. Extend `tests/app/test_streamlit_app.py` so a matching `SourceEvidence` yields:

```python
"Source 1: docs/scale.md | document | chunk scale-1 | score 0.033 | hybrid: keyword + vector"
```

Add a vector-source assertion that `retrieval_sources == ("vector",)` when the hit has no RRF metadata. Keep the current formatter tests to prove citation text remains source-grounded and path-safe.

- [ ] **Step 2: Run the provenance tests to verify they fail**

Run: `uv run pytest tests/rag/test_prompt_builder.py tests/rag/test_formatter.py tests/app/test_streamlit_app.py -v`

Expected: FAIL because `SourceEvidence` does not carry retrieval provenance and the Streamlit citation display has no method label.

- [ ] **Step 3: Carry provenance from retrieval hit to cited evidence**

Add these defaulted fields after `score` in `SourceEvidence` in `rag/models.py` so existing direct test constructors remain valid:

```python
retrieval_method: str = "vector"
retrieval_sources: tuple[str, ...] = ()
```

In `_source_evidence()` in `rag/prompt_builder.py`, derive a safe tuple without exposing the complete metadata dictionary:

```python
raw_methods = hit.metadata.get("rrf_methods")
retrieval_sources = tuple(str(method) for method in raw_methods) if isinstance(raw_methods, tuple) else (hit.retrieval_method,)
```

Pass `retrieval_method=hit.retrieval_method` and `retrieval_sources=retrieval_sources` to `SourceEvidence`. Do not add absolute paths, raw metadata, or provenance to the LLM prompt; provenance is for the user citation display only.

- [ ] **Step 4: Add the selectable Streamlit retrieval mode**

In `app/streamlit_app.py`, import `replace`, `KeywordSearchIndex`, and `build_retrieval_pipeline`. Change the cached builder to accept the mode:

```python
@st.cache_resource(show_spinner=False)
def _build_rag_service(retrieval_mode: str) -> BasicRagService:
    settings = replace(RetrievalSettings.from_env(), retrieval_mode=retrieval_mode)
    embedding_provider = build_embedding_provider()
    store = PgVectorStore(_database_url(), embedding_dimension=embedding_provider.dimension)
    keyword_index = (
        KeywordSearchIndex.from_hits(store.list_chunks())
        if settings.retrieval_mode == "hybrid"
        else None
    )
    retriever = build_retrieval_pipeline(
        store=store,
        embedding_provider=embedding_provider,
        settings=settings,
        keyword_index=keyword_index,
    )
    return BasicRagService.from_settings(retriever=retriever, generator=build_answer_generator(), settings=settings)
```

Make `_render_tools_sidebar(default_retrieval_mode: str) -> str` return a `st.selectbox("Retrieval mode", ("hybrid", "vector"), ...)` selection before rendering the existing calculator controls. In `run_app()`, obtain the default from `RetrievalSettings.from_env().retrieval_mode`, pass the selected value to `_answer_question(prompt, retrieval_mode)`, and include that same selection when retrieving the cached service. Existing calculator behavior stays unchanged.

Update `_citation_display` to append ` | {source.retrieval_method}: {' + '.join(source.retrieval_sources or (source.retrieval_method,))}`. It must continue to call `_safe_source_file()` first.

- [ ] **Step 5: Update configuration and learning documentation**

Add to `.env.example` directly after `RAG_MAX_CONTEXT_CHARS`:

```dotenv
RETRIEVAL_MODE=hybrid
HYBRID_CANDIDATE_LIMIT=10
HYBRID_RRF_K=60
HYBRID_MIN_RRF_SCORE=0.015
```

Update `README.md` so the Basic RAG flow is shown as:

```text
question -> keyword candidates + vector candidates -> RRF fusion -> bounded evidence prompt -> Ollama structured draft -> deterministic answer with citations
```

Add a short Module 1 learning note explaining lexical precision (exact identifiers such as a chemical/product name), semantic recall (related language), and RRF. Include the exact formula `1 / (60 + keyword rank) + 1 / (60 + vector rank)`, clarify that a rank starts at one, and state that this is a ranking score rather than a percentage or cosine similarity. Document `RETRIEVAL_MODE=vector` as the comparison path and that `RAG_MIN_SCORE` applies only there.

- [ ] **Step 6: Run focused UI/RAG tests and the complete suite**

Run: `uv run pytest tests/rag/test_prompt_builder.py tests/rag/test_formatter.py tests/app/test_streamlit_app.py -v`

Expected: PASS, including the hybrid provenance display, unchanged safe source handling, and cached builder behavior for both modes.

Run: `uv run pytest`

Expected: all unit tests pass; opt-in integration tests may remain skipped unless their environment flags are set.

Run: `uv run ruff check app/streamlit_app.py src/oilfield_chemical_copilot/rag tests/rag tests/app`

Expected: `All checks passed!`

- [ ] **Step 7: Commit after explicit user approval**

```powershell
git add app/streamlit_app.py src/oilfield_chemical_copilot/rag/models.py src/oilfield_chemical_copilot/rag/prompt_builder.py .env.example README.md tests/rag/test_prompt_builder.py tests/rag/test_formatter.py tests/app/test_streamlit_app.py
git commit -m "feat: expose hybrid retrieval provenance"
```

### Task 4: End-to-End Local Verification and Teaching Checkpoint

**Files:**
- Modify: `README.md` only if a command or observed configuration differs from Task 3's documented behavior.
- Test: existing `tests/rag/test_public_sample_flow.py`, `tests/rag/test_ollama_integration.py`, and all unit tests.

**Interfaces:**
- Consumes: the complete Task 1-3 hybrid pipeline, existing sample corpus, Docker Postgres, and locally running Ollama.
- Produces: a verified Module 1 hybrid answer with at least one citation showing hybrid provenance, plus a vector-mode comparison without a behavior regression.

- [ ] **Step 1: Confirm the sample corpus and PGVector index are current**

Run the documented commands only if `data/processed/chunks.jsonl` is missing or older than the sample source files:

```powershell
uv run python ingestion/ingest.py --data-dir data/sample --output-dir data/processed --max-files 20
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
uv run python ingestion/index_chunks.py --input data/processed/chunks.jsonl --database-url $env:DATABASE_URL
```

Confirm the index uses the configured embedding model before testing. Do not reset or truncate a non-test database.

- [ ] **Step 2: Run the deterministic public flow first**

Run: `uv run pytest tests/rag/test_public_sample_flow.py tests/retrieval tests/rag tests/app -v`

Expected: PASS. This proves the code paths remain deterministic before any networked verification.

- [ ] **Step 3: Run the opt-in live local smoke test**

With Docker Postgres and Ollama already reachable, run:

```powershell
$env:RUN_OLLAMA_INTEGRATION = "1"
$env:TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot_test"
uv run pytest -m integration tests/rag/test_ollama_integration.py -v
```

Expected: PASS with the local Granite answer and embedding models. If the service is unavailable, record the exact unavailable prerequisite and keep the unit-test result separate; do not substitute OpenAI.

- [ ] **Step 4: Manually compare both retrieval modes in Streamlit**

Run: `uv run streamlit run app/streamlit_app.py`

Use `How should I assess scale risk from produced water analysis?` first with `hybrid`, then with `vector`. Confirm that the hybrid response has at least one source citation labeled `hybrid: keyword + vector` or a valid single-method hybrid provenance label, the answer contains no absolute file path, and changing mode creates the correct independently cached retrieval service.

- [ ] **Step 5: Record the Module 1 teaching checkpoint in the final implementation report**

State these three observations in the implementation report: keyword retrieval catches exact terms, vector retrieval catches related wording, and RRF combines rank positions without pretending they share one score scale. State whether the hybrid answer actually differed from vector mode for the sample question; do not claim a quality improvement from a single sample.

- [ ] **Step 6: Commit after explicit user approval if README was corrected during verification**

```powershell
git add README.md
git commit -m "docs: clarify verified hybrid retrieval setup"
```

Skip this commit when Task 4 did not change `README.md`.

## Self-Review

- Spec coverage: Task 1 implements stable RRF, rank provenance, keyword index creation from PGVector records, empty-list behavior, and exact/tie tests. Task 2 implements both modes, isolated thresholds, settings validation, context limits, and safe RAG threshold handling. Task 3 implements Streamlit comparison, cited-source provenance, path privacy, configuration, and the Module 1 explanation. Task 4 verifies deterministic behavior before the local Docker/Ollama smoke test and avoids claiming a sample-based quality conclusion.
- Placeholder scan: completed; this plan contains no deferred implementation markers or unspecified test steps.
- Type consistency: `fuse_ranked_hits`, `KeywordSearchIndex.from_hits`, `HybridRetrievalPipeline`, `build_retrieval_pipeline`, `RetrievalSettings.evidence_threshold`, and the `SourceEvidence` provenance fields use the same names and types in every downstream task.
