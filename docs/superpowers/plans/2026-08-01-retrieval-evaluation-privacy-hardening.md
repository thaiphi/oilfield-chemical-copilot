# Retrieval Evaluation Privacy Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make public retrieval evaluation corpus-isolated and reproducible, while providing a separately bounded aggregate-only private evaluation mode.

**Architecture:** The evaluator derives a public-corpus manifest from `data/sample` and rejects a database that contains any unexpected chunk before retrieval. A privacy mode controls dataset location and report schema: public reports may include only public identifiers; private reports contain aggregates and sanitized provenance only. Both reports carry reproducibility metadata and disclose the current oracle topic filter.

**Tech Stack:** Python 3.11, existing parsers/chunk IDs, PGVector, `minsearch`, pytest, JSONL, Markdown.

## Global Constraints

- Do not add dependencies, tool calling, reranking, or private corpus fixtures.
- Public mode reads only `data/sample` and `eval/public_retrieval_dataset.jsonl`; reject a database with any chunk ID outside the derived public manifest before any retrieval/search call.
- Private datasets live only under ignored `eval/private/`; no private fixture or artifact may be committed.
- Public reports may contain only public question IDs, topics, chunk IDs, aggregates, and sanitized provenance. No report may contain text, excerpts, filenames, source paths, dataset/output paths, database URLs, or absolute paths.
- Private reports must contain only aggregate metrics and sanitized provenance: no per-question IDs, topics, chunk IDs, ranks, failure rows, texts, or paths.
- `--k` is exactly `5`; reject every other value.
- Existing topic filtering remains unchanged but public reports and README must call it an oracle filter.
- Commit only after explicit user instruction.

---

## File Structure

- `src/oilfield_chemical_copilot/evaluation/retrieval.py`: privacy-mode path validation, public manifest derivation, and run/result models.
- `eval/retrieval_eval.py`: mode-aware database preflight, fixed-depth CLI, provenance construction, and public/private report rendering.
- `tests/evaluation/test_retrieval.py`: pure public/private boundary and manifest tests.
- `tests/eval/test_retrieval_eval.py`: runner preflight, report-schema, provenance, and CLI regression tests using fakes.
- `.gitignore`: ignored private evaluation dataset boundary.
- `README.md`: current-state documentation, commands, privacy boundaries, and baseline interpretation.

### Task 1: Privacy Boundaries and Public Manifest

**Files:**
- Modify: `.gitignore`
- Modify: `src/oilfield_chemical_copilot/evaluation/retrieval.py`
- Modify: `tests/evaluation/test_retrieval.py`

**Interfaces:**
- Produces `EvaluationPrivacyMode = Literal["public", "private"]`.
- Produces `load_evaluation_cases(path: Path, *, privacy_mode: EvaluationPrivacyMode = "public") -> list[EvaluationCase]`.
- Produces `public_sample_chunk_ids() -> frozenset[str]`, derived from the existing `data/sample` parser/chunker output without reading any private source.
- Produces `validate_public_stored_chunk_ids(stored_chunk_ids: set[str], public_chunk_ids: frozenset[str]) -> None`.

- [ ] **Step 1: Write failing boundary tests**

Add tests that: public mode accepts `eval/public_retrieval_dataset.jsonl`; public mode rejects `eval/private/retrieval_dataset.jsonl`; private mode rejects paths outside `eval/private`; and `eval/private/**` is ignored by Git. Use temporary files only inside the allowed boundary.

Add a manifest test that derives public IDs from `data/sample`, confirms every public gold expected ID exists, and verifies this failure without echoing private-like sentinel IDs:

```python
with pytest.raises(ValueError, match="public evaluation database contains unexpected chunks"):
    validate_public_stored_chunk_ids(
        {"public-id", "private-sentinel-id"}, frozenset({"public-id"})
    )
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_retrieval.py -v`

Expected: new mode, manifest, and mixed-corpus tests fail because the current loader permits only `eval` broadly and no manifest validator exists.

- [ ] **Step 3: Implement the minimum boundary primitives**

Add `eval/private/` to `.gitignore`. Validate resolved paths against exactly `eval/` public file and `eval/private/` private directory according to `privacy_mode`. Implement public manifest derivation from the existing public parser/chunker and a validator that rejects missing or unexpected database IDs using counts only, not individual IDs.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/evaluation/test_retrieval.py -v`

Expected: all evaluation primitive tests pass.

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation tests/evaluation`

Expected: `All checks passed!`

### Task 2: Safe Report Schemas and Verified Runner

**Files:**
- Modify: `eval/retrieval_eval.py`
- Modify: `tests/eval/test_retrieval_eval.py`

**Interfaces:**
- Consumes `privacy_mode`, the public manifest validator, existing `RetrievalHit` values, and current `RetrievalSettings`.
- Produces `write_report(results_by_mode, output_dir, *, privacy_mode, provenance) -> tuple[Path, Path]`.
- Produces `RunProvenance(dataset_sha256, corpus_sha256, git_revision, retrieval_mode_settings, embedding_provider, embedding_model, embedding_dimension, k, topic_filter)` where every field is sanitized and contains no local path or credential.

- [ ] **Step 1: Write failing runner and report tests**

Add fakes with a public hit and an unexpected private hit. Assert public CLI preflight rejects the mixed list before keyword index construction, embedding creation, or any retriever call; its error and output files must omit the sentinel ID.

Assert these exact report properties:

```python
public_json = json.loads(public_json_path.read_text())
assert public_json["privacy_mode"] == "public"
assert public_json["provenance"]["topic_filter"] == "oracle_gold_topic"
assert public_json["modes"]["keyword"]["failures"][0]["question_id"] == "public-q"

private_json = json.loads(private_json_path.read_text())
assert private_json["privacy_mode"] == "private"
assert set(private_json["modes"]["keyword"]) == {
    "questions", "hit_rate_at_3", "hit_rate_at_5", "mrr_at_5", "median_latency_ms"
}
assert "private-question" not in private_json_path.read_text()
assert "private-topic" not in private_markdown_path.read_text()
assert "private-chunk" not in private_markdown_path.read_text()
```

Add tests that provenance includes nonempty SHA-256 dataset/corpus hashes, Git revision fallback `"unknown"` when unavailable, provider/model/dimension, `k == 5`, RRF/threshold settings, and topic-filter disclosure. Add parameterized CLI tests that reject `--k 4` and `--k 6`. Keep the direct-script `--help` regression test.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/eval/test_retrieval_eval.py -v`

Expected: mixed-corpus, private-report, provenance, and non-five depth tests fail because the current runner has a single public report schema and permits depths above five.

- [ ] **Step 3: Implement report and runner hardening**

Add `--privacy-mode public|private`, defaulting to `public`. Require `args.k == 5`. In public mode derive the public manifest, load stored chunks once, reject missing or unexpected IDs before constructing a keyword index, embedding provider, or pipeline. In private mode omit public-manifest preflight but enforce the private dataset directory and aggregate-only rendering.

Build provenance from hashes of dataset bytes and sorted corpus chunk IDs; sanitized current Git revision; configured retrieval mode, RRF k, candidate limit, hybrid/vector threshold, provider/model/dimension; fixed depth; and literal `oracle_gold_topic`. Never pass raw hit objects, paths, errors, or private identifiers to report rendering.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/eval/test_retrieval_eval.py tests/evaluation/test_retrieval.py -v`

Expected: all runner and primitive tests pass.

Run: `uv run ruff check eval/retrieval_eval.py tests/eval tests/evaluation src/oilfield_chemical_copilot/evaluation`

Expected: `All checks passed!`

### Task 3: Current Documentation and Hardened Public Baseline

**Files:**
- Modify: `README.md`
- Test: complete suite and local public baseline.

- [ ] **Step 1: Update README current state**

Replace stale planned/vector-only statements with the implemented hybrid RAG and retrieval evaluator. Document these commands:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
uv run python eval/retrieval_eval.py --privacy-mode public --dataset eval/public_retrieval_dataset.jsonl --output-dir data/processed/evaluation/public --modes keyword,vector,hybrid --k 5
uv run python eval/retrieval_eval.py --privacy-mode private --dataset eval/private/retrieval_dataset.jsonl --output-dir data/processed/evaluation/private --modes keyword,vector,hybrid --k 5
```

State that public mode requires a database containing only the derived public manifest; private datasets are ignored; private reports are aggregate-only; public evaluation uses an oracle gold-topic filter; and no report contains text or paths. Record the measured 18-question public baseline exactly and explain that identical perfect scores do not justify a retrieval change or establish chemistry truth, private-corpus performance, or production readiness. Remove hybrid fusion and labeled retrieval-dataset creation from next steps.

- [ ] **Step 2: Run deterministic checks**

Run: `uv run pytest`

Expected: full suite passes; opt-in integration tests may remain skipped.

Run: `uv run ruff check eval/retrieval_eval.py tests/eval tests/evaluation src/oilfield_chemical_copilot/evaluation`

Expected: `All checks passed!`

- [ ] **Step 3: Run and inspect the hardened public baseline**

Run the documented public command with Docker PGVector and the public sample index. Confirm JSON/Markdown reports exist, contain all three modes and provenance, say `oracle_gold_topic`, contain no `C:/`, and have no unexpected failure records. Do not create a private dataset or run private mode during this milestone.

- [ ] **Step 4: Final review and commit policy**

Request a whole-milestone review covering public isolation, private aggregate-only schema, provenance, fixed depth, README accuracy, and generated public report safety. Do not stage or commit unless the user explicitly instructs it.

## Self-Review

- Spec coverage: Task 1 implements path/corpus isolation; Task 2 implements runner, schemas, provenance, and fixed depth; Task 3 documents and verifies the current baseline.
- Placeholder scan: no deferred implementation steps, unspecified tests, or vague privacy rules remain.
- Type consistency: `EvaluationPrivacyMode`, `public_sample_chunk_ids`, `validate_public_stored_chunk_ids`, `RunProvenance`, and `write_report(..., privacy_mode, provenance)` are defined before consumers.
