# Public Retrieval Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure keyword, vector, and hybrid RRF retrieval on a versioned public sample question set before changing retrieval or adding tool calling.

**Architecture:** A public JSONL dataset maps each question to one or more expected chunk IDs. A deterministic evaluation library calculates Hit Rate@k and MRR from ranked `RetrievalHit` lists, while a CLI uses the current retrieval modes and writes sanitized JSON and Markdown comparison reports. The first metrics are baselines, not operational-readiness thresholds.

**Tech Stack:** Python 3.11, existing `minsearch`, PGVector retrieval APIs, pytest, JSONL, Markdown.

## Global Constraints

- Use only `data/sample`; do not read, copy, commit, or report private corpus content, local paths, credentials, or generated private artifacts.
- Dataset records contain `question`, `expected_chunk_ids`, and `topic`; expected IDs must be stable IDs from the current public `chunks.jsonl` sample.
- Compare `keyword`, `vector`, and `hybrid` on the identical question set with `k=3` and `k=5`.
- Define `Hit Rate@k` as the fraction of questions with any expected chunk in the first `k` results.
- Define reciprocal rank as `1 / first_expected_rank`, or `0.0` when no expected chunk is retrieved; MRR is its mean.
- Reports may include question IDs, topics, chunk IDs, ranks, mode, aggregate metrics, and latency; they must not include source text, excerpts, `source_path`, or absolute paths.
- Initial results establish a baseline only. Do not claim the system is ready for operational production decisions.
- Reranking is out of scope unless a later approved metric review identifies a specific retrieval gap.
- Do not implement tool calling in this milestone.
- Commit only after explicit user instruction.

---

## File Structure

- `eval/public_retrieval_dataset.jsonl`: public gold questions and expected chunk IDs.
- `src/oilfield_chemical_copilot/evaluation/retrieval.py`: dataset models, validation, metrics, and report-safe result models.
- `eval/retrieval_eval.py`: CLI that selects the current keyword, vector, or hybrid retriever and writes comparison results.
- `tests/evaluation/test_retrieval.py`: pure metric and dataset validation tests.
- `tests/eval/test_retrieval_eval.py`: CLI/report sanitization tests using fakes; no live services.
- `README.md`: public evaluation commands and the meaning of Hit Rate@k and MRR.

### Task 1: Public Gold Dataset and Metric Primitives

**Files:**
- Create: `eval/public_retrieval_dataset.jsonl`
- Create: `src/oilfield_chemical_copilot/evaluation/__init__.py`
- Create: `src/oilfield_chemical_copilot/evaluation/retrieval.py`
- Create: `tests/evaluation/test_retrieval.py`

**Interfaces:**
- Produces: `EvaluationCase(question_id: str, question: str, expected_chunk_ids: tuple[str, ...], topic: str)`.
- Produces: `load_evaluation_cases(path: Path) -> list[EvaluationCase]`.
- Produces: `first_expected_rank(hits: list[RetrievalHit], expected_chunk_ids: frozenset[str], k: int) -> int | None`.
- Produces: `hit_rate_at_k(results: list[EvaluationResult], k: int) -> float` and `mean_reciprocal_rank(results: list[EvaluationResult], k: int) -> float`.

- [ ] **Step 1: Write failing metric tests**

Create `tests/evaluation/test_retrieval.py` with a complete `RetrievalHit` helper. Assert that expected evidence at rank 2 produces `first_expected_rank(...) == 2`, `Hit Rate@3 == 1.0`, and reciprocal rank `0.5`; absence produces rank `None`, hit rate `0.0`, and reciprocal rank `0.0`. Assert `k < 1` raises `ValueError("k must be at least 1")`. Add a dataset test that rejects duplicate question IDs, empty expected IDs, and a non-public absolute path.

```python
def test_metrics_score_first_expected_result() -> None:
    result = EvaluationResult(question_id="scale-01", ranked_chunk_ids=("wrong", "scale"), latency_ms=9)
    assert first_expected_rank(result.ranked_chunk_ids, frozenset({"scale"}), k=3) == 2
    assert hit_rate_at_k([result.with_rank(2)], k=3) == 1.0
    assert mean_reciprocal_rank([result.with_rank(2)], k=3) == 0.5
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_retrieval.py -v`

Expected: collection failure because the evaluation package does not exist.

- [ ] **Step 3: Implement immutable models and metrics**

Use frozen dataclasses. `EvaluationResult` stores only `question_id`, `topic`, `ranked_chunk_ids`, `expected_rank`, and `latency_ms`; it must never hold `RetrievalHit.text`, source filenames, paths, or metadata. Reject blank values and duplicate question IDs when loading JSONL.

- [ ] **Step 4: Add the public dataset**

Create 18 JSONL cases with these expected chunk IDs, reusing each source with distinct wording:

```json
{"question_id":"scale-01","question":"How should I assess scale risk from produced water analysis?","expected_chunk_ids":["docs:scale_water_analysis_overview.md:document:0:d38047e6931f","docs:water_analysis_interpretation.md:document:0:a36132aa7fbd"],"topic":"scale"}
{"question_id":"dosage-01","question":"How do ppm, water barrels per day, and 42 gallons per barrel affect continuous treatment dosage?","expected_chunk_ids":["docs:chemical_dosage_examples.md:document:0:43031497f26b"],"topic":"dosage"}
{"question_id":"iron-01","question":"What field checks help investigate black iron sulfide deposits and THPS treatment?","expected_chunk_ids":["docs:iron_sulfide_overview.md:document:0:8a0c9656afa6"],"topic":"iron_sulfide"}
{"question_id":"corrosion-01","question":"Which observations separate corrosion under-treatment from mechanical or operating causes?","expected_chunk_ids":["docs:corrosion_root_cause.md:document:0:154c59366030"],"topic":"corrosion"}
{"question_id":"paraffin-01","question":"What operating changes can make paraffin or asphaltene deposits more likely?","expected_chunk_ids":["docs:paraffin_asphaltene_overview.md:document:0:3ded8ee54f6b"],"topic":"paraffin"}
{"question_id":"water-01","question":"Which ions and operating conditions frame a scale and corrosion water review?","expected_chunk_ids":["docs:water_analysis_interpretation.md:document:0:a36132aa7fbd","docs:scale_water_analysis_overview.md:document:0:d38047e6931f"],"topic":"water_analysis"}
```

Add these remaining twelve records, then validate that every expected ID exists in the parsed public sample:

```json
{"question_id":"scale-02","question":"What chemistry and operating changes should be screened before predicting inorganic scale?","expected_chunk_ids":["docs:scale_water_analysis_overview.md:document:0:d38047e6931f"],"topic":"scale"}
{"question_id":"scale-03","question":"Why can incompatible produced waters increase deposition risk?","expected_chunk_ids":["docs:scale_water_analysis_overview.md:document:0:d38047e6931f"],"topic":"scale"}
{"question_id":"dosage-02","question":"What conversion inputs are needed to estimate water-basis chemical gallons per day?","expected_chunk_ids":["docs:chemical_dosage_examples.md:document:0:43031497f26b"],"topic":"dosage"}
{"question_id":"dosage-03","question":"How is a continuous ppm treatment related to daily water production?","expected_chunk_ids":["docs:chemical_dosage_examples.md:document:0:43031497f26b"],"topic":"dosage"}
{"question_id":"iron-02","question":"Which history and deposit observations support an iron sulfide diagnosis?","expected_chunk_ids":["docs:iron_sulfide_overview.md:document:0:8a0c9656afa6"],"topic":"iron_sulfide"}
{"question_id":"iron-03","question":"Why might THPS be discussed when black produced-water solids appear?","expected_chunk_ids":["docs:iron_sulfide_overview.md:document:0:8a0c9656afa6"],"topic":"iron_sulfide"}
{"question_id":"corrosion-02","question":"What data should a corrosion root-cause investigation compare?","expected_chunk_ids":["docs:corrosion_root_cause.md:document:0:154c59366030"],"topic":"corrosion"}
{"question_id":"corrosion-03","question":"How do coupons, probes, wall loss, and failure location help diagnose corrosion?","expected_chunk_ids":["docs:corrosion_root_cause.md:document:0:154c59366030"],"topic":"corrosion"}
{"question_id":"paraffin-02","question":"What conditions are commonly associated with wax deposition?","expected_chunk_ids":["docs:paraffin_asphaltene_overview.md:document:0:3ded8ee54f6b"],"topic":"paraffin"}
{"question_id":"paraffin-03","question":"Which changes can destabilize asphaltenes in crude oil?","expected_chunk_ids":["docs:paraffin_asphaltene_overview.md:document:0:3ded8ee54f6b"],"topic":"paraffin"}
{"question_id":"water-02","question":"What does high chloride and TDS indicate during a brine review?","expected_chunk_ids":["docs:water_analysis_interpretation.md:document:0:a36132aa7fbd"],"topic":"water_analysis"}
{"question_id":"water-03","question":"Which water-analysis fields frame corrosion and deposit questions?","expected_chunk_ids":["docs:water_analysis_interpretation.md:document:0:a36132aa7fbd"],"topic":"water_analysis"}
```

- [ ] **Step 5: Verify GREEN and commit only on user instruction**

Run: `uv run pytest tests/evaluation/test_retrieval.py -v`

Expected: PASS.

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation tests/evaluation`

Expected: `All checks passed!`

Commit after explicit user instruction:

```powershell
git add eval/public_retrieval_dataset.jsonl src/oilfield_chemical_copilot/evaluation tests/evaluation
git commit -m "feat: add public retrieval evaluation metrics"
```

### Task 2: Mode Comparison Runner and Safe Reports

**Files:**
- Modify: `eval/retrieval_eval.py`
- Create: `tests/eval/test_retrieval_eval.py`

**Interfaces:**
- Consumes: `load_evaluation_cases`, metric functions, and an injected `Callable[[str, str | None], list[RetrievalHit]]` retriever.
- Produces: `evaluate_cases(cases: list[EvaluationCase], retrieve: RetrievalCallable, *, k: int) -> list[EvaluationResult]`.
- Produces: `write_report(results_by_mode: dict[str, list[EvaluationResult]], output_dir: Path) -> tuple[Path, Path]` for `retrieval_eval.json` and `retrieval_eval.md`.

- [ ] **Step 1: Write failing runner/report tests**

Create fakes returning `RetrievalHit` values whose `source_path` contains `C:/private/secret.md` and whose `text` contains `PRIVATE EXCERPT`. Assert `evaluate_cases` calls the retriever with each case question/topic, captures only ranked chunk IDs and integer non-negative latency, and produces expected ranks. Assert both serialized report files omit `C:/`, `PRIVATE EXCERPT`, `source_path`, and `text`, while containing `keyword`, `vector`, `hybrid`, `hit_rate_at_3`, `hit_rate_at_5`, and `mrr_at_5`.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/eval/test_retrieval_eval.py -v`

Expected: FAIL because runner functions do not exist.

- [ ] **Step 3: Implement pure evaluation and rendering**

Use `time.perf_counter()` around only the injected retrieve call. The Markdown report must contain one aggregate table with `mode`, `questions`, `Hit Rate@3`, `Hit Rate@5`, `MRR@5`, and `median latency ms`; list failures as question ID, topic, expected rank, and returned chunk IDs only. Never serialize a `RetrievalHit` object directly.

- [ ] **Step 4: Implement CLI wiring**

Give `eval/retrieval_eval.py` arguments:

```text
--dataset eval/public_retrieval_dataset.jsonl
--output-dir data/processed/evaluation
--modes keyword,vector,hybrid
--k 5
--database-url optional DATABASE_URL override
```

Load stored chunks once with `PgVectorStore.list_chunks()`. Build `KeywordSearchIndex.from_hits(stored_hits)` for keyword mode. For vector/hybrid, build the existing embedding provider, `RetrievalSettings` with the requested mode, and `build_retrieval_pipeline`; preserve the configured provider/model and do not index or migrate data. Fail before writing a report if the dataset contains an expected chunk ID absent from `stored_hits`.

- [ ] **Step 5: Verify GREEN and commit only on user instruction**

Run: `uv run pytest tests/eval/test_retrieval_eval.py tests/evaluation/test_retrieval.py -v`

Expected: PASS.

Run: `uv run ruff check eval/retrieval_eval.py tests/eval tests/evaluation src/oilfield_chemical_copilot/evaluation`

Expected: `All checks passed!`

Commit after explicit user instruction:

```powershell
git add eval/retrieval_eval.py tests/eval
git commit -m "feat: compare public retrieval modes"
```

### Task 3: Public Baseline Run and Learning Checkpoint

**Files:**
- Modify: `README.md`
- Test: all evaluation tests and opt-in local retrieval run.

**Interfaces:**
- Consumes: the Task 1 dataset and Task 2 CLI.
- Produces: gitignored `data/processed/evaluation/retrieval_eval.json` and `retrieval_eval.md` for local review.

- [ ] **Step 1: Document the exact evaluation workflow**

Add a README section explaining that the public set measures retrieval only, not chemistry truth or production readiness. Include:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
uv run python eval/retrieval_eval.py --dataset eval/public_retrieval_dataset.jsonl --output-dir data/processed/evaluation --modes keyword,vector,hybrid --k 5
```

Define Hit Rate@k and MRR in one sentence each, and state that a later private run uses a gitignored local dataset and publishes aggregate metrics only.

- [ ] **Step 2: Run deterministic checks**

Run: `uv run pytest tests/evaluation tests/eval -v`

Expected: PASS.

Run: `uv run pytest`

Expected: complete suite passes; opt-in integration tests may remain skipped.

- [ ] **Step 3: Run the local public baseline**

With Docker PGVector and the public sample index available, run the documented command. Confirm both report files exist, all three modes have metrics, no report contains `C:/`, and do not claim a winning mode until the recorded numbers are reviewed.

- [ ] **Step 4: Record teaching checkpoint and commit only on user instruction**

Report: a result with high Hit Rate but low MRR finds evidence but ranks it poorly; MRR exposes that weakness. State the baseline result, observed failure categories, and whether a follow-up retrieval change is justified.

Commit after explicit user instruction:

```powershell
git add README.md
git commit -m "docs: document public retrieval evaluation"
```

## Self-Review

- Spec coverage: Tasks 1-3 create public ground truth, calculate Hit Rate@k/MRR, compare keyword/vector/hybrid, produce sanitized reports, establish a baseline, and defer tool calling/reranking until measured evidence exists.
- Placeholder scan: complete; exact files, interfaces, test assertions, commands, metric definitions, and privacy constraints are present.
- Type consistency: `EvaluationCase`, `EvaluationResult`, `load_evaluation_cases`, `evaluate_cases`, and `write_report` retain the same names and responsibility across all tasks.
