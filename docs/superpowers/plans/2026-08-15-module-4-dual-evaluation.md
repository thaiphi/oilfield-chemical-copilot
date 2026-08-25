# Module 4 Dual Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible public teaching evaluation and a sealed local handout evaluation that exercise the actual vector and hybrid RAG paths while retaining only aggregate-safe evidence outside `.private`.

**Architecture:** Preserve the existing retrieval metrics, deterministic answer checks, and live-RAG capture. Add a narrow Module 4 contract for sealed local cases, one-shot state, and recursive aggregate-report validation. A shared runner will use the same case model for public and local scopes; public fixtures remain committed, while local fixtures and detailed diagnostics remain controller-owned under `.private/evaluation/module4_handouts/`.

**Tech Stack:** Python 3.11+, pytest, existing PGVector retrieval pipeline, local Ollama, existing evaluation package, JSONL, SHA-256.

## Global Constraints

- Use `data/sample` and committed `eval/` fixtures only for the public scope.
- Keep all local handout cases, source mappings, raw answers, retrieved evidence, detailed diagnostics, and one-shot state under `.private/evaluation/module4_handouts/`.
- Do not modify production RAG, retrieval, prompt, generator, corpus, or model settings in this module.
- Run the local handout score once per sealed dataset SHA-256; write the one-shot state before the first RAG call.
- Durable reports may contain only fixed labels, SHA-256 values, counts, numeric metrics, and pass/fail/unavailable states. They must reject source text, paths, URLs, credentials, case IDs, chunk IDs, prompts, answers, excerpts, and raw errors.
- Use local Ollama only for live runs. Do not send handout material to OpenAI.
- Any score-driven RAG change requires a separately approved experiment after results are reviewed.

---

## File Structure

- Create `src/oilfield_chemical_copilot/evaluation/module4_contract.py`: validates local case records, seals canonical JSONL, verifies digest, and enforces one-shot state.
- Create `src/oilfield_chemical_copilot/evaluation/module4_reports.py`: converts metric and deterministic outcomes to recursive aggregate-only report dictionaries and rejects unsafe values.
- Create `src/oilfield_chemical_copilot/evaluation/module4_live.py`: constructs actual vector and hybrid services, captures in-memory live answers, and returns report-safe summaries.
- Create `eval/module4_evaluation_pack.py`: CLI for public and local scopes; local mode requires a sealed fixture and a controller approval file.
- Create `tests/evaluation/test_module4_contract.py`, `tests/evaluation/test_module4_reports.py`, and `tests/evaluation/test_module4_live.py`: unit and boundary coverage.
- Create `tests/eval/test_module4_evaluation_pack.py`: CLI preflight, scope, and write-boundary coverage.
- Modify `README.md`, `docs/LEARNING_ROADMAP.md`, `docs/PROJECT_STATUS.md`, and `docs/COURSE_ALIGNED_PLAN.md`: document the actual command, metric limits, and Module 4 status only after evidence exists.
- Create `docs/superpowers/reports/2026-08-15-module-4-public-evaluation.md`: aggregate-only public evidence after the real run succeeds.

### Task 1: Sealable Module 4 Case Contract

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/module4_contract.py`
- Create: `tests/evaluation/test_module4_contract.py`

**Interfaces:**
- Consumes: a local draft JSONL file only beneath `.private/evaluation/module4_handouts/dataset/`.
- Produces: `Module4Case`, `seal_cases(draft_path, sealed_path, digest_path) -> str`, `verify_seal(sealed_path, digest_path) -> tuple[Module4Case, ...]`, and `consume_one_shot(state_path, dataset_sha256) -> None`.

- [x] **Step 1: Write failing contract tests**

```python
def test_seal_cases_canonicalizes_reviewed_local_cases(tmp_path: Path) -> None:
    draft = tmp_path / "draft.jsonl"
    draft.write_text(
        json.dumps({
            "case_id": "private-case-01", "question": "private question",
            "topic": "scale", "expected_chunk_ids": ["private-chunk-01"],
            "expect_citations": True, "expect_abstention": False, "reviewed": True,
        }) + "\n",
        encoding="utf-8",
    )
    sealed, digest = tmp_path / "sealed.jsonl", tmp_path / "sealed.sha256"

    dataset_sha256 = seal_cases(draft, sealed, digest)

    assert re.fullmatch(r"[0-9a-f]{64}", dataset_sha256)
    assert verify_seal(sealed, digest)[0].case_id == "private-case-01"


def test_consume_one_shot_rejects_a_second_attempt(tmp_path: Path) -> None:
    state = tmp_path / "state.json"

    consume_one_shot(state, "a" * 64)

    with pytest.raises(Module4ContractError, match="^ATTEMPT_UNAVAILABLE$"):
        consume_one_shot(state, "a" * 64)
```

Add tests that reject an unreviewed record, a blank question, duplicate `case_id`, duplicate expected chunk IDs, a cited-answer case with no expected IDs, a non-citation abstention case with expected IDs, a path outside the `.private` boundary, a changed sealed file, and a state file that records a different dataset SHA-256. Assert error messages use fixed codes and never echo private record values.

- [x] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/evaluation/test_module4_contract.py -q`

Expected: collection fails because `module4_contract` does not exist.

- [x] **Step 3: Implement canonical sealing and one-shot state**

```python
@dataclass(frozen=True)
class Module4Case:
    case_id: str
    question: str
    topic: str
    expected_chunk_ids: tuple[str, ...]
    expect_citations: bool
    expect_abstention: bool
    reviewed: bool


def consume_one_shot(state_path: Path, dataset_sha256: str) -> None:
    if state_path.exists():
        raise Module4ContractError("ATTEMPT_UNAVAILABLE")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"dataset_sha256": dataset_sha256}), encoding="utf-8")
```

Resolve every local path and require it to be below the passed private root. Serialize sealed records using sorted JSON keys, compact separators, and a final newline before hashing. Verify the digest with `hmac.compare_digest`. Write only the dataset hash to one-shot state; never write cases, questions, IDs, or errors there.

- [x] **Step 4: Run focused contract verification**

Run: `uv run pytest tests/evaluation/test_module4_contract.py -q`

Expected: all contract tests pass.

- [x] **Step 5: Lint the new contract**

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/module4_contract.py tests/evaluation/test_module4_contract.py`

Expected: Ruff reports no findings.

### Task 2: Aggregate-Only Evaluation Reports

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/module4_reports.py`
- Create: `tests/evaluation/test_module4_reports.py`

**Interfaces:**
- Consumes: report-safe per-mode counts, Hit Rate@5, MRR@5, a dataset SHA-256, and a fixed scope label.
- Produces: `build_module4_report(...) -> dict[str, object]` and `write_module4_report(report, destination) -> None`.

- [x] **Step 1: Write failing aggregate-report tests**

```python
def test_build_module4_report_contains_only_approved_aggregate_fields(tmp_path: Path) -> None:
    report = build_module4_report(
        scope="local",
        dataset_sha256="a" * 64,
        modes={
            "vector": ModeSummary(4, 0.75, 0.625, 3, 1, 4, 0),
            "hybrid": ModeSummary(4, 1.0, 1.0, 4, 0, 4, 0),
        },
    )

    assert report["scope"] == "local"
    assert report["modes"]["hybrid"]["retrieval"] == {"hit_rate_at_5": 1.0, "mrr_at_5": 1.0}
    write_module4_report(report, tmp_path / "aggregate.json")


def test_report_rejects_private_text_at_any_depth() -> None:
    with pytest.raises(Module4ReportError, match="^UNSAFE_REPORT$"):
        write_module4_report({"scope": "local", "nested": {"question": "private prompt"}}, Path("report.json"))
```

Add parameterized sentinels for private source text, case IDs, chunk IDs, Windows and POSIX paths, URLs, credential-shaped values, raw exception text, answer text, and evidence excerpts at nested dictionary and list depths. Add tests rejecting missing modes, unexpected modes, non-finite metrics, a non-SHA-256 digest, and counts that do not reconcile with `case_count`.

- [x] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/evaluation/test_module4_reports.py -q`

Expected: collection fails because `module4_reports` does not exist.

- [x] **Step 3: Implement strict report schema**

```python
@dataclass(frozen=True)
class ModeSummary:
    retrieval_case_count: int
    hit_rate_at_5: float
    mrr_at_5: float
    citation_pass: int
    citation_fail: int
    abstention_pass: int
    abstention_fail: int
```

Require exactly `vector` and `hybrid`. Persist only this schema:

```python
{
    "scope": "public" | "local",
    "dataset_sha256": "<64 lowercase hex characters>",
    "status": "success" | "unavailable" | "failed",
    "modes": {
        "vector": {
            "retrieval_case_count": 0,
            "retrieval": {"hit_rate_at_5": 0.0, "mrr_at_5": 0.0},
            "deterministic": {"citations": {"pass": 0, "fail": 0}, "abstention": {"pass": 0, "fail": 0}},
        },
        "hybrid": {"...": "same fixed shape"},
    },
}
```

Use a recursive allowlist validator before writing. Reject any unknown key, any string except scope/status/digest/fixed mode names, and non-finite floats. Write sorted JSON with a final newline.

- [x] **Step 4: Run focused report verification**

Run: `uv run pytest tests/evaluation/test_module4_reports.py -q`

Expected: all report tests pass.

- [x] **Step 5: Lint the report module**

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/module4_reports.py tests/evaluation/test_module4_reports.py`

Expected: Ruff reports no findings.

### Task 3: Shared Actual-RAG Evaluation Runner

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/module4_live.py`
- Create: `tests/evaluation/test_module4_live.py`

**Interfaces:**
- Consumes: `tuple[Module4Case, ...]`, a configured `PgVectorStore`, local Ollama embedding/generation settings, and `k=5`.
- Produces: `evaluate_module4_modes(cases, *, build_service) -> dict[str, ModeSummary]`.

- [x] **Step 1: Write failing runner tests with actual evaluation primitives injected**

```python
def test_evaluate_module4_modes_uses_real_rank_and_deterministic_checks() -> None:
    cases = (
        Module4Case("case-1", "question", "scale", ("expected",), True, False, True),
    )

    results = evaluate_module4_modes(cases, build_service=_fake_service_builder)

    assert results["vector"] == ModeSummary(1, 1.0, 1.0, 1, 0, 1, 0)
    assert results["hybrid"] == ModeSummary(1, 1.0, 1.0, 1, 0, 1, 0)
```

Use fake services that return an in-memory `RagAnswer` and use `RecordingRetriever` plus `RecordingAnswerGenerator` so the test observes the real boundary. Add tests proving both modes are evaluated, `first_expected_rank`, `hit_rate_at_k`, and `mean_reciprocal_rank` receive `k=5`, runtime answer text is absent from `ModeSummary`, and a RAG generation failure becomes a sanitized `Module4RuntimeError("RUNTIME_UNAVAILABLE")`.

- [x] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/evaluation/test_module4_live.py -q`

Expected: collection fails because `module4_live` does not exist.

- [x] **Step 3: Implement mode evaluation without persistent runtime material**

For each mode, construct the existing retrieval pipeline and `BasicRagService` with `apply_claim_scope_policy=True`, then call `capture_live_answer()` once per case. Convert each capture to `EvaluationResult` using only `case_id`, `topic`, ranked chunk IDs, first expected rank, and measured latency. Convert it to `DeterministicAnswerResult` with existing `evaluate_answer()`. Aggregate immediately into `ModeSummary`; discard captures before returning.

For a case expecting abstention, use an empty `expected_chunk_ids`; omit it from retrieval Hit Rate/MRR denominators while still including its citation and abstention result. For a non-abstention case, require at least one expected chunk ID at contract validation.

- [x] **Step 4: Run focused runner verification**

Run: `uv run pytest tests/evaluation/test_module4_live.py tests/evaluation/test_module4_contract.py tests/evaluation/test_module4_reports.py -q`

Expected: all focused Module 4 tests pass.

- [x] **Step 5: Lint the runner**

Run: `uv run ruff check src/oilfield_chemical_copilot/evaluation/module4_live.py tests/evaluation/test_module4_live.py`

Expected: Ruff reports no findings.

### Task 4: Public And Local Evaluation CLI

**Files:**
- Create: `eval/module4_evaluation_pack.py`
- Create: `tests/eval/test_module4_evaluation_pack.py`

**Interfaces:**
- Consumes: `--scope public|local`, `--database-url`, `--output-dir`, and local-only `--sealed-path`, `--digest-path`, `--state-path`, and `--approval-path`.
- Produces: `module4_evaluation.json` and `module4_evaluation.md` containing only `build_module4_report()` output.

- [x] **Step 1: Write failing CLI/preflight tests**

```python
def test_local_mode_rejects_unsealed_fixture_before_runtime_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module4_cli, "build_runtime", lambda: pytest.fail("runtime built"))

    with pytest.raises(Module4CliError, match="^SEAL_REQUIRED$"):
        module4_cli.run_local(_local_arguments_with_missing_digest())


def test_public_mode_uses_only_committed_public_fixtures(tmp_path: Path) -> None:
    report = module4_cli.run_public(_public_arguments(tmp_path))

    assert report["scope"] == "public"
```

Add tests that local paths must be below `.private/evaluation/module4_handouts/`, local scoring consumes state before `build_runtime`, a second score fails before RAG construction, public mode rejects an unexpected stored public-manifest ID, public mode does not accept a custom fixture path, and every serialized output rejects privacy sentinels.

- [x] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/eval/test_module4_evaluation_pack.py -q`

Expected: collection fails because `module4_evaluation_pack` does not exist.

- [x] **Step 3: Implement scope-specific preflight and execution**

Public mode must load only `eval/public_answer_evaluation.jsonl` and derive the committed sample manifest with `public_sample_chunk_ids()`. It must call `validate_public_stored_chunk_ids()` before constructing a RAG service.

Local mode must require these files below the controller directory:

```text
.private/evaluation/module4_handouts/sealed/cases.jsonl
.private/evaluation/module4_handouts/sealed/cases.sha256
.private/evaluation/module4_handouts/review/approval.json
.private/evaluation/module4_handouts/results/state.json
```

`approval.json` contains exactly `{"approved": true, "dataset_sha256": "<64 lowercase hex>"}`. Validate it against the sealed digest and call `consume_one_shot()` before constructing PostgreSQL, embeddings, retrievers, or generators.

Use a local-only `details.json` file under `.private/.../results/` for per-case statuses. Write it after scoring, then write the aggregate report through `write_module4_report()`. The public report directory may be supplied by the user; the local durable report path must be `docs/superpowers/reports/2026-08-15-module-4-local-evaluation.md` and contain aggregates only.

- [x] **Step 4: Run focused CLI verification**

Run: `uv run pytest tests/eval/test_module4_evaluation_pack.py tests/evaluation/test_module4_contract.py tests/evaluation/test_module4_reports.py tests/evaluation/test_module4_live.py -q`

Expected: all Module 4 tests pass.

- [x] **Step 5: Lint the CLI**

Run: `uv run ruff check eval/module4_evaluation_pack.py tests/eval/test_module4_evaluation_pack.py`

Expected: Ruff reports no findings.

### Task 5: Live Evidence, Documentation, And Teaching Review

**Files:**
- Modify: `README.md`
- Modify: `docs/LEARNING_ROADMAP.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `docs/COURSE_ALIGNED_PLAN.md`
- Create: `docs/superpowers/reports/2026-08-15-module-4-public-evaluation.md`
- Create locally only: `.private/evaluation/module4_handouts/dataset/`, `review/`, `sealed/`, and `results/`

**Interfaces:**
- Consumes: a passing public runner, a locally reviewed and user-approved sealed handout fixture, PostgreSQL, and local Ollama.
- Produces: aggregate-only public and local evidence, accurate module status, and a practical teaching review.

- [x] **Step 1: Run complete public verification before any live evaluation**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
git diff --check
```

Expected: all checks pass before local services or fixtures are used.

- [x] **Step 2: Run the public evaluation pack against the actual local system**

Run:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "granite4.1:8b"
$env:OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"
uv run python eval/module4_evaluation_pack.py --scope public --database-url $env:DATABASE_URL --output-dir data/processed/evaluation/module4-public
```

Record only question counts, Hit Rate@5, MRR@5, deterministic citation and abstention counts, vector/hybrid status, and limitations in the durable public report.

- [x] **Step 3: Prepare and seal the local handout fixture**

Create reviewed case records only in `.private/evaluation/module4_handouts/dataset/cases.jsonl`. Each non-abstention case maps a course-handout question to one or more expected local chunk IDs. Each abstention case has no expected chunk IDs and expects neither citations nor a direct answer. Run the sealing command, inspect only its aggregate case count and SHA-256, and obtain explicit user approval before creating `review/approval.json`.

- [x] **Step 4: Run the one-shot local handout evaluation after approval**

Run the local CLI exactly once. Confirm the state file exists before the first runtime call, the detailed result stays under `.private`, and the durable local report includes no private values. If local prerequisites fail, record only `unavailable` and the fixed failure category; do not retry after a sealed score starts. The approved 2026-08-15 local run took this failure path and was recorded `unavailable`; its sealed hash is consumed and must not be replayed.

- [x] **Step 5: Update documentation and teach the results**

Document the public and local commands, scope distinction, one-shot rule, and report limitations. Explain with the actual public result:

- ground truth as the reviewed expected evidence mapping;
- Hit Rate@5 as whether expected evidence appears in the first five results;
- MRR@5 as how early the first expected evidence appears;
- citation checks as structural evidence attribution, not chemistry truth;
- abstention checks as safe refusal behavior, not evidence of operational correctness;
- why the public sample is a transparent classroom example and the local handout pack provides the meaningful project measurement.

Set Module 4 to `active lesson` until the teaching review and both evidence sets are complete. Lock it only after the user explicitly approves the lock commit.

- [x] **Step 6: Run final verification and request lock approval**

Run:

```powershell
node --test tests/codex_hooks/agent-policy.test.cjs tests/codex_hooks/workflow-contract.test.cjs
uv run pytest
uv run ruff check .
git diff --check
```

Expected: all checks pass. Do not stage, commit, or push without explicit user approval.

## Plan Self-Review

- **Spec coverage:** Tasks 1 and 2 implement sealed local cases, one-shot protection, and aggregate-only report safety. Tasks 3 and 4 run the actual vector and hybrid RAG boundaries in public and local scopes. Task 5 creates live evidence, documents limits, and teaches the required Module 4 concepts.
- **Placeholder scan:** Each task names exact files, functions, fixed schemas, command lines, expected failures, and verification commands. No behavior is deferred without a defined boundary.
- **Type consistency:** `Module4Case` is sealed by Task 1, summarized by `ModeSummary` in Task 2, consumed by `evaluate_module4_modes` in Task 3, and loaded by the CLI in Task 4. Both public and local report paths call the same `build_module4_report()` and `write_module4_report()` functions.
