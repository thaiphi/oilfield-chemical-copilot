# Abstention and Citation Policy Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` task-by-task after the plan gate is approved. Do not begin the approved live run until the separate live-run gate is approved.

**Goal:** Test whether a small, explicit claim-scope abstention policy can remove the six under-abstentions and their six unexpected citations without changing retrieval, indexing, models, prompts, evaluation cases, or production RAG behavior.

**Architecture:** Add an evaluation-only, question-level policy and score it in shadow mode against the same single live capture already used by the vector/hybrid evaluator. The policy sees only normalized question text; frozen case labels are used only after the decision as the evaluation oracle. Reports remain aggregate-only, and a successful investigation supports a later production-hardening plan rather than deploying the policy.

**Tech Stack:** Python 3.11, existing answer/live-RAG evaluation contracts, local Ollama `granite4.1:8b`, `granite-embedding:latest`, PostgreSQL/PGVector, pytest, Ruff.

## Approval Gates

1. **Plan gate:** Do not implement Tasks 1-3 until the user explicitly approves this plan.
2. **Live-run gate:** After Tasks 1-3 pass focused and full validation, return the exact command, schemas, and privacy-test evidence. Do not run Docker/Postgres/Ollama evaluation until the user separately approves it.
3. **Interpretation gate:** A live result is interpretable only if the exact public-manifest/configuration preflight passes and the prior baseline is reproduced.
4. **Change gate:** The investigation is evaluation-only. Do not deploy a policy or change the service, prompt, retrievers, thresholds, models, index, or cases until the user approves a next branch after final review.

## Why Policy Comes First

The diagnosis is category-identical for `vector` and `hybrid`. Each mode has six `under_abstention_answered_on_insufficient_case` failures and the same six corresponding `unexpected_citation_when_abstention_expected` failures. These are one upstream decision error expressed in two metrics: the system answered, so it necessarily emitted citations on cases whose contract required no answer and no citations.

The current service abstains only when retrieval is empty/below threshold, context is empty, or generation fails. Once evidence qualifies, the current draft contract requires at least one citation. Retrieval tuning therefore cannot directly distinguish a general, evidence-bounded review from a request for a site-specific determination, field-ready prescription, or replacement for complete field inputs. Prompt tuning would simultaneously change generation behavior and refusal wording, obscuring whether an explicit answerability boundary is sufficient. This plan isolates that boundary first while leaving the two genuine citation-selection failures visible.

## Baseline Metrics to Freeze

The approved aggregate report has `baseline_reproduced: true` and the following result in both modes:

| Metric | Vector | Hybrid |
| --- | ---: | ---: |
| Questions | 12 | 12 |
| Citation pass / fail | 4 / 8 | 4 / 8 |
| Abstention pass / fail | 6 / 6 | 6 / 6 |
| Judge available | 12 | 12 |
| Under-abstention | 6 | 6 |
| Unexpected citation when abstention expected | 6 | 6 |
| Allowed evidence retrieved but not cited | 1 | 1 |
| Mixed allowed/disallowed citations | 1 | 1 |

The existing judge scores are advisory and are not acceptance criteria for this policy experiment. The experiment must reproduce the deterministic baseline and provenance before interpreting its shadow result.

## Alternatives and Selected Hypothesis

### Selected: deterministic claim-scope gate in evaluation-only shadow mode

**Hypothesis:** The six abstention-required questions ask for a claim whose scope exceeds the public sample: a site-specific determination, a field-ready/final prescription, or substitution for complete field inputs. A question-only claim-scope gate should abstain on those requests and allow general review questions, independent of retrieval mode.

The closed decision categories are:

- `general_review`: allow the existing RAG path.
- `site_specific_determination`: abstain from requests to determine, confirm, predict, or establish a named/site-specific mechanism or root cause.
- `field_ready_prescription`: abstain from requests for a field-ready dosage or final treatment plan.
- `complete_input_substitution`: abstain when the public sample is asked to replace a complete field analysis or equivalent required input set.

After lowercasing, replacing hyphens with spaces, removing other punctuation, and collapsing whitespace, use these declared token/phrase families:

| Category | Required claim-scope signal |
| --- | --- |
| `complete_input_substitution` | `replace` or `substitute` together with `complete` or `full`, and `analysis`, `data set`, or `input set` |
| `field_ready_prescription` | `field ready`, `final`, or `prescribe` together with `dosage`, `dose`, `treatment`, or `treatment plan` |
| `site_specific_determination` | `determine`, `confirm`, `predict`, `diagnose`, or `establish` together with `site specific`, `named asset`, `specific deposit`, `mechanism`, or `root cause` |
| `general_review` | no abstention signal above |

Use precedence `complete_input_substitution` -> `field_ready_prescription` -> `site_specific_determination` -> `general_review`. Rules must not use case IDs, allowed evidence IDs, topic names, or one exact sentence per fixture case. Unit tests may use synthetic paraphrases and near misses, but the canonical 12 live evaluation cases remain unchanged.

### Alternative: prompt/schema-based model refusal

Add an explicit `abstain` field and refusal instructions to the generator contract. This may eventually handle nuanced answerability, but it changes the prompt and generated schema, remains model-dependent, and confounds policy with generation. Consider it only if the deterministic claim-scope hypothesis is falsified or proves too brittle.

### Alternative: retrieval threshold, reranking, or mode tuning

Change what evidence qualifies. This is not supported as the first experiment because the failure categories and totals are identical across modes, and the current evidence threshold does not encode whether the requested claim is operationally answerable. It also risks over-abstaining on the six answerable cases. Retrieval remains frozen.

### Alternative: citation-only post-generation suppression

Suppress answers after detecting citation problems. This could address the two citation-selection failures, but it does not provide an independent answerability policy and cannot safely identify abstention-required cases from citation structure alone. Keep it as a later citation-contract branch.

## Global Constraints

- Use only `eval/public_answer_evaluation.jsonl` and a database that exactly matches `public_sample_chunk_ids()`.
- Keep the canonical 12 cases, their flags, questions, allowed evidence IDs, and ordering unchanged.
- Keep `vector` and `hybrid`, all retrieval settings, the public index, chunking, embeddings, local generation/judge models, temperature `0`, no-topic-filter behavior, prompt text, draft schema, and production RAG behavior unchanged.
- The policy may consume normalized question text only. It must not receive case ID, `evidence_sufficient`, `expect_citations`, `expect_abstention`, allowed evidence IDs, retrieved hits, scores, answers, citations, or judge output.
- Case labels are an evaluation oracle used only after the policy decision. Tests must prove that changing labels cannot change a decision for the same question.
- Apply the policy counterfactually in memory: `abstain` means the shadow result has no citations and `abstained=True`; `allow` reuses the unmodified baseline capture. Do not make a second service/model call.
- Do not add logging, tracing, snapshots, temporary captures, case-level reports, or debug dumps.
- Persist only validated safe provenance, baseline-reproduction status, policy name/version, question count, aggregate decision-category counts, and aggregate deterministic control/shadow metrics by mode.
- Never persist questions, answers, evidence, case/question IDs, allowed/retrieved/cited IDs, filenames, paths, URLs, credentials, exact match fragments, raw provider/model strings, judge details, raw errors, or per-case decisions.
- Use local providers only. Do not send any corpus material or runtime evaluation content to OpenAI or another remote provider.
- Generated reports remain under gitignored `data/processed/evaluation/live_rag`; reports and validation errors must use generic data-free failures.
- Do not claim chemistry correctness, operational safety, generalization, private-corpus quality, statistical significance, or production readiness.
- Run tasks sequentially with `fork_context: false`; never use more than two concurrent subagents. Do not commit or push unless the user explicitly requests it.

## File Map

- Create `src/oilfield_chemical_copilot/evaluation/abstention_policy.py`: pure closed claim-scope policy; no I/O and no evaluation-label inputs.
- Create `tests/evaluation/test_abstention_policy.py`: category, precedence, paraphrase, near-miss, determinism, and forbidden-input tests.
- Create `src/oilfield_chemical_copilot/evaluation/live_rag_policy.py`: counterfactual scoring, baseline reconciliation, aggregate schema validation, and JSON/Markdown writer.
- Create `tests/evaluation/test_live_rag_policy.py`: metric transformation, schema, reconciliation, and sentinel privacy tests.
- Modify `eval/live_rag_answer_eval.py`: opt-in shadow-policy integration using the same live capture and existing safe preflight.
- Modify `tests/eval/test_live_rag_answer_eval.py`: opt-in isolation, one-call behavior, mode parity, gate, and privacy integration tests.
- Generate only `live_rag_policy_investigation.json` and `live_rag_policy_investigation.md` during the separately approved live run.
- Update `README.md`, `docs/COURSE_ALIGNED_PLAN.md`, and `docs/PROJECT_STATUS.md` only after an interpretable approved run. This includes replacing README's stale instruction to diagnose vector/hybrid failures separately; that diagnosis is already complete.
- Do not modify `src/oilfield_chemical_copilot/rag/service.py`, `src/oilfield_chemical_copilot/rag/prompt_builder.py`, `src/oilfield_chemical_copilot/rag/models.py`, retrieval/storage/indexing modules, public fixtures, or their production tests.

---

### Task 1: Pure Claim-Scope Policy

**Recommended model:** Luna (`gpt-5.6-luna`) because this is isolated two-file deterministic work.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** targeted implementation.
- **Brief:** `docs/superpowers/plans/2026-08-08-abstention-citation-policy-investigation.md`, Task 1.
- **Scope:** Create only `src/oilfield_chemical_copilot/evaluation/abstention_policy.py` and `tests/evaluation/test_abstention_policy.py`. Do not touch the RAG service, prompt, runner, dataset, or reports.
- **Validation:** `uv run pytest tests/evaluation/test_abstention_policy.py -v` and `uv run ruff check src/oilfield_chemical_copilot/evaluation/abstention_policy.py tests/evaluation/test_abstention_policy.py`.
- **Report:** `.codex/reports/abstention-policy-task-1.md`; include only status, test counts, closed category names, and coarse concerns.
- **Return:** status, changed files, validation results, proof that only question text enters the policy, and concerns.

**Interfaces and tests**

- Define closed `ClaimScopeCategory = Literal["general_review", "site_specific_determination", "field_ready_prescription", "complete_input_substitution"]` and `PolicyAction = Literal["allow", "abstain"]`.
- Define immutable `AbstentionPolicyDecision(action, category)` with the invariant that `general_review` maps only to `allow` and every other category maps only to `abstain`.
- Define `classify_claim_scope(question: str) -> AbstentionPolicyDecision`. Its signature is the enforcement boundary: no case object, IDs, labels, retrieval state, or answer material.
- Normalize case and whitespace without retaining raw input. Apply precedence `complete_input_substitution` -> `field_ready_prescription` -> `site_specific_determination` -> `general_review` so a more specific safety boundary wins.
- Test every closed category with multiple synthetic paraphrases, punctuation/case variants, and near-miss general-review questions. Explicitly include general review, initial screen, indicators-to-check, and inputs-that-frame-a-review as allow cases.
- Test malformed/blank/non-string input rejection with one generic error message containing no input value.
- Test deterministic repeated calls and introspect the public signature to prove evaluation labels and retrieval fields cannot be supplied.
- Test the frozen 12 questions as a compatibility characterization only: six `allow`, six `abstain`; do not alter or copy the dataset into a new fixture.

### Task 2: Counterfactual Scorer and Aggregate-Only Report

**Recommended model:** Luna (`gpt-5.6-luna`) because the scorer/writer can be developed as a pure two-file unit behind Task 1's fixed interface.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** targeted implementation.
- **Brief:** `docs/superpowers/plans/2026-08-08-abstention-citation-policy-investigation.md`, Task 2.
- **Scope:** Create only `src/oilfield_chemical_copilot/evaluation/live_rag_policy.py` and `tests/evaluation/test_live_rag_policy.py`; consume Task 1 and existing answer-evaluation types without modifying them.
- **Validation:** `uv run pytest tests/evaluation/test_live_rag_policy.py tests/evaluation/test_abstention_policy.py -v` and `uv run ruff check src/oilfield_chemical_copilot/evaluation/live_rag_policy.py tests/evaluation/test_live_rag_policy.py`.
- **Report:** `.codex/reports/abstention-policy-task-2.md`; include aggregate fake metrics only and no sentinels or runtime values.
- **Return:** status, changed files, validation results, exact report schema, reconciliation evidence, privacy evidence, and concerns.

**Interfaces and tests**

- Define a report-safe aggregate model containing only question count, decision-category counts, control citation/abstention pass/fail counts, and shadow citation/abstention pass/fail counts.
- Define a runtime-only scorer that receives a previously computed `AbstentionPolicyDecision`, the corresponding frozen evaluation case, and the existing `LiveAnswerCapture`. It must never call the service, generator, retriever, or judge.
- For `abstain`, evaluate the counterfactual as no citations plus `abstained=True`; for `allow`, evaluate the existing capture unchanged. Reuse `evaluate_answer` rather than duplicating citation/abstention semantics.
- Require exact one-to-one case/capture/decision pairing and reconcile the control result with the existing deterministic result before aggregation. Matching totals may not hide swapped cases.
- Require exactly `vector` and `hybrid`, identical ordered policy-decision counts, 12 cases per mode, and an explicit `verified_preflight` boolean. Never infer preflight success from metrics.
- Write only `live_rag_policy_investigation.json` and `.md` with safe existing provenance, `baseline_reproduced`, policy name/version, and aggregate metrics. Use closed schema/category allowlists and generic errors.
- Add tests proving the selected policy's expected counterfactual on a synthetic 12-case aggregate: abstention `12 pass / 0 fail`, citation `10 pass / 2 fail`, decisions `6 allow / 6 abstain`, with the two answer-path citation failures unchanged.
- Add negative tests for an over-abstention, a missed abstention, mode decision mismatch, swapped pairings, unknown categories, unsafe provenance, non-boolean preflight, baseline drift, and malformed counts.
- Inject unique sentinels into every runtime-only field and prove byte-for-byte absence from JSON, Markdown, exceptions, and report models.

### Task 3: Opt-In Integration With the Existing Live Runner

**Recommended model:** Terra (`gpt-5.6-terra`) because this joins policy, live capture, deterministic evaluation, preflight, and two retrieval pipelines.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** integration implementation.
- **Brief:** `docs/superpowers/plans/2026-08-08-abstention-citation-policy-investigation.md`, Task 3.
- **Scope:** Modify only `eval/live_rag_answer_eval.py` and `tests/eval/test_live_rag_answer_eval.py`; consume Tasks 1-2 without changing production RAG/retrieval modules or the canonical dataset.
- **Validation:** `uv run pytest tests/eval/test_live_rag_answer_eval.py tests/evaluation/test_abstention_policy.py tests/evaluation/test_live_rag_policy.py tests/evaluation/test_live_rag.py tests/evaluation/test_live_rag_diagnosis.py -v`; `uv run ruff check eval/live_rag_answer_eval.py src/oilfield_chemical_copilot/evaluation/abstention_policy.py src/oilfield_chemical_copilot/evaluation/live_rag_policy.py tests/eval/test_live_rag_answer_eval.py tests/evaluation/test_abstention_policy.py tests/evaluation/test_live_rag_policy.py`; then `uv run pytest -q` and `uv run ruff check .`.
- **Report:** `.codex/reports/abstention-policy-task-3.md`; include fake aggregate counts, output filenames, and privacy assertions only.
- **Return:** status, changed files, validation results, one-call proof, preflight/gate evidence, privacy evidence, and concerns.

**Integration behavior and tests**

- Add an explicit opt-in CLI value such as `--abstention-policy-shadow claim_scope_v1`; without it, preserve the runner's existing behavior and output set exactly.
- Compute each policy decision from `case.question` alone. Only afterward may the evaluation scorer receive the case labels and capture.
- Continue calling `capture_live_answer` exactly once per case per mode. Reuse that capture for the existing control report, diagnosis, and policy counterfactual; never rerun retrieval or generation for the policy.
- Preserve canonical-dataset, public-manifest, local-provider/model, temperature-zero, no-topic-filter, and safe-provenance preflight ordering. Pass `verified_preflight=True` only after every existing check succeeds.
- Require control baseline reproduction before writing an interpretable policy report. On drift, write at most safe aggregate drift status and do not claim support for the hypothesis.
- Prove both modes receive identical questions/settings and identical policy decisions; only `retrieval_mode` differs.
- Prove case IDs, labels, allowed/retrieved/cited IDs, scores, answer/evidence text, and judge output never enter the classifier call.
- Prove failed preflight rejects before runtime construction/output, non-opt-in execution creates no policy report, and all four report files retain their existing privacy guarantees.
- Stop and report unrelated full-suite failures rather than broadening scope.

### Task 4: Separately Approved Live Public Run and Learning Record

**Recommended model:** Terra (`gpt-5.6-terra`) because this task owns local environment validation, the gated run, privacy review, and coordinated source-of-truth updates.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** live validation and documentation.
- **Brief:** `docs/superpowers/plans/2026-08-08-abstention-citation-policy-investigation.md`, Task 4, only after the user approves the live-run gate.
- **Scope:** Run the opt-in evaluation against the canonical public-only baseline; generate gitignored reports under `data/processed/evaluation/live_rag`; if interpretable, modify only `README.md`, `docs/COURSE_ALIGNED_PLAN.md`, and `docs/PROJECT_STATUS.md` with sanitized aggregate findings and the next gate.
- **Validation:** exact public-manifest/configuration preflight; exact live command below; JSON/Markdown schema and forbidden-field scan; focused tests; `uv run pytest -q`; `uv run ruff check .`; `git status --short`.
- **Report:** `.codex/reports/abstention-policy-task-4.md`; include only baseline status, aggregate control/shadow metrics, privacy/validation results, and next-branch recommendation.
- **Return:** status, generated report names, changed documentation files, aggregate metrics, acceptance decision, validation/privacy evidence, risks, and next approval gate.

After presenting this command and receiving explicit approval, run:

```powershell
$env:LLM_PROVIDER = "ollama"
$env:ANSWER_EVAL_JUDGE_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "granite4.1:8b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:EMBEDDING_PROVIDER = "ollama"
$env:OLLAMA_EMBEDDING_MODEL = "granite-embedding:latest"
uv run python eval/live_rag_answer_eval.py --dataset eval/public_answer_evaluation.jsonl --output-dir data/processed/evaluation/live_rag --database-url $env:DATABASE_URL --abstention-policy-shadow claim_scope_v1
```

- Verify the public manifest, dataset/corpus hashes, local providers/models, settings, and control baseline before interpreting shadow metrics.
- Parse every output against its exact schema and scan without printing forbidden values. Confirm generated outputs remain gitignored and unstaged.
- If the baseline drifts, mode decisions differ, privacy/schema checks fail, or any case is unpaired, stop without documentation claims or a next behavior recommendation.
- If interpretable, update the three source-of-truth docs with aggregates, limitations, and one next gate. Replace README's stale future step about diagnosing vector/hybrid failures separately with the actual reviewed next branch; do not add unrelated README changes.

### Task 5: Final Milestone Review

**Recommended model:** Sol (`gpt-5.6-sol`) because Sol is reserved for read-only final milestone review.

**Assignment packet**

- **Project:** Oilfield Chemical Troubleshooting Copilot.
- **Task class:** final milestone review.
- **Brief:** Review this plan, Tasks 1-4 reports, scoped diffs, and aggregate policy report only.
- **Scope:** Read-only review of the two new evaluation modules/tests, the live-run integration/test, generated aggregate schemas, and the three conditionally updated docs. Do not inspect private/runtime content or edit files.
- **Validation:** Reconcile every requirement to tests/results; verify baseline and candidate arithmetic, classifier input isolation, schemas, privacy scans, frozen-scope evidence, and no production behavior change.
- **Report:** `.codex/reports/abstention-policy-final-review.md`; aggregate findings only.
- **Return:** approve/block status, requirement gaps, metric reconciliation, privacy findings, validation evidence, risks, and exactly one proposed next gate. Do not implement fixes.

## Acceptance Criteria

- The control rerun passes the exact public-only/local preflight and reproduces, in both modes, 12 questions, citation `4 pass / 8 fail`, abstention `6 pass / 6 fail`, and the approved failure totals.
- The policy receives only question text, produces the same ordered decisions in both modes, and receives no evaluation oracle or runtime retrieval/generation material.
- Each case is captured once per mode; shadow scoring causes no second retrieval, generation, or judge call.
- The policy produces exactly six `allow` and six `abstain` decisions per mode.
- Shadow abstention is `12 pass / 0 fail` per mode: all six abstention-required cases are caught and none of the six sufficient-evidence cases is suppressed.
- Shadow citations are `10 pass / 2 fail` per mode: the six unexpected citations disappear, while the existing one allowed-retrieved-not-cited and one mixed-with-disallowed answer-path failure remain visible and unchanged.
- No prompt, draft schema, service, retrieval, threshold, index, embedding/model, dataset/case, allowed-evidence list, or production test changes occur.
- Reports and errors contain only the approved aggregate schema and safe provenance. Sentinel tests and the live scan prove forbidden data absent; generated reports remain gitignored and unstaged.
- Focused tests, full pytest, and Ruff pass, or unrelated pre-existing failures are reported without scope expansion.
- A passing fixture result is described as support for a bounded hypothesis, not proof of generalization or permission to deploy.

## Next Decision Branches

- **All acceptance criteria pass:** approve a separate production-hardening plan. Before deployment, require independently approved holdout/paraphrase evaluation and a service-boundary design; then separately investigate the two citation-selection failures with retrieval still frozen.
- **Under-abstention improves but any sufficient case is suppressed:** reject deployment and approve one bounded policy-rule refinement; keep prompt/retrieval/model/cases frozen for that refinement.
- **One or more abstention-required cases remain allowed:** falsify `claim_scope_v1`; next consider a structured generator refusal/schema investigation, still with retrieval frozen.
- **Policy passes but appears tied to fixture wording:** approve a new holdout-case design as a separate evaluation-change task; do not weaken the frozen-case rule inside this experiment.
- **Control baseline drifts, modes disagree on decisions, pairing fails, or privacy/schema validation fails:** stop at validity repair; make no policy, prompt, citation, or retrieval recommendation.
- **After abstention is independently validated:** choose whether to investigate the two remaining citation-selection failures through a citation-contract/generator study. Retrieval tuning remains unsupported unless new evidence identifies a retrieval gap.

## Risk Notes

- The public fixture has six paired general/overclaim questions and is small enough for lexical overfitting. Passing it is necessary for this hypothesis, not sufficient for production.
- A question-only policy is explainable but may miss implicit overclaims or reject benign wording. The separate holdout gate prevents silent deployment from fixture fit.
- Shadow counterfactual scoring proves metric effects under the frozen captures; it does not prove production integration, latency, user experience, or chemistry safety.
- Temperature `0` does not guarantee model reproducibility. The control baseline gate prevents interpreting policy metrics after runtime drift.
- The two citation-selection failures are deliberately not repaired here. Combining them with abstention work would hide whether the selected policy actually solved the upstream decision error.
