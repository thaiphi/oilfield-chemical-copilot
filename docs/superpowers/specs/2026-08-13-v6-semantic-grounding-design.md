# V6 Semantic-Grounding Evaluation Design

## Purpose

V6 is a fresh, private, one-shot evaluation of the formatter corrections made after V5. It measures the real production `format_answer()` boundary; it is not a V5 retry and it does not inspect or reuse V5 case content.

## Scope

- Harden recursive aggregate-report privacy validation and extend the closed evaluator approval scope with `semantic-grounding-v6` before any private V6 approval is created.
- Freeze the public formatter and evaluator after their focused tests pass.
- Create a fresh private 36-case fixture under `.private/evaluation/semantic_grounding_v6/`.
- Emit one aggregate-safe report at `docs/superpowers/reports/2026-08-13-semantic-grounding-v6.json`.
- Do not call retrieval, a database, Ollama, OpenAI, Docker, or a live RAG path.

## Holdout Design

The fixture has six cases in each existing category: `exact_value`, `range_bound`, `unit`, `qualifier_condition`, `conflicting_evidence`, and `no_established_threshold`. Each category contains three expected allows and three expected fallbacks.

V6 uses new synthetic wording, values, and question identities. The controller compares normalized V6 question identities with V1 through V5 before sealing. It also keeps V6 wording distinct from public formatter regression tests, so the holdout remains an independent measurement.

The private author deliberately covers the V5 aggregate failure classes without exposing their case text: changed comparison meaning, unsupported unit substitution, omitted explicit conditions, conflict erasure in one or multiple sources, and an explicitly grounded absent-threshold conclusion. The reviewer checks outcome, category, and clarity without reading formatter implementation rules.

## Execution Boundary

The author, reviewer, sealer, evaluator, and diagnostics use the controller-owned local directory:

```text
.private/evaluation/semantic_grounding_v6/
  draft/
  review/
  sealed/
  approval/
  state/
  results/
```

All V6 private artifacts remain Git-ignored. Public files contain aggregate counts, category totals, failure-class totals, and integrity gates only. Before writing a report, the evaluator recursively rejects unsafe keys and string values in the complete aggregate payload. Public files never contain a prompt, answer, excerpt, source, path, URL, identifier, or raw error.

## One-Shot Contract

After public tests pass, the controller seals the reviewed fixture, creates a digest-bound V6 approval, and runs preflight. The evaluator may score the fixture exactly once after an explicit user approval immediately before `evaluate_once`. That immediate approval is a conversation gate, not a second file: it applies only to the sealed fixture and formatter/evaluator digests verified by the immediately preceding preflight. Any artifact change requires a new preflight and a new immediate approval. The evaluator atomically records consumption before evaluating. V6 is never rerun; a failed V6 requires a new V7 fixture.

## Acceptance Rules

- **Safety pass:** `false_allows == 0`.
- **Full acceptance:** `false_allows == 0` and `false_fallbacks == 0`.
- The report must confirm the real formatter was called, the fixture was sealed, approval digests matched, and the one-shot state was consumed.

If V6 has false allows, preserve it, add distinct public regressions, make the smallest test-first correction, and design V7. If it has only false fallbacks, preserve it, diagnose the conservative behavior, add a distinct public regression only when justified, and design V7 before another measurement. No result authorizes a V6 rerun.

## Verification

Before scoring: focused formatter and evaluator tests, Ruff, private-ignore check, balanced-case validation, V1-V5 overlap check, sealing, and preflight. After scoring: verify the aggregate-report key allowlist, rerun all public tests and lint, run workflow-contract tests, run `git diff --check`, and confirm no V6 private material is staged.

## Non-Goals

V6 does not prove chemistry correctness, retrieval quality, LLM-answer quality, production readiness, or the full Module 1 curriculum. It measures only whether the formatter preserves the semantic grounding represented by this controlled holdout.
