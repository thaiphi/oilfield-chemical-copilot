# V5 Semantic-Grounding Evaluation Design

## Goal

Measure the corrected production `format_answer()` semantic-grounding boundary once with a fresh private holdout. V5 is a new measurement, not a retry or repair of V4.

## Scope

V5 evaluates whether the formatter preserves the meaning of cited evidence for numbers, ranges, units, qualifiers, conflicting evidence, and statements that lack an established threshold. It does not evaluate retrieval, source selection, answer generation, chemical correctness, or field-treatment safety.

## Fixture Design

The private fixture contains exactly 36 synthetic cases:

| Category | Allow | Fallback | Total |
| --- | ---: | ---: | ---: |
| Exact value | 3 | 3 | 6 |
| Range or bound | 3 | 3 | 6 |
| Unit | 3 | 3 | 6 |
| Qualifier or condition | 3 | 3 | 6 |
| Conflicting evidence | 3 | 3 | 6 |
| No established threshold | 3 | 3 | 6 |
| Total | 18 | 18 | 36 |

Each case uses fresh private question, evidence, answer, and failure-class wording. No V1-V4 question identity, V4 fixture wording, or public regression-test wording is reused. Before sealing, normalized private question identities are compared against V1-V4 private identities. The author and reviewer identities must differ for every case, and the reviewer validates category, expected outcome, and claimed failure class without inspecting formatter logic.

## Execution Boundary

V5 reuses the hardened semantic-grounding evaluator. For every case it constructs the production `RagDraft` and `SourceEvidence` inputs and calls the real `format_answer()` exactly once. The observed outcome is derived only from the production answer's `weak_evidence` value.

The private run lives under `.private/evaluation/semantic_grounding_v5/` with draft, review, sealed, approval, state, results, and diagnostics directories. Preflight must pass before scoring: sealed-digest validity, V1-V4 overlap check, approval artifact digests, unconsumed state, private artifact paths, and public aggregate-report path. A digest-bound approval is atomically consumed for the one permitted score.

## Outcomes And Decision Rule

The durable report remains aggregate-only: status, counts, category totals, failure-class counts, and boolean integrity gates. It contains no question, answer, excerpt, source metadata, private path, identifier, or raw error.

`false_allows` are the primary safety gate. V5 passes only with zero false allows. `false_fallbacks` are separately reported to reveal over-conservative behavior. The result is evidence about the formatter boundary only and does not establish a complete RAG-system or field-safety result.

## Failure Protocol

If V5 fails, seal, approval, state, diagnostics, and aggregate report remain unchanged. Add new public synthetic regressions with wording distinct from V5, apply the smallest formatter correction, and do not rerun V5. A later independent measurement must use a new V6 fixture.

## Validation

Before authoring, run focused evaluator and formatter tests, workflow contract tests, Ruff, Git whitespace validation, and a Git-ignore check for the V5 private root. After the one score, repeat those validations and confirm that only aggregate-safe public evidence is visible to Git.

## Approval Boundary

This design authorizes no private fixture authoring, sealing, approval, scoring, formatter change, or V5 report creation. Those actions require approval of the implementation plan.
