# Sol Review Coordination Design

## Purpose

Make bounded code reviews return a useful verdict reliably by separating evidence collection from final review judgment. This design applies only to the Codex agent workflow. It does not change application or RAG behavior.

## Problem

The existing Sol reviewer profile assigned Sol final-review responsibility while prohibiting subagent coordination. Broad review assignments consequently required Sol to collect implementation, test, privacy, evaluation-integrity, and plan-conformance evidence itself before it could synthesize a verdict. A bounded review could therefore expire without a useful result.

## Design

Terra remains the workflow owner. For a small review of one or two tightly related files, Sol reviews directly. For a medium or large review with independent evidence streams, Terra dispatches bounded Luna exploration tasks first and gives Sol their compact evidence packets.

Luna tasks are read-only and limited to named files and concrete questions. Luna returns inspected paths, line-level evidence, findings, and unanswered questions; it does not expand scope, modify files, or decide the final verdict.

Sol is the review lead. It judges the supplied evidence, verifies only dependencies needed to resolve a concrete concern, reconciles conflicts, and returns the required structured verdict. Sol does not recursively explore broad history, run broad validation unless required, or fix code during a review.

## Return Contract

Every Sol review returns Critical, Important, and Minor findings; evidence reviewed; privacy, test-coverage, and plan/spec verdicts; and one overall verdict. The allowed overall verdicts are `APPROVE`, `APPROVE WITH MINOR ISSUES`, `CHANGES REQUIRED`, and `REVIEW_INCOMPLETE`.

When time or evidence is insufficient, Sol returns `REVIEW_INCOMPLETE` with what it reviewed, what remains unresolved, and the exact reason further investigation is needed. It returns available findings rather than continuing until timeout.

## Scope Rules

1. Start with only the explicitly listed files.
2. Follow a dependency only to verify a concrete finding.
3. Do not read broad historical documentation unless the task explicitly requires it.
4. Do not run the full suite unless assigned.
5. Do not edit, fix, commit, or push.
6. Do not delegate automatically: use Luna only when independent evidence streams justify it.

## Validation

Workflow-contract tests will assert Sol's review-lead role, bounded delegation, read-only behavior, return fields, `REVIEW_INCOMPLETE`, and the small/medium/large routing rule. Hook tests will reject reviewer packets missing a structured return contract or containing broad scope entries.
