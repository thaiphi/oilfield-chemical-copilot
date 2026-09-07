# Maintained Record Reconciliation — 2026-09-06

## Purpose

This report records the first completion-plan reconciliation checkpoint. It distinguishes the reviewed project revision from local-only work so that public plans do not claim code or reports that a reviewer cannot inspect.

## Starting State

| Item | State |
| --- | --- |
| Original checkout | `acf10fc`, with unrelated retrieval edits and a large set of untracked experiment files. It was preserved unchanged. |
| Reviewed project record | `0863f99`, the parent of this isolated completion branch. |
| Private artifacts | Not opened or copied. |
| Runtime services, Drive, index, retrieval, models | Not run. |

## Recovered Public Code

The reviewed record already contained reconciliation, private-publication, E1a-3 sampling, E1a-4 mapping, and E1a-4 sampling modules. It was missing a small public dependency closure required by the active E1a-4 plan:

- evidence-state and claim-support modules;
- requirements-evidence gate;
- E1a-4 population and balanced-selection modules;
- population sealer CLI; and
- focused requirements, population, and selection tests.

The nine recovered files were present in the original checkout as local-only code. A targeted secret/path scan found no hard-coded Drive path, user profile path, credential assignment, connection URL, or API-key marker. The recovered focused suite passed `43` tests and Ruff passed in the isolated workspace.

This checkpoint establishes code availability for review. It does not claim that questions, claims, private artifacts, evaluation results, or runtime behavior have been created or validated.

## Evidence Disposition

| Evidence class | Disposition |
| --- | --- |
| Reconciliation/mapping/frame closure | Retained as reviewed public aggregate evidence. |
| Recovered E1a-4 code and tests | Committed locally in `1cb45fe`; focused and combined verification passed. |
| Historical public-intended reports present only in the original checkout | Deferred pending individual provenance, privacy, and relevance review. They are not cited as maintained evidence. |
| Missing historical report reference from the active E1a-4 plan | Replaced with the maintained curriculum and capstone backlog. |

## Integrated Revision Verification — 2026-09-07

The reviewed recovery branch was merged as `726f9df90d6220fb782ed421d5f83c8b3c675612` (PR #3). A new isolated worktree checked out that exact revision with no working-tree changes before this record was written.

- the reviewer-document suite passed: `4 passed`;
- the full Python suite passed in bounded groups: `897 passed, 14 skipped`;
- Ruff passed and `git diff --check` was clean; and
- all active plan and evidence-map paths below were present and tracked at the verified revision.

No private artifact, Drive content, runtime service, model, index, retrieval call, or holdout was opened or run for this verification.

## Active Public Evidence Map

| Purpose | Tracked public evidence at `726f9df` |
| --- | --- |
| Product scope and acceptance boundary | `docs/CAPSTONE_FOCUS.md`, `docs/LEARNING_ROADMAP.md`, `docs/CURRICULUM_REMEDIATION_BACKLOG.md`, and the consolidated completion plan. |
| Reconciliation provenance and privacy boundary | This report and `docs/CAPSTONE_EVIDENCE.md`. |
| E1a-4 experiment contract | `docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md`. |
| Recovered evaluation dependencies | claim-support, evidence-state, requirements-gate, E1a-4 population and selection modules; E1a-4 population sealer; and their focused tests. |
| Verification entry point | `tests/capstone/test_reviewer_docs.py` plus the full tracked test suite. |

Deferred historical reports remain quarantined, are not cited as maintained evidence, and do not block M1. Review an individual report only if a later milestone proposes to add it to the maintained evidence set.

## Milestone 1 Closeout

M1 is complete: the recovered public dependency closure received normal review, was integrated, and has a clean-checkout verification and active evidence map. The completion plan remains open; the next gate is M2 private coverage review under the applicable authority. No completed private reconciliation or locator audit needs to be repeated.
