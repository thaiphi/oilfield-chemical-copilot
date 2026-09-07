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

## Remaining Milestone 1 Work

1. Review the deferred historical reports individually before adding any to the maintained record.
2. Obtain normal code review for the recovered dependency closure before integration.
3. Verify the maintained revision from a clean checkout and record the resulting evidence map.

The completion plan remains open. No completed private reconciliation or locator audit needs to be repeated.
