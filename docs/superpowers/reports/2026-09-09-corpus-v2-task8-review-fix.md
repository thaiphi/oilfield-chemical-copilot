# Task 8 review correction

These corrections are uncommitted at the time of verification. The parent controller
must update this status after committing and receiving independent review.

Owner promotion now checks the exact canonical recommendation schema, binds its
index/spec/result context to the persisted artifacts, reevaluates readiness, and
reauthenticates every sealed gate. Missing gate authentication fails closed before
an owner-action record can be published.

The frozen safety policy includes the no-answer membership, expected denominator,
cohort provenance, control artifact digest, control safety, and a no-decrease rule.
An injected verifier must authenticate the cohort and control evidence before the
specification can be frozen. Results bind the same membership digest, count and
provenance. The absolute safety floor is 0.95; the frozen control safety may require
a stronger result. A candidate at 0.97 against a control at 1.0 is blocked even when
the absolute threshold is 0.95. These are synthetic contract values, not measured
candidate performance.

Verification:

- Initial expanded test run: 25 failed, 26 passed. Failures were caused by the
  old strict schema rejecting the newly required safety specification fields;
  this run is schema red evidence, not an isolated reproduction of each exploit.
- Updated evaluation tests: 54 passed, including forged recommendation rejection,
  altered cohort/denominator/provenance rejection, baseline regression, missing
  safety authentication, and exact owner gate context authentication.
- Ruff with its cache disabled passed for all three changed Python files.
- Git whitespace checks passed.
- Release tests could not finish under the sandbox: synthetic temporary-directory
  access failed with WinError 5. The parent controller must run those tests and
  broader required checks in its authorized test environment before completion.

No live evaluation, private corpus operation, external service, promotion, or
commit was performed by this correction worker.
