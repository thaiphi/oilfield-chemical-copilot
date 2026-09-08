# Corpus V2 release and runtime contract verification

Task 7 review fixes remain uncommitted in the isolated design worktree. Verification
uses synthetic fixtures only. No private document processing, network, database
connection, backup/restore, or promotion occurred.

Task 6 now seals and validates a canonical, typed index contract shared with Task 7.
The integration fixture seals a complete Task 6 publication and passes its exact
binding and index bytes into the runtime consumer. Arbitrary index-contract bytes
are rejected. The contract binds the release, database identity, embedding identity,
index manifest, source register, and index counts.

Legacy selection requires a separate authenticated legacy publication. A candidate
binding cannot authenticate legacy mode. Its separate injected database reader checks
the explicit 198-source / 4,797-chunk baseline, target, model, vector dimension, and
read-only transaction. It never queries V2 release metadata.

V2 verification independently queries the database identity, read-only status,
repeatable-read isolation, release metadata, vector column type, ready and valid
HNSW cosine index, and grouped indexed-row contract. It retains the transaction
through retrieval operations and closes it on explicit close, owner disposal, or
retrieval failure. No adapter is installed implicitly; both modes fail closed when
their reader is absent. These synthetic protocol tests do not certify a live adapter.

Red/green evidence:

- The initial Task 6 integration run failed because the shared contract API was
  absent. After implementing it, all 29 release tests passed.
- Runtime regression runs exposed missing separate legacy authentication/reader,
  isolation verification, vector-index verification, and source-register binding.
  The final owned runtime, release, and storage run passed all 59 tests.
- The broader suite initially encountered two Windows path-length failures in
  register fixtures. A short temporary path resolved them: all seven register tests
  passed. The subsequent broad run passed 233 tests with one platform skip, and
  detected the concurrently added app endpoint-cache regression. App integration
  verification is recorded separately after that fix lands.
- Ruff passed for the release/runtime modules and Corpus V2 tests.

Operational status is unchanged: no V2 corpus has been built. The legacy baseline
above is a required synthetic contract, not a fresh operational measurement.
