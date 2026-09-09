# Task 7 synthetic runtime verification

Base implementation: `e910e87` (explicit release selection).
First review correction: `4a327b179a668b938899892ba36d25d7785e429e`.
This report describes a second review correction, currently uncommitted in the
isolated `corpus-v2-rebuild-design` worktree. It does not claim a new commit.

The second review exposed six accepted, structurally valid false index contracts:
incorrect source count, chunk count, embedding provider/model, database identity,
and manifest digest. A dimension mismatch was already rejected. Canonical chunk
and embedding evidence is now reconciled with the indexed ledger dispositions;
the manifest digest is the SHA-256 of canonical chunk evidence. A new required
`index_validation` private receipt binds database identity and both artifact
digests. Producers must authenticate the actual transport before creating that
receipt. These synthetic checks establish consistency, not live provenance.

A same-name database on an unverified server was accepted in the red regression.
Runtime sessions now require `verify_connection_target(database_url)` to return
exactly True. The protocol requires authentication of actual server, port,
database, principal and required TLS identity; echoing the input does not comply.
The read-only retained session still independently queries database facts. No
production adapter is installed or certified by these tests.

The OpenAI missing-key regression reached the resource cache before the fix.
Generator settings now validate that key before each lookup and pass the captured
configuration explicitly to the builder. Warm-cache rotation creates a distinct
resource; removal rejects even with a populated cache. Cache hashing receives
fingerprints of complete settings and release identity, never raw credentials.
Changing the environment after validation does not change the constructed client.

Recorded red/green evidence:

- Structured false contracts: six expected failures and one already-passing
  dimension rejection, followed by green release tests.
- Transport mismatch: one expected failure followed by green runtime tests.
- Missing OpenAI key: cache-reached failure followed by green app tests.
- Focused release/runtime/app verification: 93 passed.
- Broader Corpus V2, app and generator-factory verification: 243 passed, 1 skipped.
- Final run including storage regression tests: 247 passed, 1 skipped in 14.22s:
  `.venv/Scripts/python.exe -m pytest tests/corpus_v2 tests/app tests/rag/test_generator_factory.py tests/storage/test_pgvector.py -q -p no:cacheprovider --basetemp=.f2final`.
- Ruff checks on changed modules and tests: passed.

All fixtures are synthetic. No private parsing, network access, live database
connection, backup/restore, promotion or legacy mutation occurred. Temporary test
fixtures required sandbox escalation within the authorized isolated worktree.
