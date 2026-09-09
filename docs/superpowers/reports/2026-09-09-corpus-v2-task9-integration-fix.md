# Task 9 integration fixes

Scope: synthetic code and tests in the isolated Corpus V2 worktree. No main checkout, private corpus, live service, database connection, network operation, or commit.

## Sealed acquisition identity

Acquisition now compares the full approved entry digest with the ledger's sealed source register before metadata, download, or resume. Batch acquisition authenticates every entry before the first client factory call. A changed Drive identifier, revision, MIME type, export policy, or steward binding under an unchanged document pseudonym fails closed with a fixed error code. The ledger refuses digest lookup before REGISTERED completion.

## Processing identity integration

Processing and typed chunk validation share one SHA256 derivation over release, source pseudonym, snapshot hash, parser policy, chunk policy, location, ordinal, and text hash. `CorpusV2Chunk.from_loaded_chunk` authenticates processing output and retains its original SHA256 identifier. A frozen private provenance record enables downstream revalidation. Existing pseudonym fixtures retain their exact source/ordinal checks.

Canonical JSONL accepts authenticated typed chunks and emits only the five public chunk fields. It never serializes private provenance or sheet labels. Raw hash-ID chunk mappings without provenance remain rejected; callers should retain/reconstruct the typed record from authenticated processing output before creating this public projection. Embedding records and the candidate store retain the original SHA256 identifier. The store also checks source bytes, location, release, text hash, character count, vector identity, and exact chunk set.

## Verification

- `python -m pytest tests/corpus_v2 -q --basetemp .t9b -p no:cacheprovider`: 278 passed, 1 skipped (Windows symlink availability).
- `python -m ruff check src/oilfield_chemical_copilot/corpus_v2 tests/corpus_v2 --no-cache`: passed.
- New tests cover changed approved entries before all client actions, preflight of the entire batch, acquisition/resume protection, processing → typed canonical manifest → embedding → candidate store → index validation, private sheet-label exclusion, and mutation of every provenance dimension.

Environment notes: the first sandboxed run could not create pytest temporary lock files. Authorized elevated synthetic checks ran successfully with short worktree-local temporary roots. An initial long temporary root caused two register initialization failures due to Windows path length; rerunning with short roots passed all tests. These temporary directories contain only synthetic test output. All changes remain uncommitted for the final integrated review.
