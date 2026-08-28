# E1a-4 Correction Application And Sampling-Frame Design

**Date:** 2026-08-27

**Status:** Approved design; implementation not yet started

## Objective

Apply the independently verified core and Iron Sulfide supplement correction proposals to a new E1a-4-only, versioned role-mapping artifact, then seal the exact 96-slot metadata sampling frame from that artifact.

This stage stops before question authoring, canonical-claim authoring, retrieval, model calls, Qdrant access, ingestion, reindexing, or production behavior changes.

## Current Evidence

- The active reconciliation seal verifies.
- The core foundational correction proposal has a verified v2 seal.
- The Iron Sulfide supplement correction proposal has a verified v2 seal and passed independent review.
- Their combined no-write projection satisfies all eight topic/role strata and the exact 96-slot allocator.
- The frozen E1a-3 allocation remains the authoritative prior-locator exclusion set.
- The earlier E1a-4 Task 2 implementation exists only as reviewed, untracked workspace files and must be recovered into the isolated branch before it can be executed or finalized.

## Selected Approach

Create a new sealed E1a-4 role mapping rather than modifying the production index, Qdrant, the reconciliation inventory, or the frozen E1a-3 role configuration.

The role mapping is a materialized projection of the authenticated reconciliation inventory after applying only the current terminal promotions from the two verified correction proposals. The sampling-frame sealer consumes this mapping directly, verifies the live index contract separately, excludes the exact sealed E1a-3 allocation, and emits the deterministic E1a-4 source register and 96-slot allocation.

This approach is selected because it:

- follows the reconciliation specification's requirement that E1a-4 corrections become a new versioned artifact;
- preserves E1a-3 and production state;
- supports locator-level mixed roles within one source;
- creates a restart-safe trust chain from reconciliation through allocation; and
- avoids treating a no-write projection as if it were already applied.

## Rejected Approaches

### Mutate `index_locators`

Rejected because the active reconciliation seal authenticates the current inventory. Updating those rows would invalidate the audited snapshot and blur evidence from applied policy.

### Rewrite The E1a-3 Role Configuration

Rejected because E1a-3 is frozen observed evidence. Its role configuration and allocation must remain immutable.

### Update The Production Database Or Qdrant

Rejected because role correction is evaluation metadata, not a corpus or retrieval change. This stage has no authority to modify production storage.

## Private Artifact Layout

The mapping application publishes exactly four files beneath the ignored private reconciliation boundary:

```text
e1a4-role-mapping/v1/sealed/
  role-mapping.v1.json
  role-mapping.v1.json.sha256
  mapping-binding.v1.json
  mapping-binding.v1.json.sha256
```

The sampling-frame sealer publishes exactly four files beneath the ignored E1a-4 evaluation boundary:

```text
e1a4/sealed/source-register.v1.json
e1a4/manifests/source-register.v1.sha256
e1a4/sealed/sampling-allocation.v1.json
e1a4/manifests/sampling-allocation.v1.sha256
```

Both artifact sets use canonical JSON, SHA-256 manifests, single-directory atomic publication, exact-file-set verification, and no overwrite. An existing complete artifact is accepted only after exact no-write verification against current authenticated state.

## Role-Mapping Payload

`role-mapping.v1.json` has exactly:

```json
{
  "schema_version": 1,
  "sources": [
    {
      "source_id": "private",
      "topic": "iron_sulfide|scale|corrosion|paraffin",
      "source_role": "foundational|supporting",
      "parser_type": "private",
      "locators": ["private"]
    }
  ]
}
```

Records are sorted by source ID, topic, role, and parser type. Locators are non-empty, unique, normalized, and sorted. The same source may appear in both foundational and supporting records when only some locators are promoted.

The payload contains only locators that are:

- available to E1a-4;
- unused by E1a-3;
- substantive in the authenticated inventory or promoted to substantive by the core proposal; and
- assigned their authenticated current role or changed to foundational by an exact verified promotion.

Retained, unresolved, out-of-prefix, unavailable, E1a-3-used, malformed, and non-substantive locators never enter the mapping.

## Mapping Binding

`mapping-binding.v1.json` binds exactly:

- schema version and reconciliation run ID;
- active reconciliation binding digest;
- core v2 correction binding digest;
- supplement v2 correction binding digest;
- frozen E1a-3 allocation payload digest;
- mapping payload digest;
- mapping source-record count and unique locator count;
- exact eight-stratum locator counts;
- exact allocator availability and slot count; and
- the fixed rule that E1a-3 locators are excluded before allocation.

The binding must not contain raw paths, questions, document text, claims, or model output.

## Application Contract

Before producing or verifying the mapping, the controller must:

1. Verify the active reconciliation snapshots against the supplied binding trust anchor.
2. Require the core and supplement audit stores to use the identical resolved reconciliation SQLite database and run ID.
3. Verify both v2 correction seals against supplied binding trust anchors.
4. Require both proposals to bind the same active reconciliation snapshot.
5. Verify the frozen E1a-3 allocation payload and manifest, then require its locator set to equal the reconciliation inventory's `e1a3_used` set.
6. Reject overlapping core and supplement promotions.
7. Require every promoted key to exist in the frozen candidate inventory with the expected pre-application role and substantive status.
8. Apply only current terminal `PROMOTE_FOUNDATIONAL` decisions in memory.
9. Build the complete E1a-4 mapping, run the exact capacity and 96-slot allocator checks, and require all eight strata sufficient with exactly 96 allocations.
10. Publish atomically without updating reconciliation, PostgreSQL, Qdrant, or any E1a-3 artifact.

Any mismatch fails closed with a sanitized E1a-4 mapping error code and no partial artifact.

## Sampling-Frame Contract

The E1a-4 sampling-frame sealer must not reuse the E1a-3 source-role configuration as its role authority. It must:

1. Run presence-only preflight without opening private payloads.
2. Verify the role-mapping artifact and its complete proposal/reconciliation trust chain.
3. Verify the approved live index contract in read-only mode.
4. Parse mapping sources with exact fields and reject duplicate or overlapping locator-role records.
5. Reverify the frozen E1a-3 allocation and reject any mapped locator reuse.
6. Build the fixed 96-slot grid and call the deterministic allocator exactly once.
7. Require exact 96-slot identity, four-topic coverage, two-role coverage, and unique source-locator allocation keys.
8. Bind the source register to the mapping binding, index contract, and E1a-3 allocation digests.
9. Bind the allocation to the source-register digest and fixed slot count.
10. Publish the source register and allocation atomically, then verify all four files without writing.

The public output is limited to status, source-record count, eight sufficient strata, and slot count.

## Recovery And Idempotence

- Mapping decisions already persist in reconciliation SQLite and are not replayed or rewritten.
- A crash before the atomic rename leaves no visible mapping seal; stale staging directories are removed before retry.
- A crash before the sampling-frame atomic publish leaves no visible sampling frame.
- A complete existing seal is verified and returned; it is never overwritten.
- A partial, extra-file, stale-manifest, or current-state mismatch blocks the run.

## Privacy Boundary

Source IDs, locators, parser metadata, allocation rows, proposal decisions, paths, and digests remain private and Git-ignored. Public plans may report aggregate counts and gate outcomes only.

No sealed holdout, private question text, canonical claims, requirements, passage text, retrieval output, or model output is accessed in this stage.

## Testing Requirements

Tests must cover:

- fake or altered reconciliation, core, supplement, and E1a-3 trust anchors;
- core/supplement stores from another database or reconciliation snapshot;
- overlapping or provenance-invalid promotions;
- exact locator-level mixed-role preservation;
- exclusion of E1a-3, unavailable, retained, and non-substantive locators;
- insufficient stratum and unavailable allocator rejection;
- canonical four-file mapping publication, crash cleanup, idempotent verification, extra files, and stale manifests;
- sampling-frame rejection of missing or altered mapping artifacts;
- exact 96-slot deterministic allocation and no locator reuse;
- atomic frame publication and aggregate-only output; and
- unchanged reconciliation locator rows, PostgreSQL data, Qdrant, E1a-3 artifacts, and tracked private-file set.

## Completion Boundary

This design is complete when the versioned role mapping and metadata-only 96-slot sampling frame are sealed and independently verified, public plans contain aggregates only, all focused and full tests pass, and independent review approves the implementation.

Question and canonical-claim authoring remain a separate, later Task 2 step requiring its own execution checkpoint.
