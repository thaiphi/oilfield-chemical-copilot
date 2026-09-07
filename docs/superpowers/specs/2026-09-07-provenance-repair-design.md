# Private Provenance Repair Design

**Goal:** Establish reproducible Drive-to-local-to-existing-index provenance without changing the approved corpus, vector index, retrieval path, model, prompts, or evaluation holdouts.

## Current State

The immutable private `v1` reconciliation seal accounts for 385 approved document identities. Coverage accounting is complete, but every identity is blocked because no verified linkage joins an approved Drive identity, a local file, and an existing index source. The current foundational-locator audit is complete but allocation remains unavailable. These are evidence gaps, not evidence that the collection is excluded, unreadable, or unusable.

## Scope And Constraints

- Preserve `v1` snapshots, manifests, coverage decisions, and prior review history unchanged.
- Work only from private metadata and private source bytes under approved authority. Public artifacts may contain only aggregate counts and states.
- A link is accepted only through an exact content digest, pre-existing source provenance corroborated by a digest, or a durable human owner decision with a stated evidence reason.
- Filename, size, topic, source role, or model suggestion may create a private review candidate but can never establish identity.
- Astra may independently review a small residual ambiguity set. It cannot create or apply a mapping, and its output is not provenance evidence.
- Fail closed on duplicate ownership, digest conflict, an index-contract mismatch, malformed payloads, a partial seal, or an incomplete candidate review.
- Do not run ingestion, reindexing, retrieval, generation, E1a-4, or Qdrant work in this repair.

## Design

### 1. Freeze The Existing Index Contract

Before collecting new mapping evidence, verify the existing private index contract: index fingerprint, source count, chunk count, embedding model, and dimension. Store the verified digest as the repair layer's base contract. Any mismatch blocks the repair; no fallback inventory may be substituted.

### 2. Create An Append-Only Repair Layer

Create a private, versioned provenance-repair store that binds to the verified `v1` snapshot-binding digest and the frozen index-contract digest. It records candidate evidence, accepted bindings, rejected candidates, owner decisions, and append-only supersessions. It does not edit `v1` tables or snapshots.

Each accepted binding has exactly one Drive identity, one local SHA-256, one existing index source, an evidence kind, an evidence digest, reviewer/controller identity, timestamp, and optional predecessor binding. A Drive identity, local SHA-256, or index source cannot have conflicting current bindings.

### 3. Generate Candidates Deterministically

Candidate generation is private and restart-safe. It first joins exact content digests and existing explicit source provenance. For still-unlinked records it may compute approved private source-byte hashes, but it does not use a language model, retrieval scores, semantic similarity, or filename similarity to accept a match.

Candidates are partitioned into exact, conflict, and unresolved states. Exact candidates are eligible for controller validation. Conflicts and unresolved candidates require a private owner-review packet; no coverage decision changes while they remain unresolved.

### 4. Review And Apply Proven Bindings

The controller validates cardinality, base digests, and evidence provenance in one transaction before accepting each binding. Only an accepted binding may supersede a `MAPPING_UNVERIFIED` coverage decision. A verified substantive local file joined to an unchanged existing index source can become `INDEXED_USABLE`; duplicates must resolve to an existing usable representative. All unproven records remain blocked.

### 5. Publish A New Versioned Reconciliation Seal

After all accepted/rejected/unresolved candidate states are durable, build a new private versioned seven-artifact reconciliation seal. The seal is bound to both the immutable `v1` snapshot-binding digest and the frozen index-contract digest. It verifies every artifact and manifest before publication, publishes atomically, and never overwrites the `v1` seal.

### 6. Reassess Capacity Without Changing The System

Re-run the read-only foundational-locator capacity calculation against the verified repair seal. It may reveal a narrower missing-evidence problem; it cannot authorize ingestion, reindexing, E1a-4, or application changes by itself.

## Acceptance Criteria

- `v1` verification remains valid before and after repair work.
- The repair layer rejects conflicting, unbound, malformed, duplicate, or stale mappings atomically.
- Every accepted current binding has verified exact provenance to one Drive identity, one local digest, and one existing index source.
- Existing index contract values are unchanged and reverified before sealing.
- The new versioned seal has all seven artifacts and manifests, verifies from a fresh process, and is bound to `v1` plus the index contract.
- Coverage decisions are superseded only for accepted bindings; unresolved identities remain blocked.
- Public reporting contains aggregates only and states that provenance repair is not an ingestion, reindexing, or product-acceptance result.

## Out Of Scope

This work does not repair or replace the vector index, download a new corpus, tune a model, modify retrieval, generate answers, consume holdouts, or claim the chatbot is ready for production use.
