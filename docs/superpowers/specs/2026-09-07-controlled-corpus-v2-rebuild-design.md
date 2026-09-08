# Controlled Corpus V2 Rebuild Design

**Date:** 2026-09-07
**Status:** Approved design; implementation is not authorized by this document
**Purpose:** Replace the unproven legacy corpus lineage with a controlled, versioned chatbot knowledge-base release built directly from the approved 385-document Google Drive collection.

## Decision

Google Drive is the authoritative source for the approved 385-document collection.
Each Corpus V2 release begins with a frozen, checksummed, versioned local snapshot
of exactly those approved Drive identities. Extraction, chunking, embeddings, the
candidate PGVector index, evaluation, and production promotion must be bound to
that snapshot.

The existing 198-source / 4,797-chunk PostgreSQL/PGVector corpus is a legacy
read-only release. It remains available only as the rollback baseline. Corpus V2
uses a separate versioned PostgreSQL/PGVector database; it must never share the
legacy chunks table or modify legacy rows, embeddings, schema, or configuration.

No Qdrant deployment, second vector-store product, silent fallback, mixed-release
table, or automatic promotion is part of this work.

## Verified Starting Point

The reconciliation investigation is closed with findings, not converted into a
false rebuild success claim:

- The approved source set has 385 Drive identities.
- The legacy inventory has 198 sources and 4,797 chunks.
- The reconciliation record does not establish verified Drive-to-index or
  local-file-to-index provenance for the legacy index.
- Existing legacy parsing/index artifacts describe 206 parsed inputs and 198
  indexed sources. They are not evidence of a 385-document build.

The investigation's seals, review records, and aggregate reports remain historical
evidence. Its findings must be recorded as the reason for this rebuild, while its
existing stage statuses and immutable artifacts remain unchanged.

## Scope and Non-Goals

In scope:

1. A reproducible, resumable Corpus V2 build pipeline and private evidence ledger.
2. Immutable release manifests joining source identity through active index release.
3. A separate candidate database, controlled promotion, and reversible rollback.
4. A fresh, pre-registered evaluation and safety decision for Corpus V2.

Out of scope:

- Rewriting, deleting, reindexing, migrating, or otherwise changing the legacy index.
- Treating E1a-3/E1a-4 private material as tuning or acceptance data for Corpus V2.
- Adding Qdrant or another vector-store framework.
- Using an LLM, including Astra, to determine approved document identity,
  provenance, or an exclusion decision.
- Publishing private document names, Drive identifiers, paths, text, hashes,
  chunks, vectors, review decisions, or raw evaluation results in tracked files.

## Ownership and Decision Boundaries

| Responsibility | Owner | Permitted action |
| --- | --- | --- |
| Approved source membership | Corpus steward (the user or named delegate) | Approve the exact 385 Drive identities and any membership change. |
| Critical-source policy | Corpus steward with subject-matter reviewer | Freeze the required/foundational source register before extraction review. |
| Mechanical provenance | Deterministic controller | Acquire exact revisions, calculate hashes, record parser facts, and validate release bindings. |
| Exception disposition | Named reviewer | Decide documented duplicate, non-text, empty, unsupported, extraction-failed, or intentionally-excluded outcomes. |
| Architecture and gate review | Independent reviewer, including Astra when requested | Review controls and aggregate gate evidence only; never establish identity, provenance, or disposition. |
| Promotion | User/release owner | Approve candidate promotion only after every gate passes. |

The controller may report factual parser and hash results, but may not silently decide
that a document is dispensable. Every authoritative identity receives a durable final
disposition.

## Release Topology

    Legacy release (read-only)                 Corpus V2 candidate release
    198 sources / 4,797 chunks                 approved 385 Drive identities
              |                                           |
              | rollback target                           v
              +------------------------------> frozen byte snapshot
                                                           |
                                                           v
                                                   parse + review ledger
                                                           |
                                                           v
                                              chunks -> embeddings -> V2 PGVector DB
                                                           |
                                                           v
                                            integrity + chatbot evaluation gates
                                                           |
                                                           v
                                            explicit production release configuration

A Corpus V2 release has an immutable release identifier. Its candidate PostgreSQL
database name, embedding provider/model/dimension, and release-manifest digest are
part of the identifier contract. The application must refuse retrieval startup if any
of those values disagree.

The application must key its cached RAG service and keyword index by release
identifier, not retrieval mode alone. This prevents a process from serving a cached
legacy service after a Corpus V2 configuration change.

## Private Release Evidence

All detailed artifacts live in an ignored private release root. Each artifact is
canonical, versioned, checksummed, and bound to the release identifier.

| Artifact | Required contents | Acceptance purpose |
| --- | --- | --- |
| Source register | Exactly 385 approved Drive identities and steward approval binding | Prevents additions, omissions, and source substitution. |
| Acquisition manifest | Drive revision evidence, downloaded-byte SHA-256, MIME type, acquisition status | Proves exact-byte snapshot provenance. |
| Frozen snapshot | Private source bytes organized only by release-safe identity | Provides reproducible parsing input. |
| Review ledger | Durable SQLite state for acquisition, parser results, reviewer decisions, and history | Survives connection loss and makes every disposition auditable. |
| Extraction manifest | Parser/version/settings, source locations, text-capability and failure facts | Separates extraction evidence from disposition decisions. |
| Chunk manifest | Chunk ID, source identity, source-byte hash, location, chunking configuration | Rejects orphan or out-of-snapshot chunks. |
| Embedding manifest | Chunk ID set, model identity/revision, vector dimension, batch integrity facts | Rejects model mismatch or partial vectorization. |
| Index contract | Database release ID, exact chunk-ID set/fingerprint, distinct source count, model and dimension | Proves the candidate database holds exactly the intended release. |
| Release binding | Digests of every prior manifest plus evaluation binding and promotion state | Supplies one verifiable end-to-end chain. |

The private review ledger records current state and append-only decision history.
Sealed manifests are immutable release evidence rather than a substitute for the
ledger. Public reporting is aggregate-only.

## Document Disposition Policy

Every approved document begins in the snapshot and must end in exactly one current
state: INDEXED, DUPLICATE, NON_TEXT, EMPTY, UNSUPPORTED, EXTRACTION_FAILED, or
INTENTIONALLY_EXCLUDED.

UNRESOLVED is permitted only while the candidate is in progress; it blocks promotion.
Duplicate detection is a mechanical hash fact, but the final duplicate disposition is
recorded by policy and linked to its representative identity. A document that is
non-searchable may remain in the approved snapshot without becoming an indexed source.
Corpus V2 is therefore not required to contain exactly 385 indexed sources.

The frozen critical-source register creates a stricter rule: a critical or foundational
document with no sufficiently searchable evidence blocks promotion. An ordinary
non-searchable document may proceed only with a reviewed final disposition and only if
the fresh evaluation shows that the released product scope remains supported.

## Build State Machine and Fail-Closed Rules

    REGISTERED -> ACQUIRED -> PARSED -> REVIEWED -> CHUNKED -> EMBEDDED
             -> INDEX_VALIDATED -> EVALUATED -> PROMOTION_READY -> PROMOTED

Any mismatch, unexpected source, changed Drive revision, missing hash, parser crash,
unresolved review item, model/dimension mismatch, orphan or extra index row, failed
evaluation threshold, or failed rollback rehearsal moves the release to BLOCKED.

The controller preserves prior successful stage artifacts and provides a resumable next
action. It must never fall back to legacy data while evaluating V2, silently skip an
input, overwrite an existing sealed release, or promote a partial candidate.

## Promotion Gates

### Gate 0 — Reconciliation closure record

Publish an aggregate closure note that preserves the findings above and names Corpus V2
as a separately authorized follow-on. Do not alter historical reconciliation statuses,
seals, or conclusions.

### Gate 1 — Legacy preservation

Before V2 writes begin, capture a recoverable legacy backup and a sealed metadata
inventory. Restore-test the backup in isolation, verify the 198-source / 4,797-chunk
legacy contract, and limit normal operational access to legacy reads. Do not run
migrations or ingestion against legacy.

### Gate 2 — Snapshot completeness

Require exact set equality between the approved register and acquisition manifest:
385 identities, no extra identities, one pinned revision and byte hash per identity,
and no unresolved acquisition result. If an item changes during acquisition, block it
rather than replacing it with a newer byte stream.

### Gate 3 — Extraction and review completeness

Require a final disposition for every one of the 385 identities and zero current
UNRESOLVED states. Require usable evidence for every frozen critical source. Require
review of parser failures, empty files, non-text items, unsupported formats, duplicate
aliases, and exclusions. No raw count target substitutes for these checks.

### Gate 4 — Candidate-index integrity

Require the V2 database to contain exactly the canonical chunk-ID set for the release:
no missing rows, no extra rows, and no mixed embedding identities. Each indexed chunk
must bind to an approved snapshot identity, downloaded-byte hash, valid source location,
and chunk manifest entry. Require finite vectors of the pinned dimension and matching
embedding model/revision. A resumed run must reproduce the same manifest without
duplicates.

### Gate 5 — Product evaluation

Run a fresh pre-registered evaluation only after the candidate release and criteria are
frozen. It must measure evidence retrieval, citation correctness, unsupported-answer
safety, abstention/no-answer behavior, latency, and registered regression families.
Keep development and unseen acceptance cohorts separate. Do not tune the candidate on
sealed historical E1a-3/E1a-4 artifacts or unseen acceptance results.

Numerical acceptance thresholds, cohort sizes, scope boundaries, and the embedding
model/execution location must be approved in the implementation plan before the first
candidate build. No threshold may be relaxed after results are observed.

### Gate 6 — Promotion and rollback rehearsal

Promote only an immutable release whose binding, database contract, and Gate 5 report
all validate. The production configuration must select one release explicitly and
include the database target, release ID, embedding configuration, and manifest digest.
Run a canary and a rollback rehearsal before broad activation. Rollback switches only
the release configuration back to legacy and refreshes release-scoped caches; it must
not modify either corpus.

## Astra Review Boundary

An Astra medium-effort independent review recommended an isolated PostgreSQL/PGVector
database, a restore-tested legacy baseline, source-to-chunk provenance binding, explicit
release selection, and a rehearsed rollback. This design adopts those architecture and
promotion-gate recommendations. Astra has no authority to establish document identity,
Drive provenance, source criticality, or final disposition.

## Implementation Preconditions

Before implementation, the release owner must approve:

1. The immutable 385-identity source register and membership-change procedure.
2. The critical/foundational source register and its evidence-sufficiency definition.
3. Drive revision acquisition policy and private-snapshot retention/backup policy.
4. Duplicate-alias policy and reviewer identity for exceptional dispositions.
5. Parser/OCR policy and review requirements for images, charts, spreadsheets, empty
   files, and extraction failures.
6. Embedding model, exact revision, execution location, vector dimension, and resource
   limits.
7. Candidate database naming, access roles, backup/restore objective, and release owner.
8. Fresh evaluation cohorts, frozen numerical acceptance thresholds, canary duration,
   rollback triggers, and production approval authority.

Only after those choices are recorded may an implementation plan authorize source
acquisition, parsing, embeddings, database writes, candidate indexing, or application
release changes.
