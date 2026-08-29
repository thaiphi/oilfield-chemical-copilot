# Authenticated Private Publication Capability Design

**Date:** 2026-08-28

**Status:** Approved in principle; written design awaiting final user review before implementation

## Objective

Replace the remaining pathname-based private-artifact publication and verification operations with one shared, cross-platform authenticated directory capability. The capability must ensure that private mapping and sampling-frame bytes can be written or verified only beneath the exact controller-approved private root, even when an output ancestor is concurrently renamed, replaced, symlinked, or converted to a Windows reparse point.

## Context

PR #2 added explicit approved-private-root validation, handle-relative sampling-frame writes, directory durability synchronization, and deterministic retarget tests. Follow-up review established three remaining gaps:

1. mapping publication validates its output path but later stages and renames through pathnames;
2. sampling-frame verification reacquires the artifact tree by pathname after the publisher has locked a stable directory capability;
3. Windows component-relative hierarchy creation does not flush each newly created name through its parent before that parent handle is released.

The correction must close these gaps without reading or rewriting existing private artifacts and without changing evaluation data, sampling decisions, retrieval, ingestion, or model behavior.

## Selected Approach

Create a focused shared module under `src/oilfield_chemical_copilot/evaluation/` that owns authenticated private-directory traversal and artifact operations. Both CLIs remain responsible for parsing controller inputs and returning fixed public-safe error codes, while the shared module owns the filesystem security and durability contract.

The shared capability will:

- accept an explicit approved private root and requested output location;
- reject lexical escape, sibling-prefix escape, a public worktree root, symlink components, and Windows reparse components;
- open one stable filesystem or volume anchor and descend to the approved root and requested publication parent component-by-component with no-follow semantics;
- create missing directories only relative to an authenticated parent handle;
- flush the authenticated parent immediately after each new directory entry is created;
- retain stable handles for lock acquisition, residue checks, staging, member creation, member reads, synchronization, and final no-replace rename;
- expose no operation that reopens an artifact member or publication directory by pathname after capability acquisition;
- use explicit single ownership for every native handle or descriptor so every resource is closed exactly once;
- fail closed with caller-mapped fixed error codes when the platform lacks a required safe primitive;
- preserve staging residue on failure and never recursively delete an untrusted path.

## Consumers

### Mapping Publication

The E1a-4 mapping publisher will receive the approved private root from the CLI and publish its versioned directory entirely through the shared capability. Locking, residue detection, staging creation, file writes, directory synchronization, and final rename must be capability-relative. Existing artifact formats, filenames, manifests, bindings, and standalone verification semantics remain unchanged.

### Sampling-Frame Publication

The sampling-frame publisher will replace its local platform publication implementation with the shared capability. Its existing fixed artifact set, fail-closed residue policy, exclusive publication rule, and durability ordering remain unchanged.

### Sampling-Frame Verification

Verification performed while the publisher lock is held must read the final frame through the same authenticated capability. Standalone verification must acquire the same capability and lock before residue checking, expected-frame reconstruction, and member verification. It must not call a pathname-based reader after capability acquisition.

## Data Flow

1. The CLI validates that all required controller arguments are present.
2. Before opening input stores, the CLI passes the approved private root and requested output root through lexical containment and public-root rejection.
3. The shared module opens a stable filesystem or volume anchor and descends to the requested publication parent using no-follow component-relative operations.
4. The capability acquires the publisher lock within that authenticated parent.
5. The caller performs residue checks, stages files, synchronizes files and directories, and publishes with no-replace rename through the capability.
6. Verification reads the exact published members through the same authenticated parent capability.
7. The capability closes every owned handle exactly once. Any acquisition, validation, synchronization, close, or verification failure returns a fixed caller-safe error without exposing a private path.

## Platform Contract

### POSIX

Use directory descriptors, `O_DIRECTORY`, `O_NOFOLLOW`, component-relative `open`/`mkdir`, descriptor-relative listing and member access, file and directory `fsync`, and a supported descriptor-relative no-replace rename. If a required primitive is unavailable, publication fails closed.

### Windows

Open only the stable volume root by pathname. Traverse and create descendants with root-relative native calls and reparse-point-safe flags. Flush a parent handle after every newly created hierarchy entry, flush staged child directories before publication, and flush the publication parent after the final rename. Native-handle ownership transfers must be explicit so constructor failure cannot cause a double close.

## Artifact And Privacy Compatibility

- No artifact schema, filename, manifest, digest, binding, allocation, or public aggregate changes.
- Existing sealed mapping and frame directories remain format-compatible.
- Implementation and automated tests use synthetic temporary artifacts only.
- No `.private` content, source identifier, locator, question, claim, label, or result is read or written during this correction.
- No existing private artifact is resealed as part of implementation.

## Failure Policy

- Every unsafe path, reparse point, symlink, race, unsupported primitive, residue, lock contention, durability failure, malformed member, or resource-close failure is fail-closed.
- Public CLI output contains fixed error codes only.
- Failed staging is preserved for manual review; automatic recursive deletion is prohibited.
- An existing final artifact is accepted only after capability-relative verification against the controller-provided expected payload and digest anchors.

## Testing Strategy

Strict test-driven development is required. Each production change must be preceded by a regression that fails for the intended reason.

Required coverage:

- mapping ancestor retarget during the real publication acquisition and staging window;
- frame ancestor retarget between lock acquisition and verification;
- standalone frame verification through the authenticated capability;
- POSIX and Windows component-relative traversal, creation, read, write, synchronization, and no-replace rename seams;
- parent synchronization after every newly created Windows hierarchy component;
- constructor-validation failure proving each native handle closes exactly once;
- sibling-prefix, public-root, symlink, junction, and reparse rejection;
- residue preservation, lock contention, partial artifacts, durability failure, and unsupported-platform failure;
- compatibility with existing synthetic mapping and frame payload contracts;
- focused suites, full repository suite, Ruff, `git diff --check`, clean worktree, and zero tracked `.private` files.

Native symlink tests may skip when the host lacks privileges only when deterministic platform seams exercise the same branch.

## Rejected Approaches

### Add More Before-And-After Path Checks

Rejected because a retarget can occur between any check and pathname operation. Additional checks narrow but do not eliminate the race.

### Duplicate The Frame Capability Inside Mapping Code

Rejected because two cross-platform implementations would diverge and require separate security review. The filesystem security boundary should have one owner.

### Reseal Existing Private Artifacts

Rejected because the artifact formats are unchanged and correction validation can be completed with synthetic fixtures. Existing private state remains untouched.

## Completion Boundary

The correction is complete only when:

1. mapping publication, frame publication, and frame verification use the shared authenticated capability;
2. all three outstanding Codex findings and the scoped double-close minor are covered by RED-to-green tests;
3. no Critical or Important scoped-review finding remains;
4. the full suite and Ruff pass on the exact pushed tree;
5. GitGuardian and Codex re-review the new head without unresolved findings; and
6. PR #2 is merged only after those gates close.
