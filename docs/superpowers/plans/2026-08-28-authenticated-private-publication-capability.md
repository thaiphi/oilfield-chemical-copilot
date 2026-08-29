# Authenticated Private Publication Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace remaining pathname-based E1a-4 private mapping/frame publication and frame verification with one cross-platform authenticated directory capability.

**Architecture:** A new evaluation utility owns stable-root traversal, component-relative creation, locking, staging, exact member reads, durability synchronization, no-replace publication, and single-owner resource cleanup. Mapping publication and sampling-frame publication/verification consume that utility; artifact formats and evaluation behavior do not change.

**Tech Stack:** Python 3.11, pathlib, POSIX descriptor-relative filesystem APIs, Windows native handle APIs through `ctypes`, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-08-28-authenticated-private-publication-capability-design.md`

## Global Constraints

- Do not inspect, read, modify, reseal, or enumerate `.private` artifacts.
- Do not run models, retrieval, ingestion, reindexing, database services, or evaluation experiments.
- Use synthetic temporary artifacts only.
- Preserve every mapping/frame artifact schema, filename, digest, manifest, binding, allocation, and public aggregate.
- Require an explicit approved private root at every public publication or frame-verification entrypoint.
- After capability acquisition, do not reopen a publication directory or artifact member by pathname.
- Traverse and create path components relative to stable directory handles with no-follow/reparse-safe semantics.
- Flush the parent after every newly created hierarchy entry, staged directories before rename, and the publication parent after rename.
- Every handle or descriptor has one owner and is closed exactly once, including constructor and close failures.
- Preserve staging residue on failure. Do not add recursive or pathname-based deletion.
- Unsafe paths, races, unsupported primitives, lock contention, residue, durability failures, malformed artifacts, and close failures are fail-closed with fixed public-safe caller codes.
- Native symlink tests may skip only when deterministic platform seams cover the same branch.

---

### Task 1: Shared Authenticated Publication Capability

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/private_artifact_publication.py`
- Create: `tests/evaluation/test_private_artifact_publication.py`
- Reference only: `eval/seal_e1a4_sampling_frame.py`

**Interfaces:**
- Produces: `PrivateArtifactPublicationError`
- Produces: `ArtifactLayout = Mapping[str, frozenset[str]]`
- Produces: `BoundStagingDirectory.mkdir(relative_name: str) -> None`
- Produces: `BoundStagingDirectory.write_exclusive(relative_name: str, content: bytes) -> None`
- Produces: `BoundStagingDirectory.sync_directory(relative_name: str) -> None`
- Produces: `BoundStagingDirectory.sync_root() -> None`
- Produces: `AuthenticatedPublicationDirectory.ensure_no_staging(prefix: str, suffix: str) -> None`
- Produces: `AuthenticatedPublicationDirectory.final_exists(final_name: str) -> bool`
- Produces: `AuthenticatedPublicationDirectory.create_staging(prefix: str, suffix: str) -> BoundStagingDirectory`
- Produces: `AuthenticatedPublicationDirectory.publish_no_replace(staging: BoundStagingDirectory, final_name: str) -> None`
- Produces: `AuthenticatedPublicationDirectory.read_exact_tree(final_name: str, layout: ArtifactLayout) -> dict[str, bytes]`
- Produces: `AuthenticatedPublicationDirectory.sync_parent() -> None`
- Produces: `authenticated_publication_directory(*, approved_private_root: Path, publication_parent: Path, lock_name: str) -> ContextManager[AuthenticatedPublicationDirectory]`

- [x] **Step 1: Write direct capability RED tests**

Add literal-behavior tests that prove:

```python
def test_capability_rejects_escape_before_opening_anchor(...): ...
def test_capability_never_binds_retargeted_ancestor_during_real_acquisition(...): ...
def test_capability_reads_final_members_through_locked_parent(...): ...
def test_windows_creation_flushes_each_parent_before_releasing_it(...): ...
def test_windows_constructor_failure_closes_each_handle_once(...): ...
def test_posix_creation_is_component_relative_nofollow_and_parent_synced(...): ...
def test_capability_preserves_residue_and_never_deletes_untrusted_entries(...): ...
def test_capability_fails_closed_without_safe_relative_rename(...): ...
```

The acquisition-race fixture must invoke the real platform constructor seam, swap the requested output ancestor before the old pathname open would occur, and assert that no synthetic secret bytes appear in the replacement. The read test must make a pathname replacement after capability acquisition and prove the capability either reads the authenticated tree or fails closed; it must never accept the replacement.

- [x] **Step 2: Run Task 1 tests and verify RED**

Run:

```powershell
uv run pytest -q tests/evaluation/test_private_artifact_publication.py
```

Expected: collection/import failure because the shared module and interfaces do not exist. After test scaffolding imports, each behavioral test must fail for its named missing behavior, not from an invalid fixture.

- [x] **Step 3: Implement the minimal shared capability**

Implement the interface above by extracting and consolidating the already reviewed platform primitives from the frame runner. The module must keep platform-specific internals private. Use this ownership rule:

```python
handle: object | None = acquired_handle
try:
    publication = _WindowsPublicationDirectory.take(handle)
    handle = None  # ownership transferred exactly once
    yield publication
finally:
    if handle is not None:
        close_handle(handle)
```

POSIX acquisition opens the filesystem root once and walks relative components with `O_DIRECTORY | O_NOFOLLOW`. Windows acquisition opens the stable volume root once and walks relative components with native root-relative calls plus reparse-safe flags. A newly created child is followed immediately by parent synchronization before traversal continues.

Do not expose raw paths in exception strings. `PrivateArtifactPublicationError` contains fixed internal codes that consumers translate to their existing public-safe codes.

- [x] **Step 4: Run Task 1 GREEN verification**

Run:

```powershell
uv run pytest -q tests/evaluation/test_private_artifact_publication.py
uv run ruff check src/oilfield_chemical_copilot/evaluation/private_artifact_publication.py tests/evaluation/test_private_artifact_publication.py
git diff --check
```

Expected: all Task 1 tests pass; Ruff and diff checks are clean.

- [x] **Step 5: Commit Task 1**

```powershell
git add src/oilfield_chemical_copilot/evaluation/private_artifact_publication.py tests/evaluation/test_private_artifact_publication.py
git commit -m "feat: add authenticated private publication capability"
```

---

### Task 2: Migrate Mapping Publication

**Files:**
- Modify: `src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py`
- Modify: `eval/apply_e1a4_role_corrections.py`
- Modify: `tests/evaluation/test_e1a4_mapping_application.py`
- Test: `tests/evaluation/test_private_artifact_publication.py`

**Interfaces:**
- Consumes: `authenticated_publication_directory(...)`
- Changes: `seal_e1a4_role_mapping(..., output_root: Path, approved_private_root: Path) -> E1A4MappingSeal`
- Changes: `_publish_mapping_directory(mapping, binding, output_root, approved_private_root) -> E1A4MappingSeal`
- Preserves: mapping filenames, canonical bytes, digest anchors, `E1A4MappingSeal`, and existing verification API

- [ ] **Step 1: Write mapping-publication RED tests**

Add focused tests:

```python
def test_mapping_publisher_never_writes_through_retargeted_ancestor(...): ...
def test_mapping_cli_passes_approved_root_to_sealer_before_inputs_open(...): ...
def test_mapping_publisher_preserves_existing_final_and_staging_residue(...): ...
def test_mapping_publisher_uses_capability_for_lock_stage_sync_and_rename(...): ...
```

The retarget test must swap during the real mapping publication acquisition/staging flow and fail on the current pathname-based implementation by detecting synthetic locator bytes in the replacement tree.

- [ ] **Step 2: Run Task 2 tests and verify RED**

Run:

```powershell
uv run pytest -q tests/evaluation/test_e1a4_mapping_application.py -k "retargeted_ancestor or approved_root_to_sealer or capability_for_lock_stage_sync or preserves_existing_final"
```

Expected: failures show that mapping publication does not accept or use the shared authenticated capability.

- [ ] **Step 3: Replace mapping pathname publication**

Pass `approved_private_root` from the CLI to `seal_e1a4_role_mapping` and `_publish_mapping_directory`. Inside the publisher:

```python
with authenticated_publication_directory(
    approved_private_root=approved_private_root,
    publication_parent=output_root / "e1a4-role-mapping",
    lock_name=".v1.publish.lock",
) as publication:
    publication.ensure_no_staging(prefix=".v1.", suffix=".tmp")
    # write the unchanged four mapping artifacts to bound staging
    # sync staging, publish_no_replace(..., "v1"), sync_parent()
    # verify using the existing canonical payload and digest anchors
```

Remove mapping publisher-only pathname lock/staging/rename/sync helpers after their consumers and tests migrate. Do not change the separately hardened mapping artifact reader unless compilation proves a private helper is shared with verification; if shared, move only that helper to the capability module without altering verification semantics.

- [ ] **Step 4: Run mapping GREEN and regressions**

Run:

```powershell
uv run pytest -q tests/evaluation/test_e1a4_mapping_application.py tests/evaluation/test_private_artifact_publication.py
uv run ruff check src/oilfield_chemical_copilot/evaluation/private_artifact_publication.py src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py eval/apply_e1a4_role_corrections.py tests/evaluation/test_private_artifact_publication.py tests/evaluation/test_e1a4_mapping_application.py
git diff --check
```

Expected: all mapping/capability tests pass with only documented platform skips; no artifact contract changes.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py eval/apply_e1a4_role_corrections.py tests/evaluation/test_e1a4_mapping_application.py
git commit -m "fix: bind e1a4 mapping publication to private root"
```

---

### Task 3: Migrate Frame Publication And Verification

**Files:**
- Modify: `eval/seal_e1a4_sampling_frame.py`
- Modify: `tests/evaluation/test_e1a4_sampling.py`
- Test: `tests/evaluation/test_private_artifact_publication.py`
- Consume: `src/oilfield_chemical_copilot/evaluation/private_artifact_publication.py`

**Interfaces:**
- Consumes: `authenticated_publication_directory(...)`
- Produces: `_verify_sampling_frame_members(*, source_register, allocation, members, expected_source_register_sha256, expected_allocation_sha256) -> E1A4SamplingFrameSeal`
- Preserves: `seal_sampling_frame(**values)`, `verify_current_sampling_frame(**values)`, fixed CLI output, exact four-file frame layout, and aggregate seal fields

- [ ] **Step 1: Write frame verification/publication RED tests**

Add tests:

```python
def test_locked_frame_verification_never_reopens_output_by_path(...): ...
def test_standalone_frame_verification_reads_through_authenticated_capability(...): ...
def test_frame_existing_final_is_verified_from_locked_tree_not_replacement(...): ...
def test_frame_publisher_preserves_directory_sync_order_with_shared_capability(...): ...
def test_frame_runner_has_no_duplicate_platform_publication_implementation(...): ...
```

The first and third tests acquire the real capability, retarget the visible pathname, place a different but structurally valid synthetic frame in the replacement, and assert that verification uses the authenticated tree or fails closed.

- [ ] **Step 2: Run Task 3 tests and verify RED**

Run:

```powershell
uv run pytest -q tests/evaluation/test_e1a4_sampling.py -k "locked_frame_verification or standalone_frame_verification_reads or locked_tree_not_replacement or shared_capability"
```

Expected: current path-based `verify_sampling_frame` accepts or opens the replacement, or the shared-capability hook is absent.

- [ ] **Step 3: Split pure validation from capability-relative reading**

Create `_verify_sampling_frame_members(...)` as a pure validator over an exact member dictionary. Both seal and standalone verify must obtain members through:

```python
members = publication.read_exact_tree(
    "v1",
    {
        "sealed": frozenset({SOURCE_REGISTER_NAME, ALLOCATION_NAME}),
        "manifests": frozenset({
            _manifest_name(SOURCE_REGISTER_NAME),
            _manifest_name(ALLOCATION_NAME),
        }),
    },
)
```

No locked call path may invoke `_read_frame_members(final_path)` or reopen `output_root`. Remove the runner's duplicated `_PosixPublicationDirectory`, `_WindowsPublicationDirectory`, acquisition, staging, rename, sync, and publisher-lock implementation after migration. Keep frame-specific payload construction and safe error translation in the runner.

- [ ] **Step 4: Run frame GREEN and combined regressions**

Run:

```powershell
uv run pytest -q tests/evaluation/test_private_artifact_publication.py tests/evaluation/test_e1a4_mapping_application.py tests/evaluation/test_e1a4_sampling.py
uv run ruff check src/oilfield_chemical_copilot/evaluation/private_artifact_publication.py src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py eval/apply_e1a4_role_corrections.py eval/seal_e1a4_sampling_frame.py tests/evaluation/test_private_artifact_publication.py tests/evaluation/test_e1a4_mapping_application.py tests/evaluation/test_e1a4_sampling.py
git diff --check
```

Expected: all capability/mapping/frame tests pass; no duplicate platform publisher remains in the frame runner; only documented privilege skips remain.

- [ ] **Step 5: Commit Task 3**

```powershell
git add eval/seal_e1a4_sampling_frame.py tests/evaluation/test_e1a4_sampling.py
git commit -m "fix: verify e1a4 frames through private capability"
```

---

### Task 4: Public Closure, Full Verification, And PR Re-Review

**Files:**
- Modify: `docs/superpowers/plans/2026-08-28-authenticated-private-publication-capability.md`
- Modify only if aggregate status changes: `docs/superpowers/plans/2026-08-27-e1a4-correction-application-and-sampling-frame.md`
- Modify only if design implementation status changes: `docs/superpowers/specs/2026-08-28-authenticated-private-publication-capability-design.md`

**Interfaces:**
- Consumes: Tasks 1-3 commits and test evidence
- Produces: aggregate-only public closure and a clean PR #2 head

- [ ] **Step 1: Run exact final-tree verification**

Run:

```powershell
uv run pytest
uv run ruff check .
git diff --check
git status --short
git ls-files | rg '(^|/)\.private(/|$)'
```

Expected: full suite passes; Ruff and diff checks pass; worktree is clean after documentation commit; private-file query returns no paths.

- [ ] **Step 2: Perform privacy and contract checks**

Verify from tracked code and tests only:

- no `.private` path or raw identifier was added;
- no artifact schema/name/digest/binding changed;
- no retrieval, ingestion, model, question, claim, label, or allocation logic changed;
- every original and second-round Codex finding has a covering regression;
- the Windows double-close minor has an exact-once ownership regression.

- [ ] **Step 3: Request scoped independent review**

Give the reviewer the spec, this plan, exact Task 1-3 diff package, implementer reports, original six GitHub findings, and deferred double-close minor. Review is public-code-only and read-only. All Critical and Important findings must be corrected and re-reviewed before push.

- [ ] **Step 4: Update aggregate-only status and commit**

Mark this plan complete and change the design status to implemented only after the scoped review is clean. Do not add private paths, identifiers, locators, hashes, source titles, or artifact contents.

```powershell
git add docs/superpowers/plans/2026-08-28-authenticated-private-publication-capability.md docs/superpowers/specs/2026-08-28-authenticated-private-publication-capability-design.md
git commit -m "docs: close authenticated publication correction"
```

- [ ] **Step 5: Push and request Codex PR re-review**

Push the verified commits to `codex/corpus-reconciliation-review`, reply in each open PR #2 review thread with commit-specific evidence, and request `@codex review` on the exact head. Merge only when GitGuardian succeeds, Codex completes without unresolved findings, the PR is mergeable, and the user-approved integration sequence remains in force.

## Execution Boundary

This plan authorizes synthetic code/test work and aggregate-only documentation only. It does not authorize reading private artifacts, resealing existing mapping/frame outputs, running evaluation data, changing the corpus, or modifying retrieval/ingestion behavior.
