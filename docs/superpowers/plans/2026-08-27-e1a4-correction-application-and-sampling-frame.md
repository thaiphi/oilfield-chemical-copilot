# E1a-4 Correction Application And Sampling Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the two verified locator-correction proposals to a new private E1a-4-only role mapping and seal the deterministic 96-slot metadata sampling frame without modifying E1a-3, reconciliation inventory, PostgreSQL, Qdrant, or retrieval.

**Architecture:** A strict sampling-contract module validates mapped source records and the frozen E1a-3 allocation. A correction-application controller authenticates reconciliation plus both v2 proposals, projects only verified promotions into locator-level roles, proves eight-stratum and allocator capacity, and atomically seals a four-file mapping artifact. A standalone sampling-frame runner verifies that complete trust chain and the live index contract, then atomically seals the metadata-only source register and allocation.

**Tech Stack:** Python 3.11+, SQLite, existing reconciliation/foundational/supplement audit contracts, existing index preflight, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-27-e1a4-correction-application-and-sampling-frame-design.md`

## Global Constraints

- Keep source IDs, locators, parser metadata, decisions, paths, allocations, and hashes beneath existing ignored private roots.
- Do not read the sealed holdout, questions, claims, requirements, passage text, retrieval output, or model output.
- Do not run retrieval, Qdrant, generation, ingestion, reindexing, or question/canonical-claim authoring.
- Do not update reconciliation `index_sources` or `index_locators`, PostgreSQL, E1a-3 artifacts, or either correction proposal.
- Authenticate the active reconciliation seal and both v2 proposal seals before application, verification, or allocation.
- Require the correction stores to share the exact resolved reconciliation SQLite path, run ID, and reconciliation snapshot binding.
- Require the sealed E1a-3 allocation keys to equal reconciliation `e1a3_used=1` keys before excluding them.
- Preserve locator-level mixed roles: promoting one locator never changes another locator from the same source.
- Publish each related four-file artifact set by one version-directory rename and reject partial, extra, altered, stale, or conflicting artifacts.
- Public output and tracked documentation contain aggregate status only.

---

### Task 1: Recover And Strengthen E1a-4 Sampling Contracts

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/e1a4_sampling.py`
- Create: `tests/evaluation/test_e1a4_sampling.py`

**Interfaces:**
- Produces: `E1A4SamplingError`, `E1A4MappedSource`, `E1A4PriorAllocation`, `load_e1a3_prior_allocation(...)`, `validate_mapping_sources(...)`, and `mapping_sources_as_sampling_metadata(...)`.
- Consumes: canonical E1a-3 sampling payloads and `E1A3SourceMetadata`.

- [x] **Step 1: Write failing tests for strict mapped-source and prior-allocation contracts**

Add tests that construct an exact 96-row E1a-3 allocation and assert:

```python
def test_prior_allocation_requires_manifest_and_exact_96_unique_keys(tmp_path: Path) -> None:
    result = load_e1a3_prior_allocation(
        payload_path=payload_path,
        manifest_path=manifest_path,
        private_root=private_root,
    )
    assert result.slot_count == 96
    assert len(result.locator_keys) == 96


def test_mapping_sources_preserve_mixed_roles_for_one_source() -> None:
    sources = validate_mapping_sources(
        (
            _mapped("source-a", "iron_sulfide", "foundational", ("page:1",)),
            _mapped("source-a", "iron_sulfide", "supporting", ("page:2",)),
        )
    )
    assert {item.source_role for item in sources} == {"foundational", "supporting"}
```

Also reject unknown fields, booleans as counts, invalid topics/roles, empty or unsorted locator lists, duplicate locator keys across records, a manifest mismatch, noncanonical payload digest, non-96 slot sets, duplicate slot identities, duplicate locator keys, and paths outside the supplied private root.

- [x] **Step 2: Run Task 1 tests and verify RED**

Run:

```powershell
python -m pytest tests/evaluation/test_e1a4_sampling.py -q
```

Expected: collection fails because `e1a4_sampling` is absent in the isolated branch.

- [x] **Step 3: Implement the minimal strict contracts**

Use these public shapes:

```python
@dataclass(frozen=True)
class E1A4MappedSource:
    source_id: str
    topic: Literal["iron_sulfide", "scale", "corrosion", "paraffin"]
    source_role: Literal["foundational", "supporting"]
    parser_type: str
    locators: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> "E1A4MappedSource":
        if not isinstance(value, dict) or set(value) != {
            "source_id", "topic", "source_role", "parser_type", "locators"
        }:
            raise E1A4SamplingError("E1A4_MAPPING_SOURCE_INVALID")
        locators = value["locators"]
        if not isinstance(locators, list):
            raise E1A4SamplingError("E1A4_MAPPING_SOURCE_INVALID")
        return cls(
            source_id=_required_text(value["source_id"]),
            topic=_topic(value["topic"]),
            source_role=_source_role(value["source_role"]),
            parser_type=_required_text(value["parser_type"]),
            locators=tuple(_required_text(item) for item in locators),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "topic": self.topic,
            "source_role": self.source_role,
            "parser_type": self.parser_type,
            "locators": list(self.locators),
        }


@dataclass(frozen=True)
class E1A4PriorAllocation:
    payload_sha256: str
    slot_count: int
    locator_keys: frozenset[str]
```

`load_e1a3_prior_allocation` must parse only the exact E1a-3 allocation fields, validate the canonical payload digest with `private_sampling_payload_digest`, require the fixed grid identities from `build_sampling_slots()`, and return keys formatted as `source_id:locator`.

`validate_mapping_sources` must sort records deterministically and reject any repeated `source_id:locator` across all records. `mapping_sources_as_sampling_metadata` converts each record to `E1A3SourceMetadata(..., eligibility_status="eligible")` without merging mixed roles.

- [x] **Step 4: Run Task 1 tests and Ruff**

```powershell
python -m pytest tests/evaluation/test_e1a4_sampling.py -q
python -m ruff check src/oilfield_chemical_copilot/evaluation/e1a4_sampling.py tests/evaluation/test_e1a4_sampling.py
```

Expected: all tests and Ruff pass.

- [x] **Step 5: Commit Task 1**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/e1a4_sampling.py tests/evaluation/test_e1a4_sampling.py docs/superpowers/plans/2026-08-27-e1a4-correction-application-and-sampling-frame.md docs/superpowers/specs/2026-08-27-e1a4-correction-application-and-sampling-frame-design.md
git commit -m "feat: add e1a4 mapping sampling contracts"
```

### Task 2: Apply Verified Corrections To A Versioned Mapping

**Files:**
- Create: `src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py`
- Create: `tests/evaluation/test_e1a4_mapping_application.py`

**Interfaces:**
- Produces: `E1A4MappingApplicationError`, `E1A4MappingArtifact`, `E1A4MappingSeal`, `build_e1a4_role_mapping(...)`, `seal_e1a4_role_mapping(...)`, and `verify_e1a4_role_mapping(...)`.
- Consumes: `ReconciliationStore`, `FoundationalAuditStore`, `IronSulfideSupplementAuditStore`, four expected binding digests, and frozen E1a-3 allocation paths.

- [ ] **Step 1: Write failing mapping-application tests**

Create a synthetic sealed reconciliation with terminal core and supplement proposals. Assert that:

```python
def test_mapping_application_preserves_inventory_and_mixed_roles(tmp_path: Path) -> None:
    before = _locator_rows(store)
    mapping, binding = build_e1a4_role_mapping(**_trusted_inputs(tmp_path))
    assert _locator_rows(store) == before
    assert _role_for(mapping, "supplement-source", "page:1") == "foundational"
    assert _role_for(mapping, "supplement-source", "page:2") == "supporting"
    assert binding["allocator_slot_count"] == 96
```

Add adversarial tests for a fake reconciliation digest, fake core/supplement binding, another SQLite database, mismatched reconciliation snapshots, altered E1a-3 allocation, inventory/allocation exclusion mismatch, overlapping proposal keys, promotion not in its frozen candidate inventory, wrong pre-application role/status, unresolved proposal, insufficient stratum, allocator failure, seal crash cleanup, extra files, altered manifests, SQLite state drift, and idempotent no-write verification.

- [ ] **Step 2: Run Task 2 tests and verify RED**

```powershell
python -m pytest tests/evaluation/test_e1a4_mapping_application.py -q
```

Expected: collection fails because `e1a4_mapping_application` is absent.

- [ ] **Step 3: Implement authenticated in-memory application**

Use this signature:

```python
def build_e1a4_role_mapping(
    *,
    store: ReconciliationStore,
    core_audit: FoundationalAuditStore,
    supplement_audit: IronSulfideSupplementAuditStore,
    expected_reconciliation_binding_sha256: str,
    expected_core_binding_sha256: str,
    expected_supplement_binding_sha256: str,
    e1a3_allocation_path: Path,
    e1a3_allocation_manifest_path: Path,
    e1a3_private_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    authenticated = _authenticate_mapping_inputs(
        store=store,
        core_audit=core_audit,
        supplement_audit=supplement_audit,
        expected_reconciliation_binding_sha256=(
            expected_reconciliation_binding_sha256
        ),
        expected_core_binding_sha256=expected_core_binding_sha256,
        expected_supplement_binding_sha256=(
            expected_supplement_binding_sha256
        ),
        e1a3_allocation_path=e1a3_allocation_path,
        e1a3_allocation_manifest_path=e1a3_allocation_manifest_path,
        e1a3_private_root=e1a3_private_root,
    )
    sources = _project_mapping_sources(authenticated)
    mapping = _mapping_payload(sources)
    binding = _mapping_binding(authenticated=authenticated, mapping=mapping)
    return mapping, binding
```

Verify all trust anchors first. Parse current decisions only from the correction artifact returned by each verified seal. Require no overlap. Query the frozen reconciliation rows, require exact E1a-3 exclusion equality, and apply only:

```python
core promotion:
    source_role == "foundational"
    substantive_status == "INELIGIBLE"
    proposed_topic in TOPICS

supplement promotion:
    source_role == "supporting"
    substantive_status == "SUBSTANTIVE"
    topic == "iron_sulfide"
```

Baseline mapped rows require `e1a4_available=1`, `e1a3_used=0`, and `substantive_status='SUBSTANTIVE'`. Core promotions enter as substantive foundational rows with their proposed topic. Supplement promotions change only their exact locator role to foundational. Group using `(source_id, topic, source_role, parser_type)` and retain other source locators in their original roles.

Validate the grouped records through Task 1, require at least 12 locators in each of eight strata, call `allocate_sampling_slots(build_sampling_slots(), sources)` once, and require 96 unique source-locator allocation keys.

- [ ] **Step 4: Implement atomic mapping seal and verification**

Use:

```python
def seal_e1a4_role_mapping(
    *,
    store: ReconciliationStore,
    core_audit: FoundationalAuditStore,
    supplement_audit: IronSulfideSupplementAuditStore,
    expected_reconciliation_binding_sha256: str,
    expected_core_binding_sha256: str,
    expected_supplement_binding_sha256: str,
    e1a3_allocation_path: Path,
    e1a3_allocation_manifest_path: Path,
    e1a3_private_root: Path,
    output_root: Path,
) -> E1A4MappingSeal:
    mapping, binding = build_e1a4_role_mapping(
        store=store,
        core_audit=core_audit,
        supplement_audit=supplement_audit,
        expected_reconciliation_binding_sha256=(
            expected_reconciliation_binding_sha256
        ),
        expected_core_binding_sha256=expected_core_binding_sha256,
        expected_supplement_binding_sha256=(
            expected_supplement_binding_sha256
        ),
        e1a3_allocation_path=e1a3_allocation_path,
        e1a3_allocation_manifest_path=e1a3_allocation_manifest_path,
        e1a3_private_root=e1a3_private_root,
    )
    return _publish_mapping_directory(mapping, binding, output_root)

def verify_e1a4_role_mapping(
    *,
    store: ReconciliationStore,
    core_audit: FoundationalAuditStore,
    supplement_audit: IronSulfideSupplementAuditStore,
    expected_reconciliation_binding_sha256: str,
    expected_core_binding_sha256: str,
    expected_supplement_binding_sha256: str,
    e1a3_allocation_path: Path,
    e1a3_allocation_manifest_path: Path,
    e1a3_private_root: Path,
    output_root: Path,
    expected_mapping_binding_sha256: str,
) -> E1A4MappingSeal:
    mapping, binding = build_e1a4_role_mapping(
        store=store,
        core_audit=core_audit,
        supplement_audit=supplement_audit,
        expected_reconciliation_binding_sha256=(
            expected_reconciliation_binding_sha256
        ),
        expected_core_binding_sha256=expected_core_binding_sha256,
        expected_supplement_binding_sha256=(
            expected_supplement_binding_sha256
        ),
        e1a3_allocation_path=e1a3_allocation_path,
        e1a3_allocation_manifest_path=e1a3_allocation_manifest_path,
        e1a3_private_root=e1a3_private_root,
    )
    return _verify_mapping_directory(
        mapping,
        binding,
        output_root,
        expected_mapping_binding_sha256,
    )
```

Implement the public seal functions with the same trust inputs as `build_e1a4_role_mapping`, plus `output_root: Path`; `verify_e1a4_role_mapping` also receives `expected_mapping_binding_sha256: str`. Both regenerate the mapping and binding through `build_e1a4_role_mapping`. The seal path calls `_publish_mapping_directory(mapping, binding, output_root)`; the verify path calls `_verify_mapping_directory(mapping, binding, output_root, expected_mapping_binding_sha256)`. These two private helpers return `E1A4MappingSeal` and implement the exact publication/verification rules below.

Publish canonical `role-mapping.v1.json`, its manifest, `mapping-binding.v1.json`, and its manifest inside one staged `e1a4-role-mapping/v1/sealed` directory. Rename the complete staged directory once. Verify exact filenames, both manifests, current regenerated payload bytes, current binding bytes, and supplied binding trust anchor. Never update reconciliation rows.

- [ ] **Step 5: Run Task 2 tests and regression checks**

```powershell
python -m pytest tests/evaluation/test_e1a4_mapping_application.py tests/evaluation/test_iron_sulfide_supplement_audit.py tests/evaluation/test_foundational_locator_audit.py tests/evaluation/test_corpus_reconciliation.py -q
python -m ruff check src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py tests/evaluation/test_e1a4_mapping_application.py
git diff --check
```

Expected: all tests, Ruff, and diff checks pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py tests/evaluation/test_e1a4_mapping_application.py
git commit -m "feat: apply verified e1a4 role corrections"
```

### Task 3: Aggregate-Only Application And Sampling-Frame Runners

**Files:**
- Create: `eval/apply_e1a4_role_corrections.py`
- Create: `eval/seal_e1a4_sampling_frame.py`
- Modify: `tests/evaluation/test_e1a4_mapping_application.py`
- Modify: `tests/evaluation/test_e1a4_sampling.py`

**Interfaces:**
- Produces CLI commands `apply`, `verify`, sampling `--preflight`, sampling seal, and sampling verify.
- Consumes the mapping APIs from Task 2, `verify_e1_index_contract`, and the fixed E1a-3 sampling primitives.

- [ ] **Step 1: Write failing CLI and frame-sealer tests**

Assert exact aggregate outputs:

```python
assert apply_output == {
    "status": "E1A4_ROLE_MAPPING_SEALED",
    "source_record_count": expected_source_records,
    "sufficient_strata_count": 8,
    "allocator_slot_count": 96,
}

assert frame_output == {
    "status": "E1A4_SAMPLING_FRAME_SEALED",
    "source_record_count": expected_source_records,
    "sufficient_strata_count": 8,
    "slot_count": 96,
}
```

Add tests that stdout/stderr never expose paths, IDs, locators, or hashes; preflight opens no private payload; malformed arguments and unexpected exceptions become fixed safe codes; mapping verification occurs before index access; an altered index contract blocks; mixed-role records survive into the source register; E1a-3 locators cannot enter allocations; and a publication failure leaves no `sampling-frame/v1` directory or staging directory.

- [ ] **Step 2: Run Task 3 tests and verify RED**

```powershell
python -m pytest tests/evaluation/test_e1a4_mapping_application.py tests/evaluation/test_e1a4_sampling.py -k "cli or frame or preflight" -q
```

Expected: failures because the runner files do not exist.

- [ ] **Step 3: Implement the mapping runner**

`eval/apply_e1a4_role_corrections.py` must accept explicit reconciliation root, run/audit IDs, expected reconciliation/core/supplement bindings, E1a-3 allocation paths/root, and `apply|verify`. It opens all stores from the same SQLite path, calls only Task 2 APIs, closes every connection, and prints aggregate JSON. It sanitizes every known or unexpected failure.

- [ ] **Step 4: Implement the standalone sampling-frame runner**

`eval/seal_e1a4_sampling_frame.py` must not import or load the E1a-3 role configuration. It accepts explicit mapping/reconciliation trust anchors, E1a-3 allocation paths, database URL, and index contract.

After mapping verification and read-only index preflight, convert the mapping records to `E1A3SourceMetadata`, build the fixed slots, allocate once, and publish under:

```text
e1a4/sampling-frame/v1/
  sealed/source-register.v1.json
  sealed/sampling-allocation.v1.json
  manifests/source-register.v1.sha256
  manifests/sampling-allocation.v1.sha256
```

The source register binds `mapping_binding_sha256`, `index_contract_sha256`, and `e1a3_allocation_sha256`. The allocation binds the source-register and E1a-3 allocation digests. Stage the complete `v1` directory beside its final path, fsync all files, rename once, then independently verify exact file set and current expected bytes.

- [ ] **Step 5: Run Task 3 tests, focused regressions, and Ruff**

```powershell
python -m pytest tests/evaluation/test_e1a4_sampling.py tests/evaluation/test_e1a4_mapping_application.py -q
python -m pytest tests/evaluation/test_iron_sulfide_supplement_audit.py tests/evaluation/test_foundational_locator_audit.py tests/evaluation/test_corpus_reconciliation.py -q
python -m ruff check src/oilfield_chemical_copilot/evaluation/e1a4_sampling.py src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py eval/apply_e1a4_role_corrections.py eval/seal_e1a4_sampling_frame.py tests/evaluation/test_e1a4_sampling.py tests/evaluation/test_e1a4_mapping_application.py
git diff --check
```

Expected: all checks pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/e1a4_sampling.py src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py eval/apply_e1a4_role_corrections.py eval/seal_e1a4_sampling_frame.py tests/evaluation/test_e1a4_sampling.py tests/evaluation/test_e1a4_mapping_application.py
git commit -m "feat: seal corrected e1a4 sampling frame"
```

### Task 4: Execute The Approved Private Application And Frame Seal

**Files:**
- Create privately only: ignored mapping and sampling-frame v1 artifact directories.

**Interfaces:**
- Consumes: current verified reconciliation, core v2, supplement v2, E1a-3 allocation, index contract, and read-only database URL.
- Produces: verified private role mapping and metadata-only 96-slot frame.

- [ ] **Step 1: Run aggregate-only preflight**

Require presence of the reconciliation database/seal, both correction seals, E1a-3 allocation plus manifest, index contract, configured database URL, absent sampling frame, and writable ignored private roots. Print booleans/status only.

- [ ] **Step 2: Seal and independently verify the role mapping**

Run the application CLI once, capture aggregate status, then run its no-write verify command against the new mapping binding digest. Require eight sufficient strata and allocator slot count 96.

- [ ] **Step 3: Seal and independently verify the sampling frame**

Run the frame sealer once, then run its no-write verification path using the direct mapping and frame binding trust anchors. Require exact four-file presence, exact manifests, source register validity, 96 unique slot identities, 96 unique source-locator keys, all four topics, both roles, and zero E1a-3 locator reuse.

- [ ] **Step 4: Confirm mutation boundaries**

Verify reconciliation `index_sources` and `index_locators` still equal the active sealed snapshots; rerun reconciliation verification; confirm E1a-3 artifact manifests are unchanged; confirm no Qdrant, PostgreSQL write, retrieval, ingestion, model, question, or claim artifact was created.

### Task 5: Public Closure, Full Verification, And Review

**Files:**
- Modify: `docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md`
- Modify: `docs/superpowers/plans/2026-08-27-e1a4-correction-application-and-sampling-frame.md`
- Modify: `docs/CURRICULUM_REMEDIATION_BACKLOG.md`

**Interfaces:**
- Consumes: aggregate verified private results only.
- Produces: synchronized public Task 2 status and a reviewed tracked commit.

- [ ] **Step 1: Update public status with aggregates only**

Record mapping/frame verification, source-record count only when disclosure-safe, eight-stratum status, exact slot count, mutation-boundary result, and the next checkpoint: private question/canonical-claim authoring. Include no identifiers, locators, paths, decisions, or hashes.

- [ ] **Step 2: Run focused and full verification**

```powershell
python -m pytest tests/evaluation/test_e1a4_sampling.py tests/evaluation/test_e1a4_mapping_application.py tests/evaluation/test_iron_sulfide_supplement_audit.py tests/evaluation/test_foundational_locator_audit.py tests/evaluation/test_corpus_reconciliation.py -q
python -m pytest -q
python -m ruff check .
git diff --check
git ls-files .private
git status --short
```

- [ ] **Step 3: Request independent public-only review**

Review the new controller, runners, tests, spec, implementation plan, aggregate public changes, and supplied private verification aggregates only. The reviewer must not inspect `.private`, mounted Drive files, raw artifacts, IDs, locators, hashes, or decisions.

- [ ] **Step 4: Address all Critical and Important findings and reverify**

Use test-first fixes, rerun focused tests and Ruff, and obtain scoped re-review approval. Record Minor findings or fix them before finalization.

- [ ] **Step 5: Commit the tracked implementation and aggregate closure**

```powershell
git add -- src/oilfield_chemical_copilot/evaluation/e1a4_sampling.py src/oilfield_chemical_copilot/evaluation/e1a4_mapping_application.py eval/apply_e1a4_role_corrections.py eval/seal_e1a4_sampling_frame.py tests/evaluation/test_e1a4_sampling.py tests/evaluation/test_e1a4_mapping_application.py docs/superpowers/specs/2026-08-27-e1a4-correction-application-and-sampling-frame-design.md docs/superpowers/plans/2026-08-27-e1a4-correction-application-and-sampling-frame.md docs/superpowers/plans/2026-08-19-e1a4-requirements-aware-evidence-gate.md docs/CURRICULUM_REMEDIATION_BACKLOG.md
git commit -m "feat: seal corrected e1a4 sampling frame"
```

## Execution Boundary

This plan ends after the metadata-only sampling frame is sealed and verified. It does not author or seal the 96 questions or canonical claims. Those remain the next explicit Task 2 checkpoint.
