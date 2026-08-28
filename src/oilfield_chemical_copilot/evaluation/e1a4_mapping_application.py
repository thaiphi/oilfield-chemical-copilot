"""Authenticated, no-write application of E1a-4 locator role corrections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import mkdtemp
from typing import Mapping

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
    CorpusReconciliationError,
    ReconciliationStore,
    TOPICS,
    verify_reconciliation_snapshots,
)
from oilfield_chemical_copilot.evaluation.e1a3_sampling import (
    E1A3SamplingError,
    allocate_sampling_slots,
    build_sampling_slots,
)
from oilfield_chemical_copilot.evaluation.e1a4_sampling import (
    E1A4SamplingError,
    load_e1a3_prior_allocation,
    mapping_sources_as_sampling_metadata,
    validate_mapping_sources,
)
from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (
    FoundationalAuditStore,
    FoundationalLocatorAuditError,
    LocatorAuditDecision,
    verify_correction_proposal,
)
from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (
    IronSulfideSupplementAuditError,
    IronSulfideSupplementAuditStore,
    SupplementLocatorDecision,
    verify_supplement_proposal,
)


MAPPING_NAME = "role-mapping.v1.json"
BINDING_NAME = "mapping-binding.v1.json"
_NAMES = frozenset((MAPPING_NAME, f"{MAPPING_NAME}.sha256", BINDING_NAME, f"{BINDING_NAME}.sha256"))


class E1A4MappingApplicationError(RuntimeError):
    """Raised with a safe mapping-application error code."""


def _fail(code: str) -> None:
    raise E1A4MappingApplicationError(code)


def _digest(value: object, code: str = "E1A4_MAPPING_AUTHENTICATION_FAILED") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _fail(code)
    return value


def _canonical(value: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


@dataclass(frozen=True)
class E1A4MappingArtifact:
    name: str
    path: Path
    manifest_path: Path
    sha256: str
    record_count: int


@dataclass(frozen=True)
class E1A4MappingSeal:
    artifacts: tuple[E1A4MappingArtifact, ...]
    binding_sha256: str


@dataclass(frozen=True)
class _Authenticated:
    run_id: str
    reconciliation_binding: str
    core_binding: str
    supplement_binding: str
    prior_digest: str
    prior_keys: frozenset[str]
    core_candidates: frozenset[tuple[str, str]]
    supplement_candidates: frozenset[tuple[str, str]]
    core: tuple[LocatorAuditDecision, ...]
    supplement: tuple[SupplementLocatorDecision, ...]


def _sealed_decisions(artifacts: object, kind: str) -> tuple[object, ...]:
    try:
        correction = next(item for item in artifacts if item.name.endswith(".jsonl"))
        lines = correction.path.read_text(encoding="utf-8").splitlines()
        parser = LocatorAuditDecision if kind == "core" else SupplementLocatorDecision
        return tuple(parser.from_mapping(json.loads(line)) for line in lines)
    except (
        AttributeError,
        FoundationalLocatorAuditError,
        IronSulfideSupplementAuditError,
        json.JSONDecodeError,
        OSError,
        StopIteration,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")


def _authenticate_mapping_inputs(**kwargs: object) -> _Authenticated:
    store = kwargs["store"]
    core = kwargs["core_audit"]
    supplement = kwargs["supplement_audit"]
    if not isinstance(store, ReconciliationStore) or not isinstance(core, FoundationalAuditStore) or not isinstance(supplement, IronSulfideSupplementAuditStore):
        _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
    try:
        expected_reconciliation = _digest(kwargs["expected_reconciliation_binding_sha256"])
        expected_core = _digest(kwargs["expected_core_binding_sha256"])
        expected_supplement = _digest(kwargs["expected_supplement_binding_sha256"])
        if core.database_path.resolve() != supplement.database_path.resolve() or core.database_path.resolve() != (store.root / "reconciliation.sqlite").resolve() or core.run_id != store.run_id or supplement.run_id != store.run_id:
            _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
        verify_reconciliation_snapshots(root=store.root, store=store, expected_binding_sha256=expected_reconciliation)
        core_snapshot = core._connection.execute(
            "select snapshot_binding_sha256 from foundational_audit_runs where run_id = ? and audit_id = ?",
            (core.run_id, core.audit_id),
        ).fetchone()
        supplement_snapshot = supplement._connection.execute(
            "select snapshot_binding_sha256 from iron_sulfide_supplement_audit_runs where run_id = ? and audit_id = ?",
            (supplement.run_id, supplement.audit_id),
        ).fetchone()
        if (
            core_snapshot is None
            or supplement_snapshot is None
            or str(core_snapshot["snapshot_binding_sha256"]) != expected_reconciliation
            or str(supplement_snapshot["snapshot_binding_sha256"]) != expected_reconciliation
        ):
            _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
        core_seal = verify_correction_proposal(audit=core, expected_binding_sha256=expected_core)
        supplement_seal = verify_supplement_proposal(audit=supplement, core_audit=core, expected_binding_sha256=expected_supplement, expected_core_binding_sha256=expected_core)
        prior = load_e1a3_prior_allocation(payload_path=kwargs["e1a3_allocation_path"], manifest_path=kwargs["e1a3_allocation_manifest_path"], private_root=kwargs["e1a3_private_root"])
        if prior.payload_sha256 != store.contract_digests()[1]:
            _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
        core_candidates = frozenset(
            (str(row["source_id"]), str(row["locator"]))
            for row in core._connection.execute(
                """
                select source_id, locator from foundational_audit_candidates
                where run_id = ? and audit_id = ?
                """,
                (core.run_id, core.audit_id),
            ).fetchall()
        )
        supplement_candidates = frozenset(
            (str(row["source_id"]), str(row["locator"]))
            for row in supplement._connection.execute(
                """
                select source_id, locator
                from iron_sulfide_supplement_audit_candidates
                where run_id = ? and audit_id = ?
                """,
                (supplement.run_id, supplement.audit_id),
            ).fetchall()
        )
        core_decisions = _sealed_decisions(core_seal.artifacts, "core")
        supplement_decisions = _sealed_decisions(
            supplement_seal.artifacts, "supplement"
        )
        rows = store._connection.execute(
            """
            select source_id, locator from index_locators
            where run_id = ? and e1a3_used = 1
            """,
            (store.run_id,),
        ).fetchall()
    except E1A4MappingApplicationError:
        raise
    except (
        CorpusReconciliationError,
        E1A4SamplingError,
        FoundationalLocatorAuditError,
        IronSulfideSupplementAuditError,
        KeyError,
        OSError,
        sqlite3.Error,
        TypeError,
        ValueError,
    ):
        _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
    if frozenset(f"{row['source_id']}:{row['locator']}" for row in rows) != prior.locator_keys:
        _fail("E1A4_MAPPING_E1A3_EXCLUSION_MISMATCH")
    return _Authenticated(
        run_id=store.run_id,
        reconciliation_binding=expected_reconciliation,
        core_binding=core_seal.binding_sha256,
        supplement_binding=supplement_seal.binding_sha256,
        prior_digest=prior.payload_sha256,
        prior_keys=prior.locator_keys,
        core_candidates=core_candidates,
        supplement_candidates=supplement_candidates,
        core=core_decisions,  # type: ignore[arg-type]
        supplement=supplement_decisions,  # type: ignore[arg-type]
    )


def _project_mapping_sources(auth: _Authenticated, *, store: ReconciliationStore) -> tuple[object, ...]:
    if any(
        item.decision not in {"PROMOTE_FOUNDATIONAL", "KEEP_INELIGIBLE"}
        for item in auth.core
    ) or any(
        item.decision not in {"PROMOTE_FOUNDATIONAL", "KEEP_SUPPORTING"}
        for item in auth.supplement
    ):
        _fail("E1A4_MAPPING_PROPOSAL_UNRESOLVED")
    core_proposal_keys = {(item.source_id, item.locator) for item in auth.core}
    supplement_proposal_keys = {
        (item.source_id, item.locator) for item in auth.supplement
    }
    if core_proposal_keys & supplement_proposal_keys:
        _fail("E1A4_MAPPING_PROMOTION_OVERLAP")
    if not core_proposal_keys.issubset(
        auth.core_candidates
    ) or not supplement_proposal_keys.issubset(auth.supplement_candidates):
        _fail("E1A4_MAPPING_PROMOTION_INVALID")
    core = tuple(item for item in auth.core if item.decision == "PROMOTE_FOUNDATIONAL")
    supplement = tuple(item for item in auth.supplement if item.decision == "PROMOTE_FOUNDATIONAL")
    core_keys = {(item.source_id, item.locator) for item in core}
    supplement_keys = {(item.source_id, item.locator) for item in supplement}
    if any(f"{source_id}:{locator}" in auth.prior_keys for source_id, locator in core_keys | supplement_keys):
        _fail("E1A4_MAPPING_PROMOTION_INVALID")
    rows = store._connection.execute("""
        select locator.source_id, locator.locator, locator.topic, locator.source_role,
               locator.substantive_status, locator.e1a3_used, locator.e1a4_available,
               source.parser_type
        from index_locators locator join index_sources source
        on source.run_id = locator.run_id and source.source_id = locator.source_id
        where locator.run_id = ? order by locator.source_id, locator.locator
    """, (store.run_id,)).fetchall()
    inventory = {(str(row["source_id"]), str(row["locator"])): row for row in rows}
    for item in core:
        row = inventory.get((item.source_id, item.locator))
        if row is None or row["source_role"] != "foundational" or row["substantive_status"] != "INELIGIBLE" or row["e1a3_used"] != 0 or item.proposed_topic not in TOPICS:
            _fail("E1A4_MAPPING_PROMOTION_INVALID")
    for item in supplement:
        row = inventory.get((item.source_id, item.locator))
        if row is None or row["source_role"] != "supporting" or row["substantive_status"] != "SUBSTANTIVE" or row["topic"] != "iron_sulfide" or row["e1a3_used"] != 0 or row["e1a4_available"] != 1:
            _fail("E1A4_MAPPING_PROMOTION_INVALID")
    grouped: dict[tuple[str, str, str, str], list[str]] = {}
    promoted = core_keys | supplement_keys
    for key, row in inventory.items():
        if key in promoted:
            continue
        if row["e1a4_available"] == 1 and row["e1a3_used"] == 0 and row["substantive_status"] == "SUBSTANTIVE":
            grouped.setdefault((str(row["source_id"]), str(row["topic"]), str(row["source_role"]), str(row["parser_type"])), []).append(str(row["locator"]))
    for item in core:
        row = inventory[(item.source_id, item.locator)]
        grouped.setdefault((item.source_id, str(item.proposed_topic), "foundational", str(row["parser_type"])), []).append(item.locator)
    for item in supplement:
        row = inventory[(item.source_id, item.locator)]
        grouped.setdefault((item.source_id, "iron_sulfide", "foundational", str(row["parser_type"])), []).append(item.locator)
    values = tuple({"source_id": source_id, "topic": topic, "source_role": role, "parser_type": parser, "locators": sorted(set(locators))} for (source_id, topic, role, parser), locators in sorted(grouped.items()))
    try:
        sources = validate_mapping_sources(values)
        counts = {(topic, role): 0 for topic in ("iron_sulfide", "scale", "corrosion", "paraffin") for role in ("foundational", "supporting")}
        for source in sources:
            counts[(source.topic, source.source_role)] += len(source.locators)
        if any(count < 12 for count in counts.values()):
            _fail("E1A4_MAPPING_STRATUM_INSUFFICIENT")
        allocations = allocate_sampling_slots(slots=build_sampling_slots(), sources=mapping_sources_as_sampling_metadata(sources))
        if len(allocations) != 96 or len({(row.source_id, row.locator) for row in allocations}) != 96:
            _fail("E1A4_MAPPING_ALLOCATION_INVALID")
    except (E1A4SamplingError, E1A3SamplingError):
        _fail("E1A4_MAPPING_ALLOCATION_UNAVAILABLE")
    return tuple(source.to_mapping() for source in sources)


def _mapping_payload(sources: tuple[object, ...]) -> dict[str, object]:
    return {"schema_version": 1, "sources": list(sources)}


def _mapping_binding(*, authenticated: _Authenticated, mapping: Mapping[str, object]) -> dict[str, object]:
    sources = validate_mapping_sources(mapping["sources"])  # type: ignore[arg-type]
    counts = {f"{topic}:{role}": sum(len(item.locators) for item in sources if item.topic == topic and item.source_role == role) for topic in ("iron_sulfide", "scale", "corrosion", "paraffin") for role in ("foundational", "supporting")}
    return {
        "schema_version": 1,
        "reconciliation_run_id": authenticated.run_id,
        "reconciliation_binding_sha256": authenticated.reconciliation_binding,
        "core_binding_sha256": authenticated.core_binding,
        "supplement_binding_sha256": authenticated.supplement_binding,
        "e1a3_allocation_sha256": authenticated.prior_digest,
        "mapping_payload_sha256": hashlib.sha256(_canonical(mapping)).hexdigest(),
        "source_record_count": len(sources),
        "unique_locator_count": sum(len(item.locators) for item in sources),
        "stratum_locator_counts": counts,
        "allocator_available": True,
        "allocator_slot_count": 96,
        "e1a3_excluded_before_allocation": True,
    }


def build_e1a4_role_mapping(*, store: ReconciliationStore, core_audit: FoundationalAuditStore, supplement_audit: IronSulfideSupplementAuditStore, expected_reconciliation_binding_sha256: str, expected_core_binding_sha256: str, expected_supplement_binding_sha256: str, e1a3_allocation_path: Path, e1a3_allocation_manifest_path: Path, e1a3_private_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    authenticated = _authenticate_mapping_inputs(store=store, core_audit=core_audit, supplement_audit=supplement_audit, expected_reconciliation_binding_sha256=expected_reconciliation_binding_sha256, expected_core_binding_sha256=expected_core_binding_sha256, expected_supplement_binding_sha256=expected_supplement_binding_sha256, e1a3_allocation_path=e1a3_allocation_path, e1a3_allocation_manifest_path=e1a3_allocation_manifest_path, e1a3_private_root=e1a3_private_root)
    mapping = _mapping_payload(_project_mapping_sources(authenticated, store=store))
    return mapping, _mapping_binding(authenticated=authenticated, mapping=mapping)


def _mapping_directory(output_root: Path) -> Path:
    return output_root / "e1a4-role-mapping" / "v1" / "sealed"


def _artifact(name: str, path: Path, count: int) -> E1A4MappingArtifact:
    return E1A4MappingArtifact(name, path, path.with_name(f"{name}.sha256"), hashlib.sha256(path.read_bytes()).hexdigest(), count)


def _verify_mapping_directory(mapping: Mapping[str, object], binding: Mapping[str, object], output_root: Path, expected_mapping_binding_sha256: str) -> E1A4MappingSeal:
    trusted = _digest(expected_mapping_binding_sha256, "E1A4_MAPPING_BINDING_MISMATCH")
    sealed = _mapping_directory(output_root)
    if not sealed.exists():
        _fail("E1A4_MAPPING_SEAL_MISSING")
    if not sealed.is_dir() or {item.name for item in sealed.iterdir()} != _NAMES:
        _fail("E1A4_MAPPING_SEAL_PARTIAL")
    paths = (sealed / MAPPING_NAME, sealed / BINDING_NAME)
    try:
        for path, expected in zip(paths, (_canonical(mapping), _canonical(binding)), strict=True):
            manifest = path.with_name(f"{path.name}.sha256")
            if path.read_bytes() != expected or (
                manifest.read_text(encoding="ascii")
                != hashlib.sha256(expected).hexdigest() + "\n"
            ):
                _fail("E1A4_MAPPING_BINDING_MISMATCH")
    except (OSError, UnicodeError):
        _fail("E1A4_MAPPING_SEAL_VERIFY_FAILED")
    if hashlib.sha256(paths[1].read_bytes()).hexdigest() != trusted:
        _fail("E1A4_MAPPING_BINDING_MISMATCH")
    return E1A4MappingSeal((_artifact(MAPPING_NAME, paths[0], len(mapping.get("sources", []))), _artifact(BINDING_NAME, paths[1], 1)), trusted)


def _publish_mapping_directory(mapping: Mapping[str, object], binding: Mapping[str, object], output_root: Path) -> E1A4MappingSeal:
    sealed = _mapping_directory(output_root)
    digest = hashlib.sha256(_canonical(binding)).hexdigest()
    if sealed.exists():
        return _verify_mapping_directory(mapping, binding, output_root, digest)
    root = sealed.parent
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob(".sealed.*.tmp"):
        shutil.rmtree(stale, ignore_errors=True)
    staged = Path(mkdtemp(prefix=".sealed.", suffix=".tmp", dir=root))
    try:
        for name, content in ((MAPPING_NAME, _canonical(mapping)), (BINDING_NAME, _canonical(binding))):
            path = staged / name
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            (staged / f"{name}.sha256").write_bytes((hashlib.sha256(content).hexdigest() + "\n").encode("ascii"))
        os.replace(staged, sealed)
        return _verify_mapping_directory(mapping, binding, output_root, digest)
    except (E1A4MappingApplicationError, OSError) as error:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise E1A4MappingApplicationError("E1A4_MAPPING_SEAL_WRITE_FAILED") from error


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
        expected_supplement_binding_sha256=expected_supplement_binding_sha256,
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
        expected_supplement_binding_sha256=expected_supplement_binding_sha256,
        e1a3_allocation_path=e1a3_allocation_path,
        e1a3_allocation_manifest_path=e1a3_allocation_manifest_path,
        e1a3_private_root=e1a3_private_root,
    )
    return _verify_mapping_directory(mapping, binding, output_root, expected_mapping_binding_sha256)
