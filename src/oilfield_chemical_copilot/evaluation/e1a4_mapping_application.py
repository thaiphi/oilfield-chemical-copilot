"""Authenticated, no-write application of E1a-4 locator role corrections."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import sys
from tempfile import mkdtemp
from typing import Iterator, Mapping

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
_PUBLISH_LOCK_NAME = ".sealed.publish.lock"
_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 0x00000102


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
class _InventoryRow:
    source_id: str
    locator: str
    topic: str
    source_role: str
    substantive_status: str
    e1a3_used: int
    e1a4_available: int
    parser_type: str


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
    inventory: tuple[_InventoryRow, ...]
    core: tuple[LocatorAuditDecision, ...]
    supplement: tuple[SupplementLocatorDecision, ...]


def _sealed_decisions(artifacts: object, kind: str) -> tuple[object, ...]:
    try:
        correction = next(item for item in artifacts if item.name.endswith(".jsonl"))
        content = correction.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != _digest(correction.sha256):
            _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
        if content and not content.endswith(b"\n"):
            _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
        lines = content.decode("utf-8").splitlines()
        parser = LocatorAuditDecision if kind == "core" else SupplementLocatorDecision
        decisions = tuple(parser.from_mapping(json.loads(line)) for line in lines)
        canonical = b"".join(
            _canonical(decision.to_mapping()) for decision in decisions
        )
        if canonical != content or len(decisions) != correction.record_count:
            _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
        return decisions
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
    snapshot_started = False
    try:
        expected_reconciliation = _digest(kwargs["expected_reconciliation_binding_sha256"])
        expected_core = _digest(kwargs["expected_core_binding_sha256"])
        expected_supplement = _digest(kwargs["expected_supplement_binding_sha256"])
        if core.database_path.resolve() != supplement.database_path.resolve() or core.database_path.resolve() != (store.root / "reconciliation.sqlite").resolve() or core.run_id != store.run_id or supplement.run_id != store.run_id:
            _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
        connections = {
            id(connection): connection
            for connection in (
                store._connection,
                core._connection,
                supplement._connection,
            )
        }
        if any(connection.in_transaction for connection in connections.values()):
            _fail("E1A4_MAPPING_AUTHENTICATION_FAILED")
        store._connection.execute("begin immediate")
        snapshot_started = True
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
        inventory = tuple(
            _InventoryRow(
                source_id=str(row["source_id"]),
                locator=str(row["locator"]),
                topic=str(row["topic"]),
                source_role=str(row["source_role"]),
                substantive_status=str(row["substantive_status"]),
                e1a3_used=int(row["e1a3_used"]),
                e1a4_available=int(row["e1a4_available"]),
                parser_type=str(row["parser_type"]),
            )
            for row in store._connection.execute(
                """
                select locator.source_id, locator.locator, locator.topic,
                       locator.source_role, locator.substantive_status,
                       locator.e1a3_used, locator.e1a4_available,
                       source.parser_type
                from index_locators locator join index_sources source
                  on source.run_id = locator.run_id
                 and source.source_id = locator.source_id
                where locator.run_id = ?
                order by locator.source_id, locator.locator
                """,
                (store.run_id,),
            ).fetchall()
        )
        e1a3_keys = frozenset(
            f"{row.source_id}:{row.locator}"
            for row in inventory
            if row.e1a3_used == 1
        )
        if e1a3_keys != prior.locator_keys:
            _fail("E1A4_MAPPING_E1A3_EXCLUSION_MISMATCH")
        authenticated = _Authenticated(
            run_id=store.run_id,
            reconciliation_binding=expected_reconciliation,
            core_binding=core_seal.binding_sha256,
            supplement_binding=supplement_seal.binding_sha256,
            prior_digest=prior.payload_sha256,
            prior_keys=prior.locator_keys,
            core_candidates=core_candidates,
            supplement_candidates=supplement_candidates,
            inventory=inventory,
            core=core_decisions,  # type: ignore[arg-type]
            supplement=supplement_decisions,  # type: ignore[arg-type]
        )
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
    finally:
        if snapshot_started:
            store._connection.rollback()
    return authenticated


def _project_mapping_sources(auth: _Authenticated) -> tuple[object, ...]:
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
    inventory = {(row.source_id, row.locator): row for row in auth.inventory}
    for item in core:
        row = inventory.get((item.source_id, item.locator))
        if row is None or row.source_role != "foundational" or row.substantive_status != "INELIGIBLE" or row.e1a3_used != 0 or item.proposed_topic not in TOPICS:
            _fail("E1A4_MAPPING_PROMOTION_INVALID")
    for item in supplement:
        row = inventory.get((item.source_id, item.locator))
        if row is None or row.source_role != "supporting" or row.substantive_status != "SUBSTANTIVE" or row.topic != "iron_sulfide" or row.e1a3_used != 0 or row.e1a4_available != 1:
            _fail("E1A4_MAPPING_PROMOTION_INVALID")
    grouped: dict[tuple[str, str, str, str], list[str]] = {}
    promoted = core_keys | supplement_keys
    for key, row in inventory.items():
        if key in promoted:
            continue
        if row.e1a4_available == 1 and row.e1a3_used == 0 and row.substantive_status == "SUBSTANTIVE":
            grouped.setdefault((row.source_id, row.topic, row.source_role, row.parser_type), []).append(row.locator)
    for item in core:
        row = inventory[(item.source_id, item.locator)]
        grouped.setdefault((item.source_id, str(item.proposed_topic), "foundational", row.parser_type), []).append(item.locator)
    for item in supplement:
        row = inventory[(item.source_id, item.locator)]
        grouped.setdefault((item.source_id, "iron_sulfide", "foundational", row.parser_type), []).append(item.locator)
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
    mapping = _mapping_payload(_project_mapping_sources(authenticated))
    return mapping, _mapping_binding(authenticated=authenticated, mapping=mapping)


def _mapping_directory(output_root: Path) -> Path:
    return output_root / "e1a4-role-mapping" / "v1" / "sealed"


def _rename_no_replace(staged: Path, final: Path) -> None:
    """Atomically rename one directory and fail if the destination exists."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
        )
        move_file.restype = wintypes.BOOL
        if move_file(str(staged), str(final), 0x00000008):
            return
        error_number = ctypes.get_last_error()
        if error_number in {80, 183}:
            raise FileExistsError(error_number, "destination exists", final)
        raise OSError(error_number, "exclusive rename failed", final)

    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(staged)
    destination = os.fsencode(final)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "exclusive rename unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source, -100, destination, 1)
    elif sys.platform == "darwin":
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOTSUP, "exclusive rename unavailable")
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source, destination, 0x00000004)
    else:
        raise OSError(errno.ENOTSUP, "exclusive rename unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number, os.strerror(error_number), final
            )
        raise OSError(error_number, os.strerror(error_number), final)


@dataclass(frozen=True)
class _WindowsSecurity:
    attributes: object
    descriptor: object


class _NativeWindowsMutex:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class SID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = (
                ("Sid", wintypes.LPVOID),
                ("Attributes", wintypes.DWORD),
            )

        class TOKEN_USER(ctypes.Structure):
            _fields_ = (("User", SID_AND_ATTRIBUTES),)

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = (
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (
            ctypes.POINTER(SECURITY_ATTRIBUTES),
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
        )
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.LocalFree.argtypes = (wintypes.HLOCAL,)
        kernel32.LocalFree.restype = wintypes.HLOCAL
        advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        )
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPDWORD,
        )
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.ConvertSidToStringSidW.argtypes = (
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        )
        advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        convert_security = (
            advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
        )
        convert_security.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            wintypes.LPDWORD,
        )
        convert_security.restype = wintypes.BOOL
        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._advapi32 = advapi32
        self._TOKEN_USER = TOKEN_USER
        self._SECURITY_ATTRIBUTES = SECURITY_ATTRIBUTES

    def owner_sid(self) -> str:
        token = self._ctypes.c_void_p()
        if not self._advapi32.OpenProcessToken(
            self._kernel32.GetCurrentProcess(),
            0x0008,
            self._ctypes.byref(token),
        ):
            raise OSError(self._ctypes.get_last_error(), "token unavailable")
        try:
            required = self._ctypes.c_ulong()
            self._advapi32.GetTokenInformation(token, 1, None, 0, required)
            if required.value == 0:
                raise OSError(
                    self._ctypes.get_last_error(), "token query failed"
                )
            buffer = self._ctypes.create_string_buffer(required.value)
            if not self._advapi32.GetTokenInformation(
                token,
                1,
                self._ctypes.cast(buffer, self._ctypes.c_void_p),
                required.value,
                required,
            ):
                raise OSError(
                    self._ctypes.get_last_error(), "token query failed"
                )
            token_user = self._ctypes.cast(
                buffer,
                self._ctypes.POINTER(self._TOKEN_USER),
            ).contents
            sid_text = self._ctypes.c_wchar_p()
            if not self._advapi32.ConvertSidToStringSidW(
                token_user.User.Sid,
                self._ctypes.byref(sid_text),
            ):
                raise OSError(
                    self._ctypes.get_last_error(), "SID conversion failed"
                )
            try:
                if sid_text.value is None:
                    raise OSError(errno.EIO, "SID conversion failed")
                return sid_text.value
            finally:
                if self._kernel32.LocalFree(sid_text):
                    raise OSError(
                        self._ctypes.get_last_error(), "SID free failed"
                    )
        finally:
            if not self._kernel32.CloseHandle(token):
                raise OSError(
                    self._ctypes.get_last_error(), "token close failed"
                )

    def build_security_attributes(self, policy: str) -> _WindowsSecurity:
        descriptor = self._ctypes.c_void_p()
        convert_security = (
            self._advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
        )
        if not convert_security(
            policy,
            1,
            self._ctypes.byref(descriptor),
            None,
        ):
            raise OSError(
                self._ctypes.get_last_error(),
                "security descriptor conversion failed",
            )
        attributes = self._SECURITY_ATTRIBUTES(
            self._ctypes.sizeof(self._SECURITY_ATTRIBUTES),
            descriptor,
            False,
        )
        return _WindowsSecurity(attributes, descriptor)

    def free_security_descriptor(self, security: _WindowsSecurity) -> None:
        if self._kernel32.LocalFree(security.descriptor):
            raise OSError(
                self._ctypes.get_last_error(),
                "security descriptor free failed",
            )

    def create_mutex(self, name: str, attributes: object) -> int:
        handle = self._kernel32.CreateMutexW(
            self._ctypes.byref(attributes), False, name
        )
        if not handle:
            raise OSError(
                self._ctypes.get_last_error(), "mutex creation failed"
            )
        return int(handle)

    def wait(self, handle: int, timeout_ms: int) -> int:
        return int(self._kernel32.WaitForSingleObject(handle, timeout_ms))

    def release_mutex(self, handle: int) -> None:
        if not self._kernel32.ReleaseMutex(handle):
            raise OSError(
                self._ctypes.get_last_error(), "mutex release failed"
            )

    def close_handle(self, handle: int) -> None:
        if not self._kernel32.CloseHandle(handle):
            raise OSError(
                self._ctypes.get_last_error(), "mutex close failed"
            )


def _windows_mutex_api() -> _NativeWindowsMutex:
    return _NativeWindowsMutex()


def _windows_mutex_name(parent: Path) -> str:
    resolved = parent.resolve(strict=True)
    canonical = os.path.normcase(os.path.normpath(str(resolved)))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"Global\\E1A4RoleMapping-{digest}"


def _windows_mutex_policy(owner_sid: str) -> str:
    parts = owner_sid.split("-")
    if len(parts) < 3 or parts[0] != "S" or not all(
        part.isdecimal() for part in parts[1:]
    ):
        raise OSError(errno.EINVAL, "invalid owner SID")
    return (
        f"O:{owner_sid}D:P"
        f"(A;;0x001F0001;;;{owner_sid})"
        "(A;;0x001F0001;;;SY)"
    )


@contextmanager
def _windows_publisher_lock(parent: Path) -> Iterator[None]:
    api = _windows_mutex_api()
    handle: int | None = None
    acquired = False
    try:
        policy = _windows_mutex_policy(api.owner_sid())
        security = api.build_security_attributes(policy)
        try:
            handle = api.create_mutex(
                _windows_mutex_name(parent), security.attributes
            )
        finally:
            api.free_security_descriptor(security)
        outcome = api.wait(handle, 0)
        if outcome not in {_WAIT_OBJECT_0, _WAIT_ABANDONED}:
            number = errno.EBUSY if outcome == _WAIT_TIMEOUT else errno.EIO
            raise OSError(number, "publisher mutex unavailable")
        acquired = True
    except Exception as error:
        if handle is not None:
            try:
                api.close_handle(handle)
            except Exception:
                pass
        raise E1A4MappingApplicationError(
            "E1A4_MAPPING_SEAL_WRITE_FAILED"
        ) from error

    try:
        yield
    finally:
        cleanup_error: Exception | None = None
        if acquired:
            try:
                assert handle is not None
                api.release_mutex(handle)
            except Exception as error:
                cleanup_error = error
        try:
            assert handle is not None
            api.close_handle(handle)
        except Exception as error:
            cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            raise E1A4MappingApplicationError(
                "E1A4_MAPPING_SEAL_WRITE_FAILED"
            ) from cleanup_error


def _validate_posix_lock_stat(observed: os.stat_result) -> None:
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise OSError(errno.EPERM, "unsafe publisher lock")


def _required_posix_flag(os_api: object, name: str) -> int:
    value = getattr(os_api, name, None)
    if type(value) is not int or value == 0:
        raise OSError(errno.ENOTSUP, "publisher lock primitive unavailable")
    return value


@contextmanager
def _posix_publisher_lock(
    parent: Path,
    *,
    os_api: object | None = None,
    flock_api: object | None = None,
) -> Iterator[None]:
    if os_api is None:
        os_api = os
    if flock_api is None:
        import fcntl

        flock_api = fcntl
    parent_descriptor: int | None = None
    lock_descriptor: int | None = None
    acquired = False
    try:
        directory_flag = _required_posix_flag(os_api, "O_DIRECTORY")
        nofollow_flag = _required_posix_flag(os_api, "O_NOFOLLOW")
        nonblock_flag = _required_posix_flag(os_api, "O_NONBLOCK")
        resolved_parent = parent.resolve(strict=True)
        initial_parent = os_api.lstat(resolved_parent)
        if not stat.S_ISDIR(initial_parent.st_mode):
            raise OSError(errno.ENOTDIR, "unsafe publisher parent")
        parent_flags = (
            os_api.O_RDONLY
            | directory_flag
            | nofollow_flag
            | getattr(os_api, "O_CLOEXEC", 0)
        )
        parent_descriptor = os_api.open(resolved_parent, parent_flags)
        opened_parent = os_api.fstat(parent_descriptor)
        if not stat.S_ISDIR(opened_parent.st_mode) or not os.path.samestat(
            initial_parent, opened_parent
        ):
            raise OSError(errno.EAGAIN, "publisher parent changed")

        lock_flags = (
            os_api.O_RDWR
            | nofollow_flag
            | nonblock_flag
            | getattr(os_api, "O_CLOEXEC", 0)
        )
        try:
            initial_lock = os_api.stat(
                _PUBLISH_LOCK_NAME,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            lock_descriptor = os_api.open(
                _PUBLISH_LOCK_NAME,
                lock_flags | os_api.O_CREAT | os_api.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
        else:
            _validate_posix_lock_stat(initial_lock)
            lock_descriptor = os_api.open(
                _PUBLISH_LOCK_NAME,
                lock_flags,
                dir_fd=parent_descriptor,
            )
            if not os.path.samestat(
                initial_lock, os_api.fstat(lock_descriptor)
            ):
                raise OSError(errno.EAGAIN, "publisher lock changed")

        opened_lock = os_api.fstat(lock_descriptor)
        _validate_posix_lock_stat(opened_lock)
        candidate = os_api.stat(
            _PUBLISH_LOCK_NAME,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _validate_posix_lock_stat(candidate)
        if not os.path.samestat(opened_lock, candidate):
            raise OSError(errno.EAGAIN, "publisher lock changed")
        flock_api.flock(
            lock_descriptor, flock_api.LOCK_EX | flock_api.LOCK_NB
        )
        acquired = True
        final_candidate = os_api.stat(
            _PUBLISH_LOCK_NAME,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _validate_posix_lock_stat(final_candidate)
        if not os.path.samestat(opened_lock, final_candidate):
            raise OSError(errno.EAGAIN, "publisher lock changed")
    except Exception as error:
        for descriptor in (lock_descriptor, parent_descriptor):
            if descriptor is not None:
                try:
                    os_api.close(descriptor)
                except Exception:
                    pass
        raise E1A4MappingApplicationError(
            "E1A4_MAPPING_SEAL_WRITE_FAILED"
        ) from error

    try:
        yield
    finally:
        cleanup_error: Exception | None = None
        if acquired:
            try:
                assert lock_descriptor is not None
                flock_api.flock(lock_descriptor, flock_api.LOCK_UN)
            except Exception as error:
                cleanup_error = error
        for descriptor in (lock_descriptor, parent_descriptor):
            if descriptor is not None:
                try:
                    os_api.close(descriptor)
                except Exception as error:
                    cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            raise E1A4MappingApplicationError(
                "E1A4_MAPPING_SEAL_WRITE_FAILED"
            ) from cleanup_error


@contextmanager
def _publisher_lock(parent: Path) -> Iterator[None]:
    if os.name == "nt":
        lock = _windows_publisher_lock(parent)
    elif os.name == "posix":
        lock = _posix_publisher_lock(parent)
    else:
        raise E1A4MappingApplicationError(
            "E1A4_MAPPING_SEAL_WRITE_FAILED"
        )
    with lock:
        yield


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        kernel32.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = create_file(
            str(directory),
            0x40000000,
            0x00000007,
            None,
            3,
            0x02000000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            raise OSError(
                ctypes.get_last_error(), "directory handle unavailable"
            )
        error: OSError | None = None
        try:
            if not kernel32.FlushFileBuffers(handle):
                error = OSError(
                    ctypes.get_last_error(), "directory sync failed"
                )
        finally:
            if not kernel32.CloseHandle(handle) and error is None:
                error = OSError(
                    ctypes.get_last_error(), "directory handle close failed"
                )
        if error is not None:
            raise error
        return

    directory_flag = _required_posix_flag(os, "O_DIRECTORY")
    nofollow_flag = _required_posix_flag(os, "O_NOFOLLOW")
    descriptor = os.open(
        directory,
        os.O_RDONLY
        | directory_flag
        | nofollow_flag
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise OSError(errno.ENOTDIR, "directory sync target unsafe")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_staging_name(name: str) -> bool:
    prefix = ".sealed."
    suffix = ".tmp"
    return (
        name.startswith(prefix)
        and name.endswith(suffix)
        and bool(name[len(prefix) : -len(suffix)])
    )


def _remove_abandoned_staging(parent: Path) -> None:
    resolved_parent = parent.resolve(strict=True)
    removed = False
    for candidate in parent.iterdir():
        if not _is_staging_name(candidate.name):
            continue
        candidate_stat = candidate.lstat()
        if stat.S_ISLNK(candidate_stat.st_mode):
            raise OSError(errno.EPERM, "unsafe staging directory")
        if not stat.S_ISDIR(candidate_stat.st_mode):
            continue
        resolved = candidate.resolve(strict=True)
        if resolved.parent != resolved_parent:
            raise OSError(errno.EPERM, "unsafe staging directory")
        shutil.rmtree(resolved)
        removed = True
    if removed:
        _fsync_directory(parent)


def _remove_owned_directory(
    directory: Path | None, identity: os.stat_result | None
) -> None:
    if directory is None or identity is None:
        return
    try:
        current = directory.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISDIR(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and os.path.samestat(identity, current)
    ):
        shutil.rmtree(directory)
        _fsync_directory(directory.parent)


def _remove_owned_publication(
    final: Path, identity: os.stat_result | None
) -> None:
    _remove_owned_directory(final, identity)


def _artifact(
    name: str, path: Path, digest: str, count: int
) -> E1A4MappingArtifact:
    return E1A4MappingArtifact(
        name,
        path,
        path.with_name(f"{name}.sha256"),
        digest,
        count,
    )


def _verify_mapping_directory(mapping: Mapping[str, object], binding: Mapping[str, object], output_root: Path, expected_mapping_binding_sha256: str) -> E1A4MappingSeal:
    trusted = _digest(expected_mapping_binding_sha256, "E1A4_MAPPING_BINDING_MISMATCH")
    sealed = _mapping_directory(output_root)
    try:
        observed_seal = sealed.lstat()
    except FileNotFoundError:
        _fail("E1A4_MAPPING_SEAL_MISSING")
    except OSError:
        _fail("E1A4_MAPPING_SEAL_VERIFY_FAILED")
    if (
        stat.S_ISLNK(observed_seal.st_mode)
        or not stat.S_ISDIR(observed_seal.st_mode)
    ):
        _fail("E1A4_MAPPING_SEAL_PARTIAL")
    try:
        if {item.name for item in sealed.iterdir()} != _NAMES:
            _fail("E1A4_MAPPING_SEAL_PARTIAL")
        prepared = (
            (MAPPING_NAME, _canonical(mapping)),
            (BINDING_NAME, _canonical(binding)),
        )
        paths: list[Path] = []
        digests: list[str] = []
        for name, expected in prepared:
            path = sealed / name
            manifest = path.with_name(f"{path.name}.sha256")
            digest = hashlib.sha256(expected).hexdigest()
            if (
                path.read_bytes() != expected
                or manifest.read_bytes()
                != f"{digest}\n".encode("ascii")
            ):
                _fail("E1A4_MAPPING_BINDING_MISMATCH")
            paths.append(path)
            digests.append(digest)
    except (OSError, UnicodeError):
        _fail("E1A4_MAPPING_SEAL_VERIFY_FAILED")
    if digests[1] != trusted:
        _fail("E1A4_MAPPING_BINDING_MISMATCH")
    return E1A4MappingSeal(
        (
            _artifact(
                MAPPING_NAME,
                paths[0],
                digests[0],
                len(mapping.get("sources", [])),
            ),
            _artifact(BINDING_NAME, paths[1], digests[1], 1),
        ),
        trusted,
    )


def _publish_mapping_directory(mapping: Mapping[str, object], binding: Mapping[str, object], output_root: Path) -> E1A4MappingSeal:
    sealed = _mapping_directory(output_root)
    digest = hashlib.sha256(_canonical(binding)).hexdigest()
    root = sealed.parent
    staged: Path | None = None
    staged_identity: os.stat_result | None = None
    published_identity: os.stat_result | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        with _publisher_lock(root):
            try:
                _remove_abandoned_staging(root)
                try:
                    sealed.lstat()
                except FileNotFoundError:
                    pass
                else:
                    return _verify_mapping_directory(
                        mapping, binding, output_root, digest
                    )

                staged = Path(
                    mkdtemp(prefix=".sealed.", suffix=".tmp", dir=root)
                )
                staged_identity = staged.lstat()
                for name, content in (
                    (MAPPING_NAME, _canonical(mapping)),
                    (BINDING_NAME, _canonical(binding)),
                ):
                    manifest_content = (
                        hashlib.sha256(content).hexdigest() + "\n"
                    ).encode("ascii")
                    for path, value in (
                        (staged / name, content),
                        (staged / f"{name}.sha256", manifest_content),
                    ):
                        with path.open("xb") as stream:
                            stream.write(value)
                            stream.flush()
                            os.fsync(stream.fileno())
                _fsync_directory(staged)
                _rename_no_replace(staged, sealed)
                published_identity = staged_identity
                staged = None
                _fsync_directory(root)
                result = _verify_mapping_directory(
                    mapping, binding, output_root, digest
                )
            except E1A4MappingApplicationError:
                _remove_owned_publication(sealed, published_identity)
                _remove_owned_directory(staged, staged_identity)
                raise
            except Exception as error:
                _remove_owned_publication(sealed, published_identity)
                _remove_owned_directory(staged, staged_identity)
                raise E1A4MappingApplicationError(
                    "E1A4_MAPPING_SEAL_WRITE_FAILED"
                ) from error
        return result
    except E1A4MappingApplicationError:
        raise
    except Exception as error:
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
