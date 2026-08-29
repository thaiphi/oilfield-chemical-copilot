"""Authenticated, no-write application of E1a-4 locator role corrections."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
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
from oilfield_chemical_copilot.evaluation.private_artifact_publication import (
    AuthenticatedPublicationDirectory,
    PrivateArtifactPublicationError,
    authenticated_publication_directory,
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


def _required_posix_flag(os_api: object, name: str) -> int:
    value = getattr(os_api, name, None)
    if type(value) is not int or value == 0:
        raise OSError(errno.ENOTSUP, "publisher lock primitive unavailable")
    return value


def _posix_member_snapshot(observed: os.stat_result) -> tuple[object, ...]:
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise OSError(errno.EPERM, "unsafe sealed member")
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _read_posix_member(descriptor: int, *, os_api: object) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os_api.read(descriptor, 65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _attempt_resource_closes(
    resources: tuple[object | None, ...],
    *,
    close: object,
    first_error: BaseException | None = None,
) -> BaseException | None:
    for resource in resources:
        if resource is None:
            continue
        try:
            close(resource)
        except BaseException as error:
            if first_error is None:
                first_error = error
    return first_error


def _read_posix_sealed_members(
    sealed: Path, *, os_api: object | None = None
) -> dict[str, bytes]:
    if os_api is None:
        os_api = os
    directory_flag = _required_posix_flag(os_api, "O_DIRECTORY")
    nofollow_flag = _required_posix_flag(os_api, "O_NOFOLLOW")
    nonblock_flag = _required_posix_flag(os_api, "O_NONBLOCK")
    directory_descriptor: int | None = None
    body_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    captured: dict[str, bytes] = {}
    try:
        directory_descriptor = os_api.open(
            sealed,
            os_api.O_RDONLY
            | directory_flag
            | nofollow_flag
            | getattr(os_api, "O_CLOEXEC", 0),
        )
        directory_stat = os_api.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise OSError(errno.ENOTDIR, "unsafe sealed directory")
        entries = set(os_api.listdir(directory_descriptor))
        if entries != _NAMES:
            _fail("E1A4_MAPPING_SEAL_PARTIAL")
        for name in sorted(_NAMES):
            descriptor: int | None = None
            current_descriptor: int | None = None
            member_error: BaseException | None = None
            try:
                before = _posix_member_snapshot(
                    os_api.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                )
                descriptor = os_api.open(
                    name,
                    os_api.O_RDONLY
                    | nofollow_flag
                    | nonblock_flag
                    | getattr(os_api, "O_CLOEXEC", 0),
                    dir_fd=directory_descriptor,
                )
                opened = _posix_member_snapshot(os_api.fstat(descriptor))
                if before != opened:
                    raise OSError(errno.EAGAIN, "sealed member changed")
                content = _read_posix_member(descriptor, os_api=os_api)
                after = _posix_member_snapshot(os_api.fstat(descriptor))
                current_preopen = _posix_member_snapshot(
                    os_api.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                )
                current_descriptor = os_api.open(
                    name,
                    os_api.O_RDONLY
                    | nofollow_flag
                    | nonblock_flag
                    | getattr(os_api, "O_CLOEXEC", 0),
                    dir_fd=directory_descriptor,
                )
                current = _posix_member_snapshot(
                    os_api.fstat(current_descriptor)
                )
                if (
                    current_preopen != current
                    or before != after
                    or before != current
                ):
                    raise OSError(errno.EAGAIN, "sealed member changed")
                captured[name] = content
            except BaseException as error:
                member_error = error
            cleanup_error = _attempt_resource_closes(
                (current_descriptor, descriptor),
                close=os_api.close,
                first_error=cleanup_error,
            )
            if member_error is not None:
                raise member_error
            if cleanup_error is not None:
                break
        final_directory = os_api.fstat(directory_descriptor)
        if not os.path.samestat(directory_stat, final_directory):
            raise OSError(errno.EAGAIN, "sealed directory changed")
    except BaseException as error:
        body_error = error
    cleanup_error = _attempt_resource_closes(
        (directory_descriptor,),
        close=os_api.close,
        first_error=cleanup_error,
    )
    if cleanup_error is not None:
        if body_error is not None:
            raise cleanup_error from body_error
        raise cleanup_error
    if body_error is not None:
        raise body_error
    return captured


class _NativeWindowsSealReader:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = (
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", wintypes.LPWSTR),
            )

        class OBJECT_ATTRIBUTES(ctypes.Structure):
            _fields_ = (
                ("Length", wintypes.ULONG),
                ("RootDirectory", wintypes.HANDLE),
                ("ObjectName", ctypes.POINTER(UNICODE_STRING)),
                ("Attributes", wintypes.ULONG),
                ("SecurityDescriptor", wintypes.LPVOID),
                ("SecurityQualityOfService", wintypes.LPVOID),
            )

        class IO_STATUS_BLOCK(ctypes.Structure):
            _fields_ = (
                ("Status", ctypes.c_ssize_t),
                ("Information", ctypes.c_size_t),
            )

        class FILE_STANDARD_INFO(ctypes.Structure):
            _fields_ = (
                ("AllocationSize", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.DWORD),
                ("DeletePending", wintypes.BOOLEAN),
                ("Directory", wintypes.BOOLEAN),
            )

        class FILE_BASIC_INFO(ctypes.Structure):
            _fields_ = (
                ("CreationTime", ctypes.c_longlong),
                ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong),
                ("ChangeTime", ctypes.c_longlong),
                ("FileAttributes", wintypes.DWORD),
            )

        class FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
            _fields_ = (
                ("FileAttributes", wintypes.DWORD),
                ("ReparseTag", wintypes.DWORD),
            )

        class FILE_ID_INFO(ctypes.Structure):
            _fields_ = (
                ("VolumeSerialNumber", ctypes.c_ulonglong),
                ("FileId", ctypes.c_ubyte * 16),
            )

        class FILE_ID_BOTH_DIR_INFO(ctypes.Structure):
            _fields_ = (
                ("NextEntryOffset", wintypes.DWORD),
                ("FileIndex", wintypes.DWORD),
                ("CreationTime", ctypes.c_longlong),
                ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong),
                ("ChangeTime", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("AllocationSize", ctypes.c_longlong),
                ("FileAttributes", wintypes.DWORD),
                ("FileNameLength", wintypes.DWORD),
                ("EaSize", wintypes.DWORD),
                ("ShortNameLength", ctypes.c_ubyte),
                ("ShortName", wintypes.WCHAR * 12),
                ("FileId", ctypes.c_longlong),
                ("FileName", wintypes.WCHAR * 1),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        kernel32.CreateFileW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.GetFileType.argtypes = (wintypes.HANDLE,)
        kernel32.GetFileType.restype = wintypes.DWORD
        kernel32.GetFileInformationByHandleEx.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = (
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPDWORD,
            wintypes.LPVOID,
        )
        kernel32.ReadFile.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        ntdll.NtCreateFile.argtypes = (
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            ctypes.POINTER(OBJECT_ATTRIBUTES),
            ctypes.POINTER(IO_STATUS_BLOCK),
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        ntdll.NtCreateFile.restype = ctypes.c_long
        ntdll.RtlNtStatusToDosError.argtypes = (ctypes.c_long,)
        ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = kernel32
        self._ntdll = ntdll
        self._UNICODE_STRING = UNICODE_STRING
        self._OBJECT_ATTRIBUTES = OBJECT_ATTRIBUTES
        self._IO_STATUS_BLOCK = IO_STATUS_BLOCK
        self._FILE_STANDARD_INFO = FILE_STANDARD_INFO
        self._FILE_BASIC_INFO = FILE_BASIC_INFO
        self._FILE_ATTRIBUTE_TAG_INFO = FILE_ATTRIBUTE_TAG_INFO
        self._FILE_ID_INFO = FILE_ID_INFO
        self._directory_info = FILE_ID_BOTH_DIR_INFO
        self._directory_name_offset = FILE_ID_BOTH_DIR_INFO.FileName.offset

    def _query(self, handle: object, info_class: int, result: object) -> None:
        if not self._kernel32.GetFileInformationByHandleEx(
            handle,
            info_class,
            self._ctypes.byref(result),
            self._ctypes.sizeof(result),
        ):
            raise OSError(
                self._ctypes.get_last_error(), "file information unavailable"
            )

    def _validate_handle(
        self, handle: object, *, directory: bool
    ) -> tuple[object, ...]:
        if self._kernel32.GetFileType(handle) != 1:
            raise OSError(errno.EPERM, "sealed member is not a disk file")
        attributes = self._FILE_ATTRIBUTE_TAG_INFO()
        standard = self._FILE_STANDARD_INFO()
        basic = self._FILE_BASIC_INFO()
        identity = self._FILE_ID_INFO()
        self._query(handle, 9, attributes)
        self._query(handle, 1, standard)
        self._query(handle, 0, basic)
        self._query(handle, 18, identity)
        if (
            bool(attributes.FileAttributes & 0x00000400)
            or bool(standard.Directory) != directory
            or (directory and not attributes.FileAttributes & 0x00000010)
            or (not directory and standard.NumberOfLinks != 1)
        ):
            raise OSError(errno.EPERM, "unsafe sealed object")
        return (
            int(identity.VolumeSerialNumber),
            bytes(identity.FileId),
            int(standard.EndOfFile),
            int(basic.LastWriteTime),
            int(basic.ChangeTime),
        )

    def open_directory(self, path: Path) -> int:
        handle = self._kernel32.CreateFileW(
            str(path),
            0x00100081,
            0x00000007,
            None,
            3,
            0x02200000,
            None,
        )
        if handle == self._ctypes.c_void_p(-1).value:
            raise OSError(
                self._ctypes.get_last_error(), "sealed directory unavailable"
            )
        try:
            self._validate_handle(handle, directory=True)
        except Exception:
            self.close_handle(handle)
            raise
        return int(handle)

    def directory_entries(self, handle: object) -> set[str]:
        names: set[str] = set()
        while True:
            buffer = self._ctypes.create_string_buffer(65536)
            if not self._kernel32.GetFileInformationByHandleEx(
                handle, 10, buffer, len(buffer)
            ):
                error_number = self._ctypes.get_last_error()
                if error_number == 18:
                    return names
                raise OSError(error_number, "directory enumeration failed")
            offset = 0
            while True:
                entry = self._directory_info.from_buffer(buffer, offset)
                name = self._ctypes.wstring_at(
                    self._ctypes.addressof(buffer)
                    + offset
                    + self._directory_name_offset,
                    entry.FileNameLength // 2,
                )
                if name not in {".", ".."}:
                    names.add(name)
                if entry.NextEntryOffset == 0:
                    break
                offset += entry.NextEntryOffset

    def open_member(self, directory: object, name: str) -> int:
        name_buffer = self._ctypes.create_unicode_buffer(name)
        name_length = len(name.encode("utf-16-le"))
        unicode_name = self._UNICODE_STRING(
            name_length,
            name_length + 2,
            self._ctypes.cast(name_buffer, self._wintypes.LPWSTR),
        )
        attributes = self._OBJECT_ATTRIBUTES(
            self._ctypes.sizeof(self._OBJECT_ATTRIBUTES),
            directory,
            self._ctypes.pointer(unicode_name),
            0x00000040,
            None,
            None,
        )
        status_block = self._IO_STATUS_BLOCK()
        handle = self._wintypes.HANDLE()
        status = self._ntdll.NtCreateFile(
            self._ctypes.byref(handle),
            0x00100081,
            self._ctypes.byref(attributes),
            self._ctypes.byref(status_block),
            None,
            0,
            0x00000007,
            1,
            0x00200060,
            None,
            0,
        )
        if status < 0:
            number = int(self._ntdll.RtlNtStatusToDosError(status))
            raise OSError(number, "sealed member open failed")
        return int(handle.value)

    def member_snapshot(self, handle: object) -> tuple[object, ...]:
        return self._validate_handle(handle, directory=False)

    def read_member(self, handle: object) -> bytes:
        chunks: list[bytes] = []
        while True:
            buffer = self._ctypes.create_string_buffer(65536)
            read = self._wintypes.DWORD()
            if not self._kernel32.ReadFile(
                handle,
                buffer,
                len(buffer),
                self._ctypes.byref(read),
                None,
            ):
                raise OSError(
                    self._ctypes.get_last_error(), "sealed member read failed"
                )
            if read.value == 0:
                return b"".join(chunks)
            chunks.append(buffer.raw[: read.value])

    def close_handle(self, handle: object) -> None:
        if not self._kernel32.CloseHandle(handle):
            raise OSError(
                self._ctypes.get_last_error(), "sealed handle close failed"
            )


def _windows_seal_reader_api() -> _NativeWindowsSealReader:
    return _NativeWindowsSealReader()


def _read_windows_sealed_members(
    sealed: Path, *, api: object | None = None
) -> dict[str, bytes]:
    if api is None:
        api = _windows_seal_reader_api()
    directory: object | None = None
    body_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    captured: dict[str, bytes] = {}
    try:
        directory = api.open_directory(sealed)
        if api.directory_entries(directory) != set(_NAMES):
            _fail("E1A4_MAPPING_SEAL_PARTIAL")
        for name in sorted(_NAMES):
            member: object | None = None
            current: object | None = None
            member_error: BaseException | None = None
            try:
                member = api.open_member(directory, name)
                before = api.member_snapshot(member)
                content = api.read_member(member)
                after = api.member_snapshot(member)
                current = api.open_member(directory, name)
                current_snapshot = api.member_snapshot(current)
                if before != after or before != current_snapshot:
                    raise OSError(errno.EAGAIN, "sealed member changed")
                captured[name] = content
            except BaseException as error:
                member_error = error
            cleanup_error = _attempt_resource_closes(
                (current, member),
                close=api.close_handle,
                first_error=cleanup_error,
            )
            if member_error is not None:
                raise member_error
            if cleanup_error is not None:
                break
    except BaseException as error:
        body_error = error
    cleanup_error = _attempt_resource_closes(
        (directory,),
        close=api.close_handle,
        first_error=cleanup_error,
    )
    if cleanup_error is not None:
        if body_error is not None:
            raise cleanup_error from body_error
        raise cleanup_error
    if body_error is not None:
        raise body_error
    return captured


def _read_sealed_members(sealed: Path) -> dict[str, bytes]:
    if os.name == "nt":
        return _read_windows_sealed_members(sealed)
    if os.name == "posix":
        return _read_posix_sealed_members(sealed)
    raise OSError(errno.ENOTSUP, "safe sealed-member open unavailable")


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
        members = _read_sealed_members(sealed)
        prepared = (
            (MAPPING_NAME, _canonical(mapping)),
            (BINDING_NAME, _canonical(binding)),
        )
        paths: list[Path] = []
        digests: list[str] = []
        for name, expected in prepared:
            path = sealed / name
            digest = hashlib.sha256(expected).hexdigest()
            if (
                members[name] != expected
                or members[f"{name}.sha256"]
                != f"{digest}\n".encode("ascii")
            ):
                _fail("E1A4_MAPPING_BINDING_MISMATCH")
            paths.append(path)
            digests.append(digest)
    except E1A4MappingApplicationError:
        raise
    except FileNotFoundError:
        _fail("E1A4_MAPPING_SEAL_MISSING")
    except (OSError, AttributeError, TypeError, ValueError) as error:
        if getattr(error, "errno", None) in {errno.ENOENT, 3} or getattr(
            error, "winerror", None
        ) in {
            2,
            3,
        }:
            _fail("E1A4_MAPPING_SEAL_MISSING")
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


def _verify_mapping_publication(
    publication: AuthenticatedPublicationDirectory,
    mapping: Mapping[str, object],
    binding: Mapping[str, object],
    output_root: Path,
    expected_mapping_binding_sha256: str,
) -> E1A4MappingSeal:
    trusted = _digest(
        expected_mapping_binding_sha256,
        "E1A4_MAPPING_BINDING_MISMATCH",
    )
    try:
        members = publication.read_exact_tree(
            "v1",
            {"sealed": _NAMES},
        )
    except PrivateArtifactPublicationError as error:
        if str(error) == "PRIVATE_ARTIFACT_TREE_INVALID":
            _fail("E1A4_MAPPING_SEAL_PARTIAL")
        raise E1A4MappingApplicationError(
            "E1A4_MAPPING_SEAL_VERIFY_FAILED"
        ) from error

    prepared = (
        (MAPPING_NAME, _canonical(mapping)),
        (BINDING_NAME, _canonical(binding)),
    )
    paths: list[Path] = []
    digests: list[str] = []
    for name, expected in prepared:
        path = _mapping_directory(output_root) / name
        digest = hashlib.sha256(expected).hexdigest()
        if (
            members[f"sealed/{name}"] != expected
            or members[f"sealed/{name}.sha256"]
            != f"{digest}\n".encode("ascii")
        ):
            _fail("E1A4_MAPPING_BINDING_MISMATCH")
        paths.append(path)
        digests.append(digest)
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


def _publish_mapping_directory(
    mapping: Mapping[str, object],
    binding: Mapping[str, object],
    output_root: Path,
    approved_private_root: Path,
) -> E1A4MappingSeal:
    digest = hashlib.sha256(_canonical(binding)).hexdigest()
    try:
        with authenticated_publication_directory(
            approved_private_root=approved_private_root,
            publication_parent=output_root / "e1a4-role-mapping",
            lock_name=".v1.publish.lock",
        ) as publication:
            publication.ensure_no_staging(prefix=".v1.", suffix=".tmp")
            if publication.final_exists("v1"):
                return _verify_mapping_publication(
                    publication,
                    mapping,
                    binding,
                    output_root,
                    digest,
                )

            staging = publication.create_staging(prefix=".v1.", suffix=".tmp")
            staging.mkdir("sealed")
            for name, content in (
                (MAPPING_NAME, _canonical(mapping)),
                (BINDING_NAME, _canonical(binding)),
            ):
                staging.write_exclusive(f"sealed/{name}", content)
                staging.write_exclusive(
                    f"sealed/{name}.sha256",
                    (hashlib.sha256(content).hexdigest() + "\n").encode(
                        "ascii"
                    ),
                )
            staging.sync_directory("sealed")
            staging.sync_root()
            publication.publish_no_replace(staging, "v1")
            publication.sync_parent()
            return _verify_mapping_publication(
                publication,
                mapping,
                binding,
                output_root,
                digest,
            )
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
    approved_private_root: Path,
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
    return _publish_mapping_directory(
        mapping,
        binding,
        output_root,
        approved_private_root,
    )


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
