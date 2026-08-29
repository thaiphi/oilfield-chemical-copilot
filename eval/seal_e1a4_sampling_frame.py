"""Seal or verify the authenticated metadata-only E1a-4 sampling frame."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from tempfile import mkdtemp
from typing import Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path[:0] = [str(PROJECT_ROOT), str(SRC_DIR)]

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (  # noqa: E402
    CorpusReconciliationError,
    ReconciliationStore,
)
from oilfield_chemical_copilot.evaluation.e1a3_sampling import (  # noqa: E402
    E1A3SamplingError,
    allocate_sampling_slots,
    build_sampling_slots,
)
from oilfield_chemical_copilot.evaluation.e1a4_mapping_application import (  # noqa: E402
    E1A4MappingApplicationError,
    verify_e1a4_role_mapping,
)
from oilfield_chemical_copilot.evaluation import (  # noqa: E402
    e1a4_mapping_application as _mapping_application,
)
from oilfield_chemical_copilot.evaluation.e1a4_sampling import (  # noqa: E402
    E1A4SamplingError,
    load_e1a3_prior_allocation,
    mapping_sources_as_sampling_metadata,
    validate_mapping_sources,
)
from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (  # noqa: E402
    FoundationalAuditStore,
    FoundationalLocatorAuditError,
)
from oilfield_chemical_copilot.evaluation.index_preflight import (  # noqa: E402
    E1IndexPreflightError,
    IndexFingerprint,
    verify_e1_index_contract,
)
from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (  # noqa: E402
    IronSulfideSupplementAuditError,
    IronSulfideSupplementAuditStore,
)


SOURCE_REGISTER_NAME = "source-register.v1.json"
ALLOCATION_NAME = "sampling-allocation.v1.json"
_PUBLISH_LOCK_NAME = ".v1.publish.lock"
_EXPECTED_PATHS = frozenset(
    {
        "sealed",
        "manifests",
        f"sealed/{SOURCE_REGISTER_NAME}",
        f"sealed/{ALLOCATION_NAME}",
        f"manifests/{SOURCE_REGISTER_NAME.removesuffix('.json')}.sha256",
        f"manifests/{ALLOCATION_NAME.removesuffix('.json')}.sha256",
    }
)
_MAPPING_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "reconciliation_run_id",
        "reconciliation_binding_sha256",
        "core_binding_sha256",
        "supplement_binding_sha256",
        "e1a3_allocation_sha256",
        "mapping_payload_sha256",
        "source_record_count",
        "unique_locator_count",
        "stratum_locator_counts",
        "allocator_available",
        "allocator_slot_count",
        "e1a3_excluded_before_allocation",
    }
)
_SAFE_CODES = frozenset(
    {
        "E1A4_SAMPLING_FRAME_ARGUMENT_INVALID",
        "E1A4_SAMPLING_FRAME_PRIVATE_ROOT_INVALID",
        "E1A4_SAMPLING_FRAME_PREFLIGHT_FAILED",
        "E1A4_SAMPLING_FRAME_MAPPING_UNTRUSTED",
        "E1A4_SAMPLING_FRAME_MAPPING_INVALID",
        "E1A4_SAMPLING_FRAME_CLOSE_FAILED",
        "E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED",
        "E1A4_SAMPLING_FRAME_E1A3_UNTRUSTED",
        "E1A4_SAMPLING_FRAME_E1A3_REUSE",
        "E1A4_SAMPLING_FRAME_STRATUM_INSUFFICIENT",
        "E1A4_SAMPLING_FRAME_ALLOCATION_INVALID",
        "E1A4_SAMPLING_FRAME_MISSING",
        "E1A4_SAMPLING_FRAME_PARTIAL",
        "E1A4_SAMPLING_FRAME_BINDING_MISMATCH",
        "E1A4_SAMPLING_FRAME_VERIFY_FAILED",
        "E1A4_SAMPLING_FRAME_WRITE_FAILED",
    }
)


class E1A4SamplingFrameError(RuntimeError):
    """Raised with a fixed public-safe sampling-frame error code."""


def _fail(code: str) -> None:
    raise E1A4SamplingFrameError(code)


def _digest(value: object, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(code)
    return value


def _canonical(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class E1A4SamplingFrameSeal:
    source_record_count: int
    sufficient_strata_count: int
    slot_count: int
    source_register_sha256: str
    allocation_sha256: str


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        _fail("E1A4_SAMPLING_FRAME_ARGUMENT_INVALID")


def _parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        description="Seal the authenticated E1a-4 metadata sampling frame."
    )
    parser.add_argument("command", nargs="?", choices=("seal", "verify"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--reconciliation-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--core-audit-id", required=True)
    parser.add_argument("--supplement-audit-id", required=True)
    parser.add_argument(
        "--expected-reconciliation-binding-sha256", required=True
    )
    parser.add_argument("--expected-core-binding-sha256", required=True)
    parser.add_argument("--expected-supplement-binding-sha256", required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--expected-mapping-binding-sha256", required=True)
    parser.add_argument("--e1a3-allocation-path", type=Path, required=True)
    parser.add_argument(
        "--e1a3-allocation-manifest-path", type=Path, required=True
    )
    parser.add_argument("--e1a3-private-root", type=Path, required=True)
    parser.add_argument("--approved-private-root", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--index-contract", dest="index_contract_path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-register-sha256")
    parser.add_argument("--expected-allocation-sha256")
    return parser


def _is_reparse_point(observed: os.stat_result) -> bool:
    attributes = getattr(observed, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _require_safe_directory_component(path: Path) -> None:
    observed = path.lstat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or _is_reparse_point(observed)
    ):
        raise OSError("unsafe private directory component")


def _existing_directory_chain(path: Path) -> tuple[Path, ...]:
    absolute = path.absolute()
    lineage = tuple(reversed((absolute, *absolute.parents)))
    existing: list[Path] = []
    for component in lineage:
        try:
            component.lstat()
        except FileNotFoundError:
            break
        existing.append(component)
    return tuple(existing)


def _repository_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        try:
            (candidate / ".git").lstat()
        except FileNotFoundError:
            continue
        return candidate
    return None


def _validate_private_paths(
    approved_private_root: Path, paths: Sequence[Path]
) -> None:
    try:
        raw_root = approved_private_root.absolute()
        for component in _existing_directory_chain(raw_root):
            _require_safe_directory_component(component)
        root = raw_root.resolve(strict=True)
        repository = _repository_root(root)
        if repository is not None:
            relative = root.relative_to(repository)
            if not relative.parts or relative.parts[0] != ".private":
                raise OSError("public worktree cannot be a private root")
        for path in paths:
            raw_path = path.absolute()
            raw_path.relative_to(raw_root)
            raw_path.resolve(strict=False).relative_to(root)
            for component in _existing_directory_chain(raw_path):
                _require_safe_directory_component(component)
    except (OSError, RuntimeError, ValueError) as error:
        raise E1A4SamplingFrameError(
            "E1A4_SAMPLING_FRAME_PRIVATE_ROOT_INVALID"
        ) from error


def _reject_unsafe_directory_ancestors(path: Path) -> None:
    for component in _existing_directory_chain(path):
        _require_safe_directory_component(component)


def _directory_identity(path: Path) -> os.stat_result:
    _reject_unsafe_directory_ancestors(path)
    observed = path.lstat()
    _require_safe_directory_component(path)
    return observed


def _require_directory_identity(path: Path, identity: os.stat_result) -> None:
    _reject_unsafe_directory_ancestors(path)
    current = path.lstat()
    if not os.path.samestat(identity, current):
        raise OSError(errno.EAGAIN, "publication directory changed")


def _sync_directory(path: Path) -> None:
    _mapping_application._fsync_directory(path)


def _presence_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    return _presence_paths_from_values(vars(args))


def _presence_paths_from_values(values: Mapping[str, object]) -> tuple[Path, ...]:
    mapping = (
        Path(values["mapping_root"])
        / "e1a4-role-mapping"
        / "v1"
        / "sealed"
    )
    return (
        Path(values["reconciliation_root"]) / "reconciliation.sqlite",
        mapping / "role-mapping.v1.json",
        mapping / "role-mapping.v1.json.sha256",
        mapping / "mapping-binding.v1.json",
        mapping / "mapping-binding.v1.json.sha256",
        Path(values["e1a3_allocation_path"]),
        Path(values["e1a3_allocation_manifest_path"]),
        Path(values["index_contract_path"]),
    )


def _presence_preflight(args: argparse.Namespace) -> None:
    _check_presence(_presence_paths(args))


def _presence_preflight_values(values: Mapping[str, object]) -> None:
    _check_presence(_presence_paths_from_values(values))


def _check_presence(paths: Sequence[Path]) -> None:
    try:
        if not all(path.is_file() for path in paths):
            _fail("E1A4_SAMPLING_FRAME_PREFLIGHT_FAILED")
    except E1A4SamplingFrameError:
        raise
    except Exception:
        _fail("E1A4_SAMPLING_FRAME_PREFLIGHT_FAILED")


def _mapping_kwargs(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "expected_reconciliation_binding_sha256": values[
            "expected_reconciliation_binding_sha256"
        ],
        "expected_core_binding_sha256": values[
            "expected_core_binding_sha256"
        ],
        "expected_supplement_binding_sha256": values[
            "expected_supplement_binding_sha256"
        ],
        "e1a3_allocation_path": values["e1a3_allocation_path"],
        "e1a3_allocation_manifest_path": values[
            "e1a3_allocation_manifest_path"
        ],
        "e1a3_private_root": values["e1a3_private_root"],
        "output_root": values["mapping_root"],
        "expected_mapping_binding_sha256": values[
            "expected_mapping_binding_sha256"
        ],
    }


def _close_mapping_stores(*connections: object | None) -> None:
    close_failed = False
    for connection in connections:
        if connection is None:
            continue
        try:
            connection.close()  # type: ignore[attr-defined]
        except Exception:
            close_failed = True
    if close_failed:
        _fail("E1A4_SAMPLING_FRAME_CLOSE_FAILED")


def _verify_mapping_trust(**values: object) -> object:
    root = Path(values["reconciliation_root"]).resolve()
    database_path = (root / "reconciliation.sqlite").resolve()
    store: ReconciliationStore | None = None
    core: FoundationalAuditStore | None = None
    supplement: IronSulfideSupplementAuditStore | None = None
    try:
        store = ReconciliationStore.open(
            root=root,
            expected_root=root,
            run_id=str(values["run_id"]),
        )
        core = FoundationalAuditStore.open(
            database_path=database_path,
            run_id=str(values["run_id"]),
            audit_id=str(values["core_audit_id"]),
        )
        supplement = IronSulfideSupplementAuditStore.open(
            database_path=database_path,
            run_id=str(values["run_id"]),
            audit_id=str(values["supplement_audit_id"]),
        )
        return verify_e1a4_role_mapping(
            store=store,
            core_audit=core,
            supplement_audit=supplement,
            **_mapping_kwargs(values),
        )
    except (
        CorpusReconciliationError,
        E1A4MappingApplicationError,
        FoundationalLocatorAuditError,
        IronSulfideSupplementAuditError,
        OSError,
        TypeError,
        ValueError,
    ):
        _fail("E1A4_SAMPLING_FRAME_MAPPING_UNTRUSTED")
    finally:
        _close_mapping_stores(supplement, core, store)


def _mapping_payload(
    seal: object, *, expected_mapping_binding_sha256: str
) -> tuple[object, ...]:
    try:
        artifacts = getattr(seal, "artifacts")
        mapping_artifact = next(
            item
            for item in artifacts
            if item.name == "role-mapping.v1.json"
        )
        binding_artifact = next(
            item
            for item in artifacts
            if item.name == "mapping-binding.v1.json"
        )
        mapping_content = mapping_artifact.path.read_bytes()
        binding_content = binding_artifact.path.read_bytes()
        binding_digest = hashlib.sha256(binding_content).hexdigest()
        if binding_digest != _digest(
            expected_mapping_binding_sha256,
            "E1A4_SAMPLING_FRAME_MAPPING_INVALID",
        ) or binding_digest != _digest(
            binding_artifact.sha256,
            "E1A4_SAMPLING_FRAME_MAPPING_INVALID",
        ):
            _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")
        binding = json.loads(binding_content)
        if (
            not isinstance(binding, dict)
            or set(binding) != _MAPPING_BINDING_FIELDS
            or type(binding["schema_version"]) is not int
            or binding["schema_version"] != 1
            or _canonical(binding) != binding_content
            or _digest(
                binding["mapping_payload_sha256"],
                "E1A4_SAMPLING_FRAME_MAPPING_INVALID",
            )
            != hashlib.sha256(mapping_content).hexdigest()
        ):
            _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")
        payload = json.loads(mapping_content)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "sources"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or not isinstance(payload["sources"], list)
            or _canonical(payload) != mapping_content
        ):
            _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")
        sources = validate_mapping_sources(payload["sources"])
        if (
            type(binding["source_record_count"]) is not int
            or len(sources) != binding["source_record_count"]
            or type(mapping_artifact.record_count) is not int
            or len(sources) != mapping_artifact.record_count
        ):
            _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")
        return sources
    except E1A4SamplingFrameError:
        raise
    except (E1A4SamplingError, AttributeError, OSError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
        _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")


def _frame_directory(output_root: Path) -> Path:
    return output_root / "e1a4" / "sampling-frame" / "v1"


def _index_fingerprint_from_bytes(content: bytes) -> IndexFingerprint:
    try:
        payload = json.loads(content.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "chunk_count",
            "distinct_source_count",
            "embedding_models",
            "embedding_dimensions",
            "inventory_sha256",
        }:
            _fail("E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED")
        chunk_count = payload["chunk_count"]
        source_count = payload["distinct_source_count"]
        models = payload["embedding_models"]
        dimensions = payload["embedding_dimensions"]
        inventory_digest = payload["inventory_sha256"]
        if (
            type(chunk_count) is not int
            or chunk_count < 1
            or type(source_count) is not int
            or source_count < 1
            or not isinstance(models, list)
            or not models
            or any(not isinstance(model, str) or not model for model in models)
            or not isinstance(dimensions, list)
            or not dimensions
            or any(
                type(dimension) is not int or dimension < 1
                for dimension in dimensions
            )
            or not isinstance(inventory_digest, str)
            or len(inventory_digest) != 64
        ):
            _fail("E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED")
        return IndexFingerprint(
            chunk_count=chunk_count,
            distinct_source_count=source_count,
            embedding_models=tuple(models),
            embedding_dimensions=tuple(dimensions),
            inventory_sha256=inventory_digest,
        )
    except E1A4SamplingFrameError:
        raise
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError):
        _fail("E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED")


def _expected_frame(**values: object) -> tuple[dict[str, object], dict[str, object]]:
    mapping_seal = _verify_mapping_trust(**values)
    expected_mapping_binding = _digest(
        values.get("expected_mapping_binding_sha256"),
        "E1A4_SAMPLING_FRAME_MAPPING_UNTRUSTED",
    )
    mapping_binding = _digest(
        getattr(mapping_seal, "binding_sha256", None),
        "E1A4_SAMPLING_FRAME_MAPPING_UNTRUSTED",
    )
    if mapping_binding != expected_mapping_binding:
        _fail("E1A4_SAMPLING_FRAME_MAPPING_UNTRUSTED")

    contract_path = Path(values["index_contract_path"])
    try:
        contract_before = contract_path.read_bytes()
        verified_fingerprint = verify_e1_index_contract(
            database_url=str(values["database_url"]),
            contract_path=contract_path,
        )
        contract_after = contract_path.read_bytes()
    except (E1IndexPreflightError, OSError, TypeError, ValueError):
        _fail("E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED")
    if (
        contract_before != contract_after
        or _index_fingerprint_from_bytes(contract_after)
        != verified_fingerprint
    ):
        _fail("E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED")
    index_contract_digest = hashlib.sha256(contract_after).hexdigest()

    sources = _mapping_payload(
        mapping_seal,
        expected_mapping_binding_sha256=expected_mapping_binding,
    )
    try:
        prior = load_e1a3_prior_allocation(
            payload_path=Path(values["e1a3_allocation_path"]),
            manifest_path=Path(values["e1a3_allocation_manifest_path"]),
            private_root=Path(values["e1a3_private_root"]),
        )
    except (E1A4SamplingError, OSError, TypeError, ValueError):
        _fail("E1A4_SAMPLING_FRAME_E1A3_UNTRUSTED")
    if prior.slot_count != 96:
        _fail("E1A4_SAMPLING_FRAME_E1A3_UNTRUSTED")
    prior_digest = _digest(
        prior.payload_sha256, "E1A4_SAMPLING_FRAME_E1A3_UNTRUSTED"
    )
    mapped_keys = {
        f"{source.source_id}:{locator}"
        for source in sources
        for locator in source.locators
    }
    if mapped_keys & prior.locator_keys:
        _fail("E1A4_SAMPLING_FRAME_E1A3_REUSE")

    metadata = mapping_sources_as_sampling_metadata(sources)
    stratum_counts = {
        (topic, role): sum(
            len(source.locators)
            for source in metadata
            if source.topic == topic and source.source_role == role
        )
        for topic in ("iron_sulfide", "scale", "corrosion", "paraffin")
        for role in ("foundational", "supporting")
    }
    if any(count < 12 for count in stratum_counts.values()):
        _fail("E1A4_SAMPLING_FRAME_STRATUM_INSUFFICIENT")

    slots = build_sampling_slots()
    try:
        allocations = allocate_sampling_slots(slots=slots, sources=metadata)
    except (E1A3SamplingError, TypeError, ValueError):
        _fail("E1A4_SAMPLING_FRAME_ALLOCATION_INVALID")
    expected_identities = {
        (
            slot.slot_id,
            slot.topic,
            slot.source_role,
            slot.question_form,
            slot.evidence_depth,
            slot.replicate,
        )
        for slot in slots
    }
    try:
        identities = {
            (
                item.slot_id,
                item.topic,
                item.source_role,
                item.question_form,
                item.evidence_depth,
                item.replicate,
            )
            for item in allocations
        }
        allocated_keys = {
            (item.source_id, item.locator) for item in allocations
        }
        allocation_rows = [item.to_mapping() for item in allocations]
    except (AttributeError, TypeError, ValueError):
        _fail("E1A4_SAMPLING_FRAME_ALLOCATION_INVALID")
    if (
        len(allocations) != 96
        or identities != expected_identities
        or len(allocated_keys) != 96
        or {item.topic for item in allocations}
        != {"iron_sulfide", "scale", "corrosion", "paraffin"}
        or {item.source_role for item in allocations}
        != {"foundational", "supporting"}
    ):
        _fail("E1A4_SAMPLING_FRAME_ALLOCATION_INVALID")

    source_register = {
        "schema_version": 1,
        "mapping_binding_sha256": mapping_binding,
        "index_contract_sha256": index_contract_digest,
        "e1a3_allocation_sha256": prior_digest,
        "source_record_count": len(metadata),
        "sources": [source.to_mapping() for source in metadata],
    }
    source_register_digest = hashlib.sha256(
        _canonical(source_register)
    ).hexdigest()
    allocation = {
        "schema_version": 1,
        "source_register_sha256": source_register_digest,
        "e1a3_allocation_sha256": prior_digest,
        "slot_count": 96,
        "allocations": allocation_rows,
    }
    return source_register, allocation


def _manifest_name(payload_name: str) -> str:
    return f"{payload_name.removesuffix('.json')}.sha256"


def _rename_no_replace(staged: Path, final: Path) -> None:
    """Atomically publish one directory and fail if the destination exists."""
    if os.name == "nt":
        os.rename(staged, final)
        return

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
            raise FileExistsError(error_number, os.strerror(error_number), final)
        raise OSError(error_number, os.strerror(error_number), final)


_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 0x00000102


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
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
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
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            wintypes.LPDWORD,
        )
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
            wintypes.BOOL
        )
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
                raise OSError(self._ctypes.get_last_error(), "token query failed")
            buffer = self._ctypes.create_string_buffer(required.value)
            if not self._advapi32.GetTokenInformation(
                token,
                1,
                self._ctypes.cast(buffer, self._ctypes.c_void_p),
                required.value,
                required,
            ):
                raise OSError(self._ctypes.get_last_error(), "token query failed")
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
                    self._ctypes.get_last_error(),
                    "SID conversion failed",
                )
            try:
                if sid_text.value is None:
                    raise OSError(errno.EIO, "SID conversion failed")
                return sid_text.value
            finally:
                if self._kernel32.LocalFree(sid_text):
                    raise OSError(
                        self._ctypes.get_last_error(),
                        "SID free failed",
                    )
        finally:
            if not self._kernel32.CloseHandle(token):
                raise OSError(self._ctypes.get_last_error(), "token close failed")

    def build_security_attributes(self, policy: str) -> _WindowsSecurity:
        descriptor = self._ctypes.c_void_p()
        if not self._advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
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
        return _WindowsSecurity(attributes=attributes, descriptor=descriptor)

    def free_security_descriptor(self, security: _WindowsSecurity) -> None:
        if self._kernel32.LocalFree(security.descriptor):
            raise OSError(
                self._ctypes.get_last_error(),
                "security descriptor free failed",
            )

    def create_mutex(self, name: str, attributes: object) -> int:
        handle = self._kernel32.CreateMutexW(
            self._ctypes.byref(attributes),
            False,
            name,
        )
        if not handle:
            raise OSError(self._ctypes.get_last_error(), "mutex creation failed")
        return int(handle)

    def wait(self, handle: int, timeout_ms: int) -> int:
        return int(self._kernel32.WaitForSingleObject(handle, timeout_ms))

    def release_mutex(self, handle: int) -> None:
        if not self._kernel32.ReleaseMutex(handle):
            raise OSError(self._ctypes.get_last_error(), "mutex release failed")

    def close_handle(self, handle: int) -> None:
        if not self._kernel32.CloseHandle(handle):
            raise OSError(self._ctypes.get_last_error(), "mutex close failed")


def _windows_mutex_api() -> _NativeWindowsMutex:
    return _NativeWindowsMutex()


def _windows_mutex_name(parent: Path) -> str:
    resolved = parent.resolve(strict=True)
    canonical = os.path.normcase(os.path.normpath(str(resolved)))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"Global\\E1A4SamplingFrame-{digest}"


def _windows_mutex_policy(owner_sid: str) -> str:
    parts = owner_sid.split("-")
    if len(parts) < 3 or parts[0] != "S" or not all(
        part.isdecimal() for part in parts[1:]
    ):
        raise OSError(errno.EINVAL, "invalid owner SID")
    # CreateMutexW reopens with full access; grant it only to owner and System.
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
                _windows_mutex_name(parent),
                security.attributes,
            )
        finally:
            api.free_security_descriptor(security)
        outcome = api.wait(handle, 0)
        if outcome not in {_WAIT_OBJECT_0, _WAIT_ABANDONED}:
            error_number = errno.EBUSY if outcome == _WAIT_TIMEOUT else errno.EIO
            raise OSError(error_number, "publisher mutex unavailable")
        acquired = True
    except Exception as error:
        if handle is not None:
            try:
                api.close_handle(handle)
            except Exception:
                pass
        raise E1A4SamplingFrameError(
            "E1A4_SAMPLING_FRAME_WRITE_FAILED"
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
            raise E1A4SamplingFrameError(
                "E1A4_SAMPLING_FRAME_WRITE_FAILED"
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
    parent: Path, *, os_api: object | None = None, flock_api: object | None = None
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
            lock_descriptor,
            flock_api.LOCK_EX | flock_api.LOCK_NB,
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
        raise E1A4SamplingFrameError(
            "E1A4_SAMPLING_FRAME_WRITE_FAILED"
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
            raise E1A4SamplingFrameError(
                "E1A4_SAMPLING_FRAME_WRITE_FAILED"
            ) from cleanup_error


@contextmanager
def _publisher_lock(parent: Path) -> Iterator[None]:
    """Hold an OS-released exclusive lock for the whole publication attempt."""
    if os.name == "nt":
        lock = _windows_publisher_lock(parent)
    elif os.name == "posix":
        lock = _posix_publisher_lock(parent)
    else:
        raise E1A4SamplingFrameError("E1A4_SAMPLING_FRAME_WRITE_FAILED")
    with lock:
        yield


def _is_staging_name(name: str) -> bool:
    middle = name.removeprefix(".v1.").removesuffix(".tmp")
    return name.startswith(".v1.") and name.endswith(".tmp") and bool(middle)


def _remove_abandoned_staging(parent: Path) -> None:
    for candidate in parent.iterdir():
        if not _is_staging_name(candidate.name):
            continue
        raise OSError(errno.EBUSY, "abandoned staging requires manual review")


def _read_posix_frame_members(
    final: Path, *, os_api: object | None = None
) -> dict[str, bytes]:
    if os_api is None:
        os_api = os
    directory_flag = _mapping_application._required_posix_flag(
        os_api, "O_DIRECTORY"
    )
    nofollow_flag = _mapping_application._required_posix_flag(
        os_api, "O_NOFOLLOW"
    )
    nonblock_flag = _mapping_application._required_posix_flag(
        os_api, "O_NONBLOCK"
    )
    root_fd: int | None = None
    opened: list[int] = []
    try:
        root_fd = os_api.open(
            final,
            os_api.O_RDONLY
            | directory_flag
            | nofollow_flag
            | getattr(os_api, "O_CLOEXEC", 0),
        )
        root_before = os_api.fstat(root_fd)
        if not stat.S_ISDIR(root_before.st_mode):
            raise OSError(errno.ENOTDIR, "unsafe frame directory")
        if set(os_api.listdir(root_fd)) != {"sealed", "manifests"}:
            _fail("E1A4_SAMPLING_FRAME_PARTIAL")
        captured: dict[str, bytes] = {}
        for dirname, names in (
            ("sealed", {SOURCE_REGISTER_NAME, ALLOCATION_NAME}),
            (
                "manifests",
                {
                    _manifest_name(SOURCE_REGISTER_NAME),
                    _manifest_name(ALLOCATION_NAME),
                },
            ),
        ):
            child_fd = os_api.open(
                dirname,
                os_api.O_RDONLY
                | directory_flag
                | nofollow_flag
                | getattr(os_api, "O_CLOEXEC", 0),
                dir_fd=root_fd,
            )
            opened.append(child_fd)
            child_before = os_api.fstat(child_fd)
            if not stat.S_ISDIR(child_before.st_mode) or set(
                os_api.listdir(child_fd)
            ) != names:
                _fail("E1A4_SAMPLING_FRAME_PARTIAL")
            for name in sorted(names):
                member_fd: int | None = None
                current_fd: int | None = None
                member_error: BaseException | None = None
                close_error: BaseException | None = None
                before = _mapping_application._posix_member_snapshot(
                    os_api.stat(name, dir_fd=child_fd, follow_symlinks=False)
                )
                try:
                    member_fd = os_api.open(
                        name,
                        os_api.O_RDONLY
                        | nofollow_flag
                        | nonblock_flag
                        | getattr(os_api, "O_CLOEXEC", 0),
                        dir_fd=child_fd,
                    )
                    if before != _mapping_application._posix_member_snapshot(
                        os_api.fstat(member_fd)
                    ):
                        raise OSError(errno.EAGAIN, "frame member changed")
                    captured[f"{dirname}/{name}"] = (
                        _mapping_application._read_posix_member(
                            member_fd, os_api=os_api
                        )
                    )
                    if before != _mapping_application._posix_member_snapshot(
                        os_api.fstat(member_fd)
                    ):
                        raise OSError(errno.EAGAIN, "frame member changed")
                    current_preopen = _mapping_application._posix_member_snapshot(
                        os_api.stat(
                            name, dir_fd=child_fd, follow_symlinks=False
                        )
                    )
                    current_fd = os_api.open(
                        name,
                        os_api.O_RDONLY
                        | nofollow_flag
                        | nonblock_flag
                        | getattr(os_api, "O_CLOEXEC", 0),
                        dir_fd=child_fd,
                    )
                    current = _mapping_application._posix_member_snapshot(
                        os_api.fstat(current_fd)
                    )
                    if (
                        before != current_preopen
                        or before != current
                    ):
                        raise OSError(errno.EAGAIN, "frame member changed")
                except BaseException as error:
                    member_error = error
                for descriptor in (current_fd, member_fd):
                    if descriptor is None:
                        continue
                    try:
                        os_api.close(descriptor)
                    except BaseException as error:
                        close_error = close_error or error
                if member_error is not None:
                    raise member_error
                if close_error is not None:
                    raise close_error
            if not os.path.samestat(child_before, os_api.fstat(child_fd)):
                raise OSError(errno.EAGAIN, "frame directory changed")
            current_child: int | None = None
            child_error: BaseException | None = None
            close_error = None
            try:
                current_preopen = os_api.stat(
                    dirname, dir_fd=root_fd, follow_symlinks=False
                )
                if not stat.S_ISDIR(current_preopen.st_mode):
                    raise OSError(errno.ENOTDIR, "frame directory changed")
                current_child = os_api.open(
                    dirname,
                    os_api.O_RDONLY
                    | directory_flag
                    | nofollow_flag
                    | getattr(os_api, "O_CLOEXEC", 0),
                    dir_fd=root_fd,
                )
                if (
                    not os.path.samestat(child_before, current_preopen)
                    or not os.path.samestat(
                        child_before, os_api.fstat(current_child)
                    )
                ):
                    raise OSError(errno.EAGAIN, "frame directory changed")
            except BaseException as error:
                child_error = error
            if current_child is not None:
                try:
                    os_api.close(current_child)
                except BaseException as error:
                    close_error = error
            if child_error is not None:
                raise child_error
            if close_error is not None:
                raise close_error
        if not os.path.samestat(root_before, os_api.fstat(root_fd)):
            raise OSError(errno.EAGAIN, "frame directory changed")
        current_root: int | None = None
        root_error: BaseException | None = None
        close_error = None
        try:
            current_root = os_api.open(
                final,
                os_api.O_RDONLY
                | directory_flag
                | nofollow_flag
                | getattr(os_api, "O_CLOEXEC", 0),
            )
            if not os.path.samestat(root_before, os_api.fstat(current_root)):
                raise OSError(errno.EAGAIN, "frame directory changed")
        except BaseException as error:
            root_error = error
        if current_root is not None:
            try:
                os_api.close(current_root)
            except BaseException as error:
                close_error = error
        if root_error is not None:
            raise root_error
        if close_error is not None:
            raise close_error
        return captured
    finally:
        close_error: BaseException | None = None
        for descriptor in reversed(opened):
            try:
                os_api.close(descriptor)
            except BaseException as error:
                close_error = close_error or error
        if root_fd is not None:
            try:
                os_api.close(root_fd)
            except BaseException as error:
                close_error = close_error or error
        if close_error is not None:
            raise close_error


def _read_windows_frame_members(final: Path) -> dict[str, bytes]:
    api = _mapping_application._windows_seal_reader_api()
    root: object | None = None
    children: list[object] = []
    try:
        root = api.open_directory(final)
        root_before = api._validate_handle(root, directory=True)
        if api.directory_entries(root) != {"sealed", "manifests"}:
            _fail("E1A4_SAMPLING_FRAME_PARTIAL")
        captured: dict[str, bytes] = {}
        for dirname, names in (
            ("sealed", {SOURCE_REGISTER_NAME, ALLOCATION_NAME}),
            (
                "manifests",
                {
                    _manifest_name(SOURCE_REGISTER_NAME),
                    _manifest_name(ALLOCATION_NAME),
                },
            ),
        ):
            child = _open_windows_child_directory(api, root, dirname)
            children.append(child)
            child_before = api._validate_handle(child, directory=True)
            if api.directory_entries(child) != names:
                _fail("E1A4_SAMPLING_FRAME_PARTIAL")
            for name in sorted(names):
                member: object | None = None
                current: object | None = None
                member_error: BaseException | None = None
                try:
                    member = api.open_member(child, name)
                    before = api.member_snapshot(member)
                    content = api.read_member(member)
                    after = api.member_snapshot(member)
                    current = api.open_member(child, name)
                    if before != after or before != api.member_snapshot(current):
                        raise OSError(errno.EAGAIN, "frame member changed")
                    captured[f"{dirname}/{name}"] = content
                except BaseException as error:
                    member_error = error
                close_error = _mapping_application._attempt_resource_closes(
                    (current, member), close=api.close_handle
                )
                if member_error is not None:
                    raise member_error
                if close_error is not None:
                    raise close_error
            if child_before != api._validate_handle(child, directory=True):
                raise OSError(errno.EAGAIN, "frame directory changed")
            current_child: object | None = None
            try:
                current_child = _open_windows_child_directory(
                    api, root, dirname
                )
                if child_before != api._validate_handle(
                    current_child, directory=True
                ):
                    raise OSError(errno.EAGAIN, "frame directory changed")
            finally:
                if current_child is not None:
                    api.close_handle(current_child)
        if root_before != api._validate_handle(root, directory=True):
            raise OSError(errno.EAGAIN, "frame directory changed")
        current_root: object | None = None
        try:
            current_root = api.open_directory(final)
            if root_before != api._validate_handle(
                current_root, directory=True
            ):
                raise OSError(errno.EAGAIN, "frame directory changed")
        finally:
            if current_root is not None:
                api.close_handle(current_root)
        return captured
    finally:
        close_error: BaseException | None = None
        for child in reversed(children):
            try:
                api.close_handle(child)
            except BaseException as error:
                close_error = close_error or error
        if root is not None:
            try:
                api.close_handle(root)
            except BaseException as error:
                close_error = close_error or error
        if close_error is not None:
            raise close_error


def _open_windows_child_directory(
    api: object, parent: object, name: str
) -> int:
    """Open a child directory relative to a verified Windows handle."""
    name_buffer = api._ctypes.create_unicode_buffer(name)
    name_length = len(name.encode("utf-16-le"))
    unicode_name = api._UNICODE_STRING(
        name_length,
        name_length + 2,
        api._ctypes.cast(name_buffer, api._wintypes.LPWSTR),
    )
    attributes = api._OBJECT_ATTRIBUTES(
        api._ctypes.sizeof(api._OBJECT_ATTRIBUTES),
        parent,
        api._ctypes.pointer(unicode_name),
        0x00000040,
        None,
        None,
    )
    status_block = api._IO_STATUS_BLOCK()
    handle = api._wintypes.HANDLE()
    status = api._ntdll.NtCreateFile(
        api._ctypes.byref(handle),
        0x00100081,
        api._ctypes.byref(attributes),
        api._ctypes.byref(status_block),
        None,
        0,
        0x00000007,
        1,
        0x00200021,
        None,
        0,
    )
    if status < 0:
        number = int(api._ntdll.RtlNtStatusToDosError(status))
        raise OSError(number, "frame child directory open failed")
    return int(handle.value)


def _read_frame_members(final: Path) -> dict[str, bytes]:
    if os.name == "posix":
        return _read_posix_frame_members(final)
    if os.name == "nt":
        return _read_windows_frame_members(final)
    raise OSError(errno.ENOTSUP, "safe frame reader unavailable")


def verify_sampling_frame(
    *,
    source_register: Mapping[str, object],
    allocation: Mapping[str, object],
    output_root: Path,
    expected_source_register_sha256: str | None = None,
    expected_allocation_sha256: str | None = None,
) -> E1A4SamplingFrameSeal:
    final = _frame_directory(output_root)
    if not final.exists():
        _fail("E1A4_SAMPLING_FRAME_MISSING")
    try:
        members = _read_frame_members(final)
        prepared = (
            (SOURCE_REGISTER_NAME, _canonical(source_register)),
            (ALLOCATION_NAME, _canonical(allocation)),
        )
        digests: list[str] = []
        for name, expected in prepared:
            digest = hashlib.sha256(expected).hexdigest()
            if (
                members[f"sealed/{name}"] != expected
                or members[f"manifests/{_manifest_name(name)}"]
                != f"{digest}\n".encode("ascii")
            ):
                _fail("E1A4_SAMPLING_FRAME_BINDING_MISMATCH")
            digests.append(digest)
    except E1A4SamplingFrameError:
        raise
    except (OSError, UnicodeError, ValueError):
        _fail("E1A4_SAMPLING_FRAME_VERIFY_FAILED")
    anchors = (
        expected_source_register_sha256,
        expected_allocation_sha256,
    )
    for anchor, actual in zip(anchors, digests, strict=True):
        if anchor is not None and _digest(
            anchor, "E1A4_SAMPLING_FRAME_BINDING_MISMATCH"
        ) != actual:
            _fail("E1A4_SAMPLING_FRAME_BINDING_MISMATCH")
    source_count = source_register.get("source_record_count")
    if type(source_count) is not int or source_count < 1:
        _fail("E1A4_SAMPLING_FRAME_BINDING_MISMATCH")
    return E1A4SamplingFrameSeal(
        source_record_count=source_count,
        sufficient_strata_count=8,
        slot_count=96,
        source_register_sha256=digests[0],
        allocation_sha256=digests[1],
    )


def _publish_sampling_frame(
    *,
    source_register: Mapping[str, object],
    allocation: Mapping[str, object],
    output_root: Path,
) -> E1A4SamplingFrameSeal:
    final = _frame_directory(output_root)
    parent = final.parent
    staged: Path | None = None
    try:
        _reject_unsafe_directory_ancestors(output_root)
        parent.mkdir(parents=True, exist_ok=True)
        parent_identity = _directory_identity(parent)
        with _publisher_lock(parent):
            try:
                _require_directory_identity(parent, parent_identity)
                _remove_abandoned_staging(parent)
                _require_directory_identity(parent, parent_identity)
                if final.exists():
                    result = verify_sampling_frame(
                        source_register=source_register,
                        allocation=allocation,
                        output_root=output_root,
                    )
                else:
                    _require_directory_identity(parent, parent_identity)
                    staged = Path(
                        mkdtemp(prefix=".v1.", suffix=".tmp", dir=parent)
                    )
                    _require_directory_identity(parent, parent_identity)
                    (staged / "sealed").mkdir()
                    (staged / "manifests").mkdir()
                    for name, content in (
                        (SOURCE_REGISTER_NAME, _canonical(source_register)),
                        (ALLOCATION_NAME, _canonical(allocation)),
                    ):
                        payload_path = staged / "sealed" / name
                        manifest_path = staged / "manifests" / _manifest_name(
                            name
                        )
                        for path, value in (
                            (payload_path, content),
                            (
                                manifest_path,
                                (
                                    hashlib.sha256(content).hexdigest() + "\n"
                                ).encode("ascii"),
                            ),
                        ):
                            with path.open("xb") as stream:
                                stream.write(value)
                                stream.flush()
                                os.fsync(stream.fileno())
                    _sync_directory(staged / "sealed")
                    _sync_directory(staged / "manifests")
                    _sync_directory(staged)
                    _require_directory_identity(parent, parent_identity)
                    _rename_no_replace(staged, final)
                    staged = None
                    _require_directory_identity(parent, parent_identity)
                    _sync_directory(parent)
                    result = verify_sampling_frame(
                        source_register=source_register,
                        allocation=allocation,
                        output_root=output_root,
                    )
            except E1A4SamplingFrameError:
                raise
            except Exception as error:
                raise E1A4SamplingFrameError(
                    "E1A4_SAMPLING_FRAME_WRITE_FAILED"
                ) from error
        return result
    except E1A4SamplingFrameError:
        raise
    except Exception as error:
        raise E1A4SamplingFrameError(
            "E1A4_SAMPLING_FRAME_WRITE_FAILED"
        ) from error


def seal_sampling_frame(**values: object) -> E1A4SamplingFrameSeal:
    _presence_preflight_values(values)
    source_register, allocation = _expected_frame(**values)
    return _publish_sampling_frame(
        source_register=source_register,
        allocation=allocation,
        output_root=Path(values["output_root"]),
    )


def verify_current_sampling_frame(**values: object) -> E1A4SamplingFrameSeal:
    _presence_preflight_values(values)
    output_root = Path(values["output_root"])
    final = _frame_directory(output_root)
    if not final.exists():
        _fail("E1A4_SAMPLING_FRAME_MISSING")
    try:
        with _publisher_lock(final.parent):
            _remove_abandoned_staging(final.parent)
            source_register, allocation = _expected_frame(**values)
            return verify_sampling_frame(
                source_register=source_register,
                allocation=allocation,
                output_root=output_root,
                expected_source_register_sha256=values.get(
                    "expected_source_register_sha256"
                ),
                expected_allocation_sha256=values.get(
                    "expected_allocation_sha256"
                ),
            )
    except E1A4SamplingFrameError:
        raise
    except Exception as error:
        raise E1A4SamplingFrameError(
            "E1A4_SAMPLING_FRAME_WRITE_FAILED"
        ) from error


def _values(args: argparse.Namespace) -> dict[str, object]:
    return {
        "reconciliation_root": args.reconciliation_root,
        "run_id": args.run_id,
        "core_audit_id": args.core_audit_id,
        "supplement_audit_id": args.supplement_audit_id,
        "expected_reconciliation_binding_sha256": (
            args.expected_reconciliation_binding_sha256
        ),
        "expected_core_binding_sha256": args.expected_core_binding_sha256,
        "expected_supplement_binding_sha256": (
            args.expected_supplement_binding_sha256
        ),
        "mapping_root": args.mapping_root,
        "expected_mapping_binding_sha256": (
            args.expected_mapping_binding_sha256
        ),
        "e1a3_allocation_path": args.e1a3_allocation_path,
        "e1a3_allocation_manifest_path": (
            args.e1a3_allocation_manifest_path
        ),
        "e1a3_private_root": args.e1a3_private_root,
        "approved_private_root": args.approved_private_root,
        "database_url": args.database_url,
        "index_contract_path": args.index_contract_path,
        "output_root": args.output_root,
        "expected_source_register_sha256": (
            args.expected_source_register_sha256
        ),
        "expected_allocation_sha256": args.expected_allocation_sha256,
    }


def _safe_code(error: Exception) -> str:
    value = str(error)
    if isinstance(error, E1A4SamplingFrameError) and value in _SAFE_CODES:
        return value
    return "E1A4_SAMPLING_FRAME_OPERATION_FAILED"


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        _validate_private_paths(
            args.approved_private_root,
            (args.mapping_root, args.output_root),
        )
        if args.preflight:
            if args.command is not None:
                _fail("E1A4_SAMPLING_FRAME_ARGUMENT_INVALID")
            _presence_preflight(args)
            output: dict[str, object] = {
                "status": "E1A4_SAMPLING_FRAME_PREFLIGHT_READY"
            }
        else:
            if args.command is None:
                _fail("E1A4_SAMPLING_FRAME_ARGUMENT_INVALID")
            _presence_preflight(args)
            values = _values(args)
            if args.command == "seal":
                sealed = seal_sampling_frame(**values)
                status = "E1A4_SAMPLING_FRAME_SEALED"
            else:
                sealed = verify_current_sampling_frame(**values)
                status = "E1A4_SAMPLING_FRAME_VERIFIED"
            output = {
                "status": status,
                "source_record_count": sealed.source_record_count,
                "sufficient_strata_count": sealed.sufficient_strata_count,
                "slot_count": sealed.slot_count,
            }
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "E1A4_SAMPLING_FRAME_BLOCKED",
                    "error_code": _safe_code(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
