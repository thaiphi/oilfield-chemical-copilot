"""Seal or verify the authenticated metadata-only E1a-4 sampling frame."""

from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp
from typing import Mapping, Sequence


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
    verify_e1_index_contract,
)
from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (  # noqa: E402
    IronSulfideSupplementAuditError,
    IronSulfideSupplementAuditStore,
)


SOURCE_REGISTER_NAME = "source-register.v1.json"
ALLOCATION_NAME = "sampling-allocation.v1.json"
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
_SAFE_CODES = frozenset(
    {
        "E1A4_SAMPLING_FRAME_ARGUMENT_INVALID",
        "E1A4_SAMPLING_FRAME_PREFLIGHT_FAILED",
        "E1A4_SAMPLING_FRAME_MAPPING_UNTRUSTED",
        "E1A4_SAMPLING_FRAME_MAPPING_INVALID",
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
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--index-contract", dest="index_contract_path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-register-sha256")
    parser.add_argument("--expected-allocation-sha256")
    return parser


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
        for connection in (supplement, core, store):
            if connection is not None:
                with suppress(Exception):
                    connection.close()


def _mapping_payload(seal: object) -> tuple[object, ...]:
    try:
        artifacts = getattr(seal, "artifacts")
        artifact = next(
            item
            for item in artifacts
            if item.name == "role-mapping.v1.json"
        )
        content = artifact.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != _digest(
            artifact.sha256, "E1A4_SAMPLING_FRAME_MAPPING_INVALID"
        ):
            _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")
        payload = json.loads(content)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "sources"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or not isinstance(payload["sources"], list)
            or _canonical(payload) != content
        ):
            _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")
        sources = validate_mapping_sources(payload["sources"])
        if (
            type(artifact.record_count) is not int
            or len(sources) != artifact.record_count
        ):
            _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")
        return sources
    except E1A4SamplingFrameError:
        raise
    except (E1A4SamplingError, AttributeError, OSError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
        _fail("E1A4_SAMPLING_FRAME_MAPPING_INVALID")


def _frame_directory(output_root: Path) -> Path:
    return output_root / "e1a4" / "sampling-frame" / "v1"


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
        verify_e1_index_contract(
            database_url=str(values["database_url"]),
            contract_path=contract_path,
        )
        contract_after = contract_path.read_bytes()
    except (E1IndexPreflightError, OSError, TypeError, ValueError):
        _fail("E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED")
    if contract_before != contract_after:
        _fail("E1A4_SAMPLING_FRAME_INDEX_UNTRUSTED")
    index_contract_digest = hashlib.sha256(contract_after).hexdigest()

    sources = _mapping_payload(mapping_seal)
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
        observed = {
            path.relative_to(final).as_posix() for path in final.rglob("*")
        }
        if not final.is_dir() or observed != _EXPECTED_PATHS:
            _fail("E1A4_SAMPLING_FRAME_PARTIAL")
        prepared = (
            (SOURCE_REGISTER_NAME, _canonical(source_register)),
            (ALLOCATION_NAME, _canonical(allocation)),
        )
        digests: list[str] = []
        for name, expected in prepared:
            payload_path = final / "sealed" / name
            digest = hashlib.sha256(expected).hexdigest()
            manifest_path = final / "manifests" / _manifest_name(name)
            if (
                payload_path.read_bytes() != expected
                or manifest_path.read_bytes()
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
    if final.exists():
        return verify_sampling_frame(
            source_register=source_register,
            allocation=allocation,
            output_root=output_root,
        )
    parent = final.parent
    staged: Path | None = None
    published = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        for stale in parent.glob(".v1.*.tmp"):
            shutil.rmtree(stale)
        if tuple(parent.glob(".v1.*.tmp")):
            _fail("E1A4_SAMPLING_FRAME_WRITE_FAILED")
        staged = Path(mkdtemp(prefix=".v1.", suffix=".tmp", dir=parent))
        (staged / "sealed").mkdir()
        (staged / "manifests").mkdir()
        for name, content in (
            (SOURCE_REGISTER_NAME, _canonical(source_register)),
            (ALLOCATION_NAME, _canonical(allocation)),
        ):
            payload_path = staged / "sealed" / name
            manifest_path = staged / "manifests" / _manifest_name(name)
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
        os.replace(staged, final)
        published = True
        staged = None
        return verify_sampling_frame(
            source_register=source_register,
            allocation=allocation,
            output_root=output_root,
        )
    except E1A4SamplingFrameError:
        if published and final.exists():
            shutil.rmtree(final, ignore_errors=True)
        if staged is not None and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise
    except Exception as error:
        if published and final.exists():
            shutil.rmtree(final, ignore_errors=True)
        if staged is not None and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
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
    source_register, allocation = _expected_frame(**values)
    return verify_sampling_frame(
        source_register=source_register,
        allocation=allocation,
        output_root=Path(values["output_root"]),
        expected_source_register_sha256=values.get(
            "expected_source_register_sha256"
        ),
        expected_allocation_sha256=values.get("expected_allocation_sha256"),
    )


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
