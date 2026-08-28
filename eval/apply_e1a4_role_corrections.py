"""Apply or verify the authenticated E1a-4 role mapping with safe output."""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path[:0] = [str(PROJECT_ROOT), str(SRC_DIR)]

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (  # noqa: E402
    CorpusReconciliationError,
    ReconciliationStore,
)
from oilfield_chemical_copilot.evaluation.e1a4_mapping_application import (  # noqa: E402
    E1A4MappingApplicationError,
    seal_e1a4_role_mapping,
    verify_e1a4_role_mapping,
)
from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (  # noqa: E402
    FoundationalAuditStore,
    FoundationalLocatorAuditError,
)
from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (  # noqa: E402
    IronSulfideSupplementAuditError,
    IronSulfideSupplementAuditStore,
)


_MAPPING_CODES = frozenset(
    {
        "E1A4_MAPPING_ALLOCATION_INVALID",
        "E1A4_MAPPING_ALLOCATION_UNAVAILABLE",
        "E1A4_MAPPING_AUTHENTICATION_FAILED",
        "E1A4_MAPPING_BINDING_MISMATCH",
        "E1A4_MAPPING_E1A3_EXCLUSION_MISMATCH",
        "E1A4_MAPPING_PROMOTION_INVALID",
        "E1A4_MAPPING_PROMOTION_OVERLAP",
        "E1A4_MAPPING_PROPOSAL_UNRESOLVED",
        "E1A4_MAPPING_SEAL_MISSING",
        "E1A4_MAPPING_SEAL_PARTIAL",
        "E1A4_MAPPING_SEAL_VERIFY_FAILED",
        "E1A4_MAPPING_SEAL_WRITE_FAILED",
        "E1A4_MAPPING_STRATUM_INSUFFICIENT",
    }
)


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise E1A4MappingApplicationError(
            "E1A4_ROLE_MAPPING_ARGUMENT_INVALID"
        )


def _parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        description="Apply authenticated E1a-4 role corrections."
    )
    parser.add_argument("command", choices=("apply", "verify"))
    parser.add_argument("--reconciliation-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--core-audit-id", required=True)
    parser.add_argument("--supplement-audit-id", required=True)
    parser.add_argument(
        "--expected-reconciliation-binding-sha256", required=True
    )
    parser.add_argument("--expected-core-binding-sha256", required=True)
    parser.add_argument("--expected-supplement-binding-sha256", required=True)
    parser.add_argument("--e1a3-allocation-path", type=Path, required=True)
    parser.add_argument(
        "--e1a3-allocation-manifest-path", type=Path, required=True
    )
    parser.add_argument("--e1a3-private-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-mapping-binding-sha256")
    return parser


def _source_record_count(seal: object) -> int:
    artifacts = getattr(seal, "artifacts", ())
    mapping = next(
        (
            artifact
            for artifact in artifacts
            if getattr(artifact, "name", None) == "role-mapping.v1.json"
        ),
        None,
    )
    count = getattr(mapping, "record_count", None)
    if type(count) is not int or count < 1:
        raise E1A4MappingApplicationError(
            "E1A4_MAPPING_AUTHENTICATION_FAILED"
        )
    return count


def _operate(args: argparse.Namespace) -> dict[str, object]:
    root = args.reconciliation_root.resolve()
    database_path = (root / "reconciliation.sqlite").resolve()
    store: ReconciliationStore | None = None
    core: FoundationalAuditStore | None = None
    supplement: IronSulfideSupplementAuditStore | None = None
    try:
        store = ReconciliationStore.open(
            root=root,
            expected_root=root,
            run_id=args.run_id,
        )
        core = FoundationalAuditStore.open(
            database_path=database_path,
            run_id=args.run_id,
            audit_id=args.core_audit_id,
        )
        supplement = IronSulfideSupplementAuditStore.open(
            database_path=database_path,
            run_id=args.run_id,
            audit_id=args.supplement_audit_id,
        )
        common = {
            "store": store,
            "core_audit": core,
            "supplement_audit": supplement,
            "expected_reconciliation_binding_sha256": (
                args.expected_reconciliation_binding_sha256
            ),
            "expected_core_binding_sha256": (
                args.expected_core_binding_sha256
            ),
            "expected_supplement_binding_sha256": (
                args.expected_supplement_binding_sha256
            ),
            "e1a3_allocation_path": args.e1a3_allocation_path,
            "e1a3_allocation_manifest_path": (
                args.e1a3_allocation_manifest_path
            ),
            "e1a3_private_root": args.e1a3_private_root,
            "output_root": args.output_root,
        }
        if args.command == "apply":
            sealed = seal_e1a4_role_mapping(**common)
            status = "E1A4_ROLE_MAPPING_SEALED"
        else:
            if args.expected_mapping_binding_sha256 is None:
                raise E1A4MappingApplicationError(
                    "E1A4_ROLE_MAPPING_ARGUMENT_INVALID"
                )
            sealed = verify_e1a4_role_mapping(
                **common,
                expected_mapping_binding_sha256=(
                    args.expected_mapping_binding_sha256
                ),
            )
            status = "E1A4_ROLE_MAPPING_VERIFIED"
        return {
            "status": status,
            "source_record_count": _source_record_count(sealed),
            "sufficient_strata_count": 8,
            "allocator_slot_count": 96,
        }
    finally:
        for connection in (supplement, core, store):
            if connection is not None:
                with suppress(Exception):
                    connection.close()


def _safe_code(error: Exception) -> str:
    value = str(error)
    if value == "E1A4_ROLE_MAPPING_ARGUMENT_INVALID":
        return value
    if isinstance(error, E1A4MappingApplicationError) and value in _MAPPING_CODES:
        return value
    if isinstance(
        error,
        (
            CorpusReconciliationError,
            FoundationalLocatorAuditError,
            IronSulfideSupplementAuditError,
        ),
    ):
        return "E1A4_ROLE_MAPPING_AUTHENTICATION_FAILED"
    return "E1A4_ROLE_MAPPING_OPERATION_FAILED"


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        output = _operate(_parser().parse_args(argv))
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "E1A4_ROLE_MAPPING_BLOCKED",
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
