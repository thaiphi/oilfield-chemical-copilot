"""Operate the private Iron Sulfide supplement audit with safe public output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path[:0] = [str(PROJECT_ROOT), str(SRC_DIR)]

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (  # noqa: E402
    ReconciliationStore,
)
from oilfield_chemical_copilot.evaluation.foundational_locator_audit import (  # noqa: E402
    FoundationalAuditStore,
)
from oilfield_chemical_copilot.evaluation.iron_sulfide_supplement_audit import (  # noqa: E402
    IronSulfideSupplementAuditError,
    IronSulfideSupplementAuditStore,
    SupplementLocatorDecision,
    bind_supplement_pages,
    calculate_combined_hypothetical_capacity,
    extract_supplement_page,
    initialize_supplement_audit,
    next_supplement_candidate,
    record_supplement_decision,
    seal_supplement_proposal,
    supplement_audit_status,
    verify_supplement_proposal,
)


DEFAULT_ROOT = PROJECT_ROOT / ".private" / "corpus-reconciliation" / "v1"
DEFAULT_RUN_ID = "corpus-reconciliation-v1"
DEFAULT_AUDIT_ID = "iron-sulfide-supplement-audit-v1"


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_ARGUMENT_INVALID"
        )


def _parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description="Private Iron Sulfide supplement audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--private-root", type=Path, default=DEFAULT_ROOT)
        child.add_argument("--run-id", default=DEFAULT_RUN_ID)
        child.add_argument("--audit-id", default=DEFAULT_AUDIT_ID)
        return child

    init = command("init")
    init.add_argument("--source-root", type=Path, required=True)
    next_command = command("next")
    next_command.add_argument("--source-root", type=Path, required=True)
    next_command.add_argument("--packet-output", type=Path, required=True)
    command("record")
    command("status")
    seal = command("seal")
    seal.add_argument("--core-binding-sha256", required=True)
    verify = command("verify")
    verify.add_argument("--expected-binding-sha256", required=True)
    verify.add_argument("--expected-core-binding-sha256", required=True)
    capacity = command("capacity")
    capacity.add_argument("--core-audit-id", required=True)
    capacity.add_argument("--expected-binding-sha256", required=True)
    capacity.add_argument("--expected-core-binding-sha256", required=True)
    return parser


def _private_root(value: Path) -> Path:
    resolved = value.resolve()
    if (
        resolved.name != "v1"
        or resolved.parent.name != "corpus-reconciliation"
        or resolved.parent.parent.name != ".private"
    ):
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_PRIVATE_ROOT_INVALID"
        )
    return resolved


def _open_audit(args: argparse.Namespace) -> IronSulfideSupplementAuditStore:
    root = _private_root(args.private_root)
    return IronSulfideSupplementAuditStore.open(
        database_path=root / "reconciliation.sqlite",
        run_id=args.run_id,
        audit_id=args.audit_id,
    )


def _read_stdin_mapping() -> dict[str, object]:
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, UnicodeError) as error:
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_STDIN_INVALID"
        ) from error
    if not isinstance(payload, dict):
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_STDIN_INVALID"
        )
    return payload


def _public_status(
    audit: IronSulfideSupplementAuditStore,
) -> dict[str, object]:
    status = supplement_audit_status(audit)
    return {
        "status": status.status,
        "source_count": status.source_count,
        "candidate_count": status.candidate_count,
        "reviewed_count": status.reviewed_count,
        "promotion_count": status.promotion_count,
        "remaining_count": status.remaining_count,
        "needs_second_review_count": status.needs_second_review_count,
    }


def _write_packet(*, root: Path, destination: Path, payload: bytes) -> None:
    resolved = destination.resolve()
    if not resolved.is_relative_to(root):
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_PACKET_PATH_INVALID"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, resolved)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise IronSulfideSupplementAuditError(
            "IRON_SULFIDE_SUPPLEMENT_PACKET_WRITE_FAILED"
        ) from error


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = _private_root(args.private_root)
        if args.command == "init":
            payload = _read_stdin_mapping()
            if set(payload) != {"snapshot_binding_sha256", "promotion_target"}:
                raise IronSulfideSupplementAuditError(
                    "IRON_SULFIDE_SUPPLEMENT_STDIN_INVALID"
                )
            store = ReconciliationStore.open(
                root=root,
                expected_root=root,
                run_id=args.run_id,
            )
            audit = initialize_supplement_audit(
                store=store,
                audit_id=args.audit_id,
                snapshot_binding_sha256=payload["snapshot_binding_sha256"],
                source_root=args.source_root,
                promotion_target=payload["promotion_target"],
            )
            bind_supplement_pages(audit=audit, source_root=args.source_root)
        else:
            audit = _open_audit(args)
        if args.command == "next":
            candidate = next_supplement_candidate(audit)
            if candidate is not None:
                packet = extract_supplement_page(
                    audit=audit,
                    source_root=args.source_root,
                    source_id=candidate.source_id,
                    locator=candidate.locator,
                )
                content = (
                    json.dumps(
                        {
                            "source_id": packet.source_id,
                            "locator": packet.locator,
                            "page_number": packet.page_number,
                            "page_text": packet.page_text,
                            "page_text_sha256": packet.page_text_sha256,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
                _write_packet(root=root, destination=args.packet_output, payload=content)
        elif args.command == "record":
            record_supplement_decision(
                audit=audit,
                record=SupplementLocatorDecision.from_mapping(_read_stdin_mapping()),
            )
        if args.command == "seal":
            sealed = seal_supplement_proposal(
                audit=audit,
                core_binding_sha256=args.core_binding_sha256,
            )
            output = {
                "status": "SEALED",
                "artifact_count": len(sealed.artifacts),
                "manifest_count": len(sealed.artifacts),
            }
        elif args.command == "verify":
            sealed = verify_supplement_proposal(
                audit=audit,
                expected_binding_sha256=args.expected_binding_sha256,
                expected_core_binding_sha256=args.expected_core_binding_sha256,
            )
            output = {
                "status": "VERIFIED",
                "artifact_count": len(sealed.artifacts),
                "manifest_count": len(sealed.artifacts),
            }
        elif args.command == "capacity":
            core = FoundationalAuditStore.open(
                database_path=audit.database_path,
                run_id=args.run_id,
                audit_id=args.core_audit_id,
            )
            try:
                report = calculate_combined_hypothetical_capacity(
                    core_audit=core,
                    supplement_audit=audit,
                    expected_core_binding_sha256=(
                        args.expected_core_binding_sha256
                    ),
                    expected_supplement_binding_sha256=(
                        args.expected_binding_sha256
                    ),
                )
            finally:
                core.close()
            output = {
                "status": "SUFFICIENT" if report.all_sufficient else "INSUFFICIENT",
                "all_sufficient": report.all_sufficient,
                "allocation_available": report.allocation_available,
                "allocation_count": report.allocation_count,
                "sufficient_strata_count": sum(
                    item.sufficient for item in report.strata
                ),
            }
        else:
            output = _public_status(audit)
        audit.close()
        print(json.dumps(output, sort_keys=True))
        return 0
    except IronSulfideSupplementAuditError as error:
        print(
            json.dumps(
                {
                    "status": "IRON_SULFIDE_SUPPLEMENT_AUDIT_BLOCKED",
                    "error_code": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except Exception:
        print(
            json.dumps(
                {
                    "status": "IRON_SULFIDE_SUPPLEMENT_AUDIT_BLOCKED",
                    "error_code": "IRON_SULFIDE_SUPPLEMENT_OPERATION_FAILED",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
