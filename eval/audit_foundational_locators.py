"""Operate the private foundational-locator audit with aggregate-only output."""

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
    FoundationalLocatorAuditError,
    LocatorAuditDecision,
    audit_status,
    extract_candidate_page,
    initialize_audit,
    next_unreviewed_locator,
    record_locator_decision,
)


DEFAULT_ROOT = PROJECT_ROOT / ".private" / "corpus-reconciliation" / "v1"
DEFAULT_RUN_ID = "corpus-reconciliation-v1"
DEFAULT_AUDIT_ID = "foundational-locator-audit-v1"


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_ARGUMENT_INVALID"
        )


def _parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description="Private foundational locator audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--private-root", type=Path, default=DEFAULT_ROOT)
        child.add_argument("--run-id", default=DEFAULT_RUN_ID)
        child.add_argument("--audit-id", default=DEFAULT_AUDIT_ID)
        return child

    command("init")
    next_command = command("next")
    next_command.add_argument("--pdf-path", type=Path, required=True)
    next_command.add_argument("--packet-output", type=Path, required=True)
    command("record")
    command("status")
    return parser


def _private_root(value: Path) -> Path:
    resolved = value.resolve()
    if resolved.name != "v1" or resolved.parent.name != "corpus-reconciliation":
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_PRIVATE_ROOT_INVALID"
        )
    if resolved.parent.parent.name != ".private":
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_PRIVATE_ROOT_INVALID"
        )
    return resolved


def _open_audit(args: argparse.Namespace) -> FoundationalAuditStore:
    root = _private_root(args.private_root)
    return FoundationalAuditStore.open(
        database_path=root / "reconciliation.sqlite",
        run_id=args.run_id,
        audit_id=args.audit_id,
    )


def _read_stdin_mapping() -> dict[str, object]:
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, UnicodeError) as error:
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_STDIN_INVALID"
        ) from error
    if not isinstance(payload, dict):
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_STDIN_INVALID"
        )
    return payload


def _public_status(audit: FoundationalAuditStore) -> dict[str, object]:
    status = audit_status(audit)
    return {
        "status": status.status,
        "candidate_count": status.candidate_count,
        "current_decision_count": status.current_decision_count,
        "remaining_count": status.remaining_count,
    }


def _write_packet(*, root: Path, destination: Path, payload: bytes) -> None:
    resolved = destination.resolve()
    if not resolved.is_relative_to(root):
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_PACKET_PATH_INVALID"
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
        raise FoundationalLocatorAuditError(
            "FOUNDATIONAL_LOCATOR_AUDIT_PACKET_WRITE_FAILED"
        ) from error


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = _private_root(args.private_root)
        if args.command == "init":
            payload = _read_stdin_mapping()
            if set(payload) != {
                "snapshot_binding_sha256",
                "source_drive_file_id",
                "source_file_sha256",
            }:
                raise FoundationalLocatorAuditError(
                    "FOUNDATIONAL_LOCATOR_AUDIT_STDIN_INVALID"
                )
            store = ReconciliationStore.open(
                root=root,
                expected_root=root,
                run_id=args.run_id,
            )
            audit = initialize_audit(
                store=store,
                audit_id=args.audit_id,
                snapshot_binding_sha256=payload["snapshot_binding_sha256"],
                source_drive_file_id=payload["source_drive_file_id"],
                source_file_sha256=payload["source_file_sha256"],
            )
            store.close()
        else:
            audit = _open_audit(args)
        if args.command == "record":
            record_locator_decision(
                audit=audit,
                record=LocatorAuditDecision.from_mapping(_read_stdin_mapping()),
            )
        elif args.command == "next":
            locator = next_unreviewed_locator(audit)
            if locator is not None:
                packet = extract_candidate_page(
                    audit=audit,
                    pdf_path=args.pdf_path,
                    locator=locator,
                )
                content = (
                    json.dumps(
                        packet.to_mapping(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
                _write_packet(
                    root=root,
                    destination=args.packet_output,
                    payload=content,
                )
        output = _public_status(audit)
        audit.close()
        print(json.dumps(output, sort_keys=True))
        return 0
    except FoundationalLocatorAuditError as error:
        print(
            json.dumps(
                {"status": "FOUNDATIONAL_LOCATOR_AUDIT_BLOCKED", "error_code": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
