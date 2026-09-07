"""Operate the private M2 coverage register with aggregate-only output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path[:0] = [str(PROJECT_ROOT), str(SRC_DIR)]

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (  # noqa: E402
    CorpusReconciliationError,
    CoverageDecisionRecord,
    ReconciliationStore,
    coverage_register_status,
    initialize_coverage_register,
    record_coverage_decision,
)


DEFAULT_ROOT = PROJECT_ROOT / ".private" / "corpus-reconciliation" / "v1"
DEFAULT_RUN_ID = "corpus-reconciliation-v1"
DEFAULT_REGISTER_ID = "m2-coverage-register-v1"


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise CorpusReconciliationError("CORPUS_RECONCILIATION_COVERAGE_ARGUMENT_INVALID")


def _parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description="Private M2 coverage register.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "record", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--private-root", type=Path, default=DEFAULT_ROOT)
        child.add_argument("--run-id", default=DEFAULT_RUN_ID)
        child.add_argument("--register-id", default=DEFAULT_REGISTER_ID)
    return parser


def _private_root(value: Path) -> Path:
    resolved = value.resolve()
    if resolved != DEFAULT_ROOT.resolve():
        raise CorpusReconciliationError("CORPUS_RECONCILIATION_PRIVATE_ROOT_INVALID")
    return resolved


def _read_stdin_mapping() -> dict[str, object]:
    try:
        payload = json.loads(sys.stdin.read())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_COVERAGE_STDIN_INVALID"
        ) from error
    if not isinstance(payload, dict):
        raise CorpusReconciliationError("CORPUS_RECONCILIATION_COVERAGE_STDIN_INVALID")
    return payload


def _public_status(status: object) -> dict[str, object]:
    return {
        "status": "COMPLETE" if status.complete else "IN_PROGRESS",
        "identity_count": status.identity_count,
        "current_decision_count": status.current_decision_count,
        "remaining_count": status.remaining_count,
        "disposition_counts": status.disposition_counts,
    }


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = _private_root(args.private_root)
        store = ReconciliationStore.open(
            root=root,
            expected_root=root,
            run_id=args.run_id,
        )
        try:
            if args.command == "init":
                status = initialize_coverage_register(
                    store=store,
                    register_id=args.register_id,
                )
            elif args.command == "record":
                status = record_coverage_decision(
                    store=store,
                    register_id=args.register_id,
                    record=CoverageDecisionRecord.from_mapping(_read_stdin_mapping()),
                )
            else:
                status = coverage_register_status(
                    store=store,
                    register_id=args.register_id,
                )
        finally:
            store.close()
        print(json.dumps(_public_status(status), sort_keys=True))
        return 0
    except CorpusReconciliationError as error:
        code = str(error)
        if re.fullmatch(r"CORPUS_RECONCILIATION_[A-Z0-9_]+", code) is None:
            code = "CORPUS_RECONCILIATION_COVERAGE_OPERATION_FAILED"
        print(
            json.dumps(
                {"status": "CORPUS_RECONCILIATION_COVERAGE_BLOCKED", "error_code": code},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except Exception:
        print(
            json.dumps(
                {
                    "status": "CORPUS_RECONCILIATION_COVERAGE_BLOCKED",
                    "error_code": "CORPUS_RECONCILIATION_COVERAGE_OPERATION_FAILED",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
