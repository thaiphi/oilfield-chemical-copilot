"""Explicit private-register initialization command for Corpus V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from oilfield_chemical_copilot.corpus_v2.models import CorpusV2ContractError, ReleaseConfig
from oilfield_chemical_copilot.corpus_v2.registers import CorpusV2RegisterError, initialize_registers


def _load_config(path: Path) -> ReleaseConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise CorpusV2RegisterError("C2_CONFIG_INVALID") from None
    try:
        return ReleaseConfig.from_mapping(payload)
    except CorpusV2ContractError:
        raise CorpusV2RegisterError("C2_CONFIG_INVALID") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize an approved private Corpus V2 register.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize-registers")
    initialize.add_argument("--release-config", type=Path, required=True)
    initialize.add_argument("--approved-register", type=Path, required=True)
    initialize.add_argument("--critical-register", type=Path, required=True)
    initialize.add_argument("--private-root", type=Path, required=True)
    initialize.add_argument("--ledger", type=Path, required=True)
    initialize.add_argument("--manifest-root", type=Path, required=True)
    process = subparsers.add_parser("process-snapshots")
    process.add_argument("--release-config", type=Path, required=True)
    process.add_argument("--ledger", type=Path, required=True)
    process.add_argument("--private-root", type=Path, required=True)
    process.add_argument("--extracted-root", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "initialize-registers":
        try:
            config = _load_config(arguments.release_config)
            initialize_registers(
                release_config=config,
                approved_register_path=arguments.approved_register,
                critical_register_path=arguments.critical_register,
                approved_private_root=arguments.private_root,
                ledger_path=arguments.ledger,
                manifest_root=arguments.manifest_root,
            )
        except CorpusV2RegisterError:
            return 2
    if arguments.command == "process-snapshots":
        # The command shape is intentional.  Real parser execution requires the
        # separately authorized operational task and is never implicit here.
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
