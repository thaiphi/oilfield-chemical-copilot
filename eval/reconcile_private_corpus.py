"""Operate the resumable private corpus reconciliation with aggregate-only output."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping

import psycopg
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path[:0] = [str(PROJECT_ROOT), str(SRC_DIR)]

from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (  # noqa: E402
    CorpusReconciliationError,
    DriveFileRecord,
    IndexLocatorRecord,
    IndexSourceRecord,
    ReconciliationStore,
    ReviewDecisionRecord,
    SEALED_SNAPSHOT_NAMES,
    TOPICS,
    calculate_locator_capacity,
    dry_run_e1a4_allocation,
    import_drive_page,
    import_index_inventory,
    inventory_local_files,
    reconcile_document_matches,
    record_review_decision,
    seal_reconciliation_snapshots,
    verify_reconciliation_snapshots,
)
from oilfield_chemical_copilot.evaluation.e1a3_sampling import (  # noqa: E402
    E1A3SamplingError,
)
from oilfield_chemical_copilot.evaluation.index_preflight import (  # noqa: E402
    E1IndexPreflightError,
    verify_e1_index_contract,
)
from oilfield_chemical_copilot.evaluation.private_retrieval import (  # noqa: E402
    PRIVATE_RETRIEVAL_ROOT,
)


DEFAULT_ROOT = PROJECT_ROOT / ".private" / "corpus-reconciliation" / "v1"
DEFAULT_RUN_ID = "corpus-reconciliation-v1"
DEFAULT_INDEX_CONTRACT = PRIVATE_RETRIEVAL_ROOT / "contracts" / "index-contract.json"
DEFAULT_E1A3_ROOT = PRIVATE_RETRIEVAL_ROOT / "e1a3"


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CorpusReconciliationError("CORPUS_RECONCILIATION_ARGUMENT_INVALID")


def _parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description="Private metadata-only corpus reconciliation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--private-root", type=Path, default=DEFAULT_ROOT)
        child.add_argument("--run-id", default=DEFAULT_RUN_ID)
        child.add_argument("--index-contract", type=Path, default=DEFAULT_INDEX_CONTRACT)
        child.add_argument("--e1a3-root", type=Path, default=DEFAULT_E1A3_ROOT)
        return child

    command("init")
    command("import-drive-page")
    local = command("inventory-local")
    local.add_argument("--local-root", type=Path, action="append", required=True)
    index = command("import-index")
    index.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""))
    command("reconcile")
    command("record-review-decision")
    command("capacity")
    command("seal")
    command("status")
    return parser


def _safe_digest_file(path: Path, *, error_code: str) -> str:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise CorpusReconciliationError(error_code) from error
    return hashlib.sha256(content).hexdigest()


def _verified_json_manifest(
    *, payload_path: Path, manifest_path: Path, error_code: str
) -> tuple[Mapping[str, object], str]:
    try:
        content = payload_path.read_bytes()
        manifest = manifest_path.read_text(encoding="ascii")
        payload = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusReconciliationError(error_code) from error
    digest = hashlib.sha256(content).hexdigest()
    if (
        not isinstance(payload, Mapping)
        or manifest != f"{digest}\n"
        or len(digest) != 64
    ):
        raise CorpusReconciliationError(error_code)
    return payload, digest


def _load_prior_locator_keys(e1a3_root: Path) -> tuple[str, ...]:
    payload, _ = _verified_json_manifest(
        payload_path=e1a3_root / "sealed" / "sampling-allocation.v4.json",
        manifest_path=e1a3_root / "manifests" / "sampling-allocation.v4.sha256",
        error_code="CORPUS_RECONCILIATION_E1A3_ALLOCATION_INVALID",
    )
    if set(payload) != {
        "schema_version",
        "source_register_sha256",
        "slot_count",
        "allocations",
    } or payload["schema_version"] != 1 or payload["slot_count"] != 96:
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_E1A3_ALLOCATION_INVALID"
        )
    allocations = payload["allocations"]
    if not isinstance(allocations, list) or len(allocations) != 96:
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_E1A3_ALLOCATION_INVALID"
        )
    try:
        keys = tuple(
            f"{item['source_id'].strip()}:{item['locator'].strip()}"
            for item in allocations
            if isinstance(item, Mapping)
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_E1A3_ALLOCATION_INVALID"
        ) from error
    if len(keys) != 96 or len(keys) != len(set(keys)):
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_E1A3_ALLOCATION_INVALID"
        )
    return keys


def _allocation_digest(e1a3_root: Path) -> str:
    _, digest = _verified_json_manifest(
        payload_path=e1a3_root / "sealed" / "sampling-allocation.v4.json",
        manifest_path=e1a3_root / "manifests" / "sampling-allocation.v4.sha256",
        error_code="CORPUS_RECONCILIATION_E1A3_ALLOCATION_INVALID",
    )
    return digest


def _index_contract_digest(path: Path) -> str:
    code = "CORPUS_RECONCILIATION_INDEX_CONTRACT_INVALID"
    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusReconciliationError(code) from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "chunk_count",
        "distinct_source_count",
        "embedding_models",
        "embedding_dimensions",
        "inventory_sha256",
    }:
        raise CorpusReconciliationError(code)
    if (
        type(payload["chunk_count"]) is not int
        or payload["chunk_count"] < 1
        or type(payload["distinct_source_count"]) is not int
        or payload["distinct_source_count"] < 1
        or not isinstance(payload["embedding_models"], list)
        or not payload["embedding_models"]
        or any(
            not isinstance(model, str) or not model.strip()
            for model in payload["embedding_models"]
        )
        or not isinstance(payload["embedding_dimensions"], list)
        or not payload["embedding_dimensions"]
        or any(
            type(dimension) is not int or dimension < 1
            for dimension in payload["embedding_dimensions"]
        )
        or not isinstance(payload["inventory_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", payload["inventory_sha256"]) is None
    ):
        raise CorpusReconciliationError(code)
    return hashlib.sha256(content).hexdigest()


def _validate_store_bindings(
    *, store: ReconciliationStore, index_contract: Path, e1a3_root: Path
) -> None:
    _load_prior_locator_keys(e1a3_root)
    current = (_index_contract_digest(index_contract), _allocation_digest(e1a3_root))
    if store.contract_digests() != current:
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_CONTRACT_BINDING_MISMATCH"
        )


def _open_store(args: argparse.Namespace) -> ReconciliationStore:
    return ReconciliationStore.open(
        root=args.private_root,
        expected_root=DEFAULT_ROOT,
        run_id=args.run_id,
    )


def _status_payload(store: ReconciliationStore) -> dict[str, object]:
    counts = {
        table: store.count(table)
        for table in (
            "drive_files",
            "local_files",
            "index_sources",
            "index_locators",
            "document_matches",
            "review_decisions",
        )
    }
    stages: dict[str, str] = {}
    for stage in (
        "drive_inventory",
        "local_inventory",
        "index_inventory",
        "document_matching",
        "locator_capacity",
        "e1a4_dry_run",
    ):
        try:
            stages[stage] = store.checkpoint(stage).status
        except CorpusReconciliationError:
            stages[stage] = "NOT_STARTED"
    snapshots = store.root / "snapshots"
    snapshot_paths = tuple(
        path
        for name in SEALED_SNAPSHOT_NAMES
        for path in (snapshots / name, snapshots / f"{name}.sha256")
    )
    presence = tuple(path.is_file() for path in snapshot_paths)
    snapshots_complete = False
    if any(presence):
        verify_reconciliation_snapshots(root=store.root, store=store)
        snapshots_complete = True
    status = (
        "CORPUS_RECONCILIATION_COMPLETE"
        if snapshots_complete and all(value == "COMPLETE" for value in stages.values())
        else "CORPUS_RECONCILIATION_IN_PROGRESS"
    )
    return {
        "status": status,
        "counts": counts,
        "stages": stages,
        "snapshots_complete": snapshots_complete,
    }


def _verified_read_only_connection(
    *, database_url: str, index_contract: Path
):
    if not isinstance(database_url, str) or not database_url.strip():
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_DATABASE_URL_MISSING"
        )
    verify_e1_index_contract(
        database_url=database_url,
        contract_path=index_contract,
    )
    return psycopg.connect(
        database_url,
        options="-c default_transaction_read_only=on",
        row_factory=dict_row,
    )


def _load_e1a3_register(e1a3_root: Path) -> tuple[Mapping[str, object], ...]:
    payload, digest = _verified_json_manifest(
        payload_path=e1a3_root / "sealed" / "source-register.v4.json",
        manifest_path=e1a3_root / "manifests" / "source-register.v4.sha256",
        error_code="CORPUS_RECONCILIATION_E1A3_REGISTER_INVALID",
    )
    sources = payload.get("sources")
    if (
        set(payload) != {"schema_version", "source_role_config_sha256", "sources"}
        or payload["schema_version"] != 1
        or not isinstance(sources, list)
        or not sources
    ):
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_E1A3_REGISTER_INVALID"
        )
    allocation, _ = _verified_json_manifest(
        payload_path=e1a3_root / "sealed" / "sampling-allocation.v4.json",
        manifest_path=e1a3_root / "manifests" / "sampling-allocation.v4.sha256",
        error_code="CORPUS_RECONCILIATION_E1A3_ALLOCATION_INVALID",
    )
    if allocation.get("source_register_sha256") != digest:
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_E1A3_REGISTER_INVALID"
        )
    return tuple(item for item in sources if isinstance(item, Mapping))


def _read_index_inventory(
    *, database_url: str, index_contract: Path, e1a3_root: Path
) -> tuple[
    tuple[IndexSourceRecord, ...],
    tuple[IndexLocatorRecord, ...],
    int,
    int,
]:
    register = _load_e1a3_register(e1a3_root)
    eligibility: dict[tuple[str, str], tuple[str, str]] = {}
    foundational_sources: set[str] = set()
    for item in register:
        try:
            source_id = str(item["source_id"]).strip()
            topic = str(item["topic"]).strip()
            role = str(item["source_role"]).strip()
            locators = item["locators"]
        except (KeyError, TypeError) as error:
            raise CorpusReconciliationError(
                "CORPUS_RECONCILIATION_E1A3_REGISTER_INVALID"
            ) from error
        if not isinstance(locators, list):
            raise CorpusReconciliationError(
                "CORPUS_RECONCILIATION_E1A3_REGISTER_INVALID"
            )
        if role == "foundational":
            foundational_sources.add(source_id)
        for locator_value in locators:
            locator = str(locator_value).strip()
            key = (source_id, locator)
            value = (topic, role)
            if key in eligibility and eligibility[key] != value:
                raise CorpusReconciliationError(
                    "CORPUS_RECONCILIATION_INDEX_LOCATOR_CONFLICT"
                )
            eligibility[key] = value

    try:
        connection = _verified_read_only_connection(
            database_url=database_url,
            index_contract=index_contract,
        )
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select source_file, parser_type, embedding_model,
                           min(topic) as topic, count(*) as chunk_count
                    from chunks
                    where embedding is not null
                    group by source_file, parser_type, embedding_model
                    order by source_file, parser_type, embedding_model
                    """
                )
                source_rows = cursor.fetchall()
                cursor.execute(
                    """
                    select distinct source_file, topic, parser_type, page_or_sheet
                    from chunks
                    where embedding is not null and page_or_sheet is not null
                    order by source_file, topic, parser_type, page_or_sheet
                    """
                )
                locator_rows = cursor.fetchall()
    except (CorpusReconciliationError, E1IndexPreflightError):
        raise
    except Exception as error:
        raise CorpusReconciliationError(
            "CORPUS_RECONCILIATION_INDEX_READ_FAILED"
        ) from error

    contract_digest = _safe_digest_file(
        index_contract, error_code="CORPUS_RECONCILIATION_INDEX_CONTRACT_INVALID"
    )
    sources = tuple(
        IndexSourceRecord.from_mapping(
            {
                "source_id": row["source_file"],
                "source_path": row["source_file"],
                "parser_type": row["parser_type"],
                "topic": (
                    str(row["topic"]).strip()
                    if str(row["topic"]).strip() in TOPICS
                    else "unassigned"
                ),
                "chunk_count": row["chunk_count"],
                "embedding_model": row["embedding_model"],
                "index_contract_sha256": contract_digest,
                "provenance_drive_file_id": None,
                "content_sha256": None,
            }
        )
        for row in source_rows
    )
    locator_candidates: dict[tuple[str, str], IndexLocatorRecord] = {}
    for row in locator_rows:
        source_id = str(row["source_file"]).strip()
        locator = str(row["page_or_sheet"]).strip()
        database_topic = str(row["topic"]).strip()
        topic, role = eligibility.get(
            (source_id, locator),
            (
                database_topic if database_topic in TOPICS else "unassigned",
                "foundational" if source_id in foundational_sources else "supporting",
            ),
        )
        status = "SUBSTANTIVE" if (source_id, locator) in eligibility else "INELIGIBLE"
        candidate = IndexLocatorRecord.from_mapping(
            {
                "source_id": source_id,
                "locator": locator,
                "topic": topic,
                "source_role": role,
                "substantive_status": status,
            }
        )
        key = (source_id, locator)
        if key in locator_candidates and locator_candidates[key] != candidate:
            raise CorpusReconciliationError(
                "CORPUS_RECONCILIATION_INDEX_LOCATOR_CONFLICT"
            )
        locator_candidates[key] = candidate
    expected_source_count = len(sources)
    expected_chunk_count = sum(source.chunk_count for source in sources)
    return (
        sources,
        tuple(locator_candidates[key] for key in sorted(locator_candidates)),
        expected_source_count,
        expected_chunk_count,
    )


def _read_drive_stdin() -> tuple[tuple[DriveFileRecord, ...], str | None, str | None]:
    code = "CORPUS_RECONCILIATION_DRIVE_STDIN_INVALID"
    try:
        payload = json.loads(sys.stdin.readline())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CorpusReconciliationError(code) from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "records",
        "page_token",
        "next_page_token",
    }:
        raise CorpusReconciliationError(code)
    records = payload["records"]
    page_token = payload["page_token"]
    token = payload["next_page_token"]
    if (
        not isinstance(records, list)
        or (page_token is not None and not isinstance(page_token, str))
        or (token is not None and not isinstance(token, str))
    ):
        raise CorpusReconciliationError(code)
    try:
        return (
            tuple(DriveFileRecord.from_mapping(item) for item in records),
            page_token,
            token,
        )
    except (CorpusReconciliationError, TypeError) as error:
        raise CorpusReconciliationError(code) from error


def _read_review_decision_stdin() -> ReviewDecisionRecord:
    code = "CORPUS_RECONCILIATION_REVIEW_STDIN_INVALID"
    try:
        payload = json.loads(sys.stdin.readline())
        if not isinstance(payload, Mapping):
            raise CorpusReconciliationError(code)
        return ReviewDecisionRecord.from_mapping(payload)
    except (UnicodeError, json.JSONDecodeError, TypeError, CorpusReconciliationError) as error:
        raise CorpusReconciliationError(code) from error


def main() -> int:
    args = _parser().parse_args()
    if args.command == "init":
        index_digest = _index_contract_digest(args.index_contract)
        _load_prior_locator_keys(args.e1a3_root)
        allocation_digest = _allocation_digest(args.e1a3_root)
        store = ReconciliationStore.create(
            root=args.private_root,
            expected_root=DEFAULT_ROOT,
            run_id=args.run_id,
            index_contract_sha256=index_digest,
            e1a3_allocation_sha256=allocation_digest,
        )
        payload = _status_payload(store)
        store.close()
        print(json.dumps(payload, sort_keys=True))
        return 0

    if args.command == "import-drive-page":
        records, page_token, token = _read_drive_stdin()
        store = _open_store(args)
        _validate_store_bindings(
            store=store,
            index_contract=args.index_contract,
            e1a3_root=args.e1a3_root,
        )
        result = import_drive_page(
            store=store,
            records=records,
            page_token=page_token,
            next_page_token=token,
        )
        store.close()
        print(
            json.dumps(
                {
                    "status": result.status,
                    "stage": result.stage,
                    "committed_records": result.committed_records,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "status":
        try:
            store = _open_store(args)
        except CorpusReconciliationError as error:
            raise CorpusReconciliationError(
                "CORPUS_RECONCILIATION_PREREQUISITES_MISSING"
            ) from error
        payload = _status_payload(store)
        store.close()
        print(json.dumps(payload, sort_keys=True))
        return 0

    store = _open_store(args)
    _validate_store_bindings(
        store=store,
        index_contract=args.index_contract,
        e1a3_root=args.e1a3_root,
    )
    if args.command == "inventory-local":
        result = inventory_local_files(store=store, roots=args.local_root)
        output: dict[str, object] = {
            "status": result.status,
            "stage": result.stage,
            "committed_records": result.committed_records,
        }
    elif args.command == "import-index":
        sources, locators, source_count, chunk_count = _read_index_inventory(
            database_url=args.database_url,
            index_contract=args.index_contract,
            e1a3_root=args.e1a3_root,
        )
        result = import_index_inventory(
            store=store,
            sources=sources,
            locators=locators,
            expected_source_count=source_count,
            expected_chunk_count=chunk_count,
        )
        output = {
            "status": result.status,
            "stage": result.stage,
            "source_count": source_count,
            "locator_count": len(locators),
        }
    elif args.command == "reconcile":
        summary = reconcile_document_matches(store=store)
        checkpoint = store.checkpoint("document_matching")
        output = {
            "status": checkpoint.status,
            "error_code": checkpoint.error_code,
            "counts": asdict(summary),
        }
    elif args.command == "record-review-decision":
        result = record_review_decision(
            store=store,
            record=_read_review_decision_stdin(),
        )
        output = {
            "status": result.status,
            "stage": result.stage,
            "committed_records": result.committed_records,
        }
    elif args.command == "capacity":
        prior = _load_prior_locator_keys(args.e1a3_root)
        report = calculate_locator_capacity(
            store=store, prior_locator_keys=prior
        )
        dry_run = dry_run_e1a4_allocation(
            store=store, prior_locator_keys=prior
        )
        output = {
            "status": dry_run.status,
            "error_code": dry_run.error_code,
            "capacity": report.to_public_mapping(),
        }
    elif args.command == "seal":
        snapshot_set = seal_reconciliation_snapshots(
            store=store, root=args.private_root
        )
        output = {
            "status": "COMPLETE",
            "snapshot_count": len(snapshot_set.artifacts),
        }
    else:
        raise CorpusReconciliationError("CORPUS_RECONCILIATION_ARGUMENT_INVALID")
    store.close()
    print(json.dumps(output, sort_keys=True))
    return 0


def cli() -> int:
    """Convert every failure to an aggregate-only closed error."""
    try:
        return main()
    except (CorpusReconciliationError, E1IndexPreflightError, E1A3SamplingError) as error:
        error_code = str(error)
        if re.fullmatch(r"CORPUS_RECONCILIATION_[A-Z0-9_]+", error_code) is None:
            error_code = "CORPUS_RECONCILIATION_FAILED"
        print(
            json.dumps(
                {
                    "status": "CORPUS_RECONCILIATION_BLOCKED",
                    "error_code": error_code,
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
                    "status": "CORPUS_RECONCILIATION_BLOCKED",
                    "error_code": "CORPUS_RECONCILIATION_FAILED",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
