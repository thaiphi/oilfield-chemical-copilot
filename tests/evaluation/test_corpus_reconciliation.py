from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def test_drive_record_requires_exact_strict_mapping() -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        DriveFileRecord,
    )

    payload = {
        "drive_file_id": "drive-1",
        "name": "private.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 10,
        "checksum_algorithm": None,
        "checksum": None,
        "modified_time": "2026-08-23T00:00:00Z",
        "parent_ids": ["folder-1"],
    }

    record = DriveFileRecord.from_mapping(payload)

    assert record.drive_file_id == "drive-1"
    assert record.parent_ids == ("folder-1",)
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_DRIVE_RECORD_INVALID"):
        DriveFileRecord.from_mapping({**payload, "content": "forbidden"})
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_DRIVE_RECORD_INVALID"):
        DriveFileRecord.from_mapping({**payload, "size_bytes": True})


def test_private_root_must_equal_expected_root(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        require_private_reconciliation_root,
    )

    expected = tmp_path / ".private" / "corpus-reconciliation" / "v1"

    assert require_private_reconciliation_root(expected, expected_root=expected) == expected.resolve()
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_PRIVATE_ROOT_INVALID"):
        require_private_reconciliation_root(tmp_path / "other", expected_root=expected)


def test_local_and_index_records_are_metadata_only() -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        IndexLocatorRecord,
        IndexSourceRecord,
        LocalFileRecord,
    )

    local = LocalFileRecord.from_mapping(
        {
            "relative_path": "approved/private.pdf",
            "sha256": "a" * 64,
            "provider_checksum_algorithm": None,
            "provider_checksum": None,
            "size_bytes": 10,
            "file_type": "pdf",
            "parser_status": "PARSED",
            "page_or_sheet_count": 2,
        }
    )
    source = IndexSourceRecord.from_mapping(
        {
            "source_id": "approved/private.pdf",
            "source_path": "approved/private.pdf",
            "parser_type": "pdf",
            "topic": "scale",
            "chunk_count": 2,
            "embedding_model": "model",
            "index_contract_sha256": "b" * 64,
        }
    )
    locator = IndexLocatorRecord.from_mapping(
        {
            "source_id": source.source_id,
            "locator": "page:1",
            "topic": "scale",
            "source_role": "supporting",
            "substantive_status": "SUBSTANTIVE",
        }
    )

    assert local.page_or_sheet_count == 2
    assert locator.source_role == "supporting"
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_LOCAL_RECORD_INVALID"):
        LocalFileRecord.from_mapping({**local.to_mapping(), "text": "forbidden"})
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_INDEX_RECORD_INVALID"):
        IndexSourceRecord.from_mapping({**source.to_mapping(), "embedding": [0.1]})


def test_store_resumes_same_contract_bound_run(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import ReconciliationStore

    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    store = ReconciliationStore.create(
        root=root,
        expected_root=root,
        run_id="run-001",
        index_contract_sha256="a" * 64,
        e1a3_allocation_sha256="b" * 64,
    )
    store.set_checkpoint(
        stage="drive_inventory",
        status="IN_PROGRESS",
        committed_records=5,
        page_token="token-2",
    )
    assert store.pragma("foreign_keys") == 1
    assert store.pragma("journal_mode").lower() == "wal"
    store.close()

    resumed = ReconciliationStore.open(root=root, expected_root=root, run_id="run-001")
    checkpoint = resumed.checkpoint("drive_inventory")

    assert checkpoint.status == "IN_PROGRESS"
    assert checkpoint.committed_records == 5
    assert checkpoint.page_token == "token-2"
    resumed.close()


def test_store_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        ReconciliationStore,
    )

    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    store = ReconciliationStore.create(
        root=root,
        expected_root=root,
        run_id="run-001",
        index_contract_sha256="a" * 64,
        e1a3_allocation_sha256="b" * 64,
    )
    store.close()
    with sqlite3.connect(root / "reconciliation.sqlite") as connection:
        connection.execute("update runs set schema_version = 2")

    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_SCHEMA_INVALID"):
        ReconciliationStore.open(root=root, expected_root=root, run_id="run-001")


def test_store_rejects_conflicting_run_for_same_contract_pair(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        ReconciliationStore,
    )

    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    first = ReconciliationStore.create(
        root=root,
        expected_root=root,
        run_id="run-001",
        index_contract_sha256="a" * 64,
        e1a3_allocation_sha256="b" * 64,
    )
    first.close()

    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_RUN_CONFLICT"):
        ReconciliationStore.create(
            root=root,
            expected_root=root,
            run_id="run-002",
            index_contract_sha256="a" * 64,
            e1a3_allocation_sha256="b" * 64,
        )
