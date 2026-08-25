from __future__ import annotations

import sqlite3
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

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
            "source_path": " approved\\private.pdf ",
            "parser_type": "pdf",
            "topic": "scale",
            "chunk_count": 2,
            "embedding_model": "model",
            "index_contract_sha256": "b" * 64,
            "provenance_drive_file_id": None,
            "content_sha256": None,
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
    assert source.source_path == "approved/private.pdf"
    assert locator.source_role == "supporting"
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_LOCAL_RECORD_INVALID"):
        LocalFileRecord.from_mapping({**local.to_mapping(), "text": "forbidden"})
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_INDEX_RECORD_INVALID"):
        IndexSourceRecord.from_mapping({**source.to_mapping(), "embedding": [0.1]})


def test_unassigned_topic_is_allowed_only_for_closed_ineligible_locator() -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        IndexLocatorRecord,
        IndexSourceRecord,
    )

    source = IndexSourceRecord.from_mapping(
        {
            "source_id": "out-of-scope.pdf",
            "source_path": "out-of-scope.pdf",
            "parser_type": "pdf",
            "topic": "unassigned",
            "chunk_count": 1,
            "embedding_model": "model",
            "index_contract_sha256": "a" * 64,
            "provenance_drive_file_id": None,
            "content_sha256": None,
        }
    )

    assert source.topic == "unassigned"
    locator = IndexLocatorRecord.from_mapping(
        {
            "source_id": source.source_id,
            "locator": "page:1",
            "topic": "unassigned",
            "source_role": "supporting",
            "substantive_status": "INELIGIBLE",
        }
    )

    assert locator.substantive_status == "INELIGIBLE"
    with pytest.raises(
        CorpusReconciliationError,
        match="CORPUS_RECONCILIATION_INDEX_RECORD_INVALID",
    ):
        IndexLocatorRecord.from_mapping(
            {
                "source_id": source.source_id,
                "locator": "page:1",
                "topic": "unassigned",
                "source_role": "supporting",
                "substantive_status": "SUBSTANTIVE",
            }
        )


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


def _create_store(tmp_path: Path):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import ReconciliationStore

    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    return ReconciliationStore.create(
        root=root,
        expected_root=root,
        run_id="run-001",
        index_contract_sha256="a" * 64,
        e1a3_allocation_sha256="b" * 64,
    )


def _drive_record(
    *,
    drive_file_id: str = "drive-1",
    size_bytes: int = 10,
    name: str | None = None,
    checksum: str | None = "c" * 32,
):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import DriveFileRecord

    return DriveFileRecord.from_mapping(
        {
            "drive_file_id": drive_file_id,
            "name": name or f"{drive_file_id}.pdf",
            "mime_type": "application/pdf",
            "size_bytes": size_bytes,
            "checksum_algorithm": "md5" if checksum is not None else None,
            "checksum": checksum,
            "modified_time": "2026-08-23T00:00:00Z",
            "parent_ids": ["folder-1"],
        }
    )


def test_drive_rescan_upserts_stable_ids_without_duplication(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import import_drive_page

    store = _create_store(tmp_path)
    page = (_drive_record(),)

    first = import_drive_page(store=store, records=page, next_page_token="token-2")
    final = import_drive_page(store=store, records=page, next_page_token=None)

    assert first.status == "IN_PROGRESS"
    assert final.status == "COMPLETE"
    assert store.count("drive_files") == 1
    assert store.checkpoint("drive_inventory").committed_records == 1
    store.close()


def test_drive_conflict_is_blocked_with_closed_error_code(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        import_drive_page,
    )

    store = _create_store(tmp_path)
    import_drive_page(store=store, records=(_drive_record(),), next_page_token="token-2")

    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_DRIVE_CONFLICT"):
        import_drive_page(
            store=store,
            records=(_drive_record(size_bytes=11),),
            next_page_token=None,
        )

    checkpoint = store.checkpoint("drive_inventory")
    assert checkpoint.status == "BLOCKED"
    assert checkpoint.error_code == "CORPUS_RECONCILIATION_DRIVE_CONFLICT"
    assert "11" not in checkpoint.error_code
    store.close()


def test_restart_keeps_committed_drive_pages(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        ReconciliationStore,
        import_drive_page,
    )

    store = _create_store(tmp_path)
    root = store.root
    import_drive_page(store=store, records=(_drive_record(),), next_page_token="expired-token")
    store.close()

    resumed = ReconciliationStore.open(root=root, expected_root=root, run_id="run-001")
    import_drive_page(
        store=resumed,
        records=(_drive_record(drive_file_id="drive-2"),),
        next_page_token=None,
    )

    assert resumed.count("drive_files") == 2
    assert resumed.checkpoint("drive_inventory").status == "COMPLETE"
    resumed.close()


def test_local_inventory_hashes_without_persisting_file_bytes(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        LocalParserMetadata,
        inventory_local_files,
    )

    approved_root = tmp_path / "approved"
    approved_root.mkdir()
    content = b"unique-private-content-that-must-not-be-stored"
    source_file = approved_root / "private.pdf"
    source_file.write_bytes(content)
    store = _create_store(tmp_path)

    result = inventory_local_files(
        store=store,
        roots=(approved_root,),
        parser_metadata={
            "approved/private.pdf": LocalParserMetadata.from_mapping(
                {"parser_status": "PARSED", "page_or_sheet_count": 2}
            )
        },
    )
    record = store.local_file("approved/private.pdf")

    assert result.status == "COMPLETE"
    assert record.sha256 == hashlib.sha256(content).hexdigest()
    assert record.provider_checksum_algorithm == "md5"
    assert record.size_bytes == len(content)
    assert record.parser_status == "PARSED"
    assert record.page_or_sheet_count == 2
    store.close()
    assert content not in (store.root / "reconciliation.sqlite").read_bytes()


def test_local_inventory_rejects_rows_for_files_missing_on_rescan(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        inventory_local_files,
    )

    approved_root = tmp_path / "approved"
    approved_root.mkdir()
    source_file = approved_root / "private.pdf"
    source_file.write_bytes(b"private-content")
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(approved_root,))
    source_file.unlink()

    with pytest.raises(
        CorpusReconciliationError,
        match="CORPUS_RECONCILIATION_LOCAL_STALE_RECORDS",
    ):
        inventory_local_files(store=store, roots=(approved_root,))

    assert store.checkpoint("local_inventory").status == "BLOCKED"
    store.close()


def test_index_inventory_requires_bound_contract_and_exact_totals(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        IndexLocatorRecord,
        IndexSourceRecord,
        import_index_inventory,
    )

    store = _create_store(tmp_path)
    source = IndexSourceRecord.from_mapping(
        {
            "source_id": "source-1",
            "source_path": "approved/private.pdf",
            "parser_type": "pdf",
            "topic": "scale",
            "chunk_count": 2,
            "embedding_model": "model",
            "index_contract_sha256": "a" * 64,
            "provenance_drive_file_id": None,
            "content_sha256": None,
        }
    )
    locator = IndexLocatorRecord.from_mapping(
        {
            "source_id": "source-1",
            "locator": "page:1",
            "topic": "scale",
            "source_role": "supporting",
            "substantive_status": "SUBSTANTIVE",
        }
    )

    result = import_index_inventory(
        store=store,
        sources=(source,),
        locators=(locator,),
        expected_source_count=1,
        expected_chunk_count=2,
    )

    assert result.status == "COMPLETE"
    assert store.count("index_sources") == 1
    assert store.count("index_locators") == 1
    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_INDEX_CONTRACT_MISMATCH"):
        import_index_inventory(
            store=store,
            sources=(IndexSourceRecord(**{**source.__dict__, "index_contract_sha256": "d" * 64}),),
            locators=(locator,),
            expected_source_count=1,
            expected_chunk_count=2,
        )
    store.close()


def test_index_duplicate_locator_conflict_fails_closed(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        IndexLocatorRecord,
        IndexSourceRecord,
        import_index_inventory,
    )

    store = _create_store(tmp_path)
    source = IndexSourceRecord(
        source_id="source-1",
        source_path="approved/private.pdf",
        parser_type="pdf",
        topic="scale",
        chunk_count=2,
        embedding_model="model",
        index_contract_sha256="a" * 64,
        provenance_drive_file_id=None,
        content_sha256=None,
    )
    locators = (
        IndexLocatorRecord("source-1", "page:1", "scale", "supporting", "SUBSTANTIVE"),
        IndexLocatorRecord("source-1", "page:1", "scale", "foundational", "SUBSTANTIVE"),
    )

    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_INDEX_LOCATOR_CONFLICT"):
        import_index_inventory(
            store=store,
            sources=(source,),
            locators=locators,
            expected_source_count=1,
            expected_chunk_count=2,
        )

    checkpoint = store.checkpoint("index_inventory")
    assert checkpoint.status == "BLOCKED"
    assert checkpoint.error_code == "CORPUS_RECONCILIATION_INDEX_LOCATOR_CONFLICT"
    assert store.count("index_sources") == 0
    store.close()


def _source_record(
    *,
    source_id: str = "source-1",
    source_path: str = "approved/private.pdf",
    drive_file_id: str | None = None,
    content_sha256: str | None = None,
):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import IndexSourceRecord

    return IndexSourceRecord.from_mapping(
        {
            "source_id": source_id,
            "source_path": source_path,
            "parser_type": "pdf",
            "topic": "scale",
            "chunk_count": 1,
            "embedding_model": "model",
            "index_contract_sha256": "a" * 64,
            "provenance_drive_file_id": drive_file_id,
            "content_sha256": content_sha256,
        }
    )


def _locator_record(*, source_id: str = "source-1"):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import IndexLocatorRecord

    return IndexLocatorRecord.from_mapping(
        {
            "source_id": source_id,
            "locator": "page:1",
            "topic": "scale",
            "source_role": "supporting",
            "substantive_status": "SUBSTANTIVE",
        }
    )


def _write_local_file(tmp_path: Path, *, name: str, content: bytes) -> Path:
    approved_root = tmp_path / "approved"
    approved_root.mkdir(exist_ok=True)
    (approved_root / name).write_bytes(content)
    return approved_root


def test_same_algorithm_checksum_creates_exact_match(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    content = b"exact-document"
    root = _write_local_file(tmp_path, name="private.pdf", content=content)
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(root,))
    import_drive_page(
        store=store,
        records=(
            _drive_record(
                size_bytes=len(content),
                name="private.pdf",
                checksum=hashlib.md5(content, usedforsecurity=False).hexdigest(),
            ),
        ),
        next_page_token=None,
    )
    import_index_inventory(
        store=store,
        sources=(_source_record(),),
        locators=(_locator_record(),),
        expected_source_count=1,
        expected_chunk_count=1,
    )

    summary = reconcile_document_matches(store=store)

    assert summary.exact_match == 1
    assert store.match_status("drive-1") == "EXACT_MATCH"
    assert store.match_method("drive-1") == "PROVIDER_CHECKSUM"
    store.close()


def test_filename_and_size_only_requires_review(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    content = b"same-size"
    root = _write_local_file(tmp_path, name="private.pdf", content=content)
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(root,))
    import_drive_page(
        store=store,
        records=(
            _drive_record(
                size_bytes=len(content), name="private.pdf", checksum=None
            ),
        ),
        next_page_token=None,
    )
    import_index_inventory(
        store=store,
        sources=(_source_record(),),
        locators=(_locator_record(),),
        expected_source_count=1,
        expected_chunk_count=1,
    )

    summary = reconcile_document_matches(store=store)

    assert summary.ambiguous_review_required == 1
    assert store.match_status("drive-1") == "AMBIGUOUS_REVIEW_REQUIRED"
    store.close()


def test_exact_drive_id_provenance_outranks_absent_checksum(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    content = b"provenance-document"
    root = _write_local_file(tmp_path, name="private.pdf", content=content)
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(root,))
    import_drive_page(
        store=store,
        records=(
            _drive_record(size_bytes=len(content), name="renamed.pdf", checksum=None),
        ),
        next_page_token=None,
    )
    import_index_inventory(
        store=store,
        sources=(_source_record(drive_file_id="drive-1"),),
        locators=(_locator_record(),),
        expected_source_count=1,
        expected_chunk_count=1,
    )

    reconcile_document_matches(store=store)

    assert store.match_status("drive-1") == "EXACT_MATCH"
    assert store.match_method("drive-1") == "DRIVE_ID_PROVENANCE"
    store.close()


def test_exact_drive_id_provenance_does_not_require_local_copy(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    empty_root = tmp_path / "approved"
    empty_root.mkdir()
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(empty_root,))
    import_drive_page(
        store=store,
        records=(_drive_record(size_bytes=20, checksum=None),),
        next_page_token=None,
    )
    import_index_inventory(
        store=store,
        sources=(
            _source_record(
                drive_file_id="drive-1", content_sha256="f" * 64
            ),
        ),
        locators=(_locator_record(),),
        expected_source_count=1,
        expected_chunk_count=1,
    )

    summary = reconcile_document_matches(store=store)

    assert summary.exact_match == 1
    assert summary.index_only == 0
    assert store.match_method("drive-1") == "DRIVE_ID_PROVENANCE"
    store.close()


def test_conflicting_high_and_lower_precedence_identity_fails_closed(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    first = b"first-document"
    second = b"second-document"
    root = _write_local_file(tmp_path, name="first.pdf", content=first)
    _write_local_file(tmp_path, name="second.pdf", content=second)
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(root,))
    import_drive_page(
        store=store,
        records=(
            _drive_record(
                size_bytes=len(second),
                name="second.pdf",
                checksum=hashlib.md5(second, usedforsecurity=False).hexdigest(),
            ),
        ),
        next_page_token=None,
    )
    import_index_inventory(
        store=store,
        sources=(
            _source_record(
                source_path="approved/first.pdf", drive_file_id="drive-1"
            ),
        ),
        locators=(_locator_record(),),
        expected_source_count=1,
        expected_chunk_count=1,
    )

    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_MATCH_CONFLICT"):
        reconcile_document_matches(store=store)

    assert store.count("document_matches") == 0
    store.close()


def test_duplicate_drive_aliases_share_one_canonical_content_identity(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    content = b"duplicate-content"
    checksum = hashlib.md5(content, usedforsecurity=False).hexdigest()
    root = _write_local_file(tmp_path, name="private.pdf", content=content)
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(root,))
    import_drive_page(
        store=store,
        records=(
            _drive_record(
                drive_file_id="drive-1",
                size_bytes=len(content),
                checksum=checksum,
            ),
            _drive_record(
                drive_file_id="drive-2",
                size_bytes=len(content),
                checksum=checksum,
            ),
        ),
        next_page_token=None,
    )
    import_index_inventory(
        store=store,
        sources=(_source_record(),),
        locators=(_locator_record(),),
        expected_source_count=1,
        expected_chunk_count=1,
    )

    summary = reconcile_document_matches(store=store)

    assert summary.exact_match == 1
    assert summary.duplicate_alias == 1
    assert store.match_status("drive-2") == "DUPLICATE_ALIAS"
    store.close()


def test_index_source_cannot_map_to_conflicting_content_hash(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    root = _write_local_file(tmp_path, name="private.pdf", content=b"actual-content")
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(root,))
    import_drive_page(store=store, records=(), next_page_token=None)
    import_index_inventory(
        store=store,
        sources=(_source_record(content_sha256="e" * 64),),
        locators=(_locator_record(),),
        expected_source_count=1,
        expected_chunk_count=1,
    )

    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_INDEX_HASH_CONFLICT"):
        reconcile_document_matches(store=store)

    assert store.count("document_matches") == 0
    store.close()


def test_ingestion_content_sha_binds_renamed_local_to_index(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    content = b"renamed-local-content"
    root = _write_local_file(tmp_path, name="renamed.pdf", content=content)
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(root,))
    import_drive_page(store=store, records=(), next_page_token=None)
    import_index_inventory(
        store=store,
        sources=(
            _source_record(
                source_path="approved/original.pdf",
                content_sha256=hashlib.sha256(content).hexdigest(),
            ),
        ),
        locators=(_locator_record(),),
        expected_source_count=1,
        expected_chunk_count=1,
    )

    summary = reconcile_document_matches(store=store)

    assert summary.local_only == 1
    assert summary.index_only == 0
    store.close()


def _capacity_inventory(*, title_only_key: str | None = None):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        IndexLocatorRecord,
        IndexSourceRecord,
    )

    topics = ("iron_sulfide", "scale", "corrosion", "paraffin")
    roles = ("foundational", "supporting")
    sources = []
    locators = []
    for topic in topics:
        for role in roles:
            source_id = f"source-{topic}-{role}"
            sources.append(
                IndexSourceRecord.from_mapping(
                    {
                        "source_id": source_id,
                        "source_path": f"approved/{source_id}.pdf",
                        "parser_type": "pdf",
                        "topic": topic,
                        "chunk_count": 12,
                        "embedding_model": "model",
                        "index_contract_sha256": "a" * 64,
                        "provenance_drive_file_id": None,
                        "content_sha256": None,
                    }
                )
            )
            for index in range(1, 13):
                locator = f"page:{index:02d}"
                locator_key = f"{source_id}:{locator}"
                locators.append(
                    IndexLocatorRecord.from_mapping(
                        {
                            "source_id": source_id,
                            "locator": locator,
                            "topic": topic,
                            "source_role": role,
                            "substantive_status": (
                                "TITLE_ONLY"
                                if locator_key == title_only_key
                                else "SUBSTANTIVE"
                            ),
                        }
                    )
                )
    return tuple(sources), tuple(locators)


def _store_with_capacity_inventory(
    tmp_path: Path, *, title_only_key: str | None = None
):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        import_index_inventory,
    )

    store = _create_store(tmp_path)
    sources, locators = _capacity_inventory(title_only_key=title_only_key)
    import_index_inventory(
        store=store,
        sources=sources,
        locators=locators,
        expected_source_count=8,
        expected_chunk_count=96,
    )
    return store


def test_capacity_requires_twelve_fresh_locators_in_every_topic_role_cell(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        calculate_locator_capacity,
    )

    store = _store_with_capacity_inventory(tmp_path)

    report = calculate_locator_capacity(store=store, prior_locator_keys=())

    assert len(report.strata) == 8
    assert all(item.required_locators == 12 for item in report.strata)
    assert all(item.fresh_locator_count == 12 for item in report.strata)
    assert report.all_sufficient is True
    assert all("fresh_locator_count" not in item for item in report.to_public_mapping()["strata"])
    store.close()


def test_capacity_excludes_exact_prior_e1a3_locator_key(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        calculate_locator_capacity,
    )

    store = _store_with_capacity_inventory(tmp_path)
    prior_key = "source-scale-supporting:page:01"

    report = calculate_locator_capacity(
        store=store, prior_locator_keys=(prior_key,)
    )
    target = next(
        item
        for item in report.strata
        if item.topic == "scale" and item.source_role == "supporting"
    )

    assert target.fresh_locator_count == 11
    assert target.sufficient is False
    assert report.all_sufficient is False
    store.close()


def test_capacity_excludes_title_only_locator(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        calculate_locator_capacity,
    )

    title_only_key = "source-corrosion-foundational:page:12"
    store = _store_with_capacity_inventory(
        tmp_path, title_only_key=title_only_key
    )

    report = calculate_locator_capacity(store=store, prior_locator_keys=())
    target = next(
        item
        for item in report.strata
        if item.topic == "corrosion" and item.source_role == "foundational"
    )

    assert target.fresh_locator_count == 11
    assert target.sufficient is False
    store.close()


def test_capacity_uses_locator_level_topic_assignment(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        IndexLocatorRecord,
        calculate_locator_capacity,
        import_index_inventory,
    )

    store = _create_store(tmp_path)
    sources, locators = _capacity_inventory()
    changed = tuple(
        IndexLocatorRecord(
            source_id=locator.source_id,
            locator=locator.locator,
            topic=(
                "corrosion"
                if locator.source_id == "source-scale-supporting"
                and locator.locator == "page:12"
                else locator.topic
            ),
            source_role=locator.source_role,
            substantive_status=locator.substantive_status,
        )
        for locator in locators
    )
    import_index_inventory(
        store=store,
        sources=sources,
        locators=changed,
        expected_source_count=8,
        expected_chunk_count=96,
    )

    report = calculate_locator_capacity(store=store, prior_locator_keys=())
    scale = next(
        item
        for item in report.strata
        if item.topic == "scale" and item.source_role == "supporting"
    )
    corrosion = next(
        item
        for item in report.strata
        if item.topic == "corrosion" and item.source_role == "supporting"
    )

    assert scale.fresh_locator_count == 11
    assert corrosion.fresh_locator_count == 13
    store.close()


def test_dry_run_produces_exact_96_slots_without_sampling_frame_write(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        dry_run_e1a4_allocation,
    )

    store = _store_with_capacity_inventory(tmp_path)

    result = dry_run_e1a4_allocation(store=store, prior_locator_keys=())

    assert result.status == "COMPLETE"
    assert result.error_code is None
    assert len(result.allocations) == 96
    assert len({(item.source_id, item.locator) for item in result.allocations}) == 96
    assert not list(store.root.rglob("*sampling-frame*"))
    store.close()


def test_dry_run_fails_closed_when_one_stratum_is_unavailable(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        dry_run_e1a4_allocation,
    )

    store = _store_with_capacity_inventory(tmp_path)

    result = dry_run_e1a4_allocation(
        store=store,
        prior_locator_keys=("source-paraffin-supporting:page:01",),
    )

    assert result.status == "BLOCKED"
    assert result.error_code == "CORPUS_RECONCILIATION_E1A4_ALLOCATION_UNAVAILABLE"
    assert result.allocations == ()
    store.close()


def _complete_snapshot_store(tmp_path: Path, *, include_dry_run: bool = True):
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        calculate_locator_capacity,
        dry_run_e1a4_allocation,
        import_drive_page,
        import_index_inventory,
        inventory_local_files,
        reconcile_document_matches,
    )

    empty_root = tmp_path / "approved"
    empty_root.mkdir()
    store = _create_store(tmp_path)
    inventory_local_files(store=store, roots=(empty_root,))
    import_drive_page(
        store=store,
        records=(
            _drive_record(drive_file_id="drive-2", checksum=None),
            _drive_record(drive_file_id="drive-1", checksum=None),
        ),
        next_page_token=None,
    )
    sources, locators = _capacity_inventory()
    import_index_inventory(
        store=store,
        sources=sources,
        locators=locators,
        expected_source_count=8,
        expected_chunk_count=96,
    )
    reconcile_document_matches(store=store)
    calculate_locator_capacity(store=store, prior_locator_keys=())
    if include_dry_run:
        dry_run = dry_run_e1a4_allocation(store=store, prior_locator_keys=())
        assert dry_run.status == "COMPLETE"
    return store


def test_snapshot_seal_requires_completed_e1a4_dry_run(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        seal_reconciliation_snapshots,
    )

    store = _complete_snapshot_store(tmp_path, include_dry_run=False)

    with pytest.raises(
        CorpusReconciliationError,
        match="CORPUS_RECONCILIATION_SNAPSHOT_INPUT_INCOMPLETE",
    ):
        seal_reconciliation_snapshots(store=store, root=store.root)

    store.close()


def test_snapshot_set_is_canonical_complete_and_idempotent(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        seal_reconciliation_snapshots,
        verify_reconciliation_snapshots,
    )

    store = _complete_snapshot_store(tmp_path)

    sealed = seal_reconciliation_snapshots(store=store, root=store.root)
    timestamps = {artifact.path.name: artifact.path.stat().st_mtime_ns for artifact in sealed.artifacts}
    verified = verify_reconciliation_snapshots(root=store.root)
    resealed = seal_reconciliation_snapshots(store=store, root=store.root)

    assert len(sealed.artifacts) == 6
    assert verified == sealed
    assert resealed == sealed
    assert {
        path.name for path in (store.root / "snapshots").glob("*.sha256")
    } == {f"{artifact.path.name}.sha256" for artifact in sealed.artifacts}
    assert {
        artifact.path.name: artifact.path.stat().st_mtime_ns
        for artifact in resealed.artifacts
    } == timestamps
    drive_lines = (store.root / "snapshots" / "drive-inventory.jsonl").read_text(
        encoding="utf-8"
    )
    assert drive_lines.endswith("\n")
    assert [json.loads(line)["drive_file_id"] for line in drive_lines.splitlines()] == [
        "drive-1",
        "drive-2",
    ]
    store.close()


def test_snapshot_verification_rejects_partial_set(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        verify_reconciliation_snapshots,
    )

    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "drive-inventory.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_SNAPSHOT_PARTIAL"):
        verify_reconciliation_snapshots(root=root)


def test_snapshot_verification_rejects_forbidden_privacy_key(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        seal_reconciliation_snapshots,
        verify_reconciliation_snapshots,
    )

    store = _complete_snapshot_store(tmp_path)
    seal_reconciliation_snapshots(store=store, root=store.root)
    snapshot = store.root / "snapshots" / "drive-inventory.jsonl"
    payload = json.loads(snapshot.read_text(encoding="utf-8").splitlines()[0])
    payload["content"] = "forbidden"
    tampered = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    snapshot.write_bytes(tampered)
    snapshot.with_name(f"{snapshot.name}.sha256").write_text(
        hashlib.sha256(tampered).hexdigest() + "\n", encoding="ascii"
    )

    with pytest.raises(CorpusReconciliationError, match="CORPUS_RECONCILIATION_SNAPSHOT_SCHEMA_INVALID"):
        verify_reconciliation_snapshots(root=store.root)
    store.close()


def test_snapshot_verification_rejects_locator_without_source(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        seal_reconciliation_snapshots,
        verify_reconciliation_snapshots,
    )

    store = _complete_snapshot_store(tmp_path)
    seal_reconciliation_snapshots(store=store, root=store.root)
    snapshot = store.root / "snapshots" / "index-inventory.jsonl"
    records = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines()]
    source_id = next(record["source_id"] for record in records if record["record_type"] == "source")
    records = [
        record
        for record in records
        if not (record["record_type"] == "source" and record["source_id"] == source_id)
    ]
    tampered = (
        "\n".join(
            sorted(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records)
        )
        + "\n"
    ).encode()
    snapshot.write_bytes(tampered)
    snapshot.with_name(f"{snapshot.name}.sha256").write_text(
        hashlib.sha256(tampered).hexdigest() + "\n", encoding="ascii"
    )

    with pytest.raises(
        CorpusReconciliationError,
        match="CORPUS_RECONCILIATION_SNAPSHOT_RELATIONSHIP_INVALID",
    ):
        verify_reconciliation_snapshots(root=store.root)

    store.close()


def test_snapshot_verification_rejects_match_without_source(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        seal_reconciliation_snapshots,
        verify_reconciliation_snapshots,
    )

    store = _complete_snapshot_store(tmp_path)
    seal_reconciliation_snapshots(store=store, root=store.root)
    snapshot = store.root / "snapshots" / "document-matches.jsonl"
    records = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines()]
    record = next(item for item in records if item["source_id"] is not None)
    record["source_id"] = "missing-source"
    tampered = (
        "\n".join(
            sorted(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in records)
        )
        + "\n"
    ).encode()
    snapshot.write_bytes(tampered)
    snapshot.with_name(f"{snapshot.name}.sha256").write_text(
        hashlib.sha256(tampered).hexdigest() + "\n", encoding="ascii"
    )

    with pytest.raises(
        CorpusReconciliationError,
        match="CORPUS_RECONCILIATION_SNAPSHOT_RELATIONSHIP_INVALID",
    ):
        verify_reconciliation_snapshots(root=store.root)

    store.close()


def test_snapshot_verification_rejects_capacity_not_derived_from_locators(
    tmp_path: Path,
) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        seal_reconciliation_snapshots,
        verify_reconciliation_snapshots,
    )

    store = _complete_snapshot_store(tmp_path)
    seal_reconciliation_snapshots(store=store, root=store.root)
    snapshot = store.root / "snapshots" / "locator-capacity.jsonl"
    records = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines()]
    records[0]["fresh_locator_count"] += 1
    tampered = (
        "\n".join(
            sorted(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in records)
        )
        + "\n"
    ).encode()
    snapshot.write_bytes(tampered)
    snapshot.with_name(f"{snapshot.name}.sha256").write_text(
        hashlib.sha256(tampered).hexdigest() + "\n", encoding="ascii"
    )

    with pytest.raises(
        CorpusReconciliationError,
        match="CORPUS_RECONCILIATION_SNAPSHOT_RELATIONSHIP_INVALID",
    ):
        verify_reconciliation_snapshots(root=store.root)

    store.close()


def test_snapshot_verification_rejects_decision_without_match(tmp_path: Path) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        CorpusReconciliationError,
        seal_reconciliation_snapshots,
        verify_reconciliation_snapshots,
    )

    store = _complete_snapshot_store(tmp_path)
    seal_reconciliation_snapshots(store=store, root=store.root)
    snapshot = store.root / "snapshots" / "review-decisions.jsonl"
    tampered = (
        json.dumps(
            {
                "decision": "ACCEPT",
                "decided_at": "2026-08-24T00:00:00Z",
                "decision_id": "decision-1",
                "match_key": "missing-match",
                "reason_code": "REVIEWED",
                "reviewer_id": "reviewer-1",
                "supersedes_decision_id": None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    snapshot.write_bytes(tampered)
    snapshot.with_name(f"{snapshot.name}.sha256").write_text(
        hashlib.sha256(tampered).hexdigest() + "\n", encoding="ascii"
    )

    with pytest.raises(
        CorpusReconciliationError,
        match="CORPUS_RECONCILIATION_SNAPSHOT_RELATIONSHIP_INVALID",
    ):
        verify_reconciliation_snapshots(root=store.root)

    store.close()


def test_snapshot_seal_rolls_back_complete_set_after_mid_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import oilfield_chemical_copilot.evaluation.corpus_reconciliation as reconciliation

    store = _complete_snapshot_store(tmp_path)
    real_replace = reconciliation.os.replace
    calls = 0

    def fail_on_third_publish(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated private path that must not escape")
        real_replace(source, destination)

    monkeypatch.setattr(reconciliation.os, "replace", fail_on_third_publish)

    with pytest.raises(
        reconciliation.CorpusReconciliationError,
        match="CORPUS_RECONCILIATION_SNAPSHOT_WRITE_FAILED",
    ):
        reconciliation.seal_reconciliation_snapshots(store=store, root=store.root)

    snapshots = store.root / "snapshots"
    assert not tuple(snapshots.glob("*.jsonl"))
    assert not tuple(snapshots.glob("*.sha256"))
    assert not tuple(snapshots.glob(".*.tmp"))
    store.close()


def _corpus_runner_module() -> object:
    path = Path(__file__).resolve().parents[2] / "eval" / "reconcile_private_corpus.py"
    spec = importlib.util.spec_from_file_location("test_reconcile_private_corpus", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_caller_selected_reconciliation_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _corpus_runner_module()
    controller_root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    public_root = tmp_path / "docs" / "reconciliation"
    monkeypatch.setattr(runner, "DEFAULT_ROOT", controller_root)
    monkeypatch.setattr(runner, "_index_contract_digest", lambda _path: "a" * 64)
    monkeypatch.setattr(runner, "_load_prior_locator_keys", lambda _root: ())
    monkeypatch.setattr(runner, "_allocation_digest", lambda _root: "b" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_private_corpus.py",
            "init",
            "--private-root",
            str(public_root),
        ],
    )

    assert runner.cli() == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "CORPUS_RECONCILIATION_BLOCKED",
        "error_code": "CORPUS_RECONCILIATION_PRIVATE_ROOT_INVALID",
    }
    assert not (public_root / "reconciliation.sqlite").exists()


def test_cli_rejects_missing_prerequisites_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _corpus_runner_module()
    root = tmp_path / ".private" / "corpus-reconciliation" / "v1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_private_corpus.py",
            "status",
            "--private-root",
            str(root),
            "--run-id",
            "run-001",
        ],
    )

    assert runner.cli() == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "CORPUS_RECONCILIATION_BLOCKED",
        "error_code": "CORPUS_RECONCILIATION_PREREQUISITES_MISSING",
    }
    assert "Traceback" not in captured.err
    assert str(root) not in captured.err


def test_cli_rejects_invalid_drive_stdin_without_echoing_private_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _corpus_runner_module()
    store = _create_store(tmp_path)
    root = store.root
    store.close()
    monkeypatch.setattr(runner, "DEFAULT_ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_private_corpus.py",
            "import-drive-page",
            "--private-root",
            str(root),
            "--run-id",
            "run-001",
        ],
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO('{"records":[],"next_page_token":null,"content":"private"}'),
    )

    assert runner.cli() == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    error = json.loads(captured.err)
    assert error == {
        "status": "CORPUS_RECONCILIATION_BLOCKED",
        "error_code": "CORPUS_RECONCILIATION_DRIVE_STDIN_INVALID",
    }
    assert "private" not in captured.err
    assert str(root) not in captured.err


def test_cli_status_is_aggregate_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _corpus_runner_module()
    store = _create_store(tmp_path)
    root = store.root
    store.close()
    monkeypatch.setattr(runner, "DEFAULT_ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_private_corpus.py",
            "status",
            "--private-root",
            str(root),
            "--run-id",
            "run-001",
        ],
    )

    assert runner.cli() == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert set(payload) == {"status", "counts", "stages", "snapshots_complete"}
    assert set(payload["counts"]) == {
        "drive_files",
        "local_files",
        "index_sources",
        "index_locators",
        "document_matches",
        "review_decisions",
    }
    assert str(root) not in captured.out
    assert "run-001" not in captured.out
    assert "a" * 64 not in captured.out


def test_cli_status_rejects_corrupted_complete_snapshot_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from oilfield_chemical_copilot.evaluation.corpus_reconciliation import (
        seal_reconciliation_snapshots,
    )

    runner = _corpus_runner_module()
    store = _complete_snapshot_store(tmp_path)
    root = store.root
    seal_reconciliation_snapshots(store=store, root=root)
    snapshot = root / "snapshots" / "drive-inventory.jsonl"
    snapshot.write_bytes(snapshot.read_bytes() + b"{}\n")
    store.close()
    monkeypatch.setattr(runner, "DEFAULT_ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_private_corpus.py",
            "status",
            "--private-root",
            str(root),
            "--run-id",
            "run-001",
        ],
    )

    assert runner.cli() == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "CORPUS_RECONCILIATION_BLOCKED",
        "error_code": "CORPUS_RECONCILIATION_SNAPSHOT_DIGEST_MISMATCH",
    }


def test_index_connection_verifies_contract_before_read_only_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _corpus_runner_module()
    events = []
    sentinel = object()
    contract = tmp_path / "index-contract.json"

    def verify(**kwargs):
        events.append(("verify", kwargs))
        return object()

    def connect(database_url: str, **kwargs):
        events.append(("connect", database_url, kwargs))
        return sentinel

    monkeypatch.setattr(runner, "verify_e1_index_contract", verify)
    monkeypatch.setattr(runner.psycopg, "connect", connect)

    connection = runner._verified_read_only_connection(
        database_url="postgresql://private",
        index_contract=contract,
    )

    assert connection is sentinel
    assert [event[0] for event in events] == ["verify", "connect"]
    assert events[1][2]["options"] == "-c default_transaction_read_only=on"
