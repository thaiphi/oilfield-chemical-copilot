from __future__ import annotations

import sqlite3
import hashlib
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
