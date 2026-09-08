"""Synthetic-only tests for the Corpus V2 immutable Drive snapshot boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from oilfield_chemical_copilot.corpus_v2.drive import (
    CorpusV2AcquisitionError,
    DriveFileMetadata,
    acquire_registered_sources,
    acquire_source,
)
from oilfield_chemical_copilot.corpus_v2.ledger import CorpusV2Ledger
from oilfield_chemical_copilot.corpus_v2.models import ApprovedSource, ReleaseConfig, Stage
from oilfield_chemical_copilot.corpus_v2.registers import ApprovedSourceRegisterEntry


SHA = "a" * 64


class FakeDriveClient:
    """Deliberately exposes only ID-keyed metadata and byte download calls."""

    def __init__(
        self,
        *,
        metadata: DriveFileMetadata | None = None,
        payload: bytes = b"approved",
        download_error: Exception | None = None,
    ) -> None:
        self._metadata = metadata or DriveFileMetadata("drive-1", "application/pdf", "revision-1")
        self._payload = payload
        self._download_error = download_error
        self.metadata_calls: list[str] = []
        self.download_calls: list[tuple[str, str | None]] = []

    def metadata(self, file_id: str) -> DriveFileMetadata:
        self.metadata_calls.append(file_id)
        return self._metadata

    def download(self, file_id: str, export_mime_type: str | None) -> bytes:
        self.download_calls.append((file_id, export_mime_type))
        if self._download_error is not None:
            raise self._download_error
        return self._payload


def _entry(
    *,
    source_id: str = "doc-1",
    file_id: str = "drive-1",
    mime_type: str = "application/pdf",
    revision: str = "revision-1",
    export_mime: str | None = None,
) -> ApprovedSourceRegisterEntry:
    return ApprovedSourceRegisterEntry(
        source_id=source_id,
        drive_file_id=file_id,
        declared_mime_type=mime_type,
        pinned_revision_token=revision,
        steward_approval_binding="approval-v1",
        approved_export_mime_type=export_mime,
    )


def _ledger(tmp_path: Path, *, source_ids: tuple[str, ...] = ("doc-1",)) -> CorpusV2Ledger:
    config = ReleaseConfig(
        release_id="corpus-v2-test",
        release_root=tmp_path,
        candidate_database_name="candidate_database",
        configured_database_name="configured_database",
        legacy_database_names=("legacy_database",),
        expected_source_count=len(source_ids),
        source_register_sha256=SHA,
        critical_source_register_sha256=SHA,
    )
    ledger = CorpusV2Ledger.create(tmp_path / "ledger.sqlite", release_config=config)
    for source_id in source_ids:
        ledger.record_source(ApprovedSource(source_id=source_id, source_sha256=SHA))
    ledger.complete_stage(Stage.REGISTERED)
    return ledger


def test_acquire_blocks_when_drive_revision_changes(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    client = FakeDriveClient(metadata=DriveFileMetadata("drive-1", "application/pdf", "new-revision"))

    with pytest.raises(CorpusV2AcquisitionError, match="C2_DRIVE_REVISION_CHANGED"):
        acquire_source(_entry(), client, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    assert not (tmp_path / "snapshots").exists()
    ledger.close()


def test_acquisition_hashes_downloaded_bytes_and_uses_only_registered_identity(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    client = FakeDriveClient(payload=b"approved")

    record = acquire_source(_entry(), client, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    assert record.byte_sha256 == hashlib.sha256(b"approved").hexdigest()
    assert record.source_id == "doc-1"
    assert record.relative_snapshot_path == "snapshots/doc-1.blob"
    assert client.metadata_calls == ["drive-1", "drive-1"]
    assert client.download_calls == [("drive-1", None)]
    assert (tmp_path / record.relative_snapshot_path).read_bytes() == b"approved"
    ledger.close()


def test_native_export_requires_explicit_approved_export_mime(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    native = "application/vnd.google-apps.document"
    client = FakeDriveClient(metadata=DriveFileMetadata("drive-1", native, "revision-1"))

    with pytest.raises(CorpusV2AcquisitionError, match="C2_DRIVE_EXPORT_POLICY_INVALID"):
        acquire_source(_entry(mime_type=native), client, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    allowed = _entry(mime_type=native, export_mime="application/pdf")
    record = acquire_source(allowed, client, snapshot_root=tmp_path / "snapshots", ledger=ledger)
    assert record.mime_type == "application/pdf"
    assert client.download_calls == [("drive-1", "application/pdf")]
    ledger.close()


def test_resume_is_idempotent_and_rejects_mismatched_observed_facts(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    first_client = FakeDriveClient(payload=b"approved")
    first = acquire_source(_entry(), first_client, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    resumed = acquire_source(_entry(), FakeDriveClient(payload=b"different"), snapshot_root=tmp_path / "snapshots", ledger=ledger)
    assert resumed == first
    assert (tmp_path / first.relative_snapshot_path).read_bytes() == b"approved"

    changed = FakeDriveClient(metadata=DriveFileMetadata("drive-1", "application/pdf", "revision-2"))
    with pytest.raises(CorpusV2AcquisitionError, match="C2_ACQUISITION_RESUME_MISMATCH"):
        acquire_source(_entry(), changed, snapshot_root=tmp_path / "snapshots", ledger=ledger)
    ledger.close()


def test_acquire_registered_sources_requires_exact_register_acquisition_identity_equality(
    tmp_path: Path,
) -> None:
    entries = (_entry(source_id="doc-1", file_id="drive-1"), _entry(source_id="doc-2", file_id="drive-2"))
    ledger = _ledger(tmp_path, source_ids=("doc-1", "doc-2"))
    clients = {
        "drive-1": FakeDriveClient(metadata=DriveFileMetadata("drive-1", "application/pdf", "revision-1")),
        "drive-2": FakeDriveClient(metadata=DriveFileMetadata("drive-2", "application/pdf", "revision-1")),
    }

    records = acquire_registered_sources(
        entries,
        client_for_source=lambda entry: clients[entry.drive_file_id],
        snapshot_root=tmp_path / "snapshots",
        ledger=ledger,
    )
    assert {record.source_id for record in records} == {"doc-1", "doc-2"}

    with pytest.raises(CorpusV2AcquisitionError, match="C2_ACQUISITION_REGISTER_MISMATCH"):
        acquire_registered_sources(
            entries[:1],
            client_for_source=lambda entry: clients[entry.drive_file_id],
            snapshot_root=tmp_path / "snapshots",
            ledger=ledger,
        )
    ledger.close()


def test_failure_cleans_staging_and_errors_never_disclose_credentials(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    client = FakeDriveClient(download_error=RuntimeError("token=secret-value"))

    with pytest.raises(CorpusV2AcquisitionError, match="C2_DRIVE_DOWNLOAD_FAILED") as error:
        acquire_source(_entry(), client, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    assert "secret-value" not in str(error.value)
    snapshots = tmp_path / "snapshots"
    assert not snapshots.exists() or not list(snapshots.iterdir())
    ledger.close()
