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
from oilfield_chemical_copilot.corpus_v2 import drive
from oilfield_chemical_copilot.corpus_v2.ledger import CorpusV2Ledger, CorpusV2LedgerError
from oilfield_chemical_copilot.corpus_v2.models import AcquisitionRecord, ApprovedSource, ReleaseConfig, Stage
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
        acquire_source(_entry(), client, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    assert not (tmp_path / "snapshots").exists()
    ledger.close()


def test_acquisition_hashes_downloaded_bytes_and_uses_only_registered_identity(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    client = FakeDriveClient(payload=b"approved")

    record = acquire_source(_entry(), client, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)

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
        acquire_source(_entry(mime_type=native), client, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    allowed = _entry(mime_type=native, export_mime="application/pdf")
    record = acquire_source(allowed, client, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)
    assert record.mime_type == "application/pdf"
    assert client.download_calls == [("drive-1", "application/pdf")]
    ledger.close()


def test_resume_is_idempotent_and_rejects_mismatched_observed_facts(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    first_client = FakeDriveClient(payload=b"approved")
    first = acquire_source(_entry(), first_client, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    resumed = acquire_source(_entry(), FakeDriveClient(payload=b"different"), release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)
    assert resumed == first
    assert (tmp_path / first.relative_snapshot_path).read_bytes() == b"approved"

    changed = FakeDriveClient(metadata=DriveFileMetadata("drive-1", "application/pdf", "revision-2"))
    with pytest.raises(CorpusV2AcquisitionError, match="C2_ACQUISITION_RESUME_MISMATCH"):
        acquire_source(_entry(), changed, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)
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

    with pytest.raises(CorpusV2AcquisitionError, match="C2_ACQUISITION_REGISTER_MISMATCH"):
        acquire_registered_sources(
            entries[:1],
            client_for_source=lambda entry: clients[entry.drive_file_id],
            release_root=tmp_path,
            snapshot_root=tmp_path / "snapshots",
            ledger=ledger,
        )
    assert ledger.completed_stages() == (Stage.REGISTERED,)

    records = acquire_registered_sources(
        entries,
        client_for_source=lambda entry: clients[entry.drive_file_id],
        release_root=tmp_path,
        snapshot_root=tmp_path / "snapshots",
        ledger=ledger,
    )
    assert {record.source_id for record in records} == {"doc-1", "doc-2"}

    ledger.close()


def test_failure_cleans_staging_and_errors_never_disclose_credentials(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    client = FakeDriveClient(download_error=RuntimeError("token=secret-value"))

    with pytest.raises(CorpusV2AcquisitionError, match="C2_DRIVE_DOWNLOAD_FAILED") as error:
        acquire_source(_entry(), client, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    assert "secret-value" not in str(error.value)
    snapshots = tmp_path / "snapshots"
    assert not snapshots.exists() or not list(snapshots.iterdir())
    ledger.close()


def test_acquisition_rejects_snapshot_root_outside_trusted_release(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    with pytest.raises(CorpusV2AcquisitionError, match="C2_SNAPSHOT_PATH_INVALID"):
        acquire_source(
            _entry(),
            FakeDriveClient(),
            release_root=tmp_path,
            snapshot_root=tmp_path.parent / "outside-snapshots",
            ledger=ledger,
        )
    ledger.close()


def test_acquisition_collision_never_overwrites_existing_snapshot(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    target = tmp_path / "snapshots" / "doc-1.blob"
    target.parent.mkdir()
    target.write_bytes(b"existing")

    with pytest.raises(CorpusV2AcquisitionError, match="C2_SNAPSHOT_DUPLICATE"):
        acquire_source(_entry(), FakeDriveClient(), release_root=tmp_path, snapshot_root=target.parent, ledger=ledger)

    assert target.read_bytes() == b"existing"
    ledger.close()


def test_crash_after_publish_resumes_without_duplicate_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _ledger(tmp_path)
    client = FakeDriveClient()

    def fail_record(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected")

    monkeypatch.setattr(ledger, "record_snapshot_acquisition", fail_record)
    with pytest.raises(CorpusV2AcquisitionError, match="C2_ACQUISITION_RECORD_FAILED"):
        acquire_source(_entry(), client, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)
    assert (tmp_path / "snapshots" / "doc-1.blob").is_file()
    assert len(client.download_calls) == 1

    monkeypatch.undo()
    recovered = acquire_source(_entry(), client, release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)
    assert recovered.byte_count == len(b"approved")
    assert len(client.download_calls) == 1
    ledger.close()


def test_post_download_revision_change_and_nonbytes_output_fail_closed(tmp_path: Path) -> None:
    class RevisionChangingClient(FakeDriveClient):
        def metadata(self, file_id: str) -> DriveFileMetadata:
            self.metadata_calls.append(file_id)
            revision = "revision-1" if len(self.metadata_calls) == 1 else "revision-2"
            return DriveFileMetadata(file_id, "application/pdf", revision)

    ledger = _ledger(tmp_path)
    with pytest.raises(CorpusV2AcquisitionError, match="C2_DRIVE_REVISION_CHANGED"):
        acquire_source(_entry(), RevisionChangingClient(), release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)
    assert not (tmp_path / "snapshots" / "doc-1.blob").exists()
    ledger.close()

    second_root = tmp_path / "second"
    second_root.mkdir()
    second = _ledger(second_root)
    class NonBytesClient(FakeDriveClient):
        def download(self, file_id: str, export_mime_type: str | None) -> bytes:
            del file_id, export_mime_type
            return "not-bytes"  # type: ignore[return-value]

    with pytest.raises(CorpusV2AcquisitionError, match="C2_SNAPSHOT_PARTIAL_OUTPUT"):
        acquire_source(_entry(), NonBytesClient(), release_root=second_root, snapshot_root=second_root / "snapshots", ledger=second)
    second.close()


def test_client_factory_error_is_sanitized(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    with pytest.raises(CorpusV2AcquisitionError, match="C2_DRIVE_CLIENT_INVALID") as error:
        acquire_registered_sources(
            (_entry(),),
            client_for_source=lambda entry: (_ for _ in ()).throw(RuntimeError("token=secret-value")),
            release_root=tmp_path,
            snapshot_root=tmp_path / "snapshots",
            ledger=ledger,
        )
    assert "secret-value" not in str(error.value)
    ledger.close()


def test_full_register_acquisition_records_artifact_then_completes_acquired_stage(tmp_path: Path) -> None:
    source_ids = tuple(f"doc-{index}" for index in range(1, 386))
    ledger = _ledger(tmp_path, source_ids=source_ids)
    entries = tuple(_entry(source_id=source_id, file_id=f"drive-{index}") for index, source_id in enumerate(source_ids, 1))

    def client_for_source(entry: ApprovedSourceRegisterEntry) -> FakeDriveClient:
        return FakeDriveClient(metadata=DriveFileMetadata(entry.drive_file_id, "application/pdf", "revision-1"))

    records = acquire_registered_sources(
        entries,
        client_for_source=client_for_source,
        release_root=tmp_path,
        snapshot_root=tmp_path / "snapshots",
        ledger=ledger,
    )

    assert len(records) == 385
    assert ledger.completed_stages() == (Stage.REGISTERED, Stage.ACQUIRED)
    assert any(event.event_type == "stage_artifact_recorded" for event in ledger.event_history())
    ledger.close()


def test_nonregular_target_is_rejected_and_never_replaced(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    target = tmp_path / "snapshots" / "doc-1.blob"
    target.mkdir(parents=True)

    with pytest.raises(CorpusV2AcquisitionError, match="C2_SNAPSHOT_NONREGULAR"):
        acquire_source(_entry(), FakeDriveClient(), release_root=tmp_path, snapshot_root=target.parent, ledger=ledger)
    assert target.is_dir()
    ledger.close()


def test_ledger_rejects_mixed_separator_or_noncanonical_snapshot_routes(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    acquisition = AcquisitionRecord("doc-1", "2026-09-07T00:00:00Z", SHA, 1)
    for path in ("snapshots\\doc-1.blob", "C:/snapshots/doc-1.blob", "../snapshots/doc-1.blob"):
        with pytest.raises(CorpusV2LedgerError, match="C2_ACQUISITION_INVALID"):
            ledger.record_snapshot_acquisition(
                acquisition,
                snapshot_relative_path=path,
                mime_type="application/pdf",
                revision_token="revision-1",
            )
    ledger.close()


def test_snapshot_root_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-snapshots"
    outside.mkdir(exist_ok=True)
    alias = tmp_path / "snapshots"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform")
    ledger = _ledger(tmp_path)
    with pytest.raises(CorpusV2AcquisitionError, match="C2_SNAPSHOT_PATH_INVALID"):
        acquire_source(_entry(), FakeDriveClient(), release_root=tmp_path, snapshot_root=alias, ledger=ledger)
    ledger.close()


@pytest.mark.parametrize(
    ("marker_source_id", "marker_release_id"),
    (("doc-2", "corpus-v2-test"), ("doc-1", "corpus-v2-other")),
)
def test_recovery_rejects_marker_not_bound_to_exact_source_and_release(
    tmp_path: Path, marker_source_id: str, marker_release_id: str
) -> None:
    ledger = _ledger(tmp_path)
    target = tmp_path / "snapshots" / "doc-1.blob"
    target.parent.mkdir()
    payload = b"approved"
    target.write_bytes(payload)
    drive._pending_path(target).write_bytes(  # noqa: SLF001
        drive._pending_payload(  # noqa: SLF001
            source_id=marker_source_id,
            release_id=marker_release_id,
            owner_nonce="a" * 32,
            digest=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
            mime_type="application/pdf",
            revision_token="revision-1",
        )
    )

    with pytest.raises(CorpusV2AcquisitionError, match="C2_ACQUISITION_RESUME_MISMATCH"):
        acquire_source(_entry(), FakeDriveClient(), release_root=tmp_path, snapshot_root=target.parent, ledger=ledger)
    ledger.close()


def test_existing_pending_marker_is_never_erased_by_another_attempt(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    pending = tmp_path / "snapshots" / ".doc-1.pending.json"
    pending.parent.mkdir()
    original = b'{"concurrent":"marker"}'
    pending.write_bytes(original)

    with pytest.raises(CorpusV2AcquisitionError, match="C2_SNAPSHOT_DUPLICATE"):
        acquire_source(_entry(), FakeDriveClient(), release_root=tmp_path, snapshot_root=pending.parent, ledger=ledger)
    assert pending.read_bytes() == original
    ledger.close()


def test_temp_cleanup_failure_after_publish_preserves_marker_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _ledger(tmp_path)
    original_unlink = drive.os.unlink
    failed_once = False

    def fail_one_temp_unlink(path: str | bytes, *args: object, **kwargs: object) -> None:
        nonlocal failed_once
        if not failed_once and str(path).endswith(".tmp"):
            failed_once = True
            raise OSError("injected temp unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(drive.os, "unlink", fail_one_temp_unlink)
    with pytest.raises(CorpusV2AcquisitionError, match="C2_SNAPSHOT_PUBLISH_FAILED"):
        acquire_source(_entry(), FakeDriveClient(), release_root=tmp_path, snapshot_root=tmp_path / "snapshots", ledger=ledger)

    target = tmp_path / "snapshots" / "doc-1.blob"
    pending = drive._pending_path(target)  # noqa: SLF001
    assert target.is_file()
    assert pending.is_file()
    monkeypatch.setattr(drive.os, "unlink", original_unlink)
    recovered = acquire_source(_entry(), FakeDriveClient(), release_root=tmp_path, snapshot_root=target.parent, ledger=ledger)
    assert recovered.source_id == "doc-1"
    ledger.close()
