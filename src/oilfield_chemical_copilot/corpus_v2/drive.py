"""Synthetic, ID-only immutable snapshot acquisition for Corpus V2.

This module deliberately defines no credential handling or Google SDK adapter.  An
operational adapter may be supplied only after a separately authorized review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
from typing import Callable, Protocol
import uuid

from .ledger import CorpusV2Ledger, CorpusV2LedgerError, SnapshotAcquisitionFacts
from .models import AcquisitionRecord
from .registers import ApprovedSourceRegisterEntry


class CorpusV2AcquisitionError(RuntimeError):
    """Raised when a source cannot safely be included in an immutable snapshot."""


@dataclass(frozen=True)
class DriveFileMetadata:
    """The exact ID-pinned facts needed before acquiring one file."""

    file_id: str
    mime_type: str
    revision_token: str


class DriveSourceClient(Protocol):
    """Narrow, injectable source client.  Identity is always the approved file ID."""

    def metadata(self, file_id: str) -> DriveFileMetadata:
        """Return current metadata for exactly ``file_id``."""

    def download(self, file_id: str, export_mime_type: str | None) -> bytes:
        """Return complete authenticated source bytes for exactly ``file_id``."""


@dataclass(frozen=True)
class SnapshotAcquisitionRecord:
    source_id: str
    relative_snapshot_path: str
    byte_sha256: str
    byte_count: int
    mime_type: str
    revision_token: str
    acquired_at: str


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _is_native(mime_type: str) -> bool:
    return mime_type.startswith("application/vnd.google-apps.")


def _validate_metadata(
    entry: ApprovedSourceRegisterEntry,
    metadata: object,
    *,
    resuming: bool,
) -> str:
    error_code = "C2_ACQUISITION_RESUME_MISMATCH" if resuming else "C2_DRIVE_METADATA_INVALID"
    if not isinstance(metadata, DriveFileMetadata):
        raise CorpusV2AcquisitionError(error_code)
    if metadata.file_id != entry.drive_file_id or metadata.mime_type != entry.declared_mime_type:
        raise CorpusV2AcquisitionError(error_code)
    if metadata.revision_token != entry.pinned_revision_token:
        raise CorpusV2AcquisitionError(
            "C2_ACQUISITION_RESUME_MISMATCH" if resuming else "C2_DRIVE_REVISION_CHANGED"
        )
    native = _is_native(metadata.mime_type)
    if native and entry.approved_export_mime_type is None:
        raise CorpusV2AcquisitionError("C2_DRIVE_EXPORT_POLICY_INVALID")
    if not native and entry.approved_export_mime_type is not None:
        raise CorpusV2AcquisitionError("C2_DRIVE_EXPORT_POLICY_INVALID")
    return entry.approved_export_mime_type if native else metadata.mime_type


def _relative_snapshot_path(snapshot_root: Path, source_id: str) -> str:
    root_name = snapshot_root.name
    if not root_name or root_name in {".", ".."}:
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_PATH_INVALID")
    return f"{root_name}/{source_id}.blob"


def _snapshot_record(facts: SnapshotAcquisitionFacts) -> SnapshotAcquisitionRecord:
    return SnapshotAcquisitionRecord(
        source_id=facts.source_id,
        relative_snapshot_path=facts.snapshot_relative_path,
        byte_sha256=facts.content_sha256,
        byte_count=facts.byte_count,
        mime_type=facts.mime_type,
        revision_token=facts.revision_token,
        acquired_at=facts.acquired_at,
    )


def _validate_existing_snapshot(
    *, facts: SnapshotAcquisitionFacts,
    entry: ApprovedSourceRegisterEntry,
    snapshot_root: Path,
    relative_path: str,
    observed_mime_type: str,
) -> SnapshotAcquisitionRecord:
    target = snapshot_root / f"{entry.source_id}.blob"
    if (
        facts.source_id != entry.source_id
        or facts.snapshot_relative_path != relative_path
        or facts.revision_token != entry.pinned_revision_token
        or facts.mime_type != observed_mime_type
        or not target.is_file()
        or target.is_symlink()
    ):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RESUME_MISMATCH")
    try:
        payload = target.read_bytes()
    except OSError:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RESUME_MISMATCH") from None
    if len(payload) != facts.byte_count or hashlib.sha256(payload).hexdigest() != facts.content_sha256:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RESUME_MISMATCH")
    return _snapshot_record(facts)


def _write_atomic_snapshot(target: Path, payload: bytes) -> tuple[str, int]:
    """Write, fsync, hash, then rename a private byte snapshot without replacement."""
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        written = temporary.read_bytes()
        digest = hashlib.sha256(written).hexdigest()
        if len(written) != len(payload):
            raise CorpusV2AcquisitionError("C2_SNAPSHOT_PARTIAL_OUTPUT")
        if target.exists() or target.is_symlink():
            raise CorpusV2AcquisitionError("C2_SNAPSHOT_DUPLICATE")
        os.rename(temporary, target)
        if not target.is_file() or target.is_symlink():
            raise CorpusV2AcquisitionError("C2_SNAPSHOT_NONREGULAR")
        return digest, len(written)
    except CorpusV2AcquisitionError:
        temporary.unlink(missing_ok=True)
        raise
    except OSError:
        temporary.unlink(missing_ok=True)
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_WRITE_FAILED") from None


def acquire_source(
    entry: ApprovedSourceRegisterEntry,
    client: DriveSourceClient,
    *,
    snapshot_root: Path,
    ledger: CorpusV2Ledger,
) -> SnapshotAcquisitionRecord:
    """Acquire one approved identity using its pinned Drive facts and nothing else."""
    if not isinstance(entry, ApprovedSourceRegisterEntry):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_INVALID")
    relative_path = _relative_snapshot_path(snapshot_root, entry.source_id)
    existing = ledger.snapshot_acquisition(entry.source_id)
    try:
        observed = client.metadata(entry.drive_file_id)
    except Exception:
        raise CorpusV2AcquisitionError("C2_DRIVE_METADATA_FAILED") from None
    output_mime_type = _validate_metadata(entry, observed, resuming=existing is not None)
    if existing is not None:
        return _validate_existing_snapshot(
            facts=existing,
            entry=entry,
            snapshot_root=snapshot_root,
            relative_path=relative_path,
            observed_mime_type=output_mime_type,
        )

    try:
        payload = client.download(entry.drive_file_id, entry.approved_export_mime_type)
    except Exception:
        raise CorpusV2AcquisitionError("C2_DRIVE_DOWNLOAD_FAILED") from None
    if not isinstance(payload, bytes):
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_PARTIAL_OUTPUT")
    try:
        snapshot_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_WRITE_FAILED") from None
    if not snapshot_root.is_dir() or snapshot_root.is_symlink():
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_NONREGULAR")
    target = snapshot_root / f"{entry.source_id}.blob"
    digest, byte_count = _write_atomic_snapshot(target, payload)
    try:
        post_download_metadata = client.metadata(entry.drive_file_id)
        _validate_metadata(entry, post_download_metadata, resuming=False)
        ledger.record_snapshot_acquisition(
            AcquisitionRecord(
                source_id=entry.source_id,
                acquired_at=_timestamp(),
                content_sha256=digest,
                byte_count=byte_count,
            ),
            snapshot_relative_path=relative_path,
            mime_type=output_mime_type,
            revision_token=entry.pinned_revision_token,
        )
    except CorpusV2AcquisitionError:
        target.unlink(missing_ok=True)
        raise
    except (CorpusV2LedgerError, OSError):
        target.unlink(missing_ok=True)
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RECORD_FAILED") from None
    facts = ledger.snapshot_acquisition(entry.source_id)
    if facts is None:
        target.unlink(missing_ok=True)
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RECORD_FAILED")
    return _snapshot_record(facts)


def acquire_registered_sources(
    entries: tuple[ApprovedSourceRegisterEntry, ...],
    *,
    client_for_source: Callable[[ApprovedSourceRegisterEntry], DriveSourceClient],
    snapshot_root: Path,
    ledger: CorpusV2Ledger,
) -> tuple[SnapshotAcquisitionRecord, ...]:
    """Resume a sealed register only when its exact acquisition identity set agrees."""
    source_ids = tuple(entry.source_id for entry in entries)
    if len(source_ids) != len(set(source_ids)):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_MISMATCH")
    existing_ids = set(ledger.registered_source_ids())
    if set(source_ids) != existing_ids:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_MISMATCH")
    return tuple(
        acquire_source(entry, client_for_source(entry), snapshot_root=snapshot_root, ledger=ledger)
        for entry in entries
    )
