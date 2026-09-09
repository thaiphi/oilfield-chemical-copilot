"""Synthetic, ID-only immutable snapshot acquisition for Corpus V2.

This module deliberately defines no credential handling or Google SDK adapter.  An
operational adapter may be supplied only after a separately authorized review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
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
    del snapshot_root
    return f"snapshots/{source_id}.blob"


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return path.is_symlink()
    return path.is_symlink() or bool(attributes & 0x400)


def _validate_snapshot_root(*, release_root: Path, snapshot_root: Path) -> Path:
    """Accept only a real release-root/snapshots directory, never an alias or escape."""
    try:
        trusted_root = release_root.resolve(strict=True)
        if not trusted_root.is_dir() or _is_reparse_or_symlink(release_root):
            raise ValueError
        candidate = snapshot_root.resolve(strict=False)
        if candidate != trusted_root / "snapshots":
            raise ValueError
        relative = snapshot_root.absolute().relative_to(release_root.absolute())
        if relative.as_posix() != "snapshots":
            raise ValueError
        current = release_root.absolute()
        for component in relative.parts:
            current /= component
            if current.exists() and _is_reparse_or_symlink(current):
                raise ValueError
    except (OSError, ValueError):
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_PATH_INVALID") from None
    return candidate


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
        or _is_reparse_or_symlink(target)
    ):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RESUME_MISMATCH")
    try:
        payload = target.read_bytes()
    except OSError:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RESUME_MISMATCH") from None
    if len(payload) != facts.byte_count or hashlib.sha256(payload).hexdigest() != facts.content_sha256:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RESUME_MISMATCH")
    return _snapshot_record(facts)


def _load_pending_recovery(
    *,
    target: Path,
    entry: ApprovedSourceRegisterEntry,
    release_id: str,
    output_mime_type: str,
) -> tuple[str, int, str]:
    """Validate a controller-owned publish marker before recovering a crash gap."""
    pending = _pending_path(target)
    if not target.exists():
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_DUPLICATE")
    if not target.is_file() or target.is_symlink() or _is_reparse_or_symlink(target):
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_NONREGULAR")
    if not pending.is_file() or pending.is_symlink() or _is_reparse_or_symlink(pending):
        raise CorpusV2AcquisitionError("C2_SNAPSHOT_DUPLICATE")
    try:
        marker = json.loads(pending.read_text(encoding="utf-8"))
        payload = target.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RESUME_MISMATCH") from None
    digest = hashlib.sha256(payload).hexdigest()
    if (
        not isinstance(marker, dict)
        or set(marker) != {
            "byte_count", "content_sha256", "mime_type", "owner_nonce", "release_id",
            "revision_token", "source_id",
        }
        or marker.get("source_id") != entry.source_id
        or marker.get("release_id") != release_id
        or not isinstance(marker.get("owner_nonce"), str)
        or len(marker["owner_nonce"]) != 32
        or any(character not in "0123456789abcdef" for character in marker["owner_nonce"])
        or marker.get("content_sha256") != digest
        or marker.get("byte_count") != len(payload)
        or marker.get("mime_type") != output_mime_type
        or marker.get("revision_token") != entry.pinned_revision_token
    ):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RESUME_MISMATCH")
    return digest, len(payload), marker["owner_nonce"]


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory durability; directory handles are not portable on Windows."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _pending_path(target: Path) -> Path:
    return target.with_name(f".{target.stem}.pending.json")


def _pending_payload(
    *,
    source_id: str,
    release_id: str,
    owner_nonce: str,
    digest: str,
    byte_count: int,
    mime_type: str,
    revision_token: str,
) -> bytes:
    return json.dumps(
        {
            "byte_count": byte_count,
            "content_sha256": digest,
            "mime_type": mime_type,
            "owner_nonce": owner_nonce,
            "release_id": release_id,
            "revision_token": revision_token,
            "source_id": source_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_atomic_snapshot(
    target: Path,
    payload: bytes,
    *,
    source_id: str,
    release_id: str,
    mime_type: str,
    revision_token: str,
) -> tuple[str, int, str]:
    """Write, fsync, mark, and publish without replacement in one directory."""
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    pending = _pending_path(target)
    owner_nonce = uuid.uuid4().hex
    marker_owned = False
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        written = temporary.read_bytes()
        digest = hashlib.sha256(written).hexdigest()
        if len(written) != len(payload):
            raise CorpusV2AcquisitionError("C2_SNAPSHOT_PARTIAL_OUTPUT")
        if target.exists() or target.is_symlink() or pending.exists():
            raise CorpusV2AcquisitionError("C2_SNAPSHOT_DUPLICATE")
        with pending.open("xb") as handle:
            handle.write(_pending_payload(
                source_id=source_id,
                release_id=release_id,
                owner_nonce=owner_nonce,
                digest=digest,
                byte_count=len(written),
                mime_type=mime_type,
                revision_token=revision_token,
            ))
            handle.flush()
            os.fsync(handle.fileno())
        marker_owned = True
        os.link(temporary, target)
        published = True
        temporary.unlink()
        _fsync_directory(target.parent)
        if not target.is_file() or target.is_symlink() or _is_reparse_or_symlink(target):
            raise CorpusV2AcquisitionError("C2_SNAPSHOT_NONREGULAR")
        return digest, len(written), owner_nonce
    except CorpusV2AcquisitionError:
        _cleanup_before_publish(temporary, pending, owner_nonce, marker_owned, published)
        raise
    except OSError:
        _cleanup_before_publish(temporary, pending, owner_nonce, marker_owned, published)
        error_code = "C2_SNAPSHOT_PUBLISH_FAILED" if published else "C2_SNAPSHOT_WRITE_FAILED"
        raise CorpusV2AcquisitionError(error_code) from None


def _safe_unlink(path: Path) -> bool:
    """Attempt removal and report whether the path is definitely absent afterward."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return not path.exists()
    return not path.exists()


def _marker_has_owner(pending: Path, owner_nonce: str) -> bool:
    try:
        marker = json.loads(pending.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(marker, dict) and marker.get("owner_nonce") == owner_nonce


def _remove_owned_pending(pending: Path, owner_nonce: str) -> None:
    if _marker_has_owner(pending, owner_nonce):
        _safe_unlink(pending)


def _cleanup_before_publish(
    temporary: Path, pending: Path, owner_nonce: str, marker_owned: bool, published: bool
) -> None:
    _safe_unlink(temporary)
    if marker_owned and not published:
        _remove_owned_pending(pending, owner_nonce)


def _authenticate_entry(entry: ApprovedSourceRegisterEntry, ledger: CorpusV2Ledger) -> None:
    if not isinstance(entry, ApprovedSourceRegisterEntry):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_INVALID")
    try:
        approved_digest = ledger.registered_source_sha256(entry.source_id)
    except CorpusV2LedgerError:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_LEDGER_INVALID") from None
    if approved_digest != entry.record_sha256:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_MISMATCH")


def acquire_source(
    entry: ApprovedSourceRegisterEntry,
    client: DriveSourceClient,
    *,
    release_root: Path,
    snapshot_root: Path,
    ledger: CorpusV2Ledger,
) -> SnapshotAcquisitionRecord:
    """Acquire one approved identity using its pinned Drive facts and nothing else."""
    if not isinstance(entry, ApprovedSourceRegisterEntry):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_INVALID")
    _authenticate_entry(entry, ledger)
    snapshot_root = _validate_snapshot_root(release_root=release_root, snapshot_root=snapshot_root)
    relative_path = _relative_snapshot_path(snapshot_root, entry.source_id)
    try:
        release_id = ledger.release_id()
    except CorpusV2LedgerError:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_LEDGER_INVALID") from None
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

    target = snapshot_root / f"{entry.source_id}.blob"
    if target.exists() or target.is_symlink():
        digest, byte_count, owner_nonce = _load_pending_recovery(
            target=target,
            entry=entry,
            release_id=release_id,
            output_mime_type=output_mime_type,
        )
        try:
            ledger.record_snapshot_acquisition(
                AcquisitionRecord(entry.source_id, _timestamp(), digest, byte_count),
                snapshot_relative_path=relative_path,
                mime_type=output_mime_type,
                revision_token=entry.pinned_revision_token,
            )
        except Exception:
            raise CorpusV2AcquisitionError("C2_ACQUISITION_RECORD_FAILED") from None
        _remove_owned_pending(_pending_path(target), owner_nonce)
        facts = ledger.snapshot_acquisition(entry.source_id)
        if facts is None:
            raise CorpusV2AcquisitionError("C2_ACQUISITION_RECORD_FAILED")
        return _snapshot_record(facts)

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
    digest, byte_count, owner_nonce = _write_atomic_snapshot(
        target,
        payload,
        source_id=entry.source_id,
        release_id=release_id,
        mime_type=output_mime_type,
        revision_token=entry.pinned_revision_token,
    )
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
        if _safe_unlink(target):
            _remove_owned_pending(_pending_path(target), owner_nonce)
        raise
    except Exception:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RECORD_FAILED") from None
    facts = ledger.snapshot_acquisition(entry.source_id)
    if facts is None:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_RECORD_FAILED")
    _remove_owned_pending(_pending_path(target), owner_nonce)
    return _snapshot_record(facts)


def acquire_registered_sources(
    entries: tuple[ApprovedSourceRegisterEntry, ...],
    *,
    client_for_source: Callable[[ApprovedSourceRegisterEntry], DriveSourceClient],
    release_root: Path,
    snapshot_root: Path,
    ledger: CorpusV2Ledger,
) -> tuple[SnapshotAcquisitionRecord, ...]:
    """Resume a sealed register only when its exact acquisition identity set agrees."""
    for entry in entries:
        _authenticate_entry(entry, ledger)
    source_ids = tuple(entry.source_id for entry in entries)
    if len(source_ids) != len(set(source_ids)):
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_MISMATCH")
    existing_ids = set(ledger.registered_source_ids())
    if set(source_ids) != existing_ids:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_MISMATCH")
    records: list[SnapshotAcquisitionRecord] = []
    for entry in entries:
        try:
            client = client_for_source(entry)
        except Exception:
            raise CorpusV2AcquisitionError("C2_DRIVE_CLIENT_INVALID") from None
        records.append(
            acquire_source(
                entry, client, release_root=release_root, snapshot_root=snapshot_root, ledger=ledger
            )
        )
    if {record.source_id for record in records} != existing_ids:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_REGISTER_MISMATCH")
    try:
        artifact_sha256 = hashlib.sha256(
            b"".join(
                json.dumps(
                    {"byte_count": record.byte_count, "content_sha256": record.byte_sha256, "mime_type": record.mime_type, "revision_token": record.revision_token, "source_id": record.source_id},
                    sort_keys=True, separators=(",", ":")
                ).encode("utf-8") + b"\n"
                for record in sorted(records, key=lambda item: item.source_id)
            )
        ).hexdigest()
        from .models import Stage, StageArtifactKind

        ledger.record_stage_artifact(
            Stage.ACQUIRED, artifact_kind=StageArtifactKind.ACQUISITION_MANIFEST, artifact_sha256=artifact_sha256
        )
        ledger.complete_stage(Stage.ACQUIRED)
    except CorpusV2LedgerError:
        raise CorpusV2AcquisitionError("C2_ACQUISITION_STAGE_FAILED") from None
    return tuple(records)
