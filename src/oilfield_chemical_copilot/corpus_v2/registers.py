"""Strict, private-input contracts for steward-approved Corpus V2 registers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from .ledger import CorpusV2Ledger, CorpusV2LedgerError
from .models import ApprovedSource, ReleaseConfig, Stage, is_valid_public_source_id


class CorpusV2RegisterError(ValueError):
    """Raised when a private source-register contract is malformed."""


_DRIVE_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
_MIME_TYPE = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_PRIVATE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_APPROVED_KEYS = frozenset(
    {
        "source_id",
        "drive_file_id",
        "declared_mime_type",
        "pinned_revision_token",
        "steward_approval_binding",
    }
)
_APPROVED_EXPORT_KEYS = _APPROVED_KEYS | {"approved_export_mime_type"}
_CRITICAL_KEYS = frozenset({"source_id", "source_register_sha256"})


@dataclass(frozen=True)
class ApprovedSourceRegisterEntry:
    """One private, steward-approved source identity without content or path data."""

    source_id: str
    drive_file_id: str
    declared_mime_type: str
    pinned_revision_token: str
    steward_approval_binding: str
    approved_export_mime_type: str | None = None

    def __post_init__(self) -> None:
        if not is_valid_public_source_id(self.source_id):
            raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
        if not _DRIVE_FILE_ID.fullmatch(self.drive_file_id):
            raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
        if not _MIME_TYPE.fullmatch(self.declared_mime_type):
            raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
        if not _PRIVATE_TOKEN.fullmatch(self.pinned_revision_token):
            raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
        if not _PRIVATE_TOKEN.fullmatch(self.steward_approval_binding):
            raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
        if self.approved_export_mime_type is not None and not _MIME_TYPE.fullmatch(
            self.approved_export_mime_type
        ):
            raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ApprovedSourceRegisterEntry":
        if not isinstance(payload, Mapping) or frozenset(payload) not in {
            _APPROVED_KEYS,
            _APPROVED_EXPORT_KEYS,
        }:
            raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
        try:
            export_mime = payload.get("approved_export_mime_type")
            if export_mime is not None and not isinstance(export_mime, str):
                raise TypeError
            return cls(
                source_id=_required_string(payload, "source_id"),
                drive_file_id=_required_string(payload, "drive_file_id"),
                declared_mime_type=_required_string(payload, "declared_mime_type"),
                pinned_revision_token=_required_string(payload, "pinned_revision_token"),
                steward_approval_binding=_required_string(payload, "steward_approval_binding"),
                approved_export_mime_type=export_mime,
            )
        except (KeyError, TypeError, CorpusV2RegisterError):
            raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID") from None

    def to_mapping(self) -> dict[str, str]:
        record = {
            "source_id": self.source_id,
            "drive_file_id": self.drive_file_id,
            "declared_mime_type": self.declared_mime_type,
            "pinned_revision_token": self.pinned_revision_token,
            "steward_approval_binding": self.steward_approval_binding,
        }
        if self.approved_export_mime_type is not None:
            record["approved_export_mime_type"] = self.approved_export_mime_type
        return record

    @property
    def record_sha256(self) -> str:
        return hashlib.sha256(_canonical_line(self.to_mapping()) + b"\n").hexdigest()


@dataclass(frozen=True)
class CriticalSourceRegisterEntry:
    """A private critical-source reference bound to the approved-register digest."""

    source_id: str
    source_register_sha256: str

    def __post_init__(self) -> None:
        if not is_valid_public_source_id(self.source_id) or not _SHA256.fullmatch(
            self.source_register_sha256
        ):
            raise CorpusV2RegisterError("C2_CRITICAL_REGISTER_INVALID")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CriticalSourceRegisterEntry":
        if not isinstance(payload, Mapping) or frozenset(payload) != _CRITICAL_KEYS:
            raise CorpusV2RegisterError("C2_CRITICAL_REGISTER_INVALID")
        try:
            return cls(
                source_id=_required_string(payload, "source_id"),
                source_register_sha256=_required_string(payload, "source_register_sha256"),
            )
        except (KeyError, TypeError, CorpusV2RegisterError):
            raise CorpusV2RegisterError("C2_CRITICAL_REGISTER_INVALID") from None

    def to_mapping(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_register_sha256": self.source_register_sha256,
        }


@dataclass(frozen=True)
class RegisterInitialization:
    """Safe aggregate facts returned after a private register seal succeeds."""

    approved_source_count: int
    critical_source_count: int
    approved_register_sha256: str
    critical_register_sha256: str
    manifest_path: Path


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str) or not value:
        raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
    return value


def _json_object(line: bytes, *, error_code: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        decoded = line.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise CorpusV2RegisterError(error_code) from None
    if not isinstance(value, dict) or _canonical_line(value) != line:
        raise CorpusV2RegisterError(error_code)
    return value


def _canonical_line(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _read_strict_jsonl(path: Path, *, error_code: str) -> tuple[Mapping[str, Any], ...]:
    payload = _read_bytes(path, error_code=error_code)
    if not payload or not payload.endswith(b"\n"):
        raise CorpusV2RegisterError(error_code)
    lines = payload[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise CorpusV2RegisterError(error_code)
    return tuple(_json_object(line, error_code=error_code) for line in lines)


def _read_bytes(path: Path, *, error_code: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        raise CorpusV2RegisterError(error_code) from None


def load_approved_register(path: Path) -> tuple[ApprovedSourceRegisterEntry, ...]:
    """Load exactly 385 canonical, private approved-register rows."""
    entries = tuple(
        ApprovedSourceRegisterEntry.from_mapping(row)
        for row in _read_strict_jsonl(path, error_code="C2_SOURCE_REGISTER_INVALID")
    )
    if len(entries) != 385:
        raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
    if len({entry.source_id for entry in entries}) != len(entries):
        raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
    if len({entry.drive_file_id for entry in entries}) != len(entries):
        raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
    return entries


def load_critical_register(path: Path) -> tuple[CriticalSourceRegisterEntry, ...]:
    """Load a nonempty canonical private critical-source register."""
    entries = tuple(
        CriticalSourceRegisterEntry.from_mapping(row)
        for row in _read_strict_jsonl(path, error_code="C2_CRITICAL_REGISTER_INVALID")
    )
    if not entries:
        raise CorpusV2RegisterError("C2_CRITICAL_REGISTER_INVALID")
    return entries


def validate_critical_register(
    approved_sources: Iterable[ApprovedSourceRegisterEntry],
    critical_sources: Iterable[CriticalSourceRegisterEntry],
    *,
    source_register_sha256: str,
) -> tuple[CriticalSourceRegisterEntry, ...]:
    """Verify critical membership, sortedness, uniqueness, and source-register binding."""
    approved = tuple(approved_sources)
    critical = tuple(critical_sources)
    if not _SHA256.fullmatch(source_register_sha256) or not critical:
        raise CorpusV2RegisterError("C2_CRITICAL_REGISTER_INVALID")
    if any(entry.source_register_sha256 != source_register_sha256 for entry in critical):
        raise CorpusV2RegisterError("C2_CRITICAL_REGISTER_INVALID")
    source_ids = tuple(entry.source_id for entry in critical)
    if source_ids != tuple(sorted(source_ids)) or len(set(source_ids)) != len(source_ids):
        raise CorpusV2RegisterError("C2_CRITICAL_REGISTER_INVALID")
    approved_ids = {entry.source_id for entry in approved}
    if not set(source_ids).issubset(approved_ids):
        raise CorpusV2RegisterError("C2_CRITICAL_REGISTER_INVALID")
    return critical


def _resolved_under_root(path: Path, root: Path) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError):
        raise CorpusV2RegisterError("C2_REGISTER_PATH_INVALID") from None
    return resolved_path


def _target_under_root(path: Path, root: Path) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=False)
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError):
        raise CorpusV2RegisterError("C2_REGISTER_PATH_INVALID") from None
    return resolved_path


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise CorpusV2RegisterError("C2_REGISTER_MANIFEST_INVALID") from None


def _register_manifest(
    *,
    release_config: ReleaseConfig,
    approved_digest: str,
    critical_digest: str,
    critical_count: int,
) -> bytes:
    record = {
        "approved_register_sha256": approved_digest,
        "approved_source_count": 385,
        "critical_register_sha256": critical_digest,
        "critical_source_count": critical_count,
        "release_id": release_config.release_id,
    }
    return _canonical_line(record) + b"\n"


def initialize_registers(
    *,
    release_config: ReleaseConfig,
    approved_register_path: Path,
    critical_register_path: Path,
    approved_private_root: Path,
    ledger_path: Path,
    manifest_root: Path,
) -> RegisterInitialization:
    """Atomically initialize a new ledger from already-approved private registers.

    This function deliberately has no Drive, parser, or database-store dependency.
    """
    approved_path = _resolved_under_root(approved_register_path, approved_private_root)
    critical_path = _resolved_under_root(critical_register_path, approved_private_root)
    target_ledger = _target_under_root(ledger_path, approved_private_root)
    target_manifest_root = _target_under_root(manifest_root, approved_private_root)
    if release_config.release_root.resolve(strict=False) != approved_private_root.resolve(strict=True):
        raise CorpusV2RegisterError("C2_REGISTER_PATH_INVALID")
    if release_config.expected_source_count != 385:
        raise CorpusV2RegisterError("C2_SOURCE_REGISTER_INVALID")
    if target_ledger.exists():
        raise CorpusV2RegisterError("C2_REGISTER_ALREADY_INITIALIZED")

    approved_payload = _read_bytes(approved_path, error_code="C2_SOURCE_REGISTER_INVALID")
    approved_digest = hashlib.sha256(approved_payload).hexdigest()
    if approved_digest != release_config.source_register_sha256:
        raise CorpusV2RegisterError("C2_REGISTER_DIGEST_MISMATCH")
    critical_payload = _read_bytes(critical_path, error_code="C2_CRITICAL_REGISTER_INVALID")
    critical_digest = hashlib.sha256(critical_payload).hexdigest()
    if critical_digest != release_config.critical_source_register_sha256:
        raise CorpusV2RegisterError("C2_REGISTER_DIGEST_MISMATCH")
    approved_sources = load_approved_register(approved_path)
    critical_sources = validate_critical_register(
        approved_sources,
        load_critical_register(critical_path),
        source_register_sha256=release_config.source_register_sha256,
    )

    if target_manifest_root.exists():
        raise CorpusV2RegisterError("C2_REGISTER_ALREADY_INITIALIZED")
    target_manifest_root.mkdir(parents=True, exist_ok=False)
    temporary_ledger = target_ledger.with_name(f".{target_ledger.name}.{uuid.uuid4().hex}.tmp")
    try:
        ledger = CorpusV2Ledger.create(temporary_ledger, release_config=release_config)
        try:
            for source in approved_sources:
                ledger.record_source(
                    ApprovedSource(source_id=source.source_id, source_sha256=source.record_sha256)
                )
            ledger.complete_stage(Stage.REGISTERED)
        finally:
            ledger.close()
        os.replace(temporary_ledger, target_ledger)
        _atomic_write(
            target_manifest_root / "registers.manifest.v1.json",
            _register_manifest(
                release_config=release_config,
                approved_digest=approved_digest,
                critical_digest=critical_digest,
                critical_count=len(critical_sources),
            ),
        )
    except (CorpusV2LedgerError, OSError, ValueError):
        temporary_ledger.unlink(missing_ok=True)
        if target_ledger.exists():
            target_ledger.unlink(missing_ok=True)
        for child in target_manifest_root.iterdir():
            child.unlink(missing_ok=True)
        target_manifest_root.rmdir()
        raise CorpusV2RegisterError("C2_REGISTER_INITIALIZATION_INVALID") from None

    return RegisterInitialization(
        approved_source_count=len(approved_sources),
        critical_source_count=len(critical_sources),
        approved_register_sha256=approved_digest,
        critical_register_sha256=critical_digest,
        manifest_path=target_manifest_root / "registers.manifest.v1.json",
    )
