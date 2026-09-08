"""Synthetic contract tests for private Corpus V2 input registers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oilfield_chemical_copilot.corpus_v2.ledger import CorpusV2Ledger
from oilfield_chemical_copilot.corpus_v2.models import ReleaseConfig, Stage
from oilfield_chemical_copilot.corpus_v2.registers import (
    CorpusV2RegisterError,
    initialize_registers,
    load_approved_register,
    load_critical_register,
    validate_critical_register,
)


def _private_jsonl(rows: list[dict[str, str]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def _sources(count: int = 385) -> list[dict[str, str]]:
    return [
        {
            "source_id": f"doc-{index}",
            "drive_file_id": f"drive-{index:03d}",
            "declared_mime_type": "application/pdf",
            "pinned_revision_token": f"revision-{index}",
            "steward_approval_binding": "approval-v1",
        }
        for index in range(1, count + 1)
    ]


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _config(root: Path, approved_payload: bytes, critical_payload: bytes = b"critical") -> ReleaseConfig:
    return ReleaseConfig(
        release_id="corpus-v2-2026-09-07",
        release_root=root,
        candidate_database_name="corpus_v2_candidate",
        configured_database_name="app_configured_database",
        legacy_database_names=("legacy_database",),
        expected_source_count=385,
        source_register_sha256=hashlib.sha256(approved_payload).hexdigest(),
        critical_source_register_sha256=hashlib.sha256(critical_payload).hexdigest(),
    )


def _critical_rows(source_digest: str, *source_ids: str) -> list[dict[str, str]]:
    return [
        {"source_id": source_id, "source_register_sha256": source_digest}
        for source_id in source_ids
    ]


def test_register_requires_exact_approved_count_and_unique_drive_ids(tmp_path: Path) -> None:
    with pytest.raises(CorpusV2RegisterError, match="C2_SOURCE_REGISTER_INVALID"):
        load_approved_register(_write(tmp_path / "approved.jsonl", _private_jsonl(_sources(384))))

    duplicate_drive = _sources()
    duplicate_drive[-1]["drive_file_id"] = duplicate_drive[0]["drive_file_id"]
    with pytest.raises(CorpusV2RegisterError, match="C2_SOURCE_REGISTER_INVALID"):
        load_approved_register(_write(tmp_path / "duplicate.jsonl", _private_jsonl(duplicate_drive)))


def test_register_rejects_unknown_keys_and_noncanonical_jsonl(tmp_path: Path) -> None:
    unknown_key = _sources()
    unknown_key[0]["path"] = "C:/private/secret.pdf"
    with pytest.raises(CorpusV2RegisterError, match="C2_SOURCE_REGISTER_INVALID"):
        load_approved_register(_write(tmp_path / "unknown.jsonl", _private_jsonl(unknown_key)))

    rows = _sources()
    noncanonical = json.dumps(rows[0], separators=(",", ":")).encode("utf-8") + b"\n"
    noncanonical += _private_jsonl(rows[1:])
    with pytest.raises(CorpusV2RegisterError, match="C2_SOURCE_REGISTER_INVALID"):
        load_approved_register(_write(tmp_path / "noncanonical.jsonl", noncanonical))


def test_critical_register_requires_sorted_unique_approved_sources_and_config_binding(
    tmp_path: Path,
) -> None:
    approved_payload = _private_jsonl(_sources())
    approved = load_approved_register(_write(tmp_path / "approved.jsonl", approved_payload))
    digest = hashlib.sha256(approved_payload).hexdigest()

    with pytest.raises(CorpusV2RegisterError, match="C2_CRITICAL_REGISTER_INVALID"):
        validate_critical_register(
            approved,
            load_critical_register(
                _write(tmp_path / "outside.jsonl", _private_jsonl(_critical_rows(digest, "doc-999")))
            ),
            source_register_sha256=digest,
        )
    with pytest.raises(CorpusV2RegisterError, match="C2_CRITICAL_REGISTER_INVALID"):
        validate_critical_register(
            approved,
            load_critical_register(
                _write(tmp_path / "duplicate.jsonl", _private_jsonl(_critical_rows(digest, "doc-2", "doc-2")))
            ),
            source_register_sha256=digest,
        )
    with pytest.raises(CorpusV2RegisterError, match="C2_CRITICAL_REGISTER_INVALID"):
        validate_critical_register(
            approved,
            load_critical_register(
                _write(tmp_path / "unsorted.jsonl", _private_jsonl(_critical_rows(digest, "doc-2", "doc-1")))
            ),
            source_register_sha256=digest,
        )
    with pytest.raises(CorpusV2RegisterError, match="C2_CRITICAL_REGISTER_INVALID"):
        validate_critical_register(
            approved,
            load_critical_register(
                _write(tmp_path / "wrong-binding.jsonl", _private_jsonl(_critical_rows("a" * 64, "doc-1")))
            ),
            source_register_sha256=digest,
        )


def test_initializer_seals_private_registers_and_registers_all_sources(tmp_path: Path) -> None:
    private_root = tmp_path / "private-release"
    approved_payload = _private_jsonl(_sources())
    approved_path = _write(private_root / "registers" / "approved.jsonl", approved_payload)
    approved_digest = hashlib.sha256(approved_payload).hexdigest()
    critical_payload = _private_jsonl(_critical_rows(approved_digest, "doc-1", "doc-2"))
    config = _config(private_root, approved_payload, critical_payload)
    critical_path = _write(private_root / "registers" / "critical.jsonl", critical_payload)

    result = initialize_registers(
        release_config=config,
        approved_register_path=approved_path,
        critical_register_path=critical_path,
        approved_private_root=private_root,
        ledger_path=private_root / "ledger.sqlite",
        manifest_root=private_root / "manifests",
    )

    assert result.approved_source_count == 385
    assert result.critical_source_count == 2
    assert result.approved_register_sha256 == config.source_register_sha256
    assert result.manifest_path.read_bytes().endswith(b"\n")
    ledger = CorpusV2Ledger.open(private_root / "ledger.sqlite")
    assert ledger.completed_stages() == (Stage.REGISTERED,)
    assert sum(event.event_type == "source_registered" for event in ledger.event_history()) == 385


def test_initializer_rejects_path_traversal_digest_mismatch_and_repeat_initialization(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private-release"
    approved_payload = _private_jsonl(_sources())
    approved_path = _write(private_root / "registers" / "approved.jsonl", approved_payload)
    approved_digest = hashlib.sha256(approved_payload).hexdigest()
    critical_payload = _private_jsonl(_critical_rows(approved_digest, "doc-1"))
    config = _config(private_root, approved_payload, critical_payload)
    critical_path = _write(
        private_root / "registers" / "critical.jsonl",
        critical_payload,
    )
    outside = _write(tmp_path / "outside.jsonl", approved_payload)

    with pytest.raises(CorpusV2RegisterError, match="C2_REGISTER_PATH_INVALID"):
        initialize_registers(
            release_config=config,
            approved_register_path=outside,
            critical_register_path=critical_path,
            approved_private_root=private_root,
            ledger_path=private_root / "ledger.sqlite",
            manifest_root=private_root / "manifests",
        )

    wrong_config = _config(private_root, b"different")
    with pytest.raises(CorpusV2RegisterError, match="C2_REGISTER_DIGEST_MISMATCH"):
        initialize_registers(
            release_config=wrong_config,
            approved_register_path=approved_path,
            critical_register_path=critical_path,
            approved_private_root=private_root,
            ledger_path=private_root / "ledger.sqlite",
            manifest_root=private_root / "manifests",
        )

    wrong_critical_config = _config(private_root, approved_payload, b"wrong-critical")
    with pytest.raises(CorpusV2RegisterError, match="C2_REGISTER_DIGEST_MISMATCH"):
        initialize_registers(
            release_config=wrong_critical_config,
            approved_register_path=approved_path,
            critical_register_path=critical_path,
            approved_private_root=private_root,
            ledger_path=private_root / "ledger.sqlite",
            manifest_root=private_root / "manifests",
        )

    initialize_registers(
        release_config=config,
        approved_register_path=approved_path,
        critical_register_path=critical_path,
        approved_private_root=private_root,
        ledger_path=private_root / "ledger.sqlite",
        manifest_root=private_root / "manifests",
    )
    with pytest.raises(CorpusV2RegisterError, match="C2_REGISTER_ALREADY_INITIALIZED"):
        initialize_registers(
            release_config=config,
            approved_register_path=approved_path,
            critical_register_path=critical_path,
            approved_private_root=private_root,
            ledger_path=private_root / "ledger.sqlite",
            manifest_root=private_root / "manifests",
        )
