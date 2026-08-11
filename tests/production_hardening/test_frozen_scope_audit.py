from __future__ import annotations

import hashlib
import json
from pathlib import Path

from production_hardening import frozen_scope_audit


def test_manifest_records_only_fixed_digest_keys(tmp_path: Path) -> None:
    policy = tmp_path / "policy.py"
    fixture = tmp_path / "fixture.jsonl"
    policy.write_bytes(b"policy-bytes")
    fixture.write_bytes(b"fixture-bytes")
    manifest = tmp_path / "manifest.json"

    frozen_scope_audit.write_frozen_scope_manifest(manifest, policy, fixture)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload == {
        "policy_sha256": hashlib.sha256(b"policy-bytes").hexdigest(),
        "public_fixture_sha256": hashlib.sha256(b"fixture-bytes").hexdigest(),
    }
    assert all("path" not in key and "content" not in key for key in payload)


def test_audit_preserves_scope_only_when_both_current_digests_match(tmp_path: Path) -> None:
    policy = tmp_path / "policy.py"
    fixture = tmp_path / "fixture.jsonl"
    policy.write_bytes(b"policy-bytes")
    fixture.write_bytes(b"fixture-bytes")
    manifest = tmp_path / "manifest.json"
    frozen_scope_audit.write_frozen_scope_manifest(manifest, policy, fixture)

    assert frozen_scope_audit.audit_frozen_scope(manifest, policy, fixture).frozen_scope_preserved is True
    fixture.write_bytes(b"changed-fixture-bytes")
    summary = frozen_scope_audit.audit_frozen_scope(manifest, policy, fixture)
    assert (summary.policy_digest_verified, summary.public_fixture_digest_verified, summary.frozen_scope_preserved) == (True, False, False)


def test_audit_detects_a_policy_change_without_consulting_git(tmp_path: Path) -> None:
    policy = tmp_path / "policy.py"
    fixture = tmp_path / "fixture.jsonl"
    policy.write_bytes(b"policy-bytes")
    fixture.write_bytes(b"fixture-bytes")
    manifest = tmp_path / "manifest.json"
    frozen_scope_audit.write_frozen_scope_manifest(manifest, policy, fixture)
    policy.write_bytes(b"changed-policy-bytes")

    summary = frozen_scope_audit.audit_frozen_scope(manifest, policy, fixture)

    assert (summary.policy_digest_verified, summary.public_fixture_digest_verified, summary.frozen_scope_preserved) == (False, True, False)


def test_default_locations_name_the_exact_frozen_artifacts() -> None:
    assert frozen_scope_audit.POLICY_SOURCE_PATH == Path("src/oilfield_chemical_copilot/evaluation/abstention_policy.py")
    assert frozen_scope_audit.PUBLIC_FIXTURE_PATH == Path("eval/public_answer_evaluation.jsonl")
