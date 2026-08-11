"""Digest-only baseline and audit for frozen Task 2 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re


POLICY_SOURCE_PATH = Path("src/oilfield_chemical_copilot/evaluation/abstention_policy.py")
PUBLIC_FIXTURE_PATH = Path("eval/public_answer_evaluation.jsonl")
MANIFEST_PATH = Path("production_hardening/frozen_scope_manifest.json")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class FrozenScopeAuditError(ValueError):
    """A sanitized frozen-scope audit error code."""


@dataclass(frozen=True)
class FrozenScopeManifest:
    policy_sha256: str
    public_fixture_sha256: str


@dataclass(frozen=True)
class FrozenScopeAuditSummary:
    policy_digest_verified: bool
    public_fixture_digest_verified: bool
    frozen_scope_preserved: bool


def _fail(code: str) -> None:
    raise FrozenScopeAuditError(code)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        _fail("FROZEN_SCOPE_READ_FAILURE")


def _validate_manifest(manifest: FrozenScopeManifest) -> None:
    values = (manifest.policy_sha256, manifest.public_fixture_sha256)
    if any(type(value) is not str or _DIGEST.fullmatch(value) is None for value in values):
        _fail("FROZEN_SCOPE_MANIFEST_INVALID")


def write_frozen_scope_manifest(
    destination: Path = MANIFEST_PATH,
    policy_path: Path = POLICY_SOURCE_PATH,
    public_fixture_path: Path = PUBLIC_FIXTURE_PATH,
) -> FrozenScopeManifest:
    manifest = FrozenScopeManifest(_sha256(policy_path), _sha256(public_fixture_path))
    try:
        destination.write_text(
            json.dumps({"policy_sha256": manifest.policy_sha256, "public_fixture_sha256": manifest.public_fixture_sha256}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        _fail("FROZEN_SCOPE_MANIFEST_WRITE_FAILURE")
    return manifest


def load_frozen_scope_manifest(path: Path = MANIFEST_PATH) -> FrozenScopeManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if type(payload) is not dict or set(payload) != {"policy_sha256", "public_fixture_sha256"}:
            _fail("FROZEN_SCOPE_MANIFEST_INVALID")
        manifest = FrozenScopeManifest(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        _fail("FROZEN_SCOPE_MANIFEST_INVALID")
    _validate_manifest(manifest)
    return manifest


def audit_frozen_scope(
    manifest_path: Path = MANIFEST_PATH,
    policy_path: Path = POLICY_SOURCE_PATH,
    public_fixture_path: Path = PUBLIC_FIXTURE_PATH,
) -> FrozenScopeAuditSummary:
    manifest = load_frozen_scope_manifest(manifest_path)
    policy_matches = hmac.compare_digest(_sha256(policy_path), manifest.policy_sha256)
    fixture_matches = hmac.compare_digest(_sha256(public_fixture_path), manifest.public_fixture_sha256)
    return FrozenScopeAuditSummary(policy_matches, fixture_matches, policy_matches and fixture_matches)
