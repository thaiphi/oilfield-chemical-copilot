"""Private release contracts. No acquisition, database writes, or backup operations.

Publication requires an already authenticated, locked publication capability.
The operator acquires that capability through authenticated_publication_directory;
this module never weakens its no-replace or exact-tree semantics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from collections.abc import Mapping, Callable
from contextlib import AbstractContextManager

from .ledger import CorpusV2Ledger
from .models import ReleaseConfig, SourceDisposition, Stage
from ..evaluation.private_artifact_publication import AuthenticatedPublicationDirectory


class CorpusV2ReleaseError(ValueError):
    """A fixed-code failure that never exposes private inputs."""


REQUIRED_DIGESTS = frozenset({
    "source_register", "acquisition", "extraction", "dispositions", "chunks",
    "embeddings", "index_contract", "evaluation_specification", "evaluation_result",
    "promotion_state", "ledger_projection", "legacy_guard",
})
STAGE_DIGESTS = {
    Stage.ACQUIRED: "acquisition", Stage.CHUNKED: "chunks", Stage.EMBEDDED: "embeddings",
    Stage.INDEX_VALIDATED: "index_contract", Stage.EVALUATED: "evaluation_result",
    Stage.PROMOTION_READY: "promotion_state",
}
_LAYOUT = {"sealed": frozenset({f"{key}.bin" for key in REQUIRED_DIGESTS} | {"binding.json"})}
_FINAL_STATES = {state.value for state in SourceDisposition if state is not SourceDisposition.UNRESOLVED}


def _invalid() -> None:
    raise CorpusV2ReleaseError("C2_RELEASE_BINDING_INVALID")


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest_valid(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


@dataclass(frozen=True)
class LegacyFingerprint:
    source_count: int
    chunk_count: int
    read_only: bool

    def __post_init__(self) -> None:
        if (type(self.source_count) is not int or type(self.chunk_count) is not int
                or self.source_count != 198 or self.chunk_count != 4797
                or self.read_only is not True):
            raise CorpusV2ReleaseError("C2_LEGACY_FINGERPRINT_INVALID")


def capture_legacy_fingerprint(
    reader: Callable[..., AbstractContextManager[Mapping[str, object]]],
) -> LegacyFingerprint:
    """Inject a reader that establishes read-only default before connecting.

    No PostgreSQL connector is supplied here. The reader must yield only counts
    and its verified transaction-read-only result, never legacy source metadata.
    """
    try:
        with reader(default_transaction_read_only=True) as facts:
            if set(facts) != {"source_count", "chunk_count", "read_only"}:
                raise ValueError
            return LegacyFingerprint(**facts)
    except Exception:
        raise CorpusV2ReleaseError("C2_LEGACY_FINGERPRINT_INVALID") from None


@dataclass(frozen=True)
class OperatorRestoreEvidence:
    backup_sha256: str
    restore_succeeded: bool
    distinct_disposable_target: bool
    restored_fingerprint: LegacyFingerprint


def validate_operator_restore_evidence(evidence: OperatorRestoreEvidence) -> None:
    """Validate operator attestations; never execute a backup or restore."""
    if (not isinstance(evidence, OperatorRestoreEvidence)
            or not _digest_valid(evidence.backup_sha256)
            or evidence.restore_succeeded is not True
            or evidence.distinct_disposable_target is not True
            or not isinstance(evidence.restored_fingerprint, LegacyFingerprint)):
        raise CorpusV2ReleaseError("C2_RESTORE_EVIDENCE_INVALID")
    LegacyFingerprint(evidence.restored_fingerprint.source_count,
                      evidence.restored_fingerprint.chunk_count,
                      evidence.restored_fingerprint.read_only)


@dataclass(frozen=True)
class LegacyGuardAttestation:
    release_id: str
    legacy_database_names: tuple[str, ...]
    fingerprint: LegacyFingerprint
    operator_evidence: OperatorRestoreEvidence


def legacy_guard_bytes(attestation: LegacyGuardAttestation) -> bytes:
    """Private operator attestation, bound into the sealed evidence digest set."""
    validate_operator_restore_evidence(attestation.operator_evidence)
    if attestation.fingerprint != attestation.operator_evidence.restored_fingerprint:
        _invalid()
    return _json(asdict(attestation))


def _validate_legacy_guard(config: ReleaseConfig, content: bytes) -> None:
    facts = json.loads(content)
    evidence = dict(facts["operator_evidence"])
    evidence["restored_fingerprint"] = LegacyFingerprint(**evidence["restored_fingerprint"])
    attestation = LegacyGuardAttestation(
        facts["release_id"], tuple(facts["legacy_database_names"]),
        LegacyFingerprint(**facts["fingerprint"]), OperatorRestoreEvidence(**evidence),
    )
    if (attestation.release_id != config.release_id
            or attestation.legacy_database_names != config.legacy_database_names
            or legacy_guard_bytes(attestation) != content):
        _invalid()


def ledger_projection_bytes(ledger: CorpusV2Ledger) -> bytes:
    return _json(ledger.release_projection())


def _final_dispositions(projection: Mapping) -> bytes:
    sources = {row["source_id"] for row in projection["source_register"]}
    current = {}
    for decision in sorted(projection["disposition_decisions"], key=lambda row: row["decision_id"]):
        current[decision["source_id"]] = decision
    if (not sources or set(current) != sources
            or any(row["disposition"] not in _FINAL_STATES for row in current.values())):
        _invalid()
    return _json([current[source] for source in sorted(sources)])


def final_dispositions_bytes(ledger: CorpusV2Ledger) -> bytes:
    return _final_dispositions(ledger.release_projection())


def _binding(config: ReleaseConfig, ledger: CorpusV2Ledger, artifacts: Mapping[str, bytes]) -> bytes:
    if (set(artifacts) != REQUIRED_DIGESTS
            or any(type(value) is not bytes or not value for value in artifacts.values())):
        _invalid()
    _validate_legacy_guard(config, artifacts["legacy_guard"])
    projection = ledger.release_projection()
    metadata = projection["release"]
    if (len(metadata) != 1 or metadata[0]["release_id"] != config.release_id
            or metadata[0]["expected_source_count"] != config.expected_source_count
            or metadata[0]["candidate_database_name"] != config.candidate_database_name
            or metadata[0]["configured_database_name"] != config.configured_database_name
            or metadata[0]["source_register_sha256"] != config.source_register_sha256
            or metadata[0]["critical_source_register_sha256"] != config.critical_source_register_sha256
            or json.loads(metadata[0]["legacy_database_names_json"]) != list(config.legacy_database_names)
            or len(projection["source_register"]) != config.expected_source_count
            or artifacts["ledger_projection"] != _json(projection)
            or artifacts["dispositions"] != _final_dispositions(projection)):
        _invalid()
    completed = {row["stage"] for row in projection["stage_checkpoints"]}
    if completed != {stage.value for stage in tuple(Stage)[:-1]}:
        _invalid()
    digests = {key: sha256(value).hexdigest() for key, value in artifacts.items()}
    if digests["source_register"] != config.source_register_sha256:
        _invalid()
    stage_artifacts = {row["stage"]: row["artifact_sha256"] for row in projection["stage_artifacts"]}
    if (set(stage_artifacts) != {stage.value for stage in STAGE_DIGESTS}
            or any(stage_artifacts[stage.value] != digests[key] for stage, key in STAGE_DIGESTS.items())):
        _invalid()
    return _json({"schema_version": 1, "release_id": config.release_id,
                  "candidate_database_name": config.candidate_database_name,
                  "critical_source_register_sha256": config.critical_source_register_sha256,
                  "digests": digests})


def seal_release_binding(*, config: ReleaseConfig, ledger: CorpusV2Ledger,
                         artifacts: Mapping[str, bytes],
                         publication: AuthenticatedPublicationDirectory) -> str:
    """Seal the exact promotion-ready evidence set via the locked capability.

    This freezes evidence, not permission to promote. A changed ledger always
    requires separate evidence; an existing release directory is never replaced.
    """
    try:
        with ledger.locked_release():
            return _seal_locked(config=config, ledger=ledger, artifacts=artifacts,
                                publication=publication)
    except Exception:
        raise CorpusV2ReleaseError("C2_RELEASE_BINDING_INVALID") from None


def _seal_locked(*, config, ledger, artifacts, publication) -> str:
    try:
        artifacts = dict(artifacts)
        binding = _binding(config, ledger, artifacts)
        publication.ensure_no_staging(".release.", ".tmp")
        if publication.final_exists(config.release_id):
            _invalid()
        staging = publication.create_staging(".release.", ".tmp")
        staging.mkdir("sealed")
        for key in sorted(artifacts):
            staging.write_exclusive(f"sealed/{key}.bin", artifacts[key])
        staging.write_exclusive("sealed/binding.json", binding)
        staging.sync_directory("sealed")
        staging.sync_root()
        staged_members = publication.read_exact_tree(staging.name, _LAYOUT)
        expected_members = {f"sealed/{key}.bin": value for key, value in artifacts.items()}
        expected_members["sealed/binding.json"] = binding
        if staged_members != expected_members:
            _invalid()
        if _binding(config, ledger, artifacts) != binding:
            _invalid()
        publication.publish_no_replace(staging, config.release_id)
        publication.sync_parent()
        return verify_release_binding(config=config, ledger=ledger, publication=publication,
                                      expected_sha256=sha256(binding).hexdigest())
    except Exception:
        raise CorpusV2ReleaseError("C2_RELEASE_BINDING_INVALID") from None


def verify_release_binding(*, config: ReleaseConfig, ledger: CorpusV2Ledger,
                           publication: AuthenticatedPublicationDirectory,
                           expected_sha256: str) -> str:
    try:
        if not _digest_valid(expected_sha256):
            _invalid()
        publication.ensure_no_staging(".release.", ".tmp")
        members = publication.read_exact_tree(config.release_id, _LAYOUT)
        artifacts = {key: members[f"sealed/{key}.bin"] for key in REQUIRED_DIGESTS}
        binding = _binding(config, ledger, artifacts)
        if (members["sealed/binding.json"] != binding
                or sha256(binding).hexdigest() != expected_sha256):
            _invalid()
        return expected_sha256
    except Exception:
        raise CorpusV2ReleaseError("C2_RELEASE_BINDING_INVALID") from None


def build_public_status(aggregates: Mapping[str, int] | None = None) -> str:
    """Pre-build closure only; an operational report needs a separate contract."""
    expected = {"legacy_sources": 198, "legacy_chunks": 4797,
                "candidate_sources": 0, "candidate_chunks": 0}
    supplied = expected if aggregates is None else dict(aggregates)
    if (supplied != expected or any(type(value) is not int for value in supplied.values())):
        raise CorpusV2ReleaseError("C2_PUBLIC_STATUS_INVALID")
    return (
        "Reconciliation closed with findings: a legacy-index provenance gap remains.\n"
        "Phase: implementation contracts. No V2 build has occurred.\n"
        "Legacy baseline: 198 sources / 4,797 chunks.\n"
        "Candidate index: 0 sources / 0 chunks. Final V2 dispositions: not recorded.\n"
        "Operational gate: BLOCKED pending release-owner authorization.\n"
        "Next approval gate: Gate 2, controlled acquisition and candidate build.\n"
    )
