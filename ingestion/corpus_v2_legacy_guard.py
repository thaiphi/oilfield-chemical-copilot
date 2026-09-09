"""Injectable legacy preflight. Deliberately no live CLI or backup commands."""

from oilfield_chemical_copilot.corpus_v2.release import (
    OperatorRestoreEvidence,
    LegacyGuardAttestation,
    capture_legacy_fingerprint,
    validate_operator_restore_evidence,
)


def legacy_guard(reader, *, config, operator_evidence: OperatorRestoreEvidence):
    """Require both operator backup/restore evidence and read-only baseline."""
    validate_operator_restore_evidence(operator_evidence)
    fingerprint = capture_legacy_fingerprint(reader)
    return LegacyGuardAttestation(config.release_id, config.legacy_database_names,
                                  fingerprint, operator_evidence)
