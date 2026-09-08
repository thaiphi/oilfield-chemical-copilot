"""Injectable legacy preflight. Deliberately no live CLI or backup commands."""

from oilfield_chemical_copilot.corpus_v2.release import (
    OperatorRestoreEvidence,
    capture_legacy_fingerprint,
    validate_operator_restore_evidence,
)


def legacy_guard(reader, *, operator_evidence: OperatorRestoreEvidence):
    """Require both operator backup/restore evidence and read-only baseline."""
    validate_operator_restore_evidence(operator_evidence)
    return capture_legacy_fingerprint(reader)
