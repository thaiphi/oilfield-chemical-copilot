from __future__ import annotations

import json
from pathlib import Path

import pytest

from oilfield_chemical_copilot.evaluation.citation_diagnostics import (
    LocalCitationDiagnostic,
    write_local_citation_diagnostics,
)


def _record() -> LocalCitationDiagnostic:
    return LocalCitationDiagnostic(
        question_id="public-case",
        policy_action="allow",
        policy_category="general_review",
        allowed_evidence_ids=("allowed",),
        retrieved_evidence_ids=("allowed", "other"),
        cited_evidence_ids=("allowed",),
        abstained=False,
        generation_outcome="succeeded",
    )


def test_local_diagnostic_writer_serializes_only_ids_and_state_flags(tmp_path: Path) -> None:
    destination = tmp_path / "diagnostics.json"

    write_local_citation_diagnostics(
        {"vector": [_record()], "hybrid": [_record()]},
        destination,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == {
        "modes": {
            "hybrid": [
                {
                    "abstained": False,
                    "allowed_evidence_ids": ["allowed"],
                    "cited_evidence_ids": ["allowed"],
                    "generation_outcome": "succeeded",
                    "policy_action": "allow",
                    "policy_category": "general_review",
                    "question_id": "public-case",
                    "retrieved_evidence_ids": ["allowed", "other"],
                }
            ],
            "vector": [
                {
                    "abstained": False,
                    "allowed_evidence_ids": ["allowed"],
                    "cited_evidence_ids": ["allowed"],
                    "generation_outcome": "succeeded",
                    "policy_action": "allow",
                    "policy_category": "general_review",
                    "question_id": "public-case",
                    "retrieved_evidence_ids": ["allowed", "other"],
                }
            ],
        },
        "schema_version": 1,
    }


def test_local_diagnostic_writer_rejects_unknown_modes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^invalid local citation diagnostics$"):
        write_local_citation_diagnostics({"vector": [_record()]}, tmp_path / "diagnostics.json")
