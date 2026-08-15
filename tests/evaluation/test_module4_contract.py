from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import oilfield_chemical_copilot.evaluation.module4_contract as contract_module
from oilfield_chemical_copilot.evaluation.module4_contract import Module4ContractError


def _configure_private_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path]:
    private_root = tmp_path / ".private" / "evaluation" / "module4_handouts"
    monkeypatch.setattr(contract_module, "PRIVATE_ROOT", private_root)
    draft = private_root / "dataset" / "cases.jsonl"
    sealed = private_root / "sealed" / "cases.jsonl"
    digest = private_root / "sealed" / "cases.sha256"
    return private_root, draft, sealed, digest


def _record(*, reviewed: bool = True, question: str = "Synthetic local question.") -> dict[str, object]:
    return {
        "case_id": "case-01",
        "question": question,
        "topic": "scale",
        "expected_chunk_ids": ["chunk-01"],
        "expect_citations": True,
        "expect_abstention": False,
        "reviewed": reviewed,
    }


def _write_draft(path: Path, *records: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_seal_cases_canonicalizes_reviewed_local_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, draft, sealed, digest = _configure_private_root(tmp_path, monkeypatch)
    _write_draft(draft, _record())

    dataset_sha256 = contract_module.seal_cases(draft, sealed, digest)

    assert re.fullmatch(r"[0-9a-f]{64}", dataset_sha256)
    cases = contract_module.verify_seal(sealed, digest)
    assert cases == (
        contract_module.Module4Case(
            "case-01", "Synthetic local question.", "scale", ("chunk-01",), True, False, True
        ),
    )


def test_sealing_rejects_unreviewed_case_without_echoing_private_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, draft, sealed, digest = _configure_private_root(tmp_path, monkeypatch)
    _write_draft(draft, _record(reviewed=False, question="private-question-sentinel"))

    with pytest.raises(Module4ContractError, match="^REVIEW_REQUIRED$") as error:
        contract_module.seal_cases(draft, sealed, digest)

    assert "private-question-sentinel" not in str(error.value)


@pytest.mark.parametrize(
    ("records", "code"),
    [
        ((_record(), _record()), "DUPLICATE_CASE_ID"),
        (
            (
                _record()
                | {"expected_chunk_ids": ["chunk-01", "chunk-01"]},
            ),
            "EXPECTED_CHUNK_IDS_INVALID",
        ),
        (
            (
                _record()
                | {"expected_chunk_ids": [], "expect_citations": False, "expect_abstention": False},
            ),
            "CASE_EXPECTATION_INVALID",
        ),
        (
            (
                _record()
                | {"expected_chunk_ids": ["chunk-01"], "expect_citations": False, "expect_abstention": True},
            ),
            "CASE_EXPECTATION_INVALID",
        ),
    ],
)
def test_sealing_rejects_invalid_case_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: tuple[dict[str, object], ...],
    code: str,
) -> None:
    _, draft, sealed, digest = _configure_private_root(tmp_path, monkeypatch)
    _write_draft(draft, *records)

    with pytest.raises(Module4ContractError, match=f"^{code}$"):
        contract_module.seal_cases(draft, sealed, digest)


def test_sealing_rejects_paths_outside_private_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, sealed, digest = _configure_private_root(tmp_path, monkeypatch)
    draft = tmp_path / "outside.jsonl"
    _write_draft(draft, _record())

    with pytest.raises(Module4ContractError, match="^PRIVATE_BOUNDARY_VIOLATION$"):
        contract_module.seal_cases(draft, sealed, digest)


def test_verify_seal_rejects_changed_sealed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, draft, sealed, digest = _configure_private_root(tmp_path, monkeypatch)
    _write_draft(draft, _record())
    contract_module.seal_cases(draft, sealed, digest)
    sealed.write_bytes(sealed.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(Module4ContractError, match="^SEAL_DIGEST_MISMATCH$"):
        contract_module.verify_seal(sealed, digest)


def test_consume_one_shot_rejects_a_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root, _, _, _ = _configure_private_root(tmp_path, monkeypatch)
    state = private_root / "results" / "state.json"

    contract_module.consume_one_shot(state, "a" * 64)

    assert json.loads(state.read_text(encoding="utf-8")) == {"dataset_sha256": "a" * 64}
    with pytest.raises(Module4ContractError, match="^ATTEMPT_UNAVAILABLE$"):
        contract_module.consume_one_shot(state, "a" * 64)
